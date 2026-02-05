#!/usr/bin/env python3
"""
worker.py — Sublobe pattern generator using GIMP-style mosaic algorithm

Algorithm (adapted from GIMP mosaic.c):
1. Generate leaf silhouette boundary
2. Define vein lines (midrib + primary + secondary veins)
3. Lay down a regular hexagonal grid over the leaf
4. grid_localize(): Move each grid vertex toward the nearest vein line
5. Build Voronoi cells from the displaced vertices
6. Clip cells to leaf boundary
7. DFM validate + subdivide failures
8. Render: fill cells, draw epoxy vein lines

Usage:
    python worker.py --worker-id 1 --batch-size 10
"""

import json
import math
import os
import uuid
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import argparse

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, LineString, Point, MultiPolygon, MultiLineString
from shapely.ops import split, unary_union, nearest_points
from shapely.validation import explain_validity
from scipy.spatial import Voronoi

WORKSPACE = Path(__file__).parent
CONFIG = None


def load_config():
    global CONFIG
    config_path = WORKSPACE / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            CONFIG = json.load(f)
    else:
        CONFIG = {
            "dfm_constraints": {"min_feature_width_mm": 3.0, "min_internal_radius_mm": 2.0, "blade_kerf_mm": 3.0},
            "output": {"render_width_px": 800, "render_height_px": 800,
                       "wood_colors": ["#D4A574", "#C4956A", "#B8860B", "#8B7355", "#6B4423",
                                       "#A0522D", "#CD853F", "#DEB887", "#D2691E", "#BC8F8F"],
                       "epoxy_color": "#1a1a1a", "gap_width_mm": 1.5},
            "segmentation": {"target_cells_min": 15, "target_cells_max": 35, "max_subdivision_iterations": 3}
        }
    return CONFIG


def setup_logging(worker_id: int):
    log_dir = WORKSPACE / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {worker_id}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_dir / f"worker_{worker_id}.log"), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# ============================================================================
# LEAF SILHOUETTES
# ============================================================================

def make_maple_leaf(cx: float, cy: float, size: float) -> Polygon:
    s = size * 0.45
    lobe_data = [
        (0, 1.0), (8, 0.65), (22, 0.85), (32, 0.50),
        (50, 0.78), (62, 0.42), (80, 0.55), (95, 0.30),
        (120, 0.25), (160, 0.20), (180, 0.35),
        (200, 0.20), (240, 0.25), (265, 0.30), (280, 0.55),
        (298, 0.42), (310, 0.78), (328, 0.50), (338, 0.85), (352, 0.65),
    ]
    points = []
    for angle_deg, r_frac in lobe_data:
        a = math.radians(90 - angle_deg)
        points.append((cx + r_frac * s * math.cos(a), cy - r_frac * s * math.sin(a)))
    poly = Polygon(points)
    return poly.buffer(0) if not poly.is_valid else poly


def make_simple_leaf(cx: float, cy: float, size: float) -> Polygon:
    s = size * 0.42
    points = []
    for i in range(48):
        angle = 2 * math.pi * i / 48
        r = s * (1 + 0.2 * math.sin(angle))
        if angle > math.pi:
            r *= max(0.1, 1.0 - 0.5 * ((angle - math.pi) / math.pi) ** 1.5)
        points.append((cx + r * math.cos(angle), cy - r * math.sin(angle) * 1.3))
    poly = Polygon(points)
    return poly.buffer(0) if not poly.is_valid else poly


def make_oak_leaf(cx: float, cy: float, size: float) -> Polygon:
    s = size * 0.35
    points = []
    for i in range(60):
        angle = 2 * math.pi * i / 60 - math.pi / 2
        base_r = s * 0.8
        lobe_r = s * 0.3 * abs(math.sin(7 * angle / 2))
        taper = max(0.25, 1.0 - 0.5 * max(0, (angle - math.pi * 0.3)) / (math.pi * 1.7))
        r = (base_r + lobe_r) * taper
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle) * 1.3))
    poly = Polygon(points)
    return poly.buffer(0) if not poly.is_valid else poly


def make_ellipse(cx: float, cy: float, size: float) -> Polygon:
    points = [(cx + size * 0.42 * math.cos(2 * math.pi * i / 48),
               cy + size * 0.33 * math.sin(2 * math.pi * i / 48)) for i in range(48)]
    return Polygon(points)


def get_random_silhouette(width: float, height: float) -> Tuple[Polygon, str]:
    cx, cy = width / 2, height / 2
    size = min(width, height) * 0.9
    choice = random.choice(["maple", "maple", "simple_leaf", "oak", "ellipse"])
    makers = {"maple": make_maple_leaf, "simple_leaf": make_simple_leaf,
              "oak": make_oak_leaf, "ellipse": make_ellipse}
    return makers[choice](cx, cy, size), choice


# ============================================================================
# VEIN GENERATION — defines the "gradient field" for grid_localize
# ============================================================================

def generate_veins(boundary: Polygon, silhouette_type: str,
                   num_primary: int = 5, num_secondary_per: int = 3) -> List[LineString]:
    """
    Generate vein lines inside the leaf boundary.
    These serve the role of image gradients in GIMP's mosaic —
    grid vertices will be attracted toward these lines.
    """
    bounds = boundary.bounds
    minx, miny, maxx, maxy = bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    w = maxx - minx
    h = maxy - miny

    veins = []

    # Stem point: bottom center
    stem_x, stem_y = cx, maxy - h * 0.02

    # Midrib: stem to top
    midrib = LineString([(stem_x, stem_y), (stem_x, miny + h * 0.05)])
    clipped = midrib.intersection(boundary)
    if not clipped.is_empty:
        for g in (clipped.geoms if hasattr(clipped, 'geoms') else [clipped]):
            if g.geom_type == 'LineString':
                veins.append(g)

    # Primary veins radiate from stem
    angle_spread = math.radians(140)
    base_angle = math.radians(90)

    for i in range(num_primary):
        t = ((i / (num_primary - 1)) - 0.5) if num_primary > 1 else 0
        angle = base_angle + t * angle_spread + random.uniform(-0.05, 0.05)
        vein_length = max(w, h) * 0.8

        # Curved vein with intermediate points
        pts = []
        for frac in np.linspace(0, 1, 8):
            curve = 0.05 * w * math.sin(frac * math.pi) * (1 if t > 0 else -1 if t < 0 else 0)
            dist = vein_length * frac
            vx = stem_x + math.cos(angle) * dist + curve * math.cos(angle + math.pi/2)
            vy = stem_y - math.sin(angle) * dist + curve * math.sin(angle + math.pi/2)
            pts.append((vx, vy))

        clipped = LineString(pts).intersection(boundary)
        if not clipped.is_empty:
            for g in (clipped.geoms if hasattr(clipped, 'geoms') else [clipped]):
                if g.geom_type == 'LineString':
                    veins.append(g)

        # Secondary veins branch off this primary
        for j in range(num_secondary_per):
            branch_frac = 0.2 + j * (0.6 / max(1, num_secondary_per - 1))
            bx = stem_x + math.cos(angle) * vein_length * branch_frac
            by = stem_y - math.sin(angle) * vein_length * branch_frac

            branch_angle = angle + (0.6 + random.uniform(-0.15, 0.15)) * (1 if j % 2 == 0 else -1)
            branch_len = vein_length * random.uniform(0.15, 0.30)
            bx2 = bx + math.cos(branch_angle) * branch_len
            by2 = by - math.sin(branch_angle) * branch_len

            clipped = LineString([(bx, by), (bx2, by2)]).intersection(boundary)
            if not clipped.is_empty:
                for g in (clipped.geoms if hasattr(clipped, 'geoms') else [clipped]):
                    if g.geom_type == 'LineString':
                        veins.append(g)

    return veins


# ============================================================================
# HEX GRID — equivalent to GIMP's grid_create_hexagons()
# ============================================================================

def create_hex_grid(boundary: Polygon, tile_size: float) -> np.ndarray:
    """Regular hexagonal grid of vertices covering the boundary."""
    minx, miny, maxx, maxy = boundary.bounds
    dx = tile_size * math.sqrt(3)
    dy = tile_size * 1.5
    margin = tile_size * 2

    vertices = []
    row = 0
    y = miny - margin
    while y < maxy + margin:
        x_offset = (dx / 2) if (row % 2) else 0
        x = minx - margin + x_offset
        while x < maxx + margin:
            vertices.append([x, y])
            x += dx
        y += dy
        row += 1

    return np.array(vertices) if vertices else np.empty((0, 2))


# ============================================================================
# GRID LOCALIZE — THE KEY STEP (GIMP's grid_localize adapted)
# ============================================================================

def grid_localize(vertices: np.ndarray, veins: List[LineString],
                  boundary: Polygon, neatness: float = 0.3,
                  attraction_radius: float = None) -> np.ndarray:
    """
    Move each grid vertex toward the nearest vein line.

    In GIMP: vertices move toward highest image gradient in local neighborhood.
    Here: vertices move toward nearest vein line, scaled by (1 - neatness).

    neatness=1.0 → perfectly regular grid (no deformation)
    neatness=0.0 → maximum attraction toward veins
    """
    if not veins or len(vertices) == 0:
        return vertices

    all_veins = unary_union(veins)
    displaced = vertices.copy()

    for i in range(len(vertices)):
        vx, vy = vertices[i]
        pt = Point(vx, vy)
        try:
            nearest_on_vein = nearest_points(pt, all_veins)[1]
            dist = pt.distance(nearest_on_vein)

            if attraction_radius and dist > attraction_radius:
                # Far from veins — small random jitter only
                jitter = attraction_radius * 0.08 * (1 - neatness)
                displaced[i] = [vx + random.uniform(-jitter, jitter),
                                vy + random.uniform(-jitter, jitter)]
            else:
                # Move toward vein
                move_frac = (1 - neatness)
                if attraction_radius and attraction_radius > 0:
                    proximity = 1.0 - (dist / attraction_radius)
                    move_frac *= (0.3 + 0.7 * proximity)

                displaced[i] = [vx + (nearest_on_vein.x - vx) * move_frac,
                                vy + (nearest_on_vein.y - vy) * move_frac]
        except Exception:
            jitter = 0.5
            displaced[i] = [vx + random.uniform(-jitter, jitter),
                            vy + random.uniform(-jitter, jitter)]

    return displaced


# ============================================================================
# VORONOI CELLS FROM DISPLACED VERTICES
# ============================================================================

def voronoi_cells_from_vertices(vertices: np.ndarray, boundary: Polygon) -> List[Polygon]:
    """Build Voronoi cells from displaced grid vertices, clipped to boundary."""
    minx, miny, maxx, maxy = boundary.bounds
    w, h = maxx - minx, maxy - miny
    margin = max(w, h) * 0.3

    mask = ((vertices[:, 0] > minx - margin) & (vertices[:, 0] < maxx + margin) &
            (vertices[:, 1] > miny - margin) & (vertices[:, 1] < maxy + margin))
    nearby = vertices[mask]

    if len(nearby) < 4:
        return []

    far = max(w, h) * 5
    far_points = np.array([[minx - far, miny - far], [maxx + far, miny - far],
                           [minx - far, maxy + far], [maxx + far, maxy + far]])
    all_points = np.vstack([nearby, far_points])

    try:
        vor = Voronoi(all_points)
    except Exception:
        return []

    cells = []
    for i in range(len(nearby)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue
        verts = [vor.vertices[j] for j in region]
        try:
            cell = Polygon(verts)
            if not cell.is_valid:
                cell = cell.buffer(0)
            clipped = cell.intersection(boundary)
            if clipped.is_empty:
                continue
            if isinstance(clipped, MultiPolygon):
                clipped = max(clipped.geoms, key=lambda g: g.area)
            if clipped.geom_type == 'Polygon' and clipped.area > 1.0:
                cells.append(clipped)
        except Exception:
            continue

    return cells


# ============================================================================
# DFM VALIDATION
# ============================================================================

def validate_cell(polygon: Polygon, config: dict) -> Tuple[bool, List[str]]:
    errors = []
    min_width = config["dfm_constraints"]["min_feature_width_mm"]

    if not polygon.is_valid:
        return False, [f"Invalid: {explain_validity(polygon)}"]
    if polygon.area < min_width * min_width:
        return False, [f"Too small: area {polygon.area:.1f}"]

    try:
        min_rect = polygon.minimum_rotated_rectangle
        if min_rect.geom_type == 'Polygon':
            coords = list(min_rect.exterior.coords)
            if len(coords) >= 4:
                e1 = math.dist(coords[0], coords[1])
                e2 = math.dist(coords[1], coords[2])
                if min(e1, e2) < min_width:
                    errors.append(f"Too narrow: {min(e1, e2):.1f}mm")
    except Exception:
        pass

    try:
        coords = list(polygon.exterior.coords)[:-1]
        for i in range(len(coords)):
            p1, p2, p3 = coords[i-1], coords[i], coords[(i+1) % len(coords)]
            v1 = (p1[0]-p2[0], p1[1]-p2[1])
            v2 = (p3[0]-p2[0], p3[1]-p2[1])
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            m1 = math.hypot(*v1)
            m2 = math.hypot(*v2)
            if m1 > 0.01 and m2 > 0.01:
                cos_a = max(-1, min(1, dot / (m1 * m2)))
                if math.degrees(math.acos(cos_a)) < 30:
                    errors.append("Sharp vertex")
                    break
    except Exception:
        pass

    return len(errors) == 0, errors


def subdivide_cell(polygon: Polygon) -> List[Polygon]:
    try:
        min_rect = polygon.minimum_rotated_rectangle
        coords = list(min_rect.exterior.coords)
        centroid = polygon.centroid
        e1 = (coords[1][0]-coords[0][0], coords[1][1]-coords[0][1])
        e2 = (coords[2][0]-coords[1][0], coords[2][1]-coords[1][1])
        len1, len2 = math.hypot(*e1), math.hypot(*e2)
        d = (e1[0]/len1, e1[1]/len1) if len1 > len2 else (e2[0]/len2, e2[1]/len2)
        perp = (-d[1], d[0])
        ext = max(len1, len2) * 2
        p1 = (centroid.x - perp[0]*ext, centroid.y - perp[1]*ext)
        p2 = (centroid.x + perp[0]*ext, centroid.y + perp[1]*ext)
        result = split(polygon, LineString([p1, p2]))
        parts = [g for g in result.geoms if g.geom_type == 'Polygon' and g.area > 1]
        return parts if len(parts) >= 2 else [polygon]
    except Exception:
        return [polygon]


# ============================================================================
# RENDERING
# ============================================================================

def render_pattern(cells: List[Polygon], boundary: Polygon,
                   veins: List[LineString],
                   width_px: int, height_px: int, config: dict) -> Image.Image:
    output = config["output"]
    wood_colors = output["wood_colors"]
    epoxy_color = output["epoxy_color"]
    gap_mm = output.get("gap_width_mm", 1.5)

    minx, miny, maxx, maxy = boundary.bounds
    cw, ch = maxx - minx, maxy - miny
    pad = max(cw, ch) * 0.1
    minx -= pad; miny -= pad; cw += 2*pad; ch += 2*pad

    scale = min(width_px / cw, height_px / ch)
    ox = (width_px - cw * scale) / 2 - minx * scale
    oy = (height_px - ch * scale) / 2 - miny * scale

    def to_px(coords):
        return [(ox + x * scale, oy + y * scale) for x, y in coords]

    gap_px = max(2, gap_mm * scale)
    img = Image.new('RGB', (width_px, height_px), '#111111')
    draw = ImageDraw.Draw(img)

    # Fill cells
    for i, cell in enumerate(cells):
        if cell.geom_type == 'Polygon':
            draw.polygon(to_px(cell.exterior.coords), fill=wood_colors[i % len(wood_colors)])

    # Cell outlines (epoxy)
    for cell in cells:
        if cell.geom_type == 'Polygon':
            c = to_px(cell.exterior.coords)
            draw.line(list(c) + [c[0]], fill=epoxy_color, width=int(gap_px), joint='curve')

    # Vein lines (slightly wider)
    for vein in veins:
        if vein.geom_type == 'LineString':
            c = to_px(vein.coords)
            if len(c) >= 2:
                draw.line(c, fill=epoxy_color, width=int(gap_px * 1.3), joint='curve')

    # Boundary outline
    if boundary.geom_type == 'Polygon':
        c = to_px(boundary.exterior.coords)
        draw.line(list(c) + [c[0]], fill=epoxy_color, width=int(gap_px * 1.8), joint='curve')

    return img


def cells_to_svg(cells, boundary, veins, config):
    output = config["output"]
    wood_colors = output["wood_colors"]
    epoxy = output["epoxy_color"]
    gap = output.get("gap_width_mm", 1.5)

    minx, miny, maxx, maxy = boundary.bounds
    pad = max(maxx-minx, maxy-miny) * 0.05
    vx, vy, vw, vh = minx-pad, miny-pad, (maxx-minx)+2*pad, (maxy-miny)+2*pad

    parts = []
    for i, cell in enumerate(cells):
        if cell.geom_type != 'Polygon': continue
        coords = list(cell.exterior.coords)
        d = f"M {coords[0][0]:.2f},{coords[0][1]:.2f}" + "".join(f" L {x:.2f},{y:.2f}" for x,y in coords[1:]) + " Z"
        parts.append(f'  <path id="cell_{i}" d="{d}" fill="{wood_colors[i%len(wood_colors)]}" stroke="{epoxy}" stroke-width="{gap}"/>')

    for j, v in enumerate(veins):
        if v.geom_type != 'LineString': continue
        coords = list(v.coords)
        d = f"M {coords[0][0]:.2f},{coords[0][1]:.2f}" + "".join(f" L {x:.2f},{y:.2f}" for x,y in coords[1:])
        parts.append(f'  <path id="vein_{j}" d="{d}" fill="none" stroke="{epoxy}" stroke-width="{gap*1.2}"/>')

    if boundary.geom_type == 'Polygon':
        coords = list(boundary.exterior.coords)
        d = f"M {coords[0][0]:.2f},{coords[0][1]:.2f}" + "".join(f" L {x:.2f},{y:.2f}" for x,y in coords[1:]) + " Z"
        parts.append(f'  <path id="boundary" d="{d}" fill="none" stroke="{epoxy}" stroke-width="{gap*1.8}"/>')

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{vw:.1f}mm" height="{vh:.1f}mm"
     viewBox="{vx:.2f} {vy:.2f} {vw:.2f} {vh:.2f}">
  <rect x="{vx}" y="{vy}" width="{vw}" height="{vh}" fill="#111111"/>
{chr(10).join(parts)}
</svg>'''


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_candidate(worker_id: int, config: dict = None, logger=None) -> Optional[dict]:
    if config is None: config = load_config()
    if logger is None: logger = logging.getLogger(__name__)

    candidate_id = str(uuid.uuid4())[:8]
    logger.info(f"Generating candidate {candidate_id}")

    seg = config["segmentation"]
    target_cells = random.randint(seg["target_cells_min"], seg["target_cells_max"])
    width_mm = random.choice([150, 200])
    height_mm = random.choice([150, 200])

    # 1. Leaf silhouette
    boundary, sil = get_random_silhouette(width_mm, height_mm)
    if boundary.is_empty or boundary.area < 10:
        return None

    # 2. Vein lines (= "gradient field")
    n_pri = random.randint(3, 7)
    n_sec = random.randint(2, 4)
    veins = generate_veins(boundary, sil, n_pri, n_sec)
    logger.info(f"  veins: {len(veins)} segments, {n_pri} primary")

    # 3. Regular hex grid
    tile_size = math.sqrt(boundary.area / target_cells) * 1.1
    vertices = create_hex_grid(boundary, tile_size)
    logger.info(f"  hex grid: {len(vertices)} vertices, tile={tile_size:.1f}mm")

    # 4. grid_localize: attract vertices to veins
    neatness = random.uniform(0.15, 0.45)
    attraction_radius = tile_size * 1.5
    displaced = grid_localize(vertices, veins, boundary, neatness, attraction_radius)

    # 5. Voronoi from displaced vertices, clipped to boundary
    cells = voronoi_cells_from_vertices(displaced, boundary)
    logger.info(f"  cells: {len(cells)} (neatness={neatness:.2f})")

    if len(cells) < 3:
        logger.warning("Too few cells, skipping")
        return None

    # 6. DFM validation + subdivision
    max_iter = seg.get("max_subdivision_iterations", 3)
    all_valid = True
    for iteration in range(max_iter):
        all_valid = True
        new_cells = []
        for cell in cells:
            valid, _ = validate_cell(cell, config)
            if valid:
                new_cells.append(cell)
            else:
                all_valid = False
                new_cells.extend(subdivide_cell(cell) if cell.area > 20 else [cell])
        cells = new_cells
        if all_valid: break
        if len(cells) > target_cells * 2.5: break

    # 7. Output
    out_dir = WORKSPACE / "candidates" / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)

    output = config["output"]
    try:
        img = render_pattern(cells, boundary, veins,
                             output.get("render_width_px", 800),
                             output.get("render_height_px", 800), config)
        img.save(out_dir / "render.png")
    except Exception as e:
        logger.error(f"Render failed: {e}")

    try:
        svg = cells_to_svg(cells, boundary, veins, config)
        with open(out_dir / "cells.svg", "w") as f: f.write(svg)
    except Exception as e:
        logger.error(f"SVG failed: {e}")

    metadata = {
        "uuid": candidate_id,
        "created": datetime.now().isoformat(),
        "worker_id": worker_id,
        "algorithm": "gimp_mosaic_adapted",
        "source": {
            "type": "procedural", "method": "hex_grid_localize",
            "silhouette": sil,
            "params": {"num_primary_veins": n_pri, "num_secondary_per_vein": n_sec,
                       "tile_size_mm": round(tile_size, 1), "neatness": round(neatness, 3),
                       "attraction_radius_mm": round(attraction_radius, 1)}
        },
        "cells": {"count": len(cells), "all_valid": all_valid, "vein_count": len(veins)},
        "dimensions": {"width_mm": width_mm, "height_mm": height_mm}
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  ✓ {candidate_id}: {len(cells)} cells, {sil}, neatness={neatness:.2f}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Sublobe pattern generator (GIMP-style)")
    parser.add_argument("--worker-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    logger = setup_logging(args.worker_id)
    config = load_config()
    logger.info(f"Starting worker {args.worker_id}, batch {args.batch_size}")

    success = fail = 0
    for _ in range(args.batch_size):
        try:
            result = generate_candidate(args.worker_id, config, logger)
            success += 1 if result else 0
            fail += 0 if result else 1
        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
            fail += 1

    logger.info(f"Done: {success} ok, {fail} failed")


if __name__ == "__main__":
    main()
