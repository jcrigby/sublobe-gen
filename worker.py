#!/usr/bin/env python3
"""
worker.py — Pattern generation worker for sublobe inlay system

This script can be run directly or imported. When run directly:
    python worker.py --worker-id 1 --batch-size 10 --seed-type mixed

When imported, use the generate_candidate() function.
"""

import json
import math
import os
import sys
import uuid
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import argparse

# These will be imported lazily to allow partial execution
np = None
Image = None
ImageDraw = None
Polygon = None
LineString = None
Point = None


def lazy_imports():
    """Import heavy dependencies only when needed."""
    global np, Image, ImageDraw, Polygon, LineString, Point
    
    if np is None:
        import numpy as _np
        np = _np
    
    if Image is None:
        from PIL import Image as _Image, ImageDraw as _ImageDraw
        Image = _Image
        ImageDraw = _ImageDraw
    
    if Polygon is None:
        from shapely.geometry import Polygon as _Polygon, LineString as _LineString, Point as _Point
        Polygon = _Polygon
        LineString = _LineString
        Point = _Point


# Configuration (loaded from config.json)
CONFIG = None
WORKSPACE = Path(__file__).parent


def load_config():
    """Load configuration from config.json."""
    global CONFIG
    config_path = WORKSPACE / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            CONFIG = json.load(f)
    else:
        # Defaults
        CONFIG = {
            "dfm_constraints": {
                "min_feature_width_mm": 3.0,
                "min_internal_radius_mm": 2.0,
                "blade_kerf_mm": 3.0
            },
            "output": {
                "render_width_px": 800,
                "render_height_px": 800,
                "wood_colors": ["#D4A574", "#C4956A", "#B8860B", "#8B7355", "#6B4423"],
                "epoxy_color": "#1a1a1a",
                "gap_width_mm": 2.0
            },
            "segmentation": {
                "target_cells_min": 15,
                "target_cells_max": 40,
                "max_subdivision_iterations": 5
            }
        }
    return CONFIG


def setup_logging(worker_id: int):
    """Configure logging for this worker."""
    log_dir = WORKSPACE / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {worker_id}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"worker_{worker_id}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# DFM VALIDATION
# ============================================================================

def validate_cell(polygon, config: dict) -> tuple[bool, list[str]]:
    """
    Validate a cell polygon against manufacturing constraints.
    
    Returns: (is_valid, list_of_errors)
    """
    lazy_imports()
    
    errors = []
    constraints = config["dfm_constraints"]
    min_width = constraints["min_feature_width_mm"]
    min_radius = constraints["min_internal_radius_mm"]
    
    # Basic validity
    if not polygon.is_valid:
        from shapely.validation import explain_validity
        errors.append(f"Invalid polygon: {explain_validity(polygon)}")
        return False, errors
    
    # Check for self-intersection
    if not polygon.is_simple:
        errors.append("Self-intersecting polygon")
        return False, errors
    
    # Minimum width check (via minimum rotated rectangle)
    try:
        min_rect = polygon.minimum_rotated_rectangle
        if min_rect.geom_type == 'Polygon':
            coords = list(min_rect.exterior.coords)
            if len(coords) >= 4:
                edge1 = math.sqrt((coords[1][0]-coords[0][0])**2 + (coords[1][1]-coords[0][1])**2)
                edge2 = math.sqrt((coords[2][0]-coords[1][0])**2 + (coords[2][1]-coords[1][1])**2)
                cell_min_width = min(edge1, edge2)
                if cell_min_width < min_width:
                    errors.append(f"Too narrow: {cell_min_width:.1f}mm < {min_width}mm")
    except Exception as e:
        errors.append(f"Width check failed: {e}")
    
    # Sharp angle check (proxy for minimum internal radius)
    try:
        coords = list(polygon.exterior.coords)[:-1]  # Remove closing duplicate
        for i in range(len(coords)):
            p1 = coords[i-1]
            p2 = coords[i]
            p3 = coords[(i+1) % len(coords)]
            
            v1 = (p1[0]-p2[0], p1[1]-p2[1])
            v2 = (p3[0]-p2[0], p3[1]-p2[1])
            
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0.001 and mag2 > 0.001:
                cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
                angle = math.degrees(math.acos(cos_angle))
                if angle < 45:  # Very sharp internal angles
                    errors.append(f"Sharp vertex ({angle:.0f}°) at ({p2[0]:.1f}, {p2[1]:.1f})")
    except Exception as e:
        errors.append(f"Angle check failed: {e}")
    
    return len(errors) == 0, errors


def subdivide_cell(polygon) -> list:
    """
    Split a failing cell into two smaller cells.
    
    Strategy: Cut along the shorter axis of the minimum bounding rectangle.
    """
    lazy_imports()
    from shapely.ops import split
    
    try:
        # Get the minimum rotated rectangle
        min_rect = polygon.minimum_rotated_rectangle
        coords = list(min_rect.exterior.coords)
        
        # Find the center
        centroid = polygon.centroid
        
        # Calculate edge vectors
        edge1 = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        edge2 = (coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
        
        len1 = math.sqrt(edge1[0]**2 + edge1[1]**2)
        len2 = math.sqrt(edge2[0]**2 + edge2[1]**2)
        
        # Use the longer edge direction for the split line
        if len1 > len2:
            direction = (edge1[0]/len1, edge1[1]/len1)
        else:
            direction = (edge2[0]/len2, edge2[1]/len2)
        
        # Create a line through centroid perpendicular to short axis
        perp = (-direction[1], direction[0])
        line_length = max(len1, len2) * 2
        
        p1 = (centroid.x - perp[0] * line_length, centroid.y - perp[1] * line_length)
        p2 = (centroid.x + perp[0] * line_length, centroid.y + perp[1] * line_length)
        
        split_line = LineString([p1, p2])
        result = split(polygon, split_line)
        
        parts = [g for g in result.geoms if g.geom_type == 'Polygon' and g.area > 1]
        
        if len(parts) >= 2:
            return parts
        else:
            # Fallback: just return original
            return [polygon]
            
    except Exception as e:
        logging.warning(f"Subdivision failed: {e}")
        return [polygon]


# ============================================================================
# IMAGE SILHOUETTE EXTRACTION
# ============================================================================

def extract_silhouette_from_image(image_path: str, target_size: tuple = (200, 200)) -> tuple:
    """
    Extract the main silhouette from an image as a Shapely polygon.

    Args:
        image_path: Path to image file
        target_size: (width_mm, height_mm) to scale the result

    Returns:
        (polygon, width, height) - Shapely Polygon and dimensions in mm
    """
    lazy_imports()
    from PIL import Image as PILImage

    # Load and convert to grayscale
    img = PILImage.open(image_path).convert('L')

    # Resize for processing (keep aspect ratio)
    img.thumbnail((500, 500), PILImage.Resampling.LANCZOS)
    img_array = np.array(img)

    # Threshold to get binary mask (assume dark = object, light = background)
    # Invert if needed based on corner pixels
    corners = [img_array[0,0], img_array[0,-1], img_array[-1,0], img_array[-1,-1]]
    if np.mean(corners) < 128:
        # Background is dark, invert
        img_array = 255 - img_array

    threshold = 200  # Pixels darker than this are part of the silhouette
    mask = img_array < threshold

    # Find contours using a simple edge-following algorithm
    # For better results, we'd use opencv, but let's keep deps minimal
    from scipy import ndimage

    # Fill holes and clean up
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, iterations=1)
    mask = ndimage.binary_dilation(mask, iterations=1)

    # Find the boundary pixels
    interior = ndimage.binary_erosion(mask, iterations=1)
    boundary = mask & ~interior

    # Get boundary coordinates
    coords = np.argwhere(boundary)
    if len(coords) < 10:
        # Fallback to rectangle
        return None, target_size[0], target_size[1]

    # Order points to form a polygon (convex hull as approximation)
    from scipy.spatial import ConvexHull
    try:
        # Use convex hull of boundary points, then simplify
        hull = ConvexHull(coords)
        hull_points = coords[hull.vertices]

        # Scale to target size
        h, w = img_array.shape
        scale_x = target_size[0] / w
        scale_y = target_size[1] / h

        # Convert from (row, col) to (x, y) and scale
        polygon_coords = [(p[1] * scale_x, p[0] * scale_y) for p in hull_points]

        polygon = Polygon(polygon_coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        return polygon, target_size[0], target_size[1]

    except Exception as e:
        logging.warning(f"Silhouette extraction failed: {e}")
        return None, target_size[0], target_size[1]


def extract_detailed_silhouette(image_path: str, target_size: tuple = (200, 200),
                                  simplify_tolerance: float = 1.0) -> tuple:
    """
    Extract a more detailed silhouette using marching squares (if scipy available).
    Falls back to convex hull if that fails.
    """
    lazy_imports()
    from PIL import Image as PILImage

    # Load and convert to grayscale
    img = PILImage.open(image_path).convert('L')

    # Resize for processing
    img.thumbnail((400, 400), PILImage.Resampling.LANCZOS)
    img_array = np.array(img)
    h, w = img_array.shape

    # Use adaptive thresholding based on image statistics
    # For images with a clear foreground object
    mean_val = np.mean(img_array)
    std_val = np.std(img_array)

    # Determine if we need to invert (check corners for background color)
    corners = [img_array[0,0], img_array[0,-1], img_array[-1,0], img_array[-1,-1]]
    corner_mean = np.mean(corners)

    # If corners are bright, object is darker than background
    # If corners are dark, object is brighter than background
    if corner_mean > mean_val:
        # Background is bright, object is dark - threshold below mean
        threshold = mean_val - std_val * 0.5
        mask = (img_array < threshold).astype(np.uint8)
    else:
        # Background is dark, object is bright - threshold above mean
        threshold = mean_val + std_val * 0.5
        mask = (img_array > threshold).astype(np.uint8)

    try:
        from scipy import ndimage

        # Clean up mask
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

        # Remove small objects (noise)
        labeled, num_features = ndimage.label(mask)
        if num_features > 1:
            # Keep only the largest connected component
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            mask = (labeled == largest_label).astype(np.uint8)

        # Find contours using skimage if available, else fall back
        try:
            from skimage import measure
            contours = measure.find_contours(mask, 0.5)

            if not contours:
                return extract_silhouette_from_image(image_path, target_size)

            # Get the largest contour
            largest = max(contours, key=len)

            # Scale to target size
            scale_x = target_size[0] / w
            scale_y = target_size[1] / h

            # Convert (row, col) to (x, y) and scale
            polygon_coords = [(p[1] * scale_x, p[0] * scale_y) for p in largest]

            polygon = Polygon(polygon_coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            # Simplify to reduce vertex count (but preserve detail)
            if simplify_tolerance > 0:
                polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

            return polygon, target_size[0], target_size[1]

        except ImportError:
            return extract_silhouette_from_image(image_path, target_size)

    except Exception as e:
        logging.warning(f"Detailed silhouette extraction failed: {e}")
        return extract_silhouette_from_image(image_path, target_size)


# ============================================================================
# SEGMENTATION METHODS
# ============================================================================

def generate_voronoi_cells(width: float, height: float, num_points: int, 
                           boundary_polygon=None, points=None) -> list:
    """
    Generate Voronoi cells within a boundary.
    
    Args:
        width, height: Bounding box dimensions
        num_points: Number of seed points (ignored if points provided)
        boundary_polygon: Optional Shapely polygon to clip to
        points: Optional pre-defined seed points (numpy array)
    
    Returns: List of Shapely Polygon objects
    """
    lazy_imports()
    from scipy.spatial import Voronoi
    
    # Generate random seed points if not provided
    if points is None:
        points = np.random.rand(num_points, 2)
        points[:, 0] *= width
        points[:, 1] *= height
    else:
        points = np.array(points)
        num_points = len(points)
    
    # Add boundary points to prevent infinite regions
    margin = max(width, height) * 0.5
    boundary_points = [
        [-margin, -margin], [width + margin, -margin],
        [-margin, height + margin], [width + margin, height + margin],
        [width/2, -margin], [width/2, height + margin],
        [-margin, height/2], [width + margin, height/2]
    ]
    all_points = np.vstack([points, boundary_points])
    
    vor = Voronoi(all_points)
    
    cells = []
    bounding_box = Polygon([
        (0, 0), (width, 0), (width, height), (0, height)
    ])
    
    if boundary_polygon is None:
        boundary_polygon = bounding_box
    
    for region_idx in vor.point_region[:num_points]:  # Only original points
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue
        
        vertices = [vor.vertices[i] for i in region]
        try:
            cell = Polygon(vertices)
            if cell.is_valid:
                clipped = cell.intersection(boundary_polygon)
                if clipped.geom_type == 'Polygon' and clipped.area > 1:
                    cells.append(clipped)
        except:
            continue
    
    return cells


def generate_grid_cells(width: float, height: float, cols: int, rows: int,
                        jitter: float = 0.3) -> list:
    """
    Generate a jittered grid of cells.
    
    Args:
        width, height: Bounding box dimensions
        cols, rows: Grid dimensions
        jitter: Random displacement factor (0-1)
    
    Returns: List of Shapely Polygon objects
    """
    lazy_imports()
    
    cell_w = width / cols
    cell_h = height / rows
    cells = []
    
    # Generate jittered grid points
    points = []
    for i in range(cols + 1):
        for j in range(rows + 1):
            x = i * cell_w + (random.random() - 0.5) * cell_w * jitter
            y = j * cell_h + (random.random() - 0.5) * cell_h * jitter
            # Clamp to bounds
            x = max(0, min(width, x))
            y = max(0, min(height, y))
            points.append((x, y))
    
    # Create Voronoi from grid points
    return generate_voronoi_cells(width, height, len(points) - 8)  # Subtract boundary points


def generate_organic_cells(width: float, height: float,
                           target_count: int, irregularity: float = 0.5,
                           boundary_polygon=None) -> list:
    """
    Generate organic-looking cells using relaxed Voronoi.

    Multiple iterations of Lloyd's algorithm smooth the cells.

    Args:
        width, height: Bounding box dimensions
        target_count: Target number of cells
        irregularity: 0 = very regular, 1 = very irregular
        boundary_polygon: Optional Shapely polygon to constrain cells
    """
    lazy_imports()
    from scipy.spatial import Voronoi

    # Start with random points
    if boundary_polygon is not None:
        # Generate points inside the boundary polygon
        minx, miny, maxx, maxy = boundary_polygon.bounds
        points = []
        attempts = 0
        while len(points) < target_count and attempts < target_count * 100:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            if boundary_polygon.contains(Point(x, y)):
                points.append([x, y])
            attempts += 1
        points = np.array(points) if points else np.random.rand(target_count, 2) * [width, height]
    else:
        points = np.random.rand(target_count, 2)
        points[:, 0] *= width
        points[:, 1] *= height

    # Lloyd relaxation (move points toward cell centroids)
    iterations = int(5 * (1 - irregularity))  # More iterations = more regular
    for _ in range(iterations):
        cells = generate_voronoi_cells(width, height, len(points), boundary_polygon)
        new_points = []
        for cell in cells:
            c = cell.centroid
            # Keep centroid inside boundary if specified
            if boundary_polygon is not None:
                if boundary_polygon.contains(Point(c.x, c.y)):
                    new_points.append([c.x, c.y])
                else:
                    # Project to nearest point inside boundary
                    new_points.append([c.x, c.y])  # Keep anyway, clipping handles it
            else:
                new_points.append([c.x, c.y])
        if len(new_points) >= len(points) * 0.8:  # Allow some loss
            points = np.array(new_points[:len(points)])

    return generate_voronoi_cells(width, height, len(points), boundary_polygon)


def generate_venation_cells(width: float, height: float,
                            num_lobes: int = 5, 
                            secondary_veins: int = 3,
                            stem_position: tuple = None) -> list:
    """
    Generate leaf-like venation pattern.
    
    Creates cells that follow natural leaf anatomy:
    - Primary veins radiate from stem
    - Secondary veins branch off primaries
    - Cells are elongated, following vein flow
    
    Args:
        width, height: Bounding dimensions
        num_lobes: Number of primary lobes (3-7 typical for maple)
        secondary_veins: Secondary veins per primary
        stem_position: (x, y) of stem attachment, default bottom center
    
    Returns: List of Shapely Polygon cells
    """
    lazy_imports()
    from shapely.ops import split, unary_union
    
    if stem_position is None:
        stem_position = (width / 2, height * 0.95)  # Bottom center
    
    stem_x, stem_y = stem_position
    
    # Generate primary vein angles (radiate upward in a fan)
    # Spread across ~140 degrees, centered on vertical
    angle_spread = math.radians(140)
    base_angle = math.radians(90)  # Pointing up
    
    primary_angles = []
    for i in range(num_lobes):
        t = (i / (num_lobes - 1)) - 0.5 if num_lobes > 1 else 0
        angle = base_angle + t * angle_spread
        # Add slight randomness
        angle += random.uniform(-0.1, 0.1)
        primary_angles.append(angle)
    
    # Create vein lines
    all_veins = []
    
    # Primary veins from stem to edge
    vein_length = max(width, height) * 0.9
    for angle in primary_angles:
        end_x = stem_x + math.cos(angle) * vein_length
        end_y = stem_y - math.sin(angle) * vein_length  # Y inverted
        
        # Add some curve to the vein
        mid_x = stem_x + math.cos(angle) * vein_length * 0.5
        mid_y = stem_y - math.sin(angle) * vein_length * 0.5
        mid_x += random.uniform(-width * 0.05, width * 0.05)
        
        vein = LineString([(stem_x, stem_y), (mid_x, mid_y), (end_x, end_y)])
        all_veins.append(vein)
        
        # Secondary veins branching off this primary
        for j in range(secondary_veins):
            # Position along primary vein
            t = 0.3 + (j / secondary_veins) * 0.5  # 30% to 80% along vein
            branch_point = vein.interpolate(t, normalized=True)
            
            # Branch angle (alternate sides)
            branch_angle = angle + (0.4 if j % 2 == 0 else -0.4)
            branch_angle += random.uniform(-0.15, 0.15)
            
            branch_length = vein_length * 0.3 * (1 - t * 0.5)  # Shorter near tip
            branch_end_x = branch_point.x + math.cos(branch_angle) * branch_length
            branch_end_y = branch_point.y - math.sin(branch_angle) * branch_length
            
            secondary = LineString([
                (branch_point.x, branch_point.y),
                (branch_end_x, branch_end_y)
            ])
            all_veins.append(secondary)
    
    # Create bounding polygon (leaf-ish shape or rectangle)
    # For simplicity, use an ellipse-ish boundary
    boundary_points = []
    for i in range(36):
        angle = 2 * math.pi * i / 36
        # Ellipse with pointed top
        r_x = width * 0.48
        r_y = height * 0.48
        if angle > math.pi:  # Bottom half - narrower
            r_x *= 0.7
        x = width/2 + r_x * math.cos(angle)
        y = height/2 + r_y * math.sin(angle)
        boundary_points.append((x, y))
    
    boundary = Polygon(boundary_points)
    if not boundary.is_valid:
        boundary = boundary.buffer(0)
    
    # Use veins to split the boundary into cells
    # This is tricky - we'll use a Voronoi-like approach but guided by veins
    
    # Sample points along veins to guide Voronoi
    vein_points = []
    for vein in all_veins:
        for t in np.linspace(0.1, 0.9, 5):
            pt = vein.interpolate(t, normalized=True)
            vein_points.append([pt.x, pt.y])
    
    # Add some fill points between veins
    num_fill = max(10, len(vein_points) // 2)
    for _ in range(num_fill):
        x = random.uniform(width * 0.1, width * 0.9)
        y = random.uniform(height * 0.1, height * 0.9)
        if boundary.contains(Point(x, y)):
            vein_points.append([x, y])
    
    if len(vein_points) < 5:
        # Fallback to regular Voronoi
        return generate_voronoi_cells(width, height, 20, boundary)
    
    # Generate Voronoi from vein-guided points
    cells = generate_voronoi_cells(width, height, len(vein_points), boundary)
    
    return cells


# ============================================================================
# RENDERING
# ============================================================================

def render_cells(cells: list, width: int, height: int, config: dict,
                 scale_mm_to_px: float = 1.0) -> 'Image':
    """
    Render cells as a preview image.
    
    Args:
        cells: List of Shapely polygons
        width, height: Image dimensions in pixels
        config: Configuration dict
        scale_mm_to_px: Conversion from mm to pixels
    
    Returns: PIL Image
    """
    lazy_imports()
    
    output_config = config["output"]
    wood_colors = output_config["wood_colors"]
    epoxy_color = output_config["epoxy_color"]
    gap_width = output_config["gap_width_mm"] * scale_mm_to_px
    
    # Create image with epoxy background
    img = Image.new('RGB', (width, height), epoxy_color)
    draw = ImageDraw.Draw(img)
    
    for i, cell in enumerate(cells):
        color = wood_colors[i % len(wood_colors)]
        
        # Scale coordinates
        coords = [(x * scale_mm_to_px, y * scale_mm_to_px) 
                  for x, y in cell.exterior.coords]
        
        # Shrink slightly to show gaps
        shrunk = cell.buffer(-gap_width / 2)
        if shrunk.geom_type == 'Polygon' and shrunk.area > 0:
            coords = [(x * scale_mm_to_px, y * scale_mm_to_px) 
                      for x, y in shrunk.exterior.coords]
        
        draw.polygon(coords, fill=color, outline=epoxy_color)
    
    return img


def cells_to_svg(cells: list, width: float, height: float, config: dict) -> str:
    """
    Convert cells to SVG string.
    
    Args:
        cells: List of Shapely polygons
        width, height: Dimensions in mm
        config: Configuration dict
    
    Returns: SVG string
    """
    output_config = config["output"]
    wood_colors = output_config["wood_colors"]
    epoxy_color = output_config["epoxy_color"]
    
    paths = []
    for i, cell in enumerate(cells):
        color = wood_colors[i % len(wood_colors)]
        coords = list(cell.exterior.coords)
        
        # Build SVG path
        d = f"M {coords[0][0]:.2f},{coords[0][1]:.2f}"
        for x, y in coords[1:]:
            d += f" L {x:.2f},{y:.2f}"
        d += " Z"
        
        paths.append(f'  <path id="cell_{i}" d="{d}" fill="{color}" stroke="{epoxy_color}" stroke-width="2"/>')
    
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{width}mm" height="{height}mm" 
     viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="{epoxy_color}"/>
{chr(10).join(paths)}
</svg>'''
    
    return svg


# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

def generate_candidate(worker_id: int, seed_type: str = "procedural",
                       config: dict = None, logger=None) -> Optional[dict]:
    """
    Generate a single pattern candidate.
    
    Args:
        worker_id: Worker identifier for logging
        seed_type: "procedural", "image", or "mixed"
        config: Configuration dict (loaded if None)
        logger: Logger instance
    
    Returns: dict with candidate info, or None on failure
    """
    lazy_imports()
    
    if config is None:
        config = load_config()
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    candidate_id = str(uuid.uuid4())[:8]
    logger.info(f"Generating candidate {candidate_id}")
    
    seg_config = config["segmentation"]
    target_cells = random.randint(seg_config["target_cells_min"],
                                   seg_config["target_cells_max"])

    # Dimensions (mm)
    width_mm = random.choice([100, 150, 200])
    height_mm = random.choice([100, 150, 200])

    # Check for reference images to use as silhouettes
    boundary_polygon = None
    reference_image = None
    references_dir = WORKSPACE / "references"
    if references_dir.exists():
        ref_images = list(references_dir.glob("*.png")) + list(references_dir.glob("*.jpg"))
        if ref_images:
            reference_image = random.choice(ref_images)
            logger.info(f"Using reference image: {reference_image.name}")
            try:
                boundary_polygon, width_mm, height_mm = extract_detailed_silhouette(
                    str(reference_image),
                    target_size=(width_mm, height_mm),
                    simplify_tolerance=0.5  # Lower = more detail preserved
                )
                if boundary_polygon is not None:
                    logger.info(f"Extracted silhouette with {len(boundary_polygon.exterior.coords)} vertices")
            except Exception as e:
                logger.warning(f"Failed to extract silhouette: {e}")
                boundary_polygon = None

    # Choose segmentation method
    method = random.choices(
        ["organic"],
        weights=[1.0]  # Organic only for now
    )[0]

    try:
        if method == "venation":
            num_lobes = random.randint(3, 7)
            secondary = random.randint(2, 4)
            cells = generate_venation_cells(width_mm, height_mm, num_lobes, secondary)
            params = {"num_lobes": num_lobes, "secondary_veins": secondary}
        elif method == "voronoi":
            cells = generate_voronoi_cells(width_mm, height_mm, target_cells, boundary_polygon)
            params = {"num_points": target_cells}
        elif method == "organic":
            irregularity = random.uniform(0.3, 0.7)
            cells = generate_organic_cells(width_mm, height_mm, target_cells, irregularity, boundary_polygon)
            params = {"target_count": target_cells, "irregularity": irregularity,
                      "reference": reference_image.name if reference_image else None}
        else:  # grid
            cols = random.randint(4, 8)
            rows = random.randint(4, 8)
            jitter = random.uniform(0.2, 0.5)
            cells = generate_grid_cells(width_mm, height_mm, cols, rows, jitter)
            params = {"cols": cols, "rows": rows, "jitter": jitter}
        
        logger.info(f"Generated {len(cells)} initial cells using {method}")
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return None
    
    # DFM Validation and Subdivision
    max_iterations = seg_config["max_subdivision_iterations"]
    iteration = 0
    
    while iteration < max_iterations:
        all_valid = True
        new_cells = []
        
        for cell in cells:
            valid, errors = validate_cell(cell, config)
            if valid:
                new_cells.append(cell)
            else:
                all_valid = False
                logger.debug(f"Cell failed: {errors}")
                subdivided = subdivide_cell(cell)
                new_cells.extend(subdivided)
        
        cells = new_cells
        iteration += 1
        
        if all_valid:
            logger.info(f"All {len(cells)} cells valid after {iteration} iterations")
            break
    
    if not all_valid:
        logger.warning(f"Some cells still invalid after {max_iterations} iterations")
    
    # Create output directory
    candidates_dir = WORKSPACE / "candidates" / candidate_id
    candidates_dir.mkdir(parents=True, exist_ok=True)
    
    # Render preview
    output_config = config["output"]
    render_w = output_config["render_width_px"]
    render_h = output_config["render_height_px"]
    scale = min(render_w / width_mm, render_h / height_mm)
    
    try:
        img = render_cells(cells, int(width_mm * scale), int(height_mm * scale), 
                          config, scale)
        img.save(candidates_dir / "render.png")
        logger.info(f"Saved render.png")
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
    
    # Save SVG
    try:
        svg = cells_to_svg(cells, width_mm, height_mm, config)
        with open(candidates_dir / "cells.svg", "w") as f:
            f.write(svg)
        logger.info(f"Saved cells.svg")
    except Exception as e:
        logger.error(f"SVG export failed: {e}")
    
    # Save metadata
    metadata = {
        "uuid": candidate_id,
        "created": datetime.now().isoformat(),
        "worker_id": worker_id,
        "source": {
            "type": seed_type,
            "method": method,
            "params": params
        },
        "cells": {
            "count": len(cells),
            "all_valid": all_valid,
            "subdivision_iterations": iteration
        },
        "dimensions": {
            "width_mm": width_mm,
            "height_mm": height_mm
        }
    }
    
    with open(candidates_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Candidate {candidate_id} complete: {len(cells)} cells")
    
    return metadata


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sublobe pattern generator worker")
    parser.add_argument("--worker-id", type=int, default=1, help="Worker ID for logging")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of candidates to generate")
    parser.add_argument("--seed-type", choices=["procedural", "image", "mixed"], 
                        default="procedural", help="Type of seeds to use")
    
    args = parser.parse_args()
    
    logger = setup_logging(args.worker_id)
    config = load_config()
    
    logger.info(f"Starting worker {args.worker_id}, batch size {args.batch_size}")
    
    success_count = 0
    fail_count = 0
    
    for i in range(args.batch_size):
        try:
            result = generate_candidate(args.worker_id, args.seed_type, config, logger)
            if result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"Candidate {i+1} failed: {e}")
            fail_count += 1
        
        # Check error rate
        total = success_count + fail_count
        if total > 5 and fail_count / total > 0.5:
            logger.error("Error rate > 50%, stopping worker")
            break
    
    logger.info(f"Worker {args.worker_id} complete: {success_count} success, {fail_count} failed")


if __name__ == "__main__":
    main()
