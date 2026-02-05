# Sublobe Inlay Pattern Generator

## What This Project Does

Generates manufacturable stained-glass-style patterns for wood inlay. Each pattern consists of:
- **Cells** (sublobes) = wood pieces, cut on table saw, shaped with 3D-printed sanding blocks
- **Gaps between cells** = epoxy "veins" (like lead came in stained glass)

The output is patterns that look organic/artistic BUT are constrained to be physically manufacturable.

## Manufacturing Process

Each sublobe stick is cut on **TWO opposing faces**:

```
1. Start with rectangular stock
2. Cut Face A (series of table saw passes at different heights/angles)
3. Sand Face A to final profile using 3D-printed sanding block
4. Place Face A into a 3D-printed CRADLE that matches its profile
5. Cradle holds the piece while you cut Face B (opposite side)
6. Sand Face B
7. Done — cross-section is now defined by two opposing curves
```

The cradle is the key jig — it's 3D-printed to match Face A's profile, so it grips the piece securely while you work on Face B.

**Future extensions:** 3-face (triangular stock), 4-face (square stock), 5-face, 6-face. More faces = more complex shapes but more cuts and more cradles.

## Manufacturing Constraints (DFM Rules)

Every cell must pass these rules:

1. **No undercuts** — every surface reachable by a sanding block approaching from above
2. **Two-face decomposable** — cross-section can be split into Face A (top) and Face B (bottom)
3. **Minimum feature width > 3mm** — narrower than this can't survive table saw kerf (~3mm blade)
4. **Minimum internal radius > 2mm** — sanding block must be printable
5. **Closed polygon** — no open paths
6. **No self-intersection** — valid simple polygon

If a cell fails validation, the fix is always: **subdivide it** (add another vein line).

## Directory Structure

```
/workspace
├── CLAUDE.md           # You are here
├── references/         # Source images (downloaded or generated)
├── candidates/         # Output patterns with renders
│   └── {uuid}/
│       ├── metadata.json
│       ├── cells.svg
│       ├── render.png
│       └── sanding_blocks/  # STL files
├── selected/           # Human-approved patterns
├── logs/               # Per-worker logs
├── config.json         # Shared configuration
└── tasks.json          # Current task state
```

## Your Job As A Worker

You are one of multiple parallel Claude instances. **Work autonomously. Do NOT ask for advice or clarification. If something is ambiguous, make a reasonable choice and continue. If something fails, try a different approach.**

Your loop:

```
1. Acquire a seed (image URL, style prompt, or random params)
2. Generate or download reference image
3. Segment into cells (watershed, Voronoi, edge detection)
4. Validate each cell against DFM rules
5. Subdivide any failures (this adds vein lines - that's good!)
6. Repeat 4-5 until ALL cells pass
7. Render preview (SVG + PNG)
8. Optionally generate sanding block STLs
9. Save to candidates/{uuid}/
10. Log completion, pick next seed
```

## Key Insight

When a cell is "too complex" to manufacture, you don't simplify it — you **split it**. The split line becomes another epoxy vein. Manufacturing constraints literally generate more visual detail. This is the core trick.

## Image Sources (In Priority Order)

1. **Wikimedia Commons** — botanical illustrations, art nouveau designs, Tiffany lamp photos
2. **Unsplash** — leaf photos, organic textures
3. **Procedural** — Voronoi diagrams, reaction-diffusion, noise-based
4. **Search queries that work well:**
   - "art nouveau stained glass pattern"
   - "tiffany lamp detail"
   - "botanical illustration leaf"
   - "ginkgo leaf silhouette"
   - "maple leaf veins"

## Reference Style

See `references/maple-leaf-reference.png` for the target aesthetic. Key properties:

1. **Sublobes follow natural venation** — veins radiate from stem, secondary veins subdivide lobes
2. **Elongated cells** — each sublobe is longer than wide (good for two-face cutting)
3. **Convex or gently curved outer edges** — no extreme concavities
4. **Sharp tips formed by vein convergence** — don't try to cut sharp wood points, let veins meet
5. **Consistent vein/epoxy width** — uniform gaps throughout

This is botanical illustration style, not random geometric division. The pattern should look like it grew, not like it was computed.

## Segmentation Approaches

Try these in order of preference:

### Best: Venation-style (radial from stem point)
```python
# 1. Define a stem point (bottom center for leaves)
# 2. Create primary veins radiating outward (3-7 main lobes)
# 3. Add secondary veins branching off primaries
# 4. Veins define cell boundaries
# 5. Each cell between veins = one sublobe

# The veins ARE the epoxy lines. Cells are elongated, 
# following the natural flow from stem to edge.
```

### Good: Voronoi with guided seeds
```python
from scipy.spatial import Voronoi
import numpy as np

# Random or image-guided seed points
points = np.random.rand(30, 2) * image_size
vor = Voronoi(points)
# Clip to leaf silhouette
```

### Medium: Watershed on edges
```python
from skimage import segmentation, filters, morphology

edges = filters.sobel(grayscale_image)
markers = morphology.label(edges < threshold)
cells = segmentation.watershed(edges, markers)
```

### Advanced: GIMP-style mosaic
```python
# Superpixel segmentation
from skimage.segmentation import slic
cells = slic(image, n_segments=50, compactness=10)
```

## DFM Validation Code

```python
from shapely.geometry import Polygon
from shapely.validation import explain_validity

MIN_WIDTH = 3.0  # mm
MIN_RADIUS = 2.0  # mm

def validate_cell(polygon: Polygon) -> tuple[bool, list[str]]:
    errors = []
    
    # Basic validity
    if not polygon.is_valid:
        errors.append(f"Invalid polygon: {explain_validity(polygon)}")
        return False, errors
    
    # Minimum width (approximated by minimum rotated rectangle)
    min_rect = polygon.minimum_rotated_rectangle
    coords = list(min_rect.exterior.coords)
    edge1 = ((coords[1][0]-coords[0][0])**2 + (coords[1][1]-coords[0][1])**2)**0.5
    edge2 = ((coords[2][0]-coords[1][0])**2 + (coords[2][1]-coords[1][1])**2)**0.5
    min_width = min(edge1, edge2)
    if min_width < MIN_WIDTH:
        errors.append(f"Too narrow: {min_width:.1f}mm < {MIN_WIDTH}mm")
    
    # Minimum internal radius (check concave vertices)
    # Simplified: if any vertex angle < 60°, likely too sharp
    coords = list(polygon.exterior.coords)[:-1]  # Remove duplicate closing point
    for i in range(len(coords)):
        p1 = coords[i-1]
        p2 = coords[i]
        p3 = coords[(i+1) % len(coords)]
        # Calculate angle at p2
        v1 = (p1[0]-p2[0], p1[1]-p2[1])
        v2 = (p3[0]-p2[0], p3[1]-p2[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = (v1[0]**2 + v1[1]**2)**0.5
        mag2 = (v2[0]**2 + v2[1]**2)**0.5
        if mag1 > 0 and mag2 > 0:
            cos_angle = dot / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp for numerical stability
            import math
            angle = math.degrees(math.acos(cos_angle))
            if angle < 60:
                errors.append(f"Sharp vertex at {p2}: {angle:.0f}° < 60°")
    
    return len(errors) == 0, errors
```

## Subdivision Strategy

When a cell fails, split it:

```python
from shapely.ops import split
from shapely.geometry import LineString

def subdivide_cell(polygon: Polygon) -> list[Polygon]:
    """Split a failing cell into two smaller cells."""
    # Find longest axis
    min_rect = polygon.minimum_rotated_rectangle
    coords = list(min_rect.exterior.coords)
    
    # Create splitting line perpendicular to short axis, through centroid
    centroid = polygon.centroid
    # ... geometry to create perpendicular line ...
    
    split_line = LineString([...])  # Line crossing the polygon
    result = split(polygon, split_line)
    return list(result.geoms)
```

## Output Format

Each candidate folder contains:

### metadata.json
```json
{
  "uuid": "abc123",
  "created": "2026-02-04T12:00:00Z",
  "source": {
    "type": "image",
    "url": "https://...",
    "search_query": "maple leaf"
  },
  "segmentation": {
    "method": "watershed",
    "params": {"threshold": 0.3}
  },
  "cells": {
    "count": 23,
    "all_valid": true,
    "subdivision_iterations": 2
  },
  "dimensions": {
    "width_mm": 150,
    "height_mm": 200
  }
}
```

### cells.svg
Vector file with each cell as a separate path, labeled with ID.

### render.png
Preview image: cells colored by simulated wood grain, gaps in dark epoxy color.

## Parallel Coordination

- Each worker writes to its own `candidates/{uuid}/` folder
- No locking needed — UUIDs prevent collision
- Workers read `config.json` for shared settings
- Workers append to `logs/{worker_id}.log`
- Orchestrator monitors `candidates/` folder count

## When To Stop

Keep generating until:
- `candidates/` contains 50+ patterns, OR
- Human triggers review (creates `REVIEW_NOW` file), OR
- Error rate exceeds 50% (something is broken)

Then notify human to pick winners.

## CLI Flags You Should Use

When spawned, you'll receive:
- `--worker-id N` — your worker number (for logging)
- `--batch-size M` — generate M candidates then exit
- `--seed-type {image|procedural|mixed}` — what kind of seeds to use

## Remember

1. **Manufacturing constraints generate beauty** — don't fight them, embrace subdivision
2. **Filesystem is memory** — write state to files, not context
3. **Fail fast, retry** — if segmentation produces garbage, try different params
4. **Log everything** — helps debug parallel issues
5. **UUID everything** — no collisions between workers
