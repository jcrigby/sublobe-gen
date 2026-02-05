# Sublobe Inlay Pattern Generator

A multi-agent system for generating manufacturable stained-glass-style wood inlay patterns.

## The Concept

**Wood = glass panes. Epoxy = lead came.**

This system generates organic patterns that:
1. Look like stained glass or Tiffany lamps
2. Are actually manufacturable with table saw + 3D-printed sanding blocks
3. Self-subdivide when a shape is "too complex" — manufacturing constraints generate visual detail

### Manufacturing Model

Each sublobe stick is cut on **two opposing faces**:

```
1. Cut Face A (series of table saw passes)
2. Sand Face A with 3D-printed sanding block
3. Put Face A into 3D-printed CRADLE (matches Face A profile)
4. Cut Face B (opposite side) while cradle holds the piece
5. Sand Face B
```

The cradle is the key jig—it grips the finished Face A profile while you work on Face B.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate patterns (single worker, 10 patterns)
python worker.py --worker-id 1 --batch-size 10

# 3. Review results
python review.py --gallery
# Opens HTML gallery in browser

# 4. Select favorites → they go to selected/
```

## Multi-Agent Mode (The Ralph Way)

For parallel generation with multiple Claude Code instances:

```bash
# Make orchestrator executable
chmod +x orchestrator.sh

# Launch 4 workers, stop when 50 candidates exist
./orchestrator.sh 4 50
```

### What the orchestrator does:
1. Spawns N Claude Code instances in parallel
2. Each reads `CLAUDE.md` for context and runs autonomously
3. Workers write to `candidates/{uuid}/` (no collisions)
4. Orchestrator monitors count, respawns dead workers
5. When target reached → triggers human review

### Key Claude Code flags used:
- `--dangerously-skip-permissions` — Don't ask for confirmation (unattended operation)
- `--print` — Output to stdout (redirected to logs)

## Manual Multi-Window Approach

If the orchestrator doesn't work for your setup, do it manually:

```bash
# Terminal 1
cd sublobe-gen
claude "Read CLAUDE.md. Generate 10 pattern candidates. Work autonomously."

# Terminal 2
cd sublobe-gen  
claude "Read CLAUDE.md. Generate 10 pattern candidates. Work autonomously."

# Terminal 3
cd sublobe-gen
claude "Read CLAUDE.md. Generate 10 pattern candidates. Work autonomously."

# ... as many as you want
```

Each instance reads `CLAUDE.md`, understands the domain, and generates patterns independently.

## Directory Structure

```
sublobe-gen/
├── CLAUDE.md           # Context file Claude reads automatically
├── config.json         # Shared settings (DFM constraints, colors, etc.)
├── worker.py           # Python implementation of generation pipeline
├── review.py           # Human review tool
├── orchestrator.sh     # Multi-agent launcher
├── requirements.txt    # Python dependencies
│
├── candidates/         # Generated patterns land here
│   └── {uuid}/
│       ├── metadata.json
│       ├── cells.svg
│       └── render.png
│
├── selected/           # Human picks go here
├── references/         # Source images (optional)
└── logs/               # Per-worker logs
```

## Configuration

Edit `config.json` to change:

```json
{
  "dfm_constraints": {
    "min_feature_width_mm": 3.0,    // Narrower than blade kerf
    "min_internal_radius_mm": 2.0,  // Sanding block must be printable
    "blade_kerf_mm": 3.0            // Your table saw blade width
  },
  "segmentation": {
    "target_cells_min": 15,         // Min cells per pattern
    "target_cells_max": 40          // Max cells per pattern
  },
  "output": {
    "wood_colors": ["#D4A574", ...], // Preview colors
    "epoxy_color": "#1a1a1a",        // Gap color
    "gap_width_mm": 2.0              // Visual gap in preview
  }
}
```

## Human Review

Three modes:

### Browser (recommended)
```bash
# Generate manifest for HTML reviewer
python review.py --manifest

# Open review.html in your browser
# (just double-click the file or use: xdg-open review.html)
```

The HTML reviewer:
- Shows all patterns in a grid
- Click 👍/👎 to rate
- Saves state in browser localStorage (survives refresh)
- Filter by: All / Pending / Liked / Disliked
- Export liked patterns to a text file
- Keyboard shortcuts: `L` like, `D` dislike, `→` next, `←` prev

### Interactive (terminal)
```bash
python review.py
```
Opens each image, you type `y`/`n` to select.

### Legacy gallery
```bash
python review.py --gallery
```
Generates old-style `gallery.html`.

## What Gets Generated

Each candidate folder contains:

| File | Purpose |
|------|---------|
| `metadata.json` | Source info, cell count, dimensions, validity |
| `cells.svg` | Vector paths for each cell (for CAD import) |
| `render.png` | Preview image with wood colors and epoxy gaps |

## The DFM Loop

The core algorithm:

```
1. Generate initial cells (Voronoi, grid, organic)
2. For each cell:
   - Check: too narrow? sharp angles? invalid polygon?
   - If fail → SUBDIVIDE (split into 2 cells)
   - Subdivision adds another "vein line" — more visual detail!
3. Repeat until all cells pass
4. Render preview
```

Manufacturing constraints literally generate beauty. A complex organic shape that can't be made becomes two simpler shapes with an epoxy line between them.

## Extending to Image-Based Generation

The current implementation uses procedural generation (Voronoi, grids). To add image-based:

1. Add to `requirements.txt`:
   ```
   scikit-image>=0.21.0
   requests>=2.28.0
   ```

2. Implement in `worker.py`:
   ```python
   from skimage import segmentation, io
   
   def segment_from_image(image_path):
       img = io.imread(image_path)
       # Use watershed, SLIC, or edge detection
       cells = segmentation.slic(img, n_segments=30)
       # Convert to Shapely polygons
       ...
   ```

3. Add image search to find references:
   ```python
   # Search Wikimedia Commons, Unsplash, etc.
   # Download to references/
   # Use as segmentation input
   ```

## Troubleshooting

**Workers crash immediately:**
- Check `logs/worker_N.log` for errors
- Usually missing dependencies: `pip install -r requirements.txt`

**No patterns generated:**
- Verify `candidates/` directory exists
- Check if worker has write permission

**All patterns look the same:**
- Random seed might be fixed — check numpy random state
- Increase `target_cells_max` in config

**Patterns too complex (many cells):**
- Subdivision is aggressive — relax DFM constraints in config
- Increase `min_feature_width_mm`

## Next Steps (After Human Selection)

Once you have 10 winners in `selected/`:

1. **Import to CAD** — Load `cells.svg` into Fusion 360 or similar
2. **Generate sanding blocks** — Extrude each cell profile, add handle
3. **Generate retainer panels** — 2D layout with cell positions + gaps
4. **Generate cast mold** — Outer boundary of full pattern

These are future extensions — the current system just generates and validates the patterns.

## Credits

Sublobe inlay system designed by John, February 2026.
Multi-agent orchestration inspired by Ralph (Jeffrey Huntley) and Claude Code task system.
