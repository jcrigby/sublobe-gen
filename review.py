#!/usr/bin/env python3
"""
review.py — Interactive review tool for selecting pattern candidates

Opens each candidate's render in sequence and lets you select favorites.
"""

import os
import sys
import json
import shutil
from pathlib import Path

WORKSPACE = Path(__file__).parent
CANDIDATES_DIR = WORKSPACE / "candidates"
SELECTED_DIR = WORKSPACE / "selected"


def get_candidates():
    """Get list of candidate directories sorted by creation time."""
    candidates = []
    for d in CANDIDATES_DIR.iterdir():
        if d.is_dir():
            metadata_file = d / "metadata.json"
            render_file = d / "render.png"
            if metadata_file.exists() and render_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                candidates.append({
                    "path": d,
                    "metadata": metadata,
                    "render": render_file
                })
    
    # Sort by creation time
    candidates.sort(key=lambda x: x["metadata"].get("created", ""))
    return candidates


def open_image(path):
    """Open an image with the system viewer."""
    import subprocess
    import platform
    
    system = platform.system()
    if system == "Darwin":  # macOS
        subprocess.run(["open", str(path)])
    elif system == "Linux":
        # Try common viewers
        for viewer in ["xdg-open", "feh", "eog", "display"]:
            if shutil.which(viewer):
                subprocess.Popen([viewer, str(path)], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
                return
        print(f"No image viewer found. File: {path}")
    else:  # Windows
        os.startfile(str(path))


def print_candidate_info(candidate, index, total):
    """Print information about a candidate."""
    m = candidate["metadata"]
    print("\n" + "="*60)
    print(f"Candidate {index+1}/{total}: {m['uuid']}")
    print("="*60)
    print(f"  Method:     {m['source']['method']}")
    print(f"  Cells:      {m['cells']['count']}")
    print(f"  Size:       {m['dimensions']['width_mm']}mm × {m['dimensions']['height_mm']}mm")
    print(f"  Iterations: {m['cells']['subdivision_iterations']}")
    print(f"  Valid:      {'✓ Yes' if m['cells']['all_valid'] else '✗ No'}")
    print("-"*60)


def interactive_review():
    """Run interactive review session."""
    SELECTED_DIR.mkdir(exist_ok=True)
    
    candidates = get_candidates()
    if not candidates:
        print("No candidates found in candidates/")
        return
    
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + "  SUBLOBE PATTERN REVIEW".center(58) + "║")
    print("╠" + "═"*58 + "╣")
    print("║" + f"  {len(candidates)} candidates to review".ljust(58) + "║")
    print("║" + "  Commands: [y]es, [n]o, [s]kip, [q]uit, [g]allery".ljust(58) + "║")
    print("╚" + "═"*58 + "╝")
    
    selected = []
    
    for i, candidate in enumerate(candidates):
        print_candidate_info(candidate, i, len(candidates))
        
        # Open the image
        open_image(candidate["render"])
        
        while True:
            response = input("\n  Select this pattern? [y/n/s/q/g]: ").strip().lower()
            
            if response == 'y':
                selected.append(candidate)
                dest = SELECTED_DIR / candidate["path"].name
                if not dest.exists():
                    shutil.copytree(candidate["path"], dest)
                print(f"  → Selected! ({len(selected)} total)")
                break
            elif response == 'n':
                print("  → Skipped")
                break
            elif response == 's':
                print("  → Skipped (will review later)")
                break
            elif response == 'q':
                print(f"\n  Exiting. {len(selected)} patterns selected.")
                return selected
            elif response == 'g':
                # Show gallery of all remaining
                for c in candidates[i:i+10]:
                    open_image(c["render"])
                print("  → Opened next 10 images")
            else:
                print("  Unknown command. Use y/n/s/q/g")
        
        if len(selected) >= 10:
            print(f"\n  Reached 10 selections. Continue? [y/n]: ", end="")
            if input().strip().lower() != 'y':
                break
    
    print("\n" + "="*60)
    print(f"REVIEW COMPLETE: {len(selected)} patterns selected")
    print(f"Selected patterns saved to: {SELECTED_DIR}")
    print("="*60)
    
    return selected


def batch_review():
    """Generate manifest.json for HTML reviewer and optionally open gallery."""
    candidates = get_candidates()
    if not candidates:
        print("No candidates found")
        return
    
    # Generate manifest.json for HTML reviewer
    manifest = []
    for c in candidates:
        m = c["metadata"]
        manifest.append({
            "uuid": m.get("uuid", c["path"].name),
            "folder": c["path"].name,
            "cells": m.get("cells", {}).get("count", "?"),
            "method": m.get("source", {}).get("method", "?"),
            "width": m.get("dimensions", {}).get("width_mm", "?"),
            "height": m.get("dimensions", {}).get("height_mm", "?")
        })
    
    manifest_path = CANDIDATES_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated manifest: {manifest_path} ({len(manifest)} candidates)")
    
    # Also generate the HTML gallery (legacy)
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Sublobe Pattern Review</title>
    <style>
        body { font-family: sans-serif; background: #1a1a1a; color: #fff; margin: 20px; }
        h1 { text-align: center; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #2a2a2a; border-radius: 8px; overflow: hidden; }
        .card img { width: 100%; height: 250px; object-fit: contain; background: #333; }
        .card .info { padding: 10px; font-size: 12px; }
        .card .info .uuid { font-weight: bold; color: #4a9; }
        .card input[type="checkbox"] { transform: scale(1.5); margin-right: 10px; }
        .card label { cursor: pointer; display: flex; align-items: center; padding: 10px; background: #333; }
        .selected { border: 3px solid #4a9; }
        #controls { position: fixed; bottom: 20px; right: 20px; background: #333; padding: 15px; border-radius: 8px; }
        #controls button { padding: 10px 20px; margin: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Sublobe Pattern Review</h1>
    <p style="text-align:center">Click patterns to select. Selected patterns will be copied to <code>selected/</code></p>
    
    <div class="gallery">
"""
    
    for c in candidates:
        m = c["metadata"]
        # Use relative path for images
        img_path = f"candidates/{c['path'].name}/render.png"
        
        html += f"""
        <div class="card" data-uuid="{m['uuid']}">
            <img src="{img_path}" alt="{m['uuid']}">
            <label>
                <input type="checkbox" class="selector" value="{c['path'].name}">
                Select this pattern
            </label>
            <div class="info">
                <div class="uuid">{m['uuid']}</div>
                <div>Method: {m['source']['method']} | Cells: {m['cells']['count']}</div>
                <div>Size: {m['dimensions']['width_mm']}×{m['dimensions']['height_mm']}mm</div>
            </div>
        </div>
"""
    
    html += """
    </div>
    
    <div id="controls">
        <div id="count">Selected: 0</div>
        <button onclick="exportSelected()">Export Selected</button>
    </div>
    
    <script>
        document.querySelectorAll('.selector').forEach(cb => {
            cb.addEventListener('change', () => {
                cb.closest('.card').classList.toggle('selected', cb.checked);
                document.getElementById('count').textContent = 
                    'Selected: ' + document.querySelectorAll('.selector:checked').length;
            });
        });
        
        function exportSelected() {
            const selected = Array.from(document.querySelectorAll('.selector:checked'))
                .map(cb => cb.value);
            
            // Create a text file with the list
            const text = selected.join('\\n');
            const blob = new Blob([text], {type: 'text/plain'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'selected_patterns.txt';
            a.click();
            
            alert('Download selected_patterns.txt, then run:\\n\\npython review.py --import selected_patterns.txt');
        }
    </script>
</body>
</html>
"""
    
    gallery_path = WORKSPACE / "gallery.html"
    with open(gallery_path, "w") as f:
        f.write(html)
    
    print(f"Gallery saved to: {gallery_path}")
    print("Open this file in a browser to review patterns visually.")
    
    # Try to open in browser
    import webbrowser
    webbrowser.open(f"file://{gallery_path}")


def import_selections(filename):
    """Import selections from a text file (one folder name per line)."""
    SELECTED_DIR.mkdir(exist_ok=True)
    
    with open(filename) as f:
        names = [line.strip() for line in f if line.strip()]
    
    count = 0
    for name in names:
        src = CANDIDATES_DIR / name
        dest = SELECTED_DIR / name
        if src.exists() and not dest.exists():
            shutil.copytree(src, dest)
            count += 1
            print(f"  Copied: {name}")
    
    print(f"\nImported {count} patterns to selected/")


def generate_manifest():
    """Generate manifest.json for HTML reviewer."""
    candidates = get_candidates()
    if not candidates:
        print("No candidates found")
        return
    
    manifest = []
    for c in candidates:
        m = c["metadata"]
        manifest.append({
            "uuid": m.get("uuid", c["path"].name),
            "folder": c["path"].name,
            "cells": m.get("cells", {}).get("count", "?"),
            "method": m.get("source", {}).get("method", "?"),
            "width": m.get("dimensions", {}).get("width_mm", "?"),
            "height": m.get("dimensions", {}).get("height_mm", "?")
        })
    
    manifest_path = CANDIDATES_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated manifest: {manifest_path} ({len(manifest)} candidates)")
    print(f"Now open review.html in your browser.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Review pattern candidates")
    parser.add_argument("--gallery", action="store_true", help="Generate HTML gallery (legacy)")
    parser.add_argument("--manifest", action="store_true", help="Generate manifest.json for review.html")
    parser.add_argument("--import", dest="import_file", help="Import selections from file")
    
    args = parser.parse_args()
    
    if args.manifest:
        generate_manifest()
    elif args.gallery:
        batch_review()
    elif args.import_file:
        import_selections(args.import_file)
    else:
        interactive_review()


if __name__ == "__main__":
    main()
