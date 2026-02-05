#!/bin/bash
# orchestrator.sh — Spawns multiple Claude Code workers in parallel
#
# Usage: ./orchestrator.sh [num_workers] [candidates_target]
# Example: ./orchestrator.sh 4 50
#
# This runs until candidates/ has enough patterns, then prompts for human review.

set -e

NUM_WORKERS=${1:-4}
TARGET_CANDIDATES=${2:-50}
WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
CANDIDATES_DIR="$WORKSPACE/candidates"
LOGS_DIR="$WORKSPACE/logs"
PIDS_FILE="$WORKSPACE/.worker_pids"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Sublobe Pattern Generator - Orchestrator      ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Workers: $NUM_WORKERS                                      ║${NC}"
echo -e "${GREEN}║  Target:  $TARGET_CANDIDATES candidates                          ║${NC}"
echo -e "${GREEN}║  Output:  $CANDIDATES_DIR${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Ensure directories exist
mkdir -p "$CANDIDATES_DIR" "$LOGS_DIR"

# Clean up function
cleanup() {
    echo -e "\n${YELLOW}Shutting down workers...${NC}"
    if [ -f "$PIDS_FILE" ]; then
        while read pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PIDS_FILE"
        rm -f "$PIDS_FILE"
    fi
    echo -e "${GREEN}Done.${NC}"
}

trap cleanup EXIT INT TERM

# Count current candidates
count_candidates() {
    find "$CANDIDATES_DIR" -maxdepth 1 -type d | wc -l | tr -d ' '
}

# Spawn a single worker
spawn_worker() {
    local worker_id=$1
    local log_file="$LOGS_DIR/worker_${worker_id}.log"
    
    echo -e "${GREEN}[Orchestrator]${NC} Spawning worker $worker_id → $log_file"
    
    # Claude Code command with worker-specific settings
    # --dangerously-skip-permissions: Don't ask for confirmation (Ralph-style)
    # --print: Output to stdout (we redirect to log)
    # The worker instructions tell Claude what to do
    
    claude --dangerously-skip-permissions \
        --print \
        "You are worker $worker_id. Read CLAUDE.md for full context. 
         Generate pattern candidates until you have created 10, then exit.
         Use --worker-id $worker_id when logging.
         Seed type: mixed (images and procedural).
         Work autonomously. Don't ask questions. Just generate.
         If something fails, log it and try a different approach.
         Save each candidate to candidates/{uuid}/ with metadata.json, cells.svg, render.png." \
        >> "$log_file" 2>&1 &
    
    echo $! >> "$PIDS_FILE"
}

# Monitor loop
monitor() {
    while true; do
        local count=$(count_candidates)
        local running=$(wc -l < "$PIDS_FILE" 2>/dev/null || echo "0")
        
        # Check for dead workers and respawn
        if [ -f "$PIDS_FILE" ]; then
            local new_pids=""
            local dead_count=0
            while read pid; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids="$new_pids$pid\n"
                else
                    ((dead_count++)) || true
                fi
            done < "$PIDS_FILE"
            echo -e "$new_pids" > "$PIDS_FILE"
            running=$(wc -l < "$PIDS_FILE" 2>/dev/null | tr -d ' ')
            
            # Respawn dead workers if we haven't hit target
            if [ "$count" -lt "$TARGET_CANDIDATES" ]; then
                for ((i=0; i<dead_count; i++)); do
                    local new_id=$((RANDOM % 1000))
                    spawn_worker "$new_id"
                done
            fi
        fi
        
        echo -e "${YELLOW}[Monitor]${NC} Candidates: $count/$TARGET_CANDIDATES | Workers: $running"
        
        # Check if we've hit target
        if [ "$count" -ge "$TARGET_CANDIDATES" ]; then
            echo -e "\n${GREEN}═══════════════════════════════════════════════${NC}"
            echo -e "${GREEN}  TARGET REACHED: $count candidates generated${NC}"
            echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
            echo ""
            echo -e "${YELLOW}Starting human review...${NC}"
            trigger_review
            break
        fi
        
        # Check for manual review trigger
        if [ -f "$WORKSPACE/REVIEW_NOW" ]; then
            rm -f "$WORKSPACE/REVIEW_NOW"
            echo -e "\n${YELLOW}Manual review triggered${NC}"
            trigger_review
            break
        fi
        
        sleep 30
    done
}

# Human review process
trigger_review() {
    # Kill remaining workers
    cleanup
    
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           HUMAN REVIEW TIME                    ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Candidates are in: $CANDIDATES_DIR"
    echo ""
    echo "Each folder contains:"
    echo "  - metadata.json  (source info, cell count)"
    echo "  - cells.svg      (vector paths)"
    echo "  - render.png     (preview image)"
    echo ""
    echo -e "${YELLOW}Instructions:${NC}"
    echo "1. Browse the render.png files"
    echo "2. Move your favorite 10 folders to: selected/"
    echo "3. Run: ./orchestrator.sh --continue"
    echo ""
    echo "Or use the review helper:"
    echo "  python review.py"
    echo ""
    
    # Generate quick gallery
    if command -v montage &> /dev/null; then
        echo "Generating gallery image..."
        find "$CANDIDATES_DIR" -name "render.png" -print0 | \
            head -z -n 50 | \
            xargs -0 montage -geometry 200x200+5+5 -tile 10x "$WORKSPACE/gallery.png" 2>/dev/null || true
        echo "Gallery saved to: $WORKSPACE/gallery.png"
    fi
}

# Main
echo "" > "$PIDS_FILE"

# Spawn initial workers
for ((i=1; i<=NUM_WORKERS; i++)); do
    spawn_worker "$i"
    sleep 2  # Stagger startup
done

# Start monitoring
monitor
