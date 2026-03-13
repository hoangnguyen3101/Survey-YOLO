#!/bin/bash

# ==============================================================================
# YOLO Parallel Inference Script (Single GPU Optimized)
# ==============================================================================
# Description: Runs YOLO inference using multiple model weights in parallel.
# ==============================================================================

# Colors for logging
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
SOURCE_DIR="inference/source"
WEIGHTS_DIR="weights"

# Models to use for inference
WEIGHTS=("yolo11n.pt" "yolo11s.pt")

echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Starting Parallel Inference Session${NC}"
echo -e "${BLUE}======================================================================${NC}"

mkdir -p "$SOURCE_DIR"

for weight in "${WEIGHTS[@]}"; do
    experiment_name="${weight%.*}_infer"
    
    echo -e "${BLUE}[INFO]${NC} Launching inference for: ${GREEN}$weight${NC}"
    
    # Run in background using infer.py
    python "$PROJECT_ROOT/modules/infer.py" \
        --model "$WEIGHTS_DIR/$weight" \
        --source "$SOURCE_DIR" \
        --project "$PROJECT_ROOT/inference" \
        --name "$experiment_name" \
        --device 0 \
        &
done

echo -e "${BLUE}[INFO]${NC} All inference processes have been launched in the background."

wait

echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Parallel Inference Completed Successfully${NC}"
echo -e "${BLUE}======================================================================${NC}"
