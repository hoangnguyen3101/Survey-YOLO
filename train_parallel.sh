#!/bin/bash

# Default models to train if no arguments provided
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("yolov3.pt" "yolov5s.pt" "yolov8s.pt" "yolov10s.pt")

echo "Starting parallel training for: ${MODELS[*]}"

for model in "${MODELS[@]}"; do
    name="${model%.*}_exp"
    echo ">> Training $model..."
    
    python modules/train.py \
        --model "$model" \
        --project runs \
        --name "$name" \
        --device 0 &
done

wait
echo "All training tasks finished."
