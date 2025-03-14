#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Define the arguments
MODEL_NAME="facebook/opt-350m"
DTYPE="float32"
USE_FLASH_ATTENTION_2=false
N_MEM_TOKENS=(1)
MAX_LENGTH=(64 96 128 192 256 384 512 768 1024 1280 1568)
NUM_ITERATIONS=5000
NUM_SAMPLES=50
LR=1e-2
WEIGHT_DECAY=0.01


# Run the Python script with the arguments
python train.py \
    --model_name "$MODEL_NAME" \
    --dtype "$DTYPE" \
    $( [ "$USE_FLASH_ATTENTION_2" = true ] && echo "--use_flash_attention_2" ) \
    --N_mem_tokens "${N_MEM_TOKENS[@]}" \
    --max_length "${MAX_LENGTH[@]}" \
    --num_iterations "$NUM_ITERATIONS" \
    --num_samples "$NUM_SAMPLES" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY"