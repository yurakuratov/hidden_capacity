#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Define the arguments
MODEL_NAME="meta-llama/Llama-3.2-3B"
DTYPE="bfloat16"
USE_FLASH_ATTENTION_2=true
N_MEM_TOKENS=(1)
MAX_LENGTH=(64 128 512 1024 1280 1568 2048 2560 3072)
NUM_ITERATIONS=5000
NUM_SAMPLES=50
LR=1e-2
WEIGHT_DECAY=0.01

TEXTS_PATH=./data/fanfics_1k_chunks.csv
SAVE_PATH=./runs_fanfics

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
    --weight_decay "$WEIGHT_DECAY" \
    --texts_path "$TEXTS_PATH" \
    --save_path "$SAVE_PATH"
