#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Define the arguments
MODEL_NAME="meta-llama/Llama-3.2-1B"
DTYPE="bfloat16"
USE_FLASH_ATTENTION_2=true
N_MEM_TOKENS=(1)
MAX_LENGTH=(768 2560) #(64 128 512 1024 1280 1568 2048 3072) #(64 128 512 1024) #(8 16 32 64 96 128 192 256 384 512 768 1024 1280 1568 2048 3072)
NUM_ITERATIONS=5000
NUM_SAMPLES=50
LR=1e-2
WEIGHT_DECAY=0.01

# Run the Python script with the arguments
# python train.py \
#     --model_name "$MODEL_NAME" \
#     --dtype "$DTYPE" \
#     $( [ "$USE_FLASH_ATTENTION_2" = true ] && echo "--use_flash_attention_2" ) \
#     --N_mem_tokens "${N_MEM_TOKENS[@]}" \
#     --max_length "${MAX_LENGTH[@]}" \
#     --num_iterations "$NUM_ITERATIONS" \
#     --num_samples "$NUM_SAMPLES" \
#     --lr "$LR" \
#     --weight_decay "$WEIGHT_DECAY"

SHUFFLED=true
NUM_ITERATIONS=5000
EARLY_STOP=2000
MAX_LENGTH=(1280) #(128 192 256 384 512 640 768)
NUM_SAMPLES=100

python train.py \
    --model_name "$MODEL_NAME" \
    --dtype "$DTYPE" \
    $( [ "$USE_FLASH_ATTENTION_2" = true ] && echo "--use_flash_attention_2" ) \
    $( [ "$SHUFFLED" = true ] && echo "--shuffled" ) \
    --N_mem_tokens "${N_MEM_TOKENS[@]}" \
    --max_length "${MAX_LENGTH[@]}" \
    --num_iterations "$NUM_ITERATIONS" \
    --num_samples "$NUM_SAMPLES" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --early_stopping_patience "$EARLY_STOP"