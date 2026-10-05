#!/usr/bin/env bash
# Two-view (left/right crop) inference with SALMA, merged into one prediction JSON per video.
# Run from models/salma:
#   bash scripts/inference.sh
set -euo pipefail

#######################################
# PATHS (edit if needed)
#######################################
DATA_ROOT="../../data"                                   # Prompting-MammAlps download (see root README)
CKPT_PATH="./output/salma_large/ckpt_cl2_500.pth"        # Output from training
OUTPUT_DIR="./output/salma_large_predictions"
NPROC=1                                                  # GPUs used by torchrun

mkdir -p "${OUTPUT_DIR}"

echo "Launching SALMA inference..."
OMP_NUM_THREADS=1 torchrun --nproc_per_node="${NPROC}" --master_port=29501 \
    run_two_view_inference.py \
    --output_dir "${OUTPUT_DIR}" \
    --csv_path "${DATA_ROOT}/metadata" \
    --label_mapping_path "${DATA_ROOT}/metadata/label_mapping.json" \
    --dense_annot_path "${DATA_ROOT}/annotations" \
    --video_root_path "${DATA_ROOT}/videos" \
    --ckpt_path "${CKPT_PATH}" \
    --model_name salma_large_patch16_224 \
    --num_frames 16 \
    --num_queries 20 \
    --batch_size 8 \
    --test_cache_size 128 \
    --temporal_stride 4

echo "Inference completed. Merged predictions: ${OUTPUT_DIR}/multi_view/"
