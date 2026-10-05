#!/usr/bin/env bash
# Two-stage curriculum training of SALMA (paper recipe). Run from models/salma:
#   bash scripts/train.sh
set -euo pipefail

#######################################
# PATHS (edit if needed)
#######################################
DATA_ROOT="../../data"                                   # Prompting-MammAlps download (see root README)
ENCODER_PATH="./pretrained_models/vit_l_hybrid_pt_800e.pth"
OUTPUT_DIR="./output/salma_large"
NPROC=8                                                  # GPUs used by torchrun

CSV_PATH="${DATA_ROOT}/metadata"
LABEL_MAPPING_PATH="${DATA_ROOT}/metadata/label_mapping.json"
DENSE_ANNOT_PATH="${DATA_ROOT}/annotations"
VIDEO_ROOT_PATH="${DATA_ROOT}/videos"

mkdir -p "${OUTPUT_DIR}"

# Arguments shared by both stages.
# Learning rates are base values, scaled by (batch_size * NPROC / 256) in train_salma.py.
COMMON_ARGS=(
    --output_dir "${OUTPUT_DIR}"
    --log_dir "${OUTPUT_DIR}"
    --csv_path "${CSV_PATH}"
    --label_mapping_path "${LABEL_MAPPING_PATH}"
    --dense_annot_path "${DENSE_ANNOT_PATH}"
    --video_root_path "${VIDEO_ROOT_PATH}"
    --model_name salma_large_patch16_224
    --num_frames 16
    --num_queries 20
    --epochs 500
    --warmup_epochs 20
    --lr 1e-4
    --min_lr 1e-5
    --warmup_lr 1e-6
    --save_ckpt_freq 100
    --vload_threads 8
    --loss_object 6.0
    --loss_bbox 4.0
    --loss_giou 4.0
    --loss_cont 4.0
    --loss_l2_norm 1e-2
    --eos_coef 20
    --rand_freq 
    --min_rts 1 
    --max_rts 30
    --jitter
    --balancing_sampling
    --amp
    --no_val
)

#######################################
# STAGE 1: localization and tracking (classification heads frozen)
#######################################
echo "Launching SALMA training, curriculum stage 1..."
OMP_NUM_THREADS=1 torchrun --nproc_per_node="${NPROC}" --master_port=29500 \
    train_salma.py "${COMMON_ARGS[@]}" \
    --experiment_name salma_large_cl1 \
    --encoder_path "${ENCODER_PATH}" \
    --out_ckpt_prefix ckpt_cl1 \
    --batch_size 8 \
    --loss_species 0.0 \
    --loss_activities 0.0 \
    --loss_actions 0.0 \
    --loss_dage 0.0 \
    --loss_dsex 0.0 \
    --loss_weather 0.0 \
    --freeze_object_heads

#######################################
# STAGE 2: all heads
#######################################
echo "Launching SALMA training, curriculum stage 2..."
OMP_NUM_THREADS=1 torchrun --nproc_per_node="${NPROC}" --master_port=29500 \
    train_salma.py "${COMMON_ARGS[@]}" \
    --experiment_name salma_large_cl2 \
    --ckpt_path "${OUTPUT_DIR}/ckpt_cl1_500.pth" \
    --out_ckpt_prefix ckpt_cl2 \
    --batch_size 6 \
    --loss_species 4.0 \
    --loss_activities 6.0 \
    --loss_actions 6.0 \
    --loss_dage 2.0 \
    --loss_dsex 2.0 \
    --loss_weather 2.0

echo "Training completed. Final checkpoint: ${OUTPUT_DIR}/ckpt_cl2_500.pth"
