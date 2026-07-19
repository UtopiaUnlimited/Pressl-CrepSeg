#!/usr/bin/env bash
# Train the cached 3D-Aware DPT baseline, then evaluate its best validation-mIoU
# checkpoint on the already-generated temporal_v2 test cache.
#
# Run from the repository root, preferably inside tmux:
#   bash scripts/train_3d_aware_then_test.sh
#
# If caches live outside the repository, pass their absolute paths, e.g.
#   bash scripts/train_3d_aware_then_test.sh \
#     --train-cache /ssd/pastis_cache/..._train \
#     --val-cache /ssd/pastis_cache/..._val \
#     --test-cache /ssd/pastis_cache/..._test

set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-presl}"
CONFIG="${CONFIG:-configs/galileo_3d_aware_dpt.yaml}"
CACHE_ROOT="${CACHE_ROOT:-data/cache/galileo-base-patch8}"
CACHE_STEM="monthly12_tile64_patch4_hl3-6-9-12_temporal-v2_tfp16"
TRAIN_CACHE="${TRAIN_CACHE:-${CACHE_ROOT}/${CACHE_STEM}_train}"
VAL_CACHE="${VAL_CACHE:-${CACHE_ROOT}/${CACHE_STEM}_val}"
TEST_CACHE="${TEST_CACHE:-${CACHE_ROOT}/${CACHE_STEM}_test}"
CHECKPOINT="${CHECKPOINT:-checkpoints/galileo_3d_aware_dpt_native_skip_late_fusion_seed42_cached/best_val_miou.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-output/3d_aware_dpt_p0_test}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
PHENOLOGY_CONFIG="${PHENOLOGY_CONFIG:-}"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_3d_aware_then_test.sh [options]

Options:
  --env NAME                 Conda environment (default: presl)
  --config PATH              Decoder config
  --train-cache PATH         Existing temporal_v2 train cache
  --val-cache PATH           Existing temporal_v2 val cache
  --test-cache PATH          Existing temporal_v2 test cache
  --checkpoint PATH          Expected best_val_miou checkpoint path
  --output-dir PATH          Directory for test prediction panels
  --epochs N                 Training epochs (default: 50)
  --test-batch-size N        Inference batch size (default: 8)
  --phenology-config PATH    Optional P1/P2 overlay; used consistently in train and test
  -h, --help                 Show this help

The script exits on a training failure. It starts test only if training exits
successfully and the expected best_val_miou.pt file exists.
EOF
}

while (($#)); do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --train-cache) TRAIN_CACHE="$2"; shift 2 ;;
    --val-cache) VAL_CACHE="$2"; shift 2 ;;
    --test-cache) TEST_CACHE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
    --test-batch-size) TEST_BATCH_SIZE="$2"; shift 2 ;;
    --phenology-config) PHENOLOGY_CONFIG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Twelve DataLoader processes already occupy most of a 14-core host. Prevent
# NumPy/BLAS from creating another pool of CPU threads inside every worker.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

for cache_dir in "$TRAIN_CACHE" "$VAL_CACHE" "$TEST_CACHE"; do
  if [[ ! -d "$cache_dir" ]] || ! compgen -G "$cache_dir/*.npz" > /dev/null; then
    echo "Missing or empty cache directory: $cache_dir" >&2
    exit 1
  fi
done

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi
if [[ -n "$PHENOLOGY_CONFIG" && ! -f "$PHENOLOGY_CONFIG" ]]; then
  echo "Phenology config not found: $PHENOLOGY_CONFIG" >&2
  exit 1
fi

run_id="$(date +%Y%m%d_%H%M%S)"
log_dir="logs/automation"
mkdir -p "$log_dir" "$OUTPUT_DIR"

common_args=(--config "$CONFIG" --cache-format temporal_v2 --temporal-dtype float16)
if [[ -n "$PHENOLOGY_CONFIG" ]]; then
  common_args+=(--phenology-config "$PHENOLOGY_CONFIG")
fi

echo "[$(date -Is)] Start training"
conda run --no-capture-output -n "$ENV_NAME" python -B scripts/train_cached.py \
  "${common_args[@]}" \
  --train-cache-dir "$TRAIN_CACHE" \
  --val-cache-dir "$VAL_CACHE" \
  --epochs "$TRAIN_EPOCHS" \
  --device cuda 2>&1 | tee "$log_dir/3d_aware_train_${run_id}.log"

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "Training finished but expected checkpoint was not found: $CHECKPOINT" >&2
  exit 1
fi

echo "[$(date -Is)] Start test using $CHECKPOINT"
conda run --no-capture-output -n "$ENV_NAME" python -B scripts/eval_cached.py \
  "${common_args[@]}" \
  --checkpoint "$CHECKPOINT" \
  --cache-dir "$TEST_CACHE" \
  --split test \
  --batch-size "$TEST_BATCH_SIZE" \
  --device cuda \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee "$log_dir/3d_aware_test_${run_id}.log"

echo "[$(date -Is)] Training and test completed successfully."
