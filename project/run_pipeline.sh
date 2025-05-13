set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 training trainig test"
  exit 1
fi

TRAIN_DIR=$1
VAL_DIR=$2
TEST_DIR=$3

# --- Pipeline settings ---
TRAIN_JSON="train_split.json"
VAL_JSON="val_split.json"
SPLIT_RATIO=0.8

NUM_CLASSES=3
IMG_SIZE=224
BS=32
LR=1e-4
EPOCHS=10
K=3

OUT_MODEL="best_ft.pth"
IDX_OUT="gallery_idx.faiss"
KEYS_OUT="gallery_keys.json"
OUT_JSON="submission.json"

# --- Run full pipeline ---
python3 src/main.py full \
  --train_dir "${TRAIN_DIR}" \
  --val_dir   "${VAL_DIR}" \
  --test_dir  "${TEST_DIR}" \
  --train_json  "${TRAIN_JSON}" \
  --val_json    "${VAL_JSON}" \
  --split_ratio "${SPLIT_RATIO}" \
  --num_classes "${NUM_CLASSES}" \
  --img_size    "${IMG_SIZE}" \
  --bs          "${BS}" \
  --lr          "${LR}" \
  --epochs      "${EPOCHS}" \
  --k           "${K}" \
  --out_model   "${OUT_MODEL}" \
  --idx_out     "${IDX_OUT}" \
  --keys_out    "${KEYS_OUT}" \
  --out_json    "${OUT_JSON}"

echo "✅ All done! Results in ${OUT_JSON}"