set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 train test"
  exit 1
fi

TRAIN_DIR=$1
TEST_DIR=$2

# --- Pipeline settings ---
TRAIN_JSON="train_split.json"
VAL_JSON="val_split.json"
SPLIT_RATIO=0.8

NUM_CLASSES=102
IMG_SIZE=224
BS=32
LR=0.1
EPOCHS=10
K=2

OUT_MODEL="best_ft.pth"
IDX_OUT="gallery_idx.faiss"
KEYS_OUT="gallery_keys.json"
OUT_JSON="submission.json"

# --- Run full pipeline ---
python3 src/main.py full \
  --train_dir "${TRAIN_DIR}" \
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