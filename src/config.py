from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw"
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "preds_test"

IMG_SIZE = 256
BATCH_SIZE = 2
LR = 1e-3
EPOCHS = 30

DEVICE = "cuda"  # or "cpu"
THRESHOLD = 0.5
