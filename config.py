from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "dataset"      # train/val/test -> fake/real
RAW_DIR = ROOT / "data" / "raw"           # raw/real, raw/fake
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
