"""Split data/raw/{real,fake} into data/dataset/{train,val,test}/{real,fake} (70/15/15)."""
import random, shutil
from pathlib import Path
ROOT = Path(__file__).parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "dataset"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(42)

if OUT.exists(): shutil.rmtree(OUT)
for cls in ("real", "fake"):
    files = [f for f in (RAW / cls).rglob("*") if f.suffix.lower() in EXT]
    if not files: raise SystemExit(f"No images found in {RAW/cls}")
    random.shuffle(files)
    n = len(files); a, b = int(.7 * n), int(.85 * n)
    for split, part in (("train", files[:a]), ("val", files[a:b]), ("test", files[b:])):
        d = OUT / split / cls; d.mkdir(parents=True, exist_ok=True)
        for f in part: shutil.copy(f, d / f"{f.parent.name}_{f.name}")
    print(f"{cls}: {n} images -> train {a}, val {b-a}, test {n-b}")
print("Dataset ready at", OUT)
