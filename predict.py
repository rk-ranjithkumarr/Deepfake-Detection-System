import argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image
from config import MODEL_DIR, OUT_DIR, IMG_SIZE
from dataset import get_transforms
from model import build_model, target_layer
from gradcam import GradCAM, overlay


def load(ckpt=MODEL_DIR / "best.pt", device="cpu"):
    ck = torch.load(ckpt, map_location=device)
    m = build_model(ck["arch"], pretrained=False)
    m.load_state_dict(ck["state"]); m.to(device).eval()
    return m, ck["arch"]


def predict_image(model, arch, img: Image.Image, device="cpu", cam=True):
    img = img.convert("RGB")
    x = get_transforms(False)(img).unsqueeze(0).to(device)
    if cam:
        c, p_real = GradCAM(model, target_layer(model, arch))(x)
        base = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        return p_real, overlay(base, c)
    with torch.no_grad():
        return torch.sigmoid(model(x)).item(), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="image file or folder")
    p.add_argument("--ckpt", default=str(MODEL_DIR / "best.pt"))
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, arch = load(a.ckpt, dev)
    path = Path(a.path)
    files = [path] if path.is_file() else sorted(f for f in path.rglob("*")
             if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    OUT_DIR.mkdir(exist_ok=True)
    for f in files:
        p_real, vis = predict_image(model, arch, Image.open(f), dev)
        label = "REAL" if p_real > 0.5 else "FAKE"
        conf = p_real if p_real > 0.5 else 1 - p_real
        print(f"{f.name}: {label} ({conf*100:.1f}% confidence)")
        Image.fromarray(vis).save(OUT_DIR / f"gradcam_{f.stem}.png")


if __name__ == "__main__":
    main()
