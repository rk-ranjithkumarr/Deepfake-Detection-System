"""Creates a SYNTHETIC toy dataset so you can test the full pipeline in minutes.
'real' = smooth natural-like images; 'fake' = same + periodic upsampling-style artifacts.
Replace with a real dataset (see README) for meaningful results."""
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
rng = np.random.default_rng(0)
RAW = Path(__file__).parent / "data" / "raw"

def base():
    small = rng.random((8, 8, 3))
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((256, 256), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(2))
    return np.asarray(img).astype(np.float32)

def add_noise(a): return np.clip(a + rng.normal(0, 3, a.shape), 0, 255)

def add_artifact(a):
    yy, xx = np.mgrid[0:256, 0:256]
    pat = 10 * np.sin(2 * np.pi * xx / rng.choice([4, 8])) * np.sin(2 * np.pi * yy / 4)
    return np.clip(a + pat[..., None] + rng.normal(0, 3, a.shape), 0, 255)

for cls, fn in (("real", add_noise), ("fake", add_artifact)):
    (RAW / cls).mkdir(parents=True, exist_ok=True)
    for i in range(300):
        Image.fromarray(fn(base()).astype(np.uint8)).save(RAW / cls / f"{cls}_{i}.png")
print("Demo data created in", RAW)
