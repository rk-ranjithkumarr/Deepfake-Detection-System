import sys
sys.path.insert(0, "src")
import streamlit as st
import torch
from PIL import Image
from predict import load, predict_image

st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️")
st.title("🕵️ Deepfake Image Detector")

@st.cache_resource
def get_model():
    return load()

try:
    model, arch = get_model()
except Exception:
    st.error("No trained model found. Run `python src/train.py` first."); st.stop()

f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
if f:
    img = Image.open(f)
    p_real, vis = predict_image(model, arch, img)
    c1, c2 = st.columns(2)
    c1.image(img, caption="Input", use_container_width=True)
    c2.image(vis, caption="Grad-CAM (regions that influenced the decision)", use_container_width=True)
    if p_real > 0.5: st.success(f"REAL — {p_real*100:.1f}% confidence")
    else: st.error(f"FAKE — {(1-p_real)*100:.1f}% confidence")
    st.caption("Detectors are probabilistic and can be wrong on unseen generators. Don't use as sole evidence.")
