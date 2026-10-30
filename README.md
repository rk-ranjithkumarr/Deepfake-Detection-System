# Deepfake Image Detection Using Deep Learning

Transfer-learning (ResNet18 / EfficientNet-B0, PyTorch) binary classifier: **REAL vs FAKE**,
with ROC/confusion-matrix evaluation, Grad-CAM explanations and a Streamlit web app.

## Project structure
```
deepfake_detector/
├── requirements.txt        # libraries
├── setup.bat / setup.sh    # one-click install (Windows / Linux-Mac)
├── generate_demo_data.py   # synthetic toy data (pipeline smoke test only)
├── prepare_dataset.py      # splits data/raw -> data/dataset (70/15/15)
├── app.py                  # Streamlit web app
├── src/ config.py dataset.py model.py train.py evaluate.py gradcam.py predict.py
├── data/  models/  outputs/
```

## Step 1 - Install Python
Python 3.9 - 3.12 (64-bit). Check: `python --version`

## Step 2 - Unzip and install libraries
Windows: double-click `setup.bat`  (or run it in CMD)
Linux/Mac: `bash setup.sh`

Manual way:
```
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
NVIDIA GPU? Install the CUDA build of PyTorch first from https://pytorch.org/get-started/locally
(then `pip install -r requirements.txt`). CPU works too, just slower.

Verify: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

## Step 3 - Get data
**Quick test (2 min):**
```
python generate_demo_data.py
```
**Real project - download a dataset and put images here:**
```
data/raw/real/   <- real face images
data/raw/fake/   <- deepfake / AI-generated images
```
Good public datasets (search on Kaggle): "140k Real and Fake Faces", "Deepfake and real images",
"CIFAKE", FaceForensics++ (frames), Celeb-DF, DFDC (frames). Aim for 5,000+ images per class.

## Step 4 - Split into train / val / test
```
python prepare_dataset.py
```

## Step 5 - Train
```
python src/train.py --model resnet18 --epochs 10
python src/train.py --model efficientnet_b0 --epochs 10 --batch 16   # alternative
```
Best weights -> `models/best.pt`, curves -> `outputs/training_curves.png`.
(Pretrained ImageNet weights download automatically; no internet = random init with a warning.
For the synthetic demo add `--no-pretrained --epochs 5`.)

## Step 6 - Evaluate
```
python src/evaluate.py
```
Prints precision/recall/F1/ROC-AUC; saves `outputs/confusion_matrix.png`, `outputs/roc_curve.png`.

## Step 7 - Predict on your own images
```
python src/predict.py path/to/image.jpg
python src/predict.py path/to/folder
```
Prints REAL/FAKE + confidence and saves a Grad-CAM heatmap to `outputs/`.

## Step 8 - Web app
```
streamlit run app.py
```
Opens http://localhost:8501 - upload an image, get verdict + heatmap.

## How it works
1. Images are resized to 224x224 and normalised.
2. A pretrained CNN backbone is fine-tuned; the head outputs one logit, sigmoid = P(real).
3. Loss: BCE-with-logits, AdamW optimizer, cosine LR schedule, best-val checkpointing.
4. Grad-CAM shows which regions drove the decision.
Labels follow ImageFolder alphabetical order: fake=0, real=1.

## Troubleshooting
| Problem | Fix |
|---|---|
| `No images found in data/raw/real` | Put images in the folders first (Step 3) |
| CUDA out of memory | `--batch 8` |
| `ModuleNotFoundError: config` | Run scripts from the project root: `python src/train.py` |
| DataLoader hangs on Windows | Set `workers=0` in `src/dataset.py` |
| Very high accuracy on demo data | Expected - it is synthetic and easy |

## Limitations
Detectors generalise poorly to generators not seen in training, and compression/resizing can hide
artifacts. Treat output as a probabilistic signal, never as sole proof.
