import argparse
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from config import DATA_DIR, MODEL_DIR, OUT_DIR
from dataset import get_loaders
from model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--ckpt", default=str(MODEL_DIR / "best.pt"))
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state"]); model.to(device).eval()
    loaders, _ = get_loaders(Path(a.data))

    probs, labels = [], []
    with torch.no_grad():
        for x, y in loaders["test"]:
            probs += torch.sigmoid(model(x.to(device))).squeeze(1).cpu().tolist()
            labels += y.tolist()
    probs, labels = np.array(probs), np.array(labels)
    preds = (probs > 0.5).astype(int)

    print(classification_report(labels, preds, target_names=["fake", "real"], digits=4))
    print("ROC-AUC:", round(roc_auc_score(labels, probs), 4))
    OUT_DIR.mkdir(exist_ok=True)
    ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=["fake", "real"]).plot()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=120); plt.close()
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve")
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=120)
    print("Saved plots to", OUT_DIR)


if __name__ == "__main__":
    main()
