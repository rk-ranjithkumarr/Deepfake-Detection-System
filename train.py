import argparse, json, time
from pathlib import Path
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import DATA_DIR, MODEL_DIR, OUT_DIR
from dataset import get_loaders
from model import build_model


def run_epoch(model, loader, device, crit, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    loss_sum = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            with torch.autocast(device.type, enabled=scaler is not None):
                out = model(x)
                loss = crit(out, y)
            if train:
                opt.zero_grad()
                if scaler:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += ((out > 0).float() == y).sum().item()
            n += x.size(0)
    return loss_sum / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--no-pretrained", action="store_true")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    loaders, c2i = get_loaders(Path(a.data), a.batch)
    print("Classes:", c2i)  # fake=0, real=1

    model = build_model(a.model, not a.no_pretrained).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    MODEL_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
    hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
    best = 0
    for ep in range(1, a.epochs + 1):
        t = time.time()
        tl, ta = run_epoch(model, loaders["train"], device, crit, opt, scaler)
        vl, va = run_epoch(model, loaders["val"], device, crit)
        sched.step()
        for k, v in zip(hist, (tl, vl, ta, va)): hist[k].append(v)
        print(f"Epoch {ep}/{a.epochs} | train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} | {time.time()-t:.0f}s")
        if va >= best:
            best = va
            torch.save({"state": model.state_dict(), "arch": a.model,
                        "class_to_idx": c2i}, MODEL_DIR / "best.pt")
            print("  saved best model")

    json.dump(hist, open(OUT_DIR / "history.json", "w"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["va_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["tr_acc"], label="train"); ax[1].plot(hist["va_acc"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUT_DIR / "training_curves.png", dpi=120)
    print(f"Done. Best val acc: {best:.4f}")


if __name__ == "__main__":
    main()
