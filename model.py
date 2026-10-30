import torch.nn as nn
from torchvision import models


def build_model(name="resnet18", pretrained=True):
    """Binary classifier: 1 output logit -> sigmoid = P(real)."""
    if name == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = _load(models.resnet18, w)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 1))
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = _load(models.efficientnet_b0, w)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, 1))
    else:
        raise ValueError("model must be resnet18 or efficientnet_b0")
    return m


def _load(fn, weights):
    try:
        return fn(weights=weights)
    except Exception as e:  # offline / download failure
        print(f"[warn] could not load pretrained weights ({e}); using random init")
        return fn(weights=None)


def target_layer(model, name):
    return model.layer4[-1] if name == "resnet18" else model.features[-1]
