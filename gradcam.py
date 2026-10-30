import numpy as np, cv2, torch


class GradCAM:
    """Highlights image regions that drove the prediction."""
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0].detach()))

    def __call__(self, x):
        self.model.zero_grad()
        out = self.model(x)
        out.sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1)).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.sigmoid(out).item()


def overlay(img_rgb, cam):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (0.55 * img_rgb + 0.45 * heat).astype(np.uint8)
