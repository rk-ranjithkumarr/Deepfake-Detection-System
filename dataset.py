from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import IMG_SIZE, MEAN, STD


def get_transforms(train: bool):
    # NOTE: no blur/heavy resize augmentation on purpose - it destroys the
    # subtle artifacts that deepfake detectors rely on.
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_loaders(data_dir, batch_size=32, workers=2):
    sets = {s: datasets.ImageFolder(str(data_dir / s), get_transforms(s == "train"))
            for s in ("train", "val", "test")}
    loaders = {s: DataLoader(d, batch_size=batch_size, shuffle=(s == "train"),
                             num_workers=workers) for s, d in sets.items()}
    return loaders, sets["train"].class_to_idx   # {'fake': 0, 'real': 1}
