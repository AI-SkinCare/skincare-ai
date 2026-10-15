"""split CSV 기반 Dataset / DataLoader."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import config as C


def build_transforms(train, image_size=C.IMAGE_SIZE):
    # 정면 1024² / 측면 512² 모두 정사각형 → 비율 유지한 채 image_size 로 리사이즈
    resize = T.Resize((image_size, image_size), antialias=True)
    norm = [T.ToTensor(), T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)]
    if not train:
        return T.Compose([resize, *norm])
    return T.Compose([
        resize,
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.1, contrast=0.1),  # 색조(hue)는 진단 단서라 건드리지 않음
        *norm,
    ])


class SkinDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.paths = self.df["image_path"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.labels[i], i


def build_loader(split, split_dir=C.SPLIT_DIR, batch_size=C.BATCH_SIZE,
                 num_workers=C.NUM_WORKERS, image_size=C.IMAGE_SIZE, train=None, seed=C.SEED):
    is_train = (split == "train") if train is None else train
    ds = SkinDataset(Path(split_dir) / f"{split}.csv", build_transforms(is_train, image_size))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False, generator=g,
        persistent_workers=num_workers > 0,
    )
