import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from functools import lru_cache
from augmentations import random_flip, random_crop

class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]

    def __len__(self):
        return len(self.image_paths)

    @lru_cache(maxsize=512)
    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        img = img.astype(np.float32) / 255.0

        img = random_flip(img)
        img = random_crop(img)

        return torch.from_numpy(img).permute(2, 0, 1)
