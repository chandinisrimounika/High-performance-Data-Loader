import os
import sys
import numpy as np
import cv2
import torch

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import ImageDataset


def make_temp_images(tmpdir, count=4, size=(256, 256)):
    for i in range(count):
        img = (np.random.rand(size[0], size[1], 3) * 255).astype(np.uint8)
        p = os.path.join(tmpdir, f"img_{i}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def test_image_dataset_basic(tmp_path):
    tmpdir = str(tmp_path)
    make_temp_images(tmpdir, count=3)

    ds = ImageDataset(tmpdir)
    assert len(ds) == 3

    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert item.dtype == torch.float32
    assert item.ndim == 3
    assert item.shape[0] == 3

    # values should be approximately in 0..1
    eps = 1e-6
    assert item.max() <= 1.0 + eps
    assert item.min() >= 0.0 - eps
