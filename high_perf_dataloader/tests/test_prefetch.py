import os
import sys
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import ImageDataset
from loader import HighPerfDataLoader, PrefetchDataLoader


def make_temp_images(tmpdir, count=6, size=(128, 128)):
    for i in range(count):
        img = (np.random.rand(size[0], size[1], 3) * 255).astype(np.uint8)
        p = os.path.join(tmpdir, f"img_{i}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def test_prefetch_cpu(tmp_path):
    tmpdir = str(tmp_path)
    make_temp_images(tmpdir)

    ds = ImageDataset(tmpdir)
    base = HighPerfDataLoader(ds, batch_size=2, num_workers=0)
    loader = PrefetchDataLoader(base, device='cpu')

    for batch in loader:
        # ensure data are CPU tensors
        if torch.is_tensor(batch):
            assert batch.device.type == 'cpu'
        else:
            # batch could be list/tuple of tensors
            def check(obj):
                if torch.is_tensor(obj):
                    assert obj.device.type == 'cpu'
                elif isinstance(obj, (list, tuple)):
                    for x in obj:
                        check(x)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        check(v)
            check(batch)
        break


import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prefetch_gpu(tmp_path):
    tmpdir = str(tmp_path)
    make_temp_images(tmpdir)

    ds = ImageDataset(tmpdir)
    base = HighPerfDataLoader(ds, batch_size=2, num_workers=0)
    loader = PrefetchDataLoader(base, device='cuda')

    for batch in loader:
        if torch.is_tensor(batch):
            assert batch.device.type == 'cuda'
        else:
            def check(obj):
                if torch.is_tensor(obj):
                    assert obj.device.type == 'cuda'
                elif isinstance(obj, (list, tuple)):
                    for x in obj:
                        check(x)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        check(v)
            check(batch)
        break
