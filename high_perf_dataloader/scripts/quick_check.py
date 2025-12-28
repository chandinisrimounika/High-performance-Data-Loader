import tempfile
import os
import numpy as np
import cv2
import torch
from dataset import ImageDataset
from loader import HighPerfDataLoader, PrefetchDataLoader

# create temp images
with tempfile.TemporaryDirectory() as tmp:
    for i in range(4):
        img = (np.random.rand(256, 256, 3) * 255).astype('uint8')
        p = os.path.join(tmp, f"img_{i}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    ds = ImageDataset(tmp)
    assert len(ds) == 4

    base = HighPerfDataLoader(ds, batch_size=2, num_workers=0)
    pre = PrefetchDataLoader(base, device='cpu')

    for batch in pre:
        # if tensor or nested
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
        print('OK')
        break

print('Quick check finished')