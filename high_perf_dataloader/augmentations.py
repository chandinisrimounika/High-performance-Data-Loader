import numpy as np

def random_flip(image):
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
    return image.copy()

def random_crop(image, size=224):
    h, w, _ = image.shape
    if h < size or w < size:
        return image.copy()
    top = np.random.randint(0, h - size)
    left = np.random.randint(0, w - size)
    return image[top:top + size, left:left + size].copy()
