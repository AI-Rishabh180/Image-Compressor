
import numpy as np

def psnr(original, compressed):
    orig = np.array(original).astype(float)
    comp = np.array(compressed).astype(float)

    mse = np.mean((orig - comp) ** 2)
    if mse == 0:
        return 100

    return round(20 * np.log10(255.0 / np.sqrt(mse)), 2)
