import numpy as np
from PIL import Image

def resize_keep_ratio(img, max_size=256):
    w, h = img.size

    if max(w, h) <= max_size:
        return img

    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)

    return img.resize((new_w, new_h))



def apply_svd_color(img, k):
    img = img.convert("RGB")
    img = resize_keep_ratio(img, max_size=256)

    A = np.array(img).astype(float)

    R, G, B = A[:, :, 0], A[:, :, 1], A[:, :, 2]

    def svd_channel(channel, k):
        h, w = channel.shape
        k = min(k, h, w)
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        return (U[:, :k] * S[:k]) @ Vt[:k, :]

    Rk = svd_channel(R, k)
    Gk = svd_channel(G, k)
    Bk = svd_channel(B, k)

    Ak = np.stack([Rk, Gk, Bk], axis=2)
    Ak = np.clip(Ak, 0, 255).astype("uint8")

    return Image.fromarray(Ak)
