












"""   
import gradio as gr
import numpy as np
from PIL import Image
import io


def apply_svd_color(img, k):
    img = img.resize((256, 256))   
    img = img.convert("RGB")       

    A = np.array(img).astype(float)

    R, G, B = A[:,:,0], A[:,:,1], A[:,:,2]

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


def psnr(original, compressed):
    orig = np.array(original).astype(float)
    comp = np.array(compressed).astype(float)

    mse = np.mean((orig - comp) ** 2)
    if mse == 0:
        return 100
    return round(20 * np.log10(255.0 / np.sqrt(mse)), 2)


def compress_ui(img, k):
    if img is None:
        return None, "Please upload an image"

    compressed = apply_svd_color(img, k)
    return compressed, "Color SVD compression applied"



gr.Interface(
    fn=compress_ui,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(5, 100, value=30, step=5, label="k (rank)")
    ],
    outputs=[
        gr.Image(label="Compressed Image"),
        gr.Textbox(label="Quality")
    ],
    title="Image Compression using SVD (WORKING)",
    description="Guaranteed working SVD demo for learning Linear Algebra"
).launch()
"""