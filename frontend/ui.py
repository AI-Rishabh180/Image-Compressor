
import gradio as gr
from backend.svd_logic import apply_svd_color
from database.metrices import psnr

def compress_ui(img, k):
    if img is None:
        return None, "Please upload an image"

    compressed = apply_svd_color(img, k)
    return compressed, "Compression successful (aspect ratio preserved)"


def create_ui():
    return gr.Interface(
        fn=compress_ui,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Slider(5, 100, value=30, step=5, label="k (rank)")
        ],
        outputs=[
            gr.Image(label="Compressed Image"),
            gr.Textbox(label="Quality")
        ],
        title="Image Compression using SVD (Color)",
        description="Color image compression using SVD (channel-wise)"
    )

