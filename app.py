# app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# ========== Streamlit Page Setup ==========
st.set_page_config(page_title="Bone Fracture Enhancement", layout="wide")
st.markdown(
    "<h1 style='text-align:center; color:#2c3e50;'>Interactive Bone X-ray Transformations</h1>",
    unsafe_allow_html=True
)

# ========== Utility Functions ==========
def load_image_gray(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

def apply_negative(img: np.ndarray) -> np.ndarray:
    return 255 - img

def apply_gamma(img: np.ndarray, g: float) -> np.ndarray:
    norm = img / 255.0
    return np.uint8(255 * (norm ** g))

def plot_image_and_hist(image: np.ndarray, title: str):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title(title)
    ax[0].axis('off')

    hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist_scaled = hist / hist.max() * 0.9   # make bars fill most of the panel

    ax[1].bar(bins[:-1], hist_scaled, width=1, color='gray')
    ax[1].set_title("Histogram")
    ax[1].set_xlim([0, 256])
    ax[1].set_ylim([0, 1])        # fixed axis so bars are always visible
    ax[1].set_ylabel("Relative Count")

    plt.tight_layout()
    st.pyplot(fig)

# ========== UI Controls ==========
uploaded = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
st.sidebar.header("Controls")
gamma_val = st.sidebar.slider("Gamma Value", 0.1, 5.0, 1.0, 0.1)
choice = st.sidebar.radio("Choose Operation", ["Original", "Negative", "Gamma", "Negative + Gamma"])

# ========== Main Logic ==========
# try to locate a default sample in same folder
DEMO_PATH = pathlib.Path("sample.jpg")

if uploaded:
    image = load_image_gray(uploaded.read())
else:
    if DEMO_PATH.exists():
        with open(DEMO_PATH, "rb") as f:
            image = load_image_gray(f.read())
        st.info("Showing built-in demo X-ray (upload your own to override).")
    else:
        st.warning("No image uploaded and no demo file found.")
        st.stop()

# apply transformation
if choice == "Negative":
    processed = apply_negative(image)
    title = "Negative"
elif choice == "Gamma":
    processed = apply_gamma(image, gamma_val)
    title = f"Gamma (γ={gamma_val:.1f})"
elif choice == "Negative + Gamma":
    processed = apply_gamma(apply_negative(image), gamma_val)
    title = f"Negative + Gamma (γ={gamma_val:.1f})"
else:
    processed = image
    title = "Original"

plot_image_and_hist(processed, title)
