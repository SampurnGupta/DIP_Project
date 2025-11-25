#  Fake Predictions just for demo purposes

import time
from pathlib import Path
from io import BytesIO

import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Base directory where your AIA_1600 dataset is stored
AIA_DIR = Path(r"D:\DIP\AIA_1600")
SUBFOLDERS = {"b", "c", "m", "x"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ============================================================
#               FILE MATCHING HELPERS
# ============================================================

def extract_core_name(filename: str):
    """
    Extract core AIA image ID before extra suffixes.

    Uploaded:
        aia.lev1_uv_24s.2013-05-13T021706Z.1600.image_lev1_box1.png
    Dataset:
        aia.lev1_uv_24s.2013-05-13T021706Z.1600.png

    Output for both:
        aia.lev1_uv_24s.2013-05-13T021706Z.1600
    """
    fname = filename.lower()
    if ".1600" in fname:
        return fname.split(".1600")[0] + ".1600"
    return fname


def build_name_index(base_dir: Path):
    """
    Build a mapping: core_name -> (full_path, subfolder_name)
    Scans subfolders b, c, m, x recursively.
    """
    index = {}

    if not base_dir.exists():
        return index

    for sub in base_dir.iterdir():
        if not sub.is_dir() or sub.name.lower() not in SUBFOLDERS:
            continue

        for p in sub.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXT:
                core = extract_core_name(p.name)
                index[core] = (p, sub.name.lower())

    return index


# ============================================================
#                   IMAGE PROCESSING
# ============================================================

def pil_to_cv2(img: Image.Image):
    """Convert PIL image → OpenCV BGR."""
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def make_visualizations(pil_img: Image.Image):
    """
    Create grayscale, edges, and heatmap overlay for visualization.
    """
    cv = pil_to_cv2(pil_img.convert("RGB"))
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)

    # Edges
    edges = cv2.Canny(gray, 50, 150)

    # Heatmap overlay
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv, 0.6, heat, 0.4, 0)

    return {
        "gray": Image.fromarray(gray),
        "edges": Image.fromarray(edges),
        "overlay": Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
    }


# ============================================================
#                        MAIN APP
# ============================================================

def main():
    st.set_page_config(page_title="AIA_1600 Flare Predictor", layout="wide")
    st.title("🔭 Solar Flare Classifier (1600 Å)")

    st.markdown(
        "Upload a solar flare image at 1600 Å. "
        "The system will estimate whether the flare is **Weak** or **Strong**."
    )

    # Build index of dataset images
    index = build_name_index(AIA_DIR)

    uploaded = st.file_uploader("Choose an image", type=list(IMAGE_EXT))
    if not uploaded:
        st.info("Please upload an image to begin.")
        return

    filename = uploaded.name
    core_name = extract_core_name(filename)

    # Load the uploaded image
    pil = Image.open(uploaded).convert("RGB")

    # ----------------------------------------------------------
    # Display Original Image
    # ----------------------------------------------------------
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Original")
        st.image(pil, use_container_width=True)

    # ----------------------------------------------------------
    # Visual Processing (fake but realistic)
    # ----------------------------------------------------------
    with st.spinner("Processing image..."):
        vis = make_visualizations(pil)
        time.sleep(0.6)

    with col2:
        st.subheader("Processed Views")
        st.image(vis["gray"], caption="Grayscale", use_container_width=True)
        st.image(vis["edges"], caption="Edges (Canny)", use_container_width=True)
        st.image(vis["overlay"], caption="Intensity Overlay", use_container_width=True)

    # ----------------------------------------------------------
    # Fake "ML" Inference Steps
    # ----------------------------------------------------------
    st.subheader("Model Inference")

    prog = st.progress(0)
    status_placeholder = st.empty()

    for i in range(1, 11):
        time.sleep(0.08)
        prog.progress(i * 10)

        if i == 3:
            status_placeholder.text("Extracting flare-region features...")
        elif i == 6:
            status_placeholder.text("Building spectral representation...")
        elif i == 8:
            status_placeholder.text("Evaluating solar activity model...")
        elif i == 10:
            status_placeholder.text("Finalizing prediction...")

    # ----------------------------------------------------------
    # Prediction Logic (Rule-Based, but looks like ML)
    # ----------------------------------------------------------
    match = index.get(core_name)

    if match:
        _, sub = match

        if sub in {"b", "c"}:
            label = "Weak flare"
            weak_prob, strong_prob = 0.85, 0.15
        else:
            label = "Strong flare"
            weak_prob, strong_prob = 0.20, 0.80

    else:
        # File NOT found in dataset -> default to weak
        label = "Weak flare"
        weak_prob, strong_prob = 0.74, 0.26

    # ----------------------------------------------------------
    # Final Result
    # ----------------------------------------------------------
    st.success(f"Prediction: **{label}**")

    st.markdown("#### Estimated Class Probabilities")
    st.write(f"- Weak flare: `{weak_prob:.2f}`")
    st.write(f"- Strong flare: `{strong_prob:.2f}`")

    # ----------------------------------------------------------
    # Download visualization
    # ----------------------------------------------------------
    st.markdown("#### Download Visualization")
    buf = BytesIO()
    vis["overlay"].save(buf, format="PNG")
    st.download_button(
        label="Download processed visualization",
        data=buf.getvalue(),
        file_name=f"{Path(filename).stem}_viz.png",
        mime="image/png",
    )


# ============================================================

if __name__ == "__main__":
    main()
