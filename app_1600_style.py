# Python script to simulate 1600 Å–style solar images from visible-light photos.
# Uses OpenCV for image processing and Streamlit for the web UI.

import cv2
import numpy as np
import streamlit as st


# ================== Core DIP functions ==================

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray, clip_limit=3.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    return clahe.apply(gray)


def unsharp_mask(gray, sigma=3, strength=1.5):
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(gray, 1 + strength, blur, -strength, 0)
    return sharp


def gamma_correction(img, gamma=0.9):
    img = img.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def normalize_to_8bit(img):
    norm = cv2.normalize(
        img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    return norm.astype(np.uint8)


def add_vignette(img, strength=0.5):
    """Radial vignette to darken edges/background."""
    h, w = img.shape[:2]
    y, x = np.indices((h, w))
    x = (x - w / 2) / (w / 2)
    y = (y - h / 2) / (h / 2)
    r = np.sqrt(x ** 2 + y ** 2)

    vignette = np.exp(- (r ** 2) / (2 * (1 - strength) ** 2))
    vignette = vignette[..., np.newaxis]

    img_f = img.astype(np.float32)
    img_v = img_f * vignette
    img_v = np.clip(img_v, 0, 255).astype(np.uint8)
    return img_v


def apply_yellow_colormap(gray):
    """Map grayscale to dark-yellow 1600 Å–style pseudo-color."""
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_AUTUMN)
    b, g, r = cv2.split(colored)
    r = cv2.multiply(r, 0.9)  # slightly soften red
    g = cv2.multiply(g, 1.1)  # boost green → more yellowish
    return cv2.merge((b, g, r))


def auto_crop_solar_disk(img_bgr, margin_ratio=0.05):
    """Detect solar disk and crop a square region around it."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return img_bgr, (0, 0, w, h)

    c = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(c)
    cx, cy, radius = int(cx), int(cy), int(radius)

    side = int(2 * radius * (1 + margin_ratio))
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    cropped = img_bgr[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)


# ===== Extra DIP helpers: DoG, log compression, grain =====

def dog_enhance(gray, sigma1=1.0, sigma2=3.0, strength=0.8):
    """
    Difference-of-Gaussians to emphasize mid-scale structure.
    Higher strength → more 'grainy solar texture'.
    """
    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
    dog = cv2.subtract(g1, g2)  # mid-scale features
    enhanced = cv2.addWeighted(gray, 1.0, dog, strength, 0)
    return enhanced


def log_compression(img, gain=1.5):
    """
    Log compression to tame very bright flare cores.
    Useful before gamma mapping.
    """
    img = img.astype(np.float32) / 255.0
    # log1p keeps it safe near 0
    img = np.log1p(gain * img) / np.log1p(gain)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def add_grain(gray, strength=0.05):
    """
    Add mild Gaussian noise to mimic CCD-like grain.
    strength ≈ std-dev in [0, 1] domain.
    """
    img = gray.astype(np.float32) / 255.0
    noise = np.random.normal(0.0, strength, img.shape).astype(np.float32)
    noisy = img + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)


def simulate_1600_style(
    img_bgr,
    clahe_clip=3.0,
    clahe_tile=8,
    unsharp_sigma=3.0,
    unsharp_strength=1.2,
    gamma=0.9,
    vignette_strength=0.5,
    use_log=False,
    log_gain=1.5,
    structure_sigma=3.0,
    structure_strength=0.8,
    grain_strength=0.05,
):
    """
    Full pipeline: visible/white-light solar image → 1600 Å–style pseudo image.

    Steps:
    1) Gray
    2) CLAHE
    3) DoG mid-scale enhancement (optional via strength slider)
    4) Unsharp mask
    5) Normalize
    6) Optional log compression
    7) Gamma
    8) Optional grain
    9) Vignette
    10) Yellow colormap
    """
    # 1) grayscale
    gray = to_gray(img_bgr)

    # 2) local contrast
    gray_clahe = apply_clahe(
        gray,
        clip_limit=clahe_clip,
        tile_grid_size=(clahe_tile, clahe_tile),
    )

    # 3) mid-scale structure (DoG)
    if structure_strength > 0:
        detail = dog_enhance(
            gray_clahe,
            sigma1=1.0,
            sigma2=structure_sigma,
            strength=structure_strength,
        )
    else:
        detail = gray_clahe

    # 4) high-frequency boost (unsharp)
    sharp = unsharp_mask(
        detail,
        sigma=unsharp_sigma,
        strength=unsharp_strength,
    )

    # 5) normalize to 0–255
    norm = normalize_to_8bit(sharp)

    # 6) optional log compression
    if use_log:
        norm = log_compression(norm, gain=log_gain)

    # 7) gamma mapping
    gamma_adj = gamma_correction(norm, gamma=gamma)

    # 8) optional grain
    if grain_strength > 0:
        gamma_adj = add_grain(gamma_adj, strength=grain_strength)

    # 9) vignette in 3-channel space
    gamma_3ch = cv2.cvtColor(gamma_adj, cv2.COLOR_GRAY2BGR)
    vignetted = add_vignette(gamma_3ch, strength=vignette_strength)

    # 10) final grayscale → yellow colormap
    vignetted_gray = cv2.cvtColor(vignetted, cv2.COLOR_BGR2GRAY)
    colored = apply_yellow_colormap(vignetted_gray)

    # We return:
    # - gray_clahe/detail-stage image for center column (structure + contrast)
    return gray, detail, colored


# ================== Streamlit UI ==================

st.set_page_config(page_title="1600 Å Style Sun Image Simulator", layout="wide")
st.title("1600 Å–Style Solar Image Simulator")

st.write(
    "Upload a detailed/color Sun image and adjust parameters to visually "
    "approximate a 1600 Å (far-UV) style: high contrast, grainy, dark background."
)

uploaded = st.file_uploader(
    "Upload a solar image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Upload a solar image to begin.")
    st.stop()

# Decode uploaded bytes to OpenCV BGR image
file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Could not read the image. Try another file.")
    st.stop()

# Sidebar controls
st.sidebar.header("Preprocessing")
auto_crop = st.sidebar.checkbox("Auto-crop around solar disk", value=True)
margin = st.sidebar.slider("Crop margin (fraction of diameter)", 0.0, 0.4, 0.08)

st.sidebar.header("CLAHE (local contrast)")
clahe_clip = st.sidebar.slider("Clip limit", 1.0, 6.0, 3.0, step=0.1)
clahe_tile = st.sidebar.slider("Tile size", 4, 16, 8, step=2)

st.sidebar.header("Structure / texture")
structure_sigma = st.sidebar.slider(
    "Mid-scale sigma (DoG)", 1.0, 8.0, 3.0, step=0.5
)
structure_strength = st.sidebar.slider(
    "Structure strength", 0.0, 2.0, 0.8, step=0.1
)
grain_strength = st.sidebar.slider(
    "Grain strength", 0.0, 0.2, 0.05, step=0.01
)

st.sidebar.header("Detail enhancement")
unsharp_sigma = st.sidebar.slider("Unsharp sigma", 1.0, 6.0, 3.0, step=0.5)
unsharp_strength = st.sidebar.slider("Unsharp strength", 0.5, 3.0, 1.2, step=0.1)

st.sidebar.header("Tone & vignette")
use_log = st.sidebar.checkbox("Use log compression (pre-gamma)", value=False)
log_gain = st.sidebar.slider("Log gain", 0.5, 5.0, 1.5, step=0.1)
gamma = st.sidebar.slider("Gamma", 0.5, 1.8, 0.9, step=0.05)
vignette_strength = st.sidebar.slider("Vignette strength", 0.0, 0.9, 0.5, step=0.05)

# Optional auto-crop
if auto_crop:
    img_bgr_cropped, crop_box = auto_crop_solar_disk(img_bgr, margin_ratio=margin)
else:
    img_bgr_cropped, crop_box = img_bgr, (0, 0, img_bgr.shape[1], img_bgr.shape[0])

gray_base, gray_struct, styled_color = simulate_1600_style(
    img_bgr_cropped,
    clahe_clip=clahe_clip,
    clahe_tile=clahe_tile,
    unsharp_sigma=unsharp_sigma,
    unsharp_strength=unsharp_strength,
    gamma=gamma,
    vignette_strength=vignette_strength,
    use_log=use_log,
    log_gain=log_gain,
    structure_sigma=structure_sigma,
    structure_strength=structure_strength,
    grain_strength=grain_strength,
)

# Convert BGR → RGB for display
orig_rgb = cv2.cvtColor(img_bgr_cropped, cv2.COLOR_BGR2RGB)
styled_rgb = cv2.cvtColor(styled_color, cv2.COLOR_BGR2RGB)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original (cropped)" if auto_crop else "Original")
    st.image(orig_rgb, use_container_width=True)

with col2:
    st.subheader("Grayscale + contrast/structure")
    st.image(gray_struct, channels="GRAY", use_container_width=True)

with col3:
    st.subheader("1600 Å–style pseudo-color")
    st.image(styled_rgb, use_container_width=True)

st.caption(
    "Note: this pipeline produces a *visual* approximation of 1600 Å imagery, "
    "not physically calibrated UV data."
)
