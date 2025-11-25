# app.py just with Download buttons refactored to helper function
# Download buttons for all images produced in the app via DIP pipeline
# and original/cropped images.

from pathlib import Path
from io import BytesIO

from PIL import Image, ImageDraw, ImageOps
import streamlit as st

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor
from skimage import exposure

# ==== CONFIGURE THESE ====
IMAGES_DIR = Path("D:\\DIP\\Folder1")   # Folder 1: images
LABELS_DIR = Path("D:\\DIP\\Folder2")   # Folder 2: labels
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
# ==========================


def parse_label_file(label_path, img_width, img_height):
    """
    Parse custom label format:
    sunspot 0 0 0 x1 y1 x2 y2 ... (rest ignored)
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            label = parts[0]
            try:
                x1 = int(float(parts[4]))
                y1 = int(float(parts[5]))
                x2 = int(float(parts[6]))
                y2 = int(float(parts[7]))
            except ValueError:
                continue

            # Clip to image bounds
            x1 = max(0, min(img_width - 1, x1))
            y1 = max(0, min(img_height - 1, y1))
            x2 = max(0, min(img_width - 1, x2))
            y2 = max(0, min(img_height - 1, y2))

            if x2 > x1 and y2 > y1:
                boxes.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return boxes


def draw_boxes_on_image(image: Image.Image, boxes):
    """
    Draw rectangle overlays on PIL image.
    Enhanced boundary: thicker outline.
    """
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    for box in boxes:
        draw.rectangle(
            [(box["x1"], box["y1"]), (box["x2"], box["y2"])],
            outline="red",
            width=8,   # bold boundary
        )

    return img_with_boxes


def preprocess_patch(
    gray_np: np.ndarray,
    do_eq: bool,
    do_clahe: bool,
    do_blur: bool,
    do_sharp: bool,
):
    """
    Basic enhancement pipeline applied IN ORDER to the selected patch:
    1) Histogram equalization
    2) CLAHE (local contrast)
    3) Gaussian blur
    4) Unsharp mask (sharpen)
    """
    img = gray_np.copy().astype(np.uint8)

    # 1) Histogram equalization
    if do_eq:
        img = cv2.equalizeHist(img)

    # 2) CLAHE
    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    # 3) Gaussian blur
    if do_blur:
        img = cv2.GaussianBlur(img, (5, 5), 1.0)

    # 4) Unsharp mask
    if do_sharp:
        blur = cv2.GaussianBlur(img, (0, 0), 1.0)
        img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    return img

# Helper: convert image (PIL or numpy) to PNG bytes and add a download button
def _image_bytes(img):
    """
    Accepts a PIL.Image or numpy array and returns PNG bytes.
    """
    if isinstance(img, Image.Image):
        pil = img
    elif isinstance(img, np.ndarray):
        arr = img
        # If float or other dtypes, rescale to 0-255 and convert to uint8
        if arr.dtype != np.uint8:
            arr = exposure.rescale_intensity(arr, out_range=(0, 255)).astype("uint8")
        # If single-channel 2D -> 'L', else let PIL infer (RGB/RGBA)
        if arr.ndim == 2:
            pil = Image.fromarray(arr, mode="L")
        else:
            pil = Image.fromarray(arr)
    else:
        raise TypeError("Unsupported image type for download")

    buf = BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def add_download_button_for_image(img, filename, key=None, label=None):
    data = _image_bytes(img)
    btn_label = label or f"Download {filename}"
    st.download_button(label=btn_label, data=data, file_name=filename, mime="image/png", key=key)

def main():
    st.title("Solar Flare/Sunspot Identifier and DIP Pipeline")

    st.sidebar.header("Settings")
    enhance_contrast = st.sidebar.checkbox(
        "Apply autocontrast", value=True
    )

    st.header("Select an Image")

    # File picker
    uploaded_file = st.file_uploader(
        "Choose an image file from Folder 1", type=list(IMAGE_EXTENSIONS)
    )

    if not uploaded_file:
        st.info("Please choose an image from your local machine.")
        return

    file_path = Path(uploaded_file.name)
    actual_image_path = IMAGES_DIR / file_path.name

    # Validate chosen file is actually in Folder1
    if not actual_image_path.exists():
        st.error(f"❌ The file **{file_path.name}** is NOT present in Folder1:\n\n{IMAGES_DIR}")
        return

    # Validate label exists
    label_path = LABELS_DIR / f"{file_path.stem}.txt"
    if not label_path.exists():
        st.error(f"❌ Label file not found in Folder2 for:\n{file_path.name}")
        return

    # Load image from Folder1
    image = Image.open(actual_image_path).convert("L")
    if enhance_contrast:
        image = ImageOps.autocontrast(image)

    img_width, img_height = image.size

    # Parse label boxes
    boxes = parse_label_file(label_path, img_width, img_height)

    boxes_to_draw = boxes
    selected_box_index = None

    if boxes:
        show_all = st.checkbox("Show all bounding boxes", value=True)

        if not show_all and len(boxes) > 1:
            options = [
                f"Box {i+1} – {b['label']} "
                f"({b['x1']}, {b['y1']}) → ({b['x2']}, {b['y2']})"
                for i, b in enumerate(boxes)
            ]

            selected_box_index = st.selectbox(
                "Select a single box to display:",
                list(range(len(boxes))),
                format_func=lambda i: options[i],
            )

            boxes_to_draw = [boxes[selected_box_index]]

        elif not show_all and len(boxes) == 1:
            boxes_to_draw = [boxes[0]]

    # Display images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, caption=file_path.name, use_container_width=True)
        # download for original
        add_download_button_for_image(image, f"{file_path.stem}_original.png", key=f"dl_original_{file_path.stem}")

    cropped_region = None  # will hold crop if a single box is active

    with col2:
        st.subheader("With Bounding Boxes")
        if boxes_to_draw:
            img_with_boxes = draw_boxes_on_image(image.convert("RGB"), boxes_to_draw)
            caption_suffix = (
                f"{len(boxes_to_draw)} box(es)"
                if len(boxes_to_draw) > 1
                else "1 selected box"
            )
            st.image(img_with_boxes, caption=caption_suffix, use_container_width=True)
            # download for image with boxes
            add_download_button_for_image(img_with_boxes, f"{file_path.stem}_with_boxes.png", key=f"dl_with_boxes_{file_path.stem}")

            # If exactly one box currently being drawn, prepare crop + download
            if len(boxes_to_draw) == 1:
                b = boxes_to_draw[0]
                cropped_region = image.crop((b["x1"], b["y1"], b["x2"], b["y2"]))
        else:
            st.info("No valid bounding boxes found.")
            img_with_boxes = None

    # Show box details
    if boxes:
        st.markdown("### Bounding Box Details")
        for i, box in enumerate(boxes, start=1):
            tag = " 🔹 (selected)" if selected_box_index == (i - 1) else ""
            st.write(
                f"**Box {i}{tag}** — `{box['label']}` "
                f"({box['x1']}, {box['y1']}) → ({box['x2']}, {box['y2']})"
            )

    # ===== Crop download + DIP on selected region =====
    if cropped_region is not None:
        st.markdown("## Selected Region (cropped from image)")
        st.image(cropped_region, use_container_width=True)

        # Download button (keeps existing behavior but via helper)
        cropped_region_rgb = cropped_region.convert("RGB")
        crop_index = (
            selected_box_index + 1
            if selected_box_index is not None
            else 1
        )
        add_download_button_for_image(cropped_region_rgb, f"{file_path.stem}_box{crop_index}.png", key=f"dl_cropped_{file_path.stem}_{crop_index}")

        # ---------- DIP PIPELINE ON SELECTED PATCH ----------
        st.markdown("## Digital Image Processing on Selected Region")

        gray_np = np.array(cropped_region.convert("L"))

        st.markdown("**Preprocessing pipeline (applied in order):**")
        col_eq, col_clahe, col_blur, col_sharp = st.columns(4)
        with col_eq:
            do_eq = st.checkbox("Equalization", value=True)
        with col_clahe:
            do_clahe = st.checkbox("CLAHE", value=False)
        with col_blur:
            do_blur = st.checkbox("Gaussian blur", value=False)
        with col_sharp:
            do_sharp = st.checkbox("Sharpen", value=False)

        preprocessed = preprocess_patch(
            gray_np, do_eq=do_eq, do_clahe=do_clahe, do_blur=do_blur, do_sharp=do_sharp
        )
        st.image(preprocessed, channels="GRAY",
                 caption="Preprocessed region", use_container_width=True)
        add_download_button_for_image(preprocessed, f"{file_path.stem}_box{crop_index}_preprocessed.png", key=f"dl_pre_{file_path.stem}_{crop_index}")

        st.markdown("**Feature / analysis views (computed on preprocessed image):**")
        show_edges = st.checkbox("Edges (Canny)", value=True)
        show_lbp = st.checkbox("Local Binary Pattern (LBP)")
        show_gabor = st.checkbox("Gabor filter")
        show_hog = st.checkbox("HOG visualization")

        col_a, col_b, col_c, col_d = st.columns(4)

        # Canny edges
        if show_edges:
            edges = cv2.Canny(preprocessed, 50, 150)
            col_a.image(
                edges,
                channels="GRAY",
                caption="Canny edges",
                use_container_width=True,
            )
            add_download_button_for_image(edges, f"{file_path.stem}_box{crop_index}_edges.png", key=f"dl_edges_{file_path.stem}_{crop_index}")

        # LBP
        if show_lbp:
            lbp = local_binary_pattern(preprocessed, P=8, R=1, method="uniform")
            lbp_img = exposure.rescale_intensity(
                lbp, in_range=(0, lbp.max()), out_range=(0, 255)
            ).astype("uint8")
            col_b.image(
                lbp_img,
                channels="GRAY",
                caption="LBP",
                use_container_width=True,
            )
            add_download_button_for_image(lbp_img, f"{file_path.stem}_box{crop_index}_lbp.png", key=f"dl_lbp_{file_path.stem}_{crop_index}")

        # Gabor
        if show_gabor:
            real, imag = gabor(preprocessed, frequency=0.2)
            mag = np.sqrt(real**2 + imag**2)
            gabor_img = exposure.rescale_intensity(
                mag, in_range=(0, mag.max()), out_range=(0, 255)
            ).astype("uint8")
            col_c.image(
                gabor_img,
                channels="GRAY",
                caption="Gabor magnitude",
                use_container_width=True,
            )
            add_download_button_for_image(gabor_img, f"{file_path.stem}_box{crop_index}_gabor.png", key=f"dl_gabor_{file_path.stem}_{crop_index}")

        # HOG visualization
        if show_hog:
            try:
                _, hog_image = hog(
                    preprocessed,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    visualize=True,
                    feature_vector=True,
                )
                hog_img = exposure.rescale_intensity(
                    hog_image, in_range=(0, hog_image.max()), out_range=(0, 255)
                ).astype("uint8")
                col_d.image(
                    hog_img,
                    channels="GRAY",
                    caption="HOG visualization",
                    use_container_width=True,
                )
                add_download_button_for_image(hog_img, f"{file_path.stem}_box{crop_index}_hog.png", key=f"dl_hog_{file_path.stem}_{crop_index}")
            except Exception as e:
                st.warning(f"HOG failed on this patch: {e}")


if __name__ == "__main__":
    main()
