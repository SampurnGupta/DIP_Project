# Prediction based on Actual Models

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T

import pickle  # or joblib if you used it


# ============ CONFIG ============
MODEL_PATH = Path("models") / "best_model_graphsage.pth"
PCA_PATH = Path("models") / "pca_transform.pkl"   # optional
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping: adjust to your training
IDX_TO_LABEL = {
    0: "Weak flare",
    1: "Strong flare",
}
# ================================


@st.cache_resource
def load_pca():
    """Load PCA transform if available."""
    if PCA_PATH.exists():
        with open(PCA_PATH, "rb") as f:
            pca = pickle.load(f)
        return pca
    return None


class DummyClassifier(nn.Module):
    """
    Placeholder model definition.

    ❗ IMPORTANT:
    Replace this with the exact model class you used in training,
    or use torch.load(MODEL_PATH) directly if you saved the whole model.
    """
    def __init__(self, in_features=512, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (batch, in_features)
        return self.fc(x)


@st.cache_resource
def load_model():
    """
    Load your trained model onto CPU/GPU.
    Adjust based on how you saved it.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Option 1: you saved entire model with torch.save(model, path)
    try:
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        model.eval()
        return model
    except Exception:
        # Option 2: you saved only state_dict
        # ❗ CHANGE in_features to match your PCA or feature vector size
        model = DummyClassifier(in_features=512, num_classes=2)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model


def preprocess_image(pil_img: Image.Image, target_size=256):
    """
    Basic preprocessing: convert to grayscale, resize, tensor.

    ❗ IMPORTANT:
    Change target_size / transforms to match what you did in training.
    If you trained on 4096x4096 cropped patches, reflect that here.
    """
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((target_size, target_size)),
        T.ToTensor(),          # -> [0,1], shape (1, H, W)
    ])
    tensor = transform(pil_img)  # (1, H, W)
    return tensor


def extract_features(image_tensor: torch.Tensor):
    """
    Turn preprocessed image tensor into a 1D feature vector.

    ❗ IMPORTANT:
    In your training, you may have used:
      - A CNN feature extractor
      - Precomputed features (like cnn_features_pca)
    Reproduce that logic here.

    For now, as a placeholder, we:
      - Flatten the tensor: (1, H, W) -> (H*W)
      - Optionally apply PCA later.
    """
    # (1, H, W) -> (H*W,)
    flat = image_tensor.view(-1).numpy().astype(np.float32)
    return flat  # shape (N,)


def apply_pca_if_available(features: np.ndarray, pca):
    """
    Apply PCA to features if PCA is provided.
    """
    if pca is None:
        return features[None, :]  # (1, N)
    # pca expects 2D (batch, features)
    features = features[None, :]  # (1, N)
    reduced = pca.transform(features)
    return reduced  # (1, D)


def predict_single_image(pil_img: Image.Image):
    """
    Full pipeline: PIL image -> tensor -> features -> (PCA) -> model -> logits
    """
    model = load_model()
    pca = load_pca()

    # 1) Preprocess
    img_tensor = preprocess_image(pil_img)          # (1, H, W)
    # 2) Extract features
    features = extract_features(img_tensor)         # (N,)
    # 3) PCA (if available)
    feat_vec = apply_pca_if_available(features, pca)  # (1, D)

    # 4) To tensor for model
    feat_tensor = torch.from_numpy(feat_vec).float().to(DEVICE)  # (1, D)

    with torch.no_grad():
        logits = model(feat_tensor)                 # (1, num_classes)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = IDX_TO_LABEL.get(pred_idx, f"class_{pred_idx}")
    return pred_label, probs


# ============ Streamlit UI ============

st.set_page_config(page_title="Solar Flare Strength Classifier", layout="centered")
st.title("🔭 Solar Flare Strength Classifier (1600 Å)")

st.write(
    "Upload a **1600 Å solar flare image** (grayscale PNG/JPG) and the model "
    "will classify it as **Weak** or **Strong** based on the trained ML model."
)

uploaded_file = st.file_uploader(
    "Upload a 1600 Å solar flare image", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Read image
    image_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")

    st.image(pil_img, caption="Uploaded image", use_container_width=True)

    if st.button("Predict flare strength"):
        try:
            label, probs = predict_single_image(pil_img)
            st.success(f"Predicted: **{label}**")
            st.write("Class probabilities:")

            for idx, p in enumerate(probs):
                name = IDX_TO_LABEL.get(idx, f"class_{idx}")
                st.write(f"- {name}: `{p:.4f}`")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.info("Please upload a 1600 Å flare image to begin.")
