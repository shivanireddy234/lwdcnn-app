import streamlit as st
import numpy as np
from PIL import Image

# ── Page config ───────────────────────────────────
st.set_page_config(
    page_title="LWDCNN — Breast Ultrasound Classifier",
    page_icon="🩺",
    layout="wide"
)

# ── Load weights ──────────────────────────────────
@st.cache_resource
def load_weights():
    try:
        w = np.load("weights.npz")
        return dict(w)
    except FileNotFoundError:
        st.error("weights.npz not found!")
        st.stop()

# ── Pure numpy inference ──────────────────────────
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def batchnorm(x, gamma, beta, mean, var, eps=1e-3):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def conv2d(x, w):
    """
    x: (H, W, C)
    w: (kH, kW, C, F)
    returns: (H-kH+1, W-kW+1, F)
    """
    H, W, C   = x.shape
    kH, kW, _, F = w.shape
    oH = H - kH + 1
    oW = W - kW + 1
    # reshape kernel to (kH*kW*C, F)
    w_flat = w.reshape(-1, F)                          # (kH*kW*C, F)
    # build im2col matrix: (oH*oW, kH*kW*C)
    cols = np.zeros((oH * oW, kH * kW * C), dtype=np.float32)
    for i in range(oH):
        for j in range(oW):
            patch = x[i:i+kH, j:j+kW, :]             # (kH, kW, C)
            cols[i*oW + j] = patch.flatten()           # (kH*kW*C,)
    out = cols @ w_flat                                # (oH*oW, F)
    return out.reshape(oH, oW, F)

def maxpool(x, size=3, stride=3):
    H, W, C = x.shape
    oH = (H - size) // stride + 1
    oW = (W - size) // stride + 1
    out = np.zeros((oH, oW, C), dtype=np.float32)
    for h in range(oH):
        for w in range(oW):
            out[h, w, :] = np.max(
                x[h*stride:h*stride+size, w*stride:w*stride+size, :],
                axis=(0, 1)
            )
    return out

def predict_numpy(img_array, w):
    x = img_array.astype(np.float32) / 255.0

    # Conv1 + BN + ReLU + Pool
    x = conv2d(x, w['conv1_w'])
    x = batchnorm(x, w['bn1_gamma'], w['bn1_beta'], w['bn1_mean'], w['bn1_var'])
    x = relu(x)
    x = maxpool(x)

    # Conv2 + BN + ReLU + Pool
    x = conv2d(x, w['conv2_w'])
    x = batchnorm(x, w['bn2_gamma'], w['bn2_beta'], w['bn2_mean'], w['bn2_var'])
    x = relu(x)
    x = maxpool(x)

    # Conv3 + BN + ReLU + Pool
    x = conv2d(x, w['conv3_w'])
    x = batchnorm(x, w['bn3_gamma'], w['bn3_beta'], w['bn3_mean'], w['bn3_var'])
    x = relu(x)
    x = maxpool(x)

    # Flatten
    x = x.flatten()

    # Dense1 + BN + ReLU
    x = x @ w['dense1_w'] + w['dense1_b']
    x = batchnorm(x, w['bn4_gamma'], w['bn4_beta'], w['bn4_mean'], w['bn4_var'])
    x = relu(x)

    # Dense2 + Sigmoid
    x = x @ w['dense2_w'] + w['dense2_b']
    prob = float(sigmoid(x).flatten()[0])
    return prob

# ── Preprocess ────────────────────────────────────
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    return np.array(img, dtype=np.float32)

# ── UI ────────────────────────────────────────────
st.title("🩺 LWDCNN — Breast Ultrasound Classifier")
st.markdown("Lightweight Deep CNN for breast ultrasound classification — **Pure NumPy inference**")
st.divider()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Model Info")
    st.metric("Parameters", "1,853")
    st.metric("Accuracy", "94.00%")
    st.metric("AUC-ROC", "97.30%")
    st.metric("Platform", "Streamlit Cloud")
    st.markdown("**Architecture:**")
    st.markdown("- Conv(4)→BN→Pool\n- Conv(4)→BN→Pool\n- Conv(8)→BN→Pool\n- Dense(16)→BN\n- Dense(1)→Sigmoid")

with col_right:
    weights = load_weights()

    uploaded = st.file_uploader(
        "Upload a breast ultrasound image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        img = Image.open(uploaded)

        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption="Uploaded BUS Image", use_column_width=True)

        with c2:
            with st.spinner("Running inference..."):
                img_array = preprocess(img)
                prob = predict_numpy(img_array, weights)

            label    = "Malignant" if prob >= 0.5 else "Benign"
            color    = "🔴" if prob >= 0.5 else "🟢"
            conf     = prob if prob >= 0.5 else 1 - prob

            st.subheader("Prediction")
            st.metric("Classification", f"{color} {label}")
            st.metric("Confidence", f"{conf*100:.1f}%")
            st.progress(float(prob), text=f"Malignant probability: {prob:.4f}")

            if prob >= 0.5:
                st.error("⚠️ Likely **Malignant** — please consult a medical professional.")
            else:
                st.success("✅ Likely **Benign** — please consult a medical professional for confirmation.")

            st.caption("⚕️ For research purposes only. Not a medical diagnosis.")

st.divider()
st.caption("Built with LWDCNN | Streamlit Cloud | Pure NumPy — no TensorFlow required")
