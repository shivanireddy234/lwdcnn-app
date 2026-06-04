import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import pandas as pd
from scipy.special import expit  # sigmoid

st.set_page_config(
    page_title="LWDCNN — Breast Cancer Detection",
    page_icon="🔬", layout="centered")

# ── Pure numpy inference ──────────────────────────
@st.cache_resource
def load_weights():
    """Load model weights from numpy arrays embedded in app."""
    import urllib.request, io
    # Weights are stored as base64 in the app
    # We implement forward pass in pure numpy
    return True

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return expit(x)

def batchnorm(x, gamma, beta, mean, var, eps=1e-3):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def conv2d(x, w):
    """x: HxWxC, w: KHxKWxCxF"""
    H,W,C = x.shape
    KH,KW,_,F = w.shape
    OH,OW = H-KH+1, W-KW+1
    out = np.zeros((OH,OW,F), dtype=np.float32)
    for f in range(F):
        for kh in range(KH):
            for kw in range(KW):
                out[:,:,f] += x[kh:kh+OH, kw:kw+OW, :] @ w[kh,kw,:,f:f+1]
    return out

def maxpool(x, size=3, stride=3):
    H,W,C = x.shape
    OH,OW = H//stride, W//stride
    out = np.zeros((OH,OW,C), dtype=np.float32)
    for h in range(OH):
        for w in range(OW):
            out[h,w,:] = x[h*stride:h*stride+size,
                           w*stride:w*stride+size,:].max(axis=(0,1))
    return out

def predict_numpy(img, weights):
    """Full LWDCNN forward pass in numpy."""
    w = weights

    # Conv1 + BN + ReLU + Pool
    x = conv2d(img, w['conv1_w'])
    x = batchnorm(x, w['bn1_gamma'], w['bn1_beta'],
                     w['bn1_mean'],  w['bn1_var'])
    x = relu(x)
    x = maxpool(x)

    # Conv2 + BN + ReLU + Pool
    x = conv2d(x, w['conv2_w'])
    x = batchnorm(x, w['bn2_gamma'], w['bn2_beta'],
                     w['bn2_mean'],  w['bn2_var'])
    x = relu(x)
    x = maxpool(x)

    # Conv3 + BN + ReLU + Pool
    x = conv2d(x, w['conv3_w'])
    x = batchnorm(x, w['bn3_gamma'], w['bn3_beta'],
                     w['bn3_mean'],  w['bn3_var'])
    x = relu(x)
    x = maxpool(x)

    # Flatten
    x = x.flatten()

    # Dense1 + BN + ReLU
    x = x @ w['dense1_w'] + w['dense1_b']
    x = batchnorm(x, w['bn4_gamma'], w['bn4_beta'],
                     w['bn4_mean'],  w['bn4_var'])
    x = relu(x)

    # Dense2 + Sigmoid
    x = x @ w['dense2_w'] + w['dense2_b']
    return float(sigmoid(x[0]))

@st.cache_resource
def load_model_weights():
    weights = np.load('weights.npz', allow_pickle=True)
    return dict(weights)

# ── UI ────────────────────────────────────────────
st.title("🔬 LWDCNN Breast Cancer Detection")
st.markdown("""
**Lightweight Deep CNN** | IEEE Access 2024 |
Parameters: **1,853** | Accuracy: **94.00%**
""")
st.divider()

with st.sidebar:
    st.title("📊 Model Info")
    st.metric("Parameters", "1,853")
    st.metric("Accuracy",   "94.00%")
    st.metric("AUC-ROC",    "97.30%")
    st.metric("Platform",   "Streamlit Cloud")
    st.divider()
    st.markdown("**Architecture:**")
    st.markdown("- Conv(4)→BN→Pool")
    st.markdown("- Conv(4)→BN→Pool")
    st.markdown("- Conv(8)→BN→Pool")
    st.markdown("- Dense(16)→Dense(1)")

try:
    weights = load_model_weights()
    st.success("✓ Model loaded successfully")
except:
    st.error("weights.npz not found in repo")
    st.stop()

st.subheader("📤 Upload Breast Ultrasound Image")
uploaded = st.file_uploader(
    "Choose a BUS image (PNG/JPG)",
    type=["png","jpg","jpeg"])

if uploaded is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="Uploaded BUS Image",
                 use_column_width=True)

    img_np  = np.array(img_pil, dtype=np.float32)
    img_res = cv2.resize(img_np, (128,128)) / 255.0

    start = time.time()
    pred  = predict_numpy(img_res, weights)
    end   = time.time()
    ms    = (end-start)*1000

    with col2:
        st.subheader("🎯 Prediction")
        if pred >= 0.5:
            st.error("🔴 MALIGNANT TUMOR DETECTED")
            conf = pred*100
        else:
            st.success("🟢 BENIGN TUMOR")
            conf = (1-pred)*100
        st.metric("Confidence",      f"{conf:.2f}%")
        st.metric("Raw Score",       f"{pred:.4f}")
        st.metric("Inference Time",  f"{ms:.1f} ms")

    st.subheader("Malignancy Probability")
    st.progress(pred)
    st.caption(f"Benign ←————→ Malignant | {pred:.4f}")

    st.divider()
    st.subheader("📊 Platform Comparison (Table 7)")
    df = pd.DataFrame({
        "Platform"      : ["☁️ Streamlit","📱 Android","⚡ FPGA"],
        "Framework"     : ["NumPy","TFLite INT8","HLS4ML"],
        "Accuracy"      : ["94.00%","93.76%","93.16%"],
        "Speed"         : ["~1331 sec/1624","~1072 sec/1624","6.44 sec/1624"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("👆 Upload a breast ultrasound image to get started")

st.divider()
st.caption("LWDCNN — IEEE Access 2024 | Pure NumPy inference")
