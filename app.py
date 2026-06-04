
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import time
import pandas as pd

st.set_page_config(
    page_title="LWDCNN — Breast Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lwdcnn_db1.keras")

model = load_model()

# ── Header ────────────────────────────────────────
st.title("🔬 LWDCNN Breast Cancer Detection")
st.markdown("""
Lightweight Deep CNN for automated detection of malignant
tumors using Breast Ultrasound (BUS) images.
**IEEE Access 2024** — Parameters: 1,853 | Accuracy: 94.00%
""")
st.divider()

# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.title("📊 Model Info")
    st.metric("Parameters",  "1,853")
    st.metric("Accuracy",    "94.00%")
    st.metric("AUC-ROC",     "97.30%")
    st.metric("F1-Score",    "93.27%")
    st.metric("Input Size",  "128×128×3")
    st.divider()
    st.markdown("**Architecture:**")
    st.markdown("- Conv2D(4) → BN → MaxPool")
    st.markdown("- Conv2D(4) → BN → MaxPool")
    st.markdown("- Conv2D(8) → BN → MaxPool")
    st.markdown("- Dense(16) → BN")
    st.markdown("- Dense(1) → Sigmoid")
    st.divider()
    st.markdown("**Platform:** Streamlit Cloud (CPU)")

# ── Upload ────────────────────────────────────────
st.subheader("📤 Upload Breast Ultrasound Image")
uploaded = st.file_uploader(
    "Choose a BUS image (PNG/JPG)",
    type=["png","jpg","jpeg"]
)

if uploaded is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="Uploaded BUS Image",
                 use_column_width=True)

    # Preprocess
    img_np  = np.array(img_pil)
    img_res = cv2.resize(img_np, (128,128))
    img_in  = img_res.astype(np.float32) / 255.0
    img_in  = np.expand_dims(img_in, axis=0)

    # Inference
    start = time.time()
    pred  = float(model.predict(img_in, verbose=0)[0][0])
    end   = time.time()
    ms    = (end - start) * 1000

    with col2:
        st.subheader("🎯 Prediction")
        if pred >= 0.5:
            st.error("🔴 MALIGNANT TUMOR DETECTED")
            conf = pred * 100
        else:
            st.success("🟢 BENIGN TUMOR")
            conf = (1 - pred) * 100

        st.metric("Confidence",     f"{conf:.2f}%")
        st.metric("Raw Score",      f"{pred:.4f}")
        st.metric("Inference Time", f"{ms:.1f} ms")

    # Probability bar
    st.subheader("Malignancy Probability")
    st.progress(pred)
    st.caption(
        f"Benign ←————→ Malignant | Score: {pred:.4f}")

    # CAM
    st.subheader("🗺️ Grad-CAM Visualization")
    last_conv  = [l for l in model.layers
                  if isinstance(l,tf.keras.layers.Conv2D)][-1]
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[last_conv.output, model.output])

    with tf.GradientTape() as tape:
        img_t = tf.cast(img_in, tf.float32)
        conv_out, preds = grad_model(img_t)
        loss = preds[:,0]

    grads  = tape.gradient(loss, conv_out)[0]
    pooled = tf.reduce_mean(grads, axis=(0,1))
    cam    = np.zeros(conv_out.shape[1:3], dtype=np.float32)
    for i,w in enumerate(pooled):
        cam += w.numpy() * conv_out[0,:,:,i].numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (128,128))

    heatmap = cv2.applyColorMap(
        (cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(
        img_res.astype(np.uint8), 0.6,
        heatmap, 0.4, 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img_res,  caption="Original", clamp=True)
    with c2:
        st.image((cam*255).astype(np.uint8),
                 caption="CAM Heatmap", clamp=True)
    with c3:
        st.image(overlay, caption="Overlay",
                 use_column_width=True)

    st.divider()

    # Platform comparison table
    st.subheader("📊 Platform Comparison (Paper Table 7)")
    df = pd.DataFrame({
        "Platform"      : [
            "☁️ Streamlit Cloud",
            "📱 Android Device",
            "⚡ PYNQ-Z2 FPGA"
        ],
        "Framework"     : [
            "TensorFlow FP32",
            "TFLite INT8",
            "HLS4ML FDP<20,8>"
        ],
        "Accuracy"      : ["94.15%","93.76%","93.16%"],
        "Inference Time": [
            "1331.68 sec/1624 imgs",
            "1071.84 sec/1624 imgs",
            "6.44 sec/1624 imgs"
        ],
        "Parameters"    : ["1,837","1,837","1,837"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("👆 Upload a breast ultrasound image to get started")
    st.markdown("""
    ### How it works
    1. Upload a breast ultrasound (BUS) image
    2. LWDCNN classifies it as **Benign** or **Malignant**
    3. Grad-CAM shows which regions influenced the decision
    4. Results match paper accuracy of **94.15%**
    """)

st.divider()
st.caption("LWDCNN — IEEE Access 2024 | Streamlit Cloud CPU")
