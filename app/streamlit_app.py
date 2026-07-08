import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---- Config ----
BASELINE_MODEL_PATH = "models/baseline_cnn.keras"
RESNET_MODEL_PATH = "models/resnet_finetuned.keras"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")

@st.cache_resource
def load_models():
    baseline = tf.keras.models.load_model(BASELINE_MODEL_PATH)
    resnet = tf.keras.models.load_model(RESNET_MODEL_PATH)
    return baseline, resnet

baseline_model, resnet_model = load_models()


def preprocess(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, img_array):
    prob = model.predict(img_array, verbose=0)[0][0]
    label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = float(prob) if prob > 0.5 else float(1 - prob)
    return label, confidence


st.title("🫁 Chest X-Ray Pneumonia Detector")
st.write("Upload a chest X-ray image to compare predictions from both models — a baseline CNN and a fine-tuned ResNet.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = preprocess(image)

    st.image(image, caption="Uploaded X-ray", width="stretch")

    with st.spinner("Analyzing with both models..."):
        baseline_label, baseline_conf = predict_image(baseline_model, img_array)
        resnet_label, resnet_conf = predict_image(resnet_model, img_array)

    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Baseline CNN")
        if baseline_label == "PNEUMONIA":
            st.error(f"**{baseline_label}**")
        else:
            st.success(f"**{baseline_label}**")
        st.metric("Confidence", f"{baseline_conf*100:.2f}%")
        st.progress(baseline_conf)

    with col2:
        st.markdown("### ResNet (Transfer Learning)")
        if resnet_label == "PNEUMONIA":
            st.error(f"**{resnet_label}**")
        else:
            st.success(f"**{resnet_label}**")
        st.metric("Confidence", f"{resnet_conf*100:.2f}%")
        st.progress(resnet_conf)

    if baseline_label != resnet_label:
        st.warning("⚠️ The two models disagree on this prediction — worth a closer look.")

st.markdown("---")
st.caption("Note: This tool is for educational/portfolio purposes only and is not a substitute for professional medical diagnosis.")