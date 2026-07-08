import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---- Config ----
MODEL_PATH = "models/resnet_finetuned.keras"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")

# ---- Load model (cached so it doesn't reload every interaction) ----
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---- Prediction function ----
def predict_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array, verbose=0)[0][0]
    label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = float(prob) if prob > 0.5 else float(1 - prob)

    return label, confidence

# ---- UI ----
st.title("🫁 Chest X-Ray Pneumonia Detector")
st.write("Upload a chest X-ray image and the model will predict whether it shows signs of pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    with st.spinner("Analyzing..."):
        label, confidence = predict_image(image)

    st.subheader("Prediction")
    if label == "PNEUMONIA":
        st.error(f"**{label}** — Confidence: {confidence*100:.2f}%")
    else:
        st.success(f"**{label}** — Confidence: {confidence*100:.2f}%")

    st.progress(confidence)

st.markdown("---")
st.caption("Note: This tool is for educational/portfolio purposes only and is not a substitute for professional medical diagnosis.")