import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

_model = None


def load_model(model_path="models/baseline_cnn.keras"):
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(model_path)
    return _model


def predict_image(image: Image.Image, model_path="models/baseline_cnn.keras"):
    model = load_model(model_path)

    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array, verbose=0)[0][0]
    label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = float(prob) if prob > 0.5 else float(1 - prob)

    return {"class": label, "confidence": round(confidence, 4)}