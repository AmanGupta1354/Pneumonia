from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict_image

app = FastAPI(title="Chest X-Ray Pneumonia Detector (TensorFlow)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = predict_image(image)
    return result