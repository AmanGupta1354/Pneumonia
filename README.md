# Chest X-Ray Pneumonia Detector

A CNN-based image classifier that detects pneumonia from chest X-ray images.
Includes a baseline CNN trained from scratch and a fine-tuned ResNet18 transfer
learning model, served through a FastAPI backend.

## Project Structure

chest-xray-pneumonia-detector/<br>
├── data/               # dataset (train/val/test folders)/<br>
├── src/                # training, evaluation, inference code/<br>
├── models/             # saved trained models (.keras files)/<br>
├── app/                # FastAPI backend/<br>
├── requirements.txt/<br>
├── Dockerfile/<br>
└── README.md/<br>

## Step 1: Get the Dataset

1. Download the dataset from Kaggle: "Chest X-Ray Images (Pneumonia)" by Paul Mooney
   <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

2. Extract it and place the folders like this:

data/<br>
├── train/<br>
│   ├── NORMAL/<br>
│   └── PNEUMONIA/<br>
├── val/<br>
│   ├── NORMAL/<br>
│   └── PNEUMONIA/<br>
└── test/<br>
├── NORMA/<br>
└── PNEUMONIA/<br>

Note: The original Kaggle val folder only has ~16 images. For better validation,
move roughly 15% of images from train/ into val/ for both classes before training.

## Step 2: Install Dependencies

Create a virtual environment first (recommended):

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

## Step 3: Train the Models

Run training from the project root:

```bash
cd src
python train.py
```

This will:

- Train the Baseline CNN for 10 epochs
- Train the ResNet18 transfer learning model for 10 epochs
- Save the best version of each model into the `models/` folder as
  `baseline_cnn.pt` and `resnet_finetuned.pt`

Training progress (loss/accuracy per epoch) will print to the console.

## Step 4: Evaluate the Models

Still inside `src/`, run:

```bash
python evaluate.py
```

This prints Accuracy, Precision, Recall, F1-score, ROC-AUC, and the
confusion matrix for both models on the test set — use this to compare
the baseline CNN vs the transfer learning model.

## Step 5: Run the API Locally

From the project root:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Test it:

- Open `http://localhost:8000/health` in a browser — should return `{"status": "ok"}`
- Send a test prediction using curl:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path_to_xray_image.jpg"
```

Response example:

```json
{"class": "PNEUMONIA", "confidence": 0.94}
```

## Step 6: Run with Docker (optional, for deployment)

Build the image:

```bash
docker build -t pneumonia-detector .
```

Run the container:

```bash
docker run -p 8000:8000 pneumonia-detector
```

The API will be available at `http://localhost:8000`.

## Step 7: Deploy

- **Hugging Face Spaces:** Push this repo (with a Gradio/Streamlit wrapper instead
  of FastAPI if you want the simplest deploy path) to a new Space.
- **Render/Railway:** Connect your GitHub repo, set the start command to
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, and deploy directly
  using the Dockerfile provided.

## Results (fill in after training)

| Model                 | Accuracy | Precision | Recall | F1   | ROC-AUC |
|-----------------------|----------|-----------|--------|------|---------|
| Baseline CNN          |  0.8942  |   0.9091  | 0.9231 |0.9160|  0.9547 |
| ResNet18 (fine-tuned) |  0.9151  |   0.9120  | 0.9564 |0.9337|  0.9706 |

## Classification Report of Baseline Model:
              precision    recall  f1-score   support

      NORMAL       0.87      0.85      0.86       234
   PNEUMONIA       0.91      0.92      0.92       390

    accuracy                           0.89       624
   macro avg       0.89      0.88      0.89       624
weighted avg       0.89      0.89      0.89       624

## Classification Report of ResNet18 (fine-tuned model)
              precision    recall  f1-score   support

      NORMAL       0.92      0.85      0.88       234
   PNEUMONIA       0.91      0.96      0.93       390

    accuracy                           0.92       624
   macro avg       0.92      0.90      0.91       624
weighted avg       0.92      0.92      0.91       624
## Notes

- Recall is prioritized over precision for the PNEUMONIA class, since missing
  a true pneumonia case is more costly than a false alarm.
- Class weighting is applied during training to handle dataset imbalance.
