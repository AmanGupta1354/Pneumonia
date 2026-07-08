import tensorflow as tf
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, classification_report)
from data_loader import get_dataloaders


def evaluate_model(model_path, test_ds):
    model = tf.keras.models.load_model(model_path)

    all_labels, all_probs = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


if __name__ == "__main__":
    _, _, test_ds, class_names = get_dataloaders("data")

    print("=== Baseline CNN ===")
    evaluate_model("models/baseline_cnn.keras", test_ds)

    print("\n=== Transfer Learning (ResNet50) ===")
    evaluate_model("models/resnet_finetuned.keras", test_ds)