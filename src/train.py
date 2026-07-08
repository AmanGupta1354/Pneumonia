import tensorflow as tf
import numpy as np
import os
from data_loader import get_dataloaders
from model import build_baseline_cnn, build_transfer_model


def compute_class_weights(train_ds):
    normal, pneumonia = 0, 0
    for _, labels in train_ds.unbatch():
        if labels.numpy() == 0:
            normal += 1
        else:
            pneumonia += 1
    total = normal + pneumonia
    weight_for_0 = total / (2 * normal)
    weight_for_1 = total / (2 * pneumonia)
    return {0: weight_for_0, 1: weight_for_1}


def train_model(model, train_ds, val_ds, epochs, lr, model_name, class_weights):
    os.makedirs("models", exist_ok=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"models/{model_name}.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    return model, history


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = get_dataloaders("data")
    print("Classes:", class_names)

    class_weights = compute_class_weights(train_ds)
    print("Class weights:", class_weights)

    print("\n--- Training Baseline CNN ---")
    baseline = build_baseline_cnn()
    train_model(baseline, train_ds, val_ds, epochs=10, lr=1e-3,
                model_name="baseline_cnn", class_weights=class_weights)

    print("\n--- Training Transfer Learning Model (ResNet50) ---")
    transfer_model = build_transfer_model()
    train_model(transfer_model, train_ds, val_ds, epochs=10, lr=1e-4,
                model_name="resnet_finetuned", class_weights=class_weights)