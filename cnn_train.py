#!/usr/bin/env python3
"""
-------------------------------------------------------------------------------------------------
cnn_train.py - Transfer Learning & Classification Pipeline using MobileNetV2

Features:
- MobileNetV2 pretrained backbone (ImageNet weights) for transfer learning
- Custom classification head with dropout and L2 regularization
- Data augmentation pipeline (RandomFlip, RandomRotation, RandomZoom, RandomBrightness)
- Structured argparse CLI for flexible training execution
- Learning curves and confusion matrix evaluation figures saved to images/cnn/
- Model checkpointing and early stopping
-------------------------------------------------------------------------------------------------
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Avoid GPU memory allocation hogging
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train object classifier using MobileNetV2 transfer learning on stereo crops."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "crops" / "labeled",
        help="Path to labeled class folders",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=PROJECT_ROOT / "crop_cnn.keras",
        help="Output path for trained Keras model",
    )
    parser.add_argument(
        "--images-out",
        type=Path,
        default=PROJECT_ROOT / "images" / "cnn",
        help="Output directory for plots and figures",
    )
    parser.add_argument("--img-size", type=int, default=128, help="Target image size (height & width)")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Enable fine-tuning of top MobileNetV2 layers after initial feature extraction",
    )
    return parser


def save_fig(fig_id, images_path, tight_layout=True, fig_extension="jpg", resolution=300):
    images_path.mkdir(parents=True, exist_ok=True)
    path = images_path / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="jpeg", dpi=resolution)
    print(f"Saved figure to {path}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    model_out = args.model_out.resolve()
    images_out = args.images_out.resolve()
    images_out.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. Run prepare_labeled_crops.py first."
        )

    img_height = args.img_size
    img_width = args.img_size
    batch_size = args.batch_size
    epochs = args.epochs

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 1.0 - train_ratio - val_ratio
    total_val_test = val_ratio + test_ratio

    print(f"Loading dataset from: {dataset_dir}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        validation_split=total_val_test,
        subset="training",
    )

    temp_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        validation_split=total_val_test,
        subset="validation",
    )

    temp_card_np = tf.data.experimental.cardinality(temp_ds).numpy()
    try:
        temp_card = int(temp_card_np)
    except Exception:
        temp_card = int(np.asarray(temp_card_np).item())

    val_batches = int(round(temp_card * (val_ratio / total_val_test))) if temp_card > 0 else 0
    val_ds = temp_ds.take(val_batches)
    test_ds = temp_ds.skip(val_batches)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    # Data Augmentation pipeline for small dataset robustness
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ],
        name="data_augmentation",
    )

    # MobileNetV2 Base Backbone (Pretrained on ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze base weights initially

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="MobileNetV2_TransferLearning")
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
    )

    print("\n--- Phase 1: Feature Extraction Training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1,
    )

    if args.fine_tune:
        print("\n--- Phase 2: Fine-Tuning Top MobileNetV2 Layers ---")
        base_model.trainable = True
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr / 10.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        fine_tune_epochs = 15
        total_epochs = len(history.history["loss"]) + fine_tune_epochs
        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            initial_epoch=history.epoch[-1],
            callbacks=[early_stopping],
            verbose=1,
        )
        # Merge histories
        for k in history.history:
            history.history[k].extend(history_fine.history[k])

    print("\n--- Model Evaluation ---")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Final Test Loss:     {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

    model.save(model_out)
    print(f"Model saved to {model_out}")

    # Plot learning curves
    pd.DataFrame(history.history).plot(
        figsize=(8, 5),
        grid=True,
        xlabel="Epoch",
        title="MobileNetV2 Training & Validation Performance",
    )
    plt.legend(loc="lower left")
    save_fig("learning_curves", images_out)
    plt.close()

    # Confusion matrix evaluation
    y_true_batches = []
    y_pred_batches = []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true_batches.append(labels.numpy())
        y_pred_batches.append(preds)

    if y_true_batches:
        y_true = np.concatenate(y_true_batches)
        y_pred = np.concatenate(y_pred_batches)

        conf_mat = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

        plt.figure(figsize=(8, 6))
        plt.imshow(conf_mat, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix (MobileNetV2)")
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        threshold = conf_mat.max() / 2.0 if conf_mat.size > 0 else 0
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                plt.text(
                    j,
                    i,
                    str(conf_mat[i, j]),
                    horizontalalignment="center",
                    color="white" if conf_mat[i, j] > threshold else "black",
                )
        save_fig("confusion_matrix", images_out)
        plt.close()


if __name__ == "__main__":
    main()
