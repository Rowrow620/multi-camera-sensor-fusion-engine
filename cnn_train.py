"""
-------------------------------------------------------------------------------------------------
cnn_train.py (this was annbest.py)


trains a simple cnn on a small dataset of cropped images labeled by types

added some features (as recommened by copilot ask agent) to the original annbest.py, such as:
- instead of the fashion mnist dataset, it uses cropped images from stereo_roi_finder
- uses 128x128 input size instead of 28x28, and 3 colors instead of grayscale
- added conv2d and max pooling layeers, creating a more complex model than the original annbest.py which only had dense layers.
- still using the ann hyperparameters such as learning rate and epochs, but they can be adjusted as needed.
- learning curves and confusion matrix are saved to the images/cnn directory
- added dataset directory and model saving paths

some tutorials I used:
https://www.tensorflow.org/tutorials/load_data/images
https://www.tensorflow.org/tutorials/images/data_augmentation
https://www.tensorflow.org/tutorials/images/transfer_learning
https://www.tensorflow.org/guide/data_performance

uses the cropped inputs produced by the stereo roi finder

-------------------------------------------------------------------------------------------------
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

print("Avoid memory hogging")
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "crops" / "labeled"
IMAGES_PATH = PROJECT_ROOT / "images" / "cnn"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "crop_cnn.keras"
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 8
EPOCHS = 30
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def save_fig(fig_id, tight_layout=True, fig_extension="jpg", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="jpeg", dpi=resolution)
    print(f"Saved figure to {path}")

IMAGES_PATH = Path() / "images" / "cnn"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

# load dataset and split into train, val, test
TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO
if TEST_RATIO < 0:
    raise ValueError("TRAIN_RATIO + VAL_RATIO must be <= 1.0")
total_val_test = VAL_RATIO + TEST_RATIO
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=total_val_test,
    subset="training",
)

temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=total_val_test,
    subset="validation",
)

# added this to handle the case where temp_ds is empty
temp_card_np = tf.data.experimental.cardinality(temp_ds).numpy()
try:
    temp_card = int(temp_card_np)
except Exception:
    temp_card = int(np.asarray(temp_card_np).item())

val_batches = int(round(temp_card * (VAL_RATIO / total_val_test))) if temp_card > 0 else 0
val_ds = temp_ds.take(val_batches)
test_ds = temp_ds.skip(val_batches)

class_names = train_ds.class_names
num_classes = len(class_names)

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
autotune = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(autotune)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(autotune)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(autotune)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=1,
)

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

model.save(MODEL_PATH)
pd.DataFrame(history.history).plot(
    figsize=(8, 5),
    xlim=[0, len(history.history["loss"]) - 1],
    ylim=[0, 1],
    grid=True,
    xlabel="Epoch",
)
plt.legend(loc="lower left")
save_fig("learning_curves")
plt.close()

y_true_batches = []
y_pred_batches = []
for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true_batches.append(labels.numpy())
    y_pred_batches.append(preds)

y_true = np.concatenate(y_true_batches)
y_pred = np.concatenate(y_pred_batches)

conf_mat = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

plt.figure(figsize=(8, 6))
plt.imshow(conf_mat, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted label")
plt.ylabel("True label")

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

save_fig("confusion_matrix")
plt.close()
