
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "processed" / "Training"
TEST_DIR = BASE_DIR / "data" / "processed" / "Testing"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

CLASS_NAMES = ["No_Tumor", "Tumor"]

print("TensorFlow:", tf.__version__)

# --------------------------------------------------
# 3. LOAD TRAINING DATA
# --------------------------------------------------

train_full_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=CLASS_NAMES,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    shuffle=True
)

# --------------------------------------------------
# 4. LOAD VALIDATION DATA
# --------------------------------------------------

validation_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=CLASS_NAMES,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    shuffle=False
)

# --------------------------------------------------
# 5. LOAD ORIGINAL TEST DATA
# --------------------------------------------------

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=CLASS_NAMES,
    shuffle=False
)

print("Class mapping:", CLASS_NAMES)

# --------------------------------------------------
# 6. CLASS WEIGHTS
# --------------------------------------------------

labels = np.concatenate([
    y.numpy().flatten()
    for _, y in train_full_ds
])

classes = np.unique(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("Class weights:", class_weights)

# --------------------------------------------------
# 7. NORMALIZATION
# --------------------------------------------------

normalization = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_full_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_ds = validation_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# --------------------------------------------------
# 8. BUILD CNN MODEL
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# 9. EARLY STOPPING
# --------------------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# --------------------------------------------------
# 10. TRAIN CNN
# --------------------------------------------------

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# --------------------------------------------------
# 11. FINAL TEST EVALUATION
# --------------------------------------------------

print("\nEvaluating on untouched testing dataset...")

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# --------------------------------------------------
# 12. CONFUSION MATRIX AND REPORT
# --------------------------------------------------

y_true = np.concatenate([
    y.numpy().flatten()
    for _, y in test_ds
]).astype(int)

predictions = model.predict(test_ds)
y_pred = (predictions.flatten() >= 0.5).astype(int)

print("\nClassification Report:")

print(classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(cmap="Blues", values_format="d")
plt.title("Brain Tumor CNN - Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.close()

# --------------------------------------------------
# 13. SAVE MODEL
# --------------------------------------------------

model_path = MODEL_DIR / "brain_tumor_cnn.keras"
model.save(model_path)

print("\nModel saved:", model_path)

# --------------------------------------------------
# 14. ACCURACY GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(RESULTS_DIR / "accuracy.png")
plt.close()

# --------------------------------------------------
# 15. LOSS GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(RESULTS_DIR / "loss.png")
plt.close()

print("\nCNN TRAINING COMPLETED SUCCESSFULLY!")
print("Accuracy graph:", RESULTS_DIR / "accuracy.png")
print("Loss graph:", RESULTS_DIR / "loss.png")
print("Confusion matrix:", RESULTS_DIR / "confusion_matrix.png")

