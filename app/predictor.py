
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# Project configuration
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "brain_tumor_cnn.keras"

IMAGE_SIZE = (128, 128)

# Load the trained model once
model = None


def load_model():
    """Load the trained CNN model."""

    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        model = tf.keras.models.load_model(MODEL_PATH)

    return model


def validate_mri_image(image):
    """
    Perform basic MRI-like image screening.

    This is a heuristic, not a reliable MRI detector.
    It rejects strongly coloured and nearly blank images.
    """

    image = image.convert("RGB")
    sample = image.resize((128, 128))

    # Check colour saturation
    hsv_image = np.array(sample.convert("HSV"))
    saturation = hsv_image[:, :, 1]

    mean_saturation = float(np.mean(saturation))

    if mean_saturation > 45:
        raise ValueError(
            "This image appears strongly coloured and may not be "
            "a brain MRI scan. Please upload a suitable MRI image."
        )

    # Check whether the image is nearly blank
    grayscale = np.array(sample.convert("L"))
    image_variation = float(np.std(grayscale))

    if image_variation < 8:
        raise ValueError(
            "The uploaded image appears blank or has insufficient "
            "visual information. Please upload another MRI image."
        )

    return image


def preprocess_image(uploaded_file):
    """Validate and prepare an uploaded MRI-like image."""

    image = Image.open(uploaded_file).convert("RGB")

    # Basic screening before CNN prediction
    image = validate_mri_image(image)

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)

    # Normalize pixel values to 0-1
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_mri(uploaded_file):
    """Predict whether an MRI-like image shows Tumor or No Tumor."""

    cnn_model = load_model()

    processed_image = preprocess_image(uploaded_file)

    prediction = cnn_model.predict(
        processed_image,
        verbose=0
    )

    tumor_probability = float(prediction[0][0])

    if tumor_probability >= 0.5:
        result = "Tumor"
        confidence = tumor_probability * 100
    else:
        result = "No Tumor"
        confidence = (1 - tumor_probability) * 100

    return {
        "prediction": result,
        "confidence": round(confidence, 2),
        "tumor_probability": round(
            tumor_probability * 100, 2
        ),
        "no_tumor_probability": round(
            (1 - tumor_probability) * 100, 2
        )
    }
