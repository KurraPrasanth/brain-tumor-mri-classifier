
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from predictor import predict_mri

# ---------------------------------------
# Page configuration
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------
# Application heading
# ---------------------------------------

st.title("🧠 Brain Tumor MRI Classifier")

st.write(
    "An educational CNN-based application that "
    "classifies uploaded brain MRI images into "
    "Tumor or No Tumor categories."
)

st.warning(
    "Educational demonstration only. This application "
    "is not a medical diagnostic tool. Always consult "
    "a qualified medical professional."
)

st.divider()

# ---------------------------------------
# MRI image upload
# ---------------------------------------

st.header("Upload Brain MRI Image")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded MRI Image")

        st.image(
            uploaded_file,
            caption="Uploaded Brain MRI",
            use_container_width=True
        )

    with col2:
        st.subheader("CNN Prediction")

        if st.button("Analyze MRI Image", type="primary"):

            try:
                with st.spinner("Analyzing MRI image..."):

                    uploaded_file.seek(0)
                    result = predict_mri(uploaded_file)

                st.success("Prediction completed!")

                st.metric(
                    "Predicted Category",
                    result["prediction"]
                )

                st.metric(
                    "Prediction Confidence",
                    f'{result["confidence"]:.2f}%'
                )

                st.write(
                    "Tumor probability:",
                    f'{result["tumor_probability"]:.2f}%'
                )

                st.write(
                    "No Tumor probability:",
                    f'{result["no_tumor_probability"]:.2f}%'
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

# ---------------------------------------
# Model performance section
# ---------------------------------------

st.divider()

st.header("Model Performance")

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Test Accuracy", "94.50%")
metric2.metric("Test Loss", "0.1837")
metric3.metric("Model", "CNN")

# ---------------------------------------
# Display accuracy graph
# ---------------------------------------

st.subheader("Training and Validation Accuracy")

accuracy_path = RESULTS_DIR / "accuracy.png"

if accuracy_path.exists():
    st.image(
        str(accuracy_path),
        use_container_width=True
    )

# ---------------------------------------
# Display loss graph
# ---------------------------------------

st.subheader("Training and Validation Loss")

loss_path = RESULTS_DIR / "loss.png"

if loss_path.exists():
    st.image(
        str(loss_path),
        use_container_width=True
    )

# ---------------------------------------
# Display confusion matrix
# ---------------------------------------

st.subheader("Confusion Matrix")

confusion_path = RESULTS_DIR / "confusion_matrix.png"

if confusion_path.exists():
    st.image(
        str(confusion_path),
        use_container_width=True
    )

# ---------------------------------------
# Footer
# ---------------------------------------

st.divider()

st.caption(
    "Brain Tumor MRI Classifier | "
    "Built using Python, TensorFlow, CNN and Streamlit"
)
