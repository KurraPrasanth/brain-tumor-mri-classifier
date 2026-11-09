
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

from pathlib import Path
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from predictor import predict_mri

# ---------------------------------------
# Paths and page configuration
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

st.set_page_config(
    page_title="NeuroScan AI | MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------
# Custom dashboard styling
# ---------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111a30 55%, #101426 100%);
        color: #eef2ff;
    }

    [data-testid="stSidebar"] {
        background: #0b1224;
        border-right: 1px solid #26334f;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172b50, #31265c);
        border: 1px solid #40517b;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #c4d2f0;
        font-size: 15px;
        margin-bottom: 0;
    }

    .eyebrow {
        color: #9fafff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .section-title {
        color: #eef2ff;
        font-size: 22px;
        font-weight: 700;
        margin: 18px 0 12px 0;
    }

    .info-card {
        background: #17233b;
        border: 1px solid #2d4063;
        border-radius: 14px;
        padding: 20px;
        min-height: 110px;
    }

    .info-label {
        color: #9eafd0;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .info-value {
        color: #ffffff;
        font-size: 25px;
        font-weight: 700;
    }

    .result-card {
        background: linear-gradient(135deg, #1c3152, #29264f);
        border: 1px solid #5968a0;
        border-radius: 16px;
        padding: 22px;
        margin: 12px 0;
    }

    .result-label {
        color: #b8c7e8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 700;
        margin-top: 8px;
    }

    .muted {
        color: #aab8d5;
        font-size: 13px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 46px;
        border: 1px solid #7786ff;
        background: linear-gradient(90deg, #5865f2, #805ad5);
        color: white;
    }

    .stButton > button:hover {
        border-color: #a5b4fc;
        color: white;
        background: linear-gradient(90deg, #6975ff, #956be8);
    }

    [data-testid="stFileUploader"] {
        background: #17233b;
        border: 1px dashed #6678b5;
        border-radius: 14px;
        padding: 12px;
    }

    div[data-testid="stMetric"] {
        background: #17233b;
        border: 1px solid #2d4063;
        border-radius: 12px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #b7c5e2;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    .footer {
        color: #8798ba;
        text-align: center;
        font-size: 12px;
        padding: 22px 0;
    }

    hr {
        border-color: #2d4063;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Sidebar navigation
# ---------------------------------------

with st.sidebar:
    st.markdown("# 🧠 NeuroScan AI")
    st.caption("Brain MRI Intelligence Platform")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Model Performance", "About Project"],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### Model information")
    st.markdown("**Architecture:** CNN")
    st.markdown("**Input size:** 128 × 128")
    st.markdown("**Classes:** 2")
    st.markdown("**Framework:** TensorFlow")

    st.divider()
    st.caption("Educational AI research project")

# ---------------------------------------
# Reusable image display helper
# ---------------------------------------

def show_result_image(image_path, caption):
    if image_path.exists():
        st.image(
            str(image_path),
            caption=caption,
            width="stretch"
        )
    else:
        st.info(f"{caption} is not available yet.")

# ---------------------------------------
# Dashboard page
# ---------------------------------------

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Artificial Intelligence · Medical Imaging</div>
        <h1>Brain Tumor MRI Classifier</h1>
        <p>
            Analyze brain MRI images using a Convolutional Neural Network.
            Upload an image to explore the model's prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Overview</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <div class="info-label">Test Accuracy</div>
            <div class="info-value">94.50%</div>
            <div class="muted">Held-out test dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <div class="info-label">Model Architecture</div>
            <div class="info-value">CNN</div>
            <div class="muted">Deep learning classifier</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
            <div class="info-label">Classification</div>
            <div class="info-value">Binary</div>
            <div class="muted">Tumor / No Tumor</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-title">MRI Image Analysis</div>',
                unsafe_allow_html=True)

    st.write(
        "Upload a brain MRI image in JPG, JPEG or PNG format."
    )

    uploaded_file = st.file_uploader(
        "Choose your MRI image",
        type=["jpg", "jpeg", "png"],
        key="mri_upload"
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        col1, col2 = st.columns([1, 1.15], gap="large")

        with col1:
            st.markdown("#### Uploaded Image")

            try:
                with Image.open(BytesIO(image_bytes)) as img:
                    img.verify()

                with Image.open(BytesIO(image_bytes)) as img:
                    preview = img.convert("RGB")
                    st.image(
                        preview,
                        caption=uploaded_file.name,
                        width="stretch"
                    )

                st.success("Image file successfully loaded.")

                if len(image_bytes) > 10 * 1024 * 1024:
                    st.error("Image exceeds the 10 MB upload limit.")
                    valid_image = False
                else:
                    valid_image = True

            except (UnidentifiedImageError, OSError, ValueError):
                st.error("Invalid image. Please upload a valid JPG or PNG file.")
                valid_image = False

        with col2:
            st.markdown("#### AI Analysis")

            st.markdown(
                "The CNN model will process the uploaded image "
                "and estimate the probability of each class."
            )

            analyze = st.button(
                "🧠  Analyze MRI Image",
                type="primary",
                use_container_width=True,
                disabled=not valid_image
            )

            if analyze:

                try:
                    with st.spinner("Processing image with CNN..."):
                        uploaded_file.seek(0)
                        result = predict_mri(uploaded_file)

                    prediction = str(result["prediction"])
                    confidence = float(result["confidence"])
                    tumor_prob = float(result["tumor_probability"])
                    no_tumor_prob = float(result["no_tumor_probability"])

                    if not all(
                        0 <= value <= 100
                        for value in [confidence, tumor_prob, no_tumor_prob]
                    ):
                        raise ValueError("Model returned invalid probability values.")

                    st.session_state["last_result"] = {
                        "prediction": prediction,
                        "confidence": confidence,
                        "tumor_probability": tumor_prob,
                        "no_tumor_probability": no_tumor_prob,
                        "filename": uploaded_file.name
                    }
                    # Trigger automatic scrolling after the result is rendered.
                    st.session_state["scroll_to_results"] = True

                except Exception as error:
                    st.error(f"Analysis failed: {error}")

        # Keep the result visible after Streamlit reruns.
        result = st.session_state.get("last_result")

        if result is not None:

            # Stable target for the automatic scroll action.
            st.markdown(
                '<div id="prediction-results" style="scroll-margin-top: 24px;"></div>',
                unsafe_allow_html=True
            )

            st.divider()
            st.markdown("### Prediction Results")

            r1, r2 = st.columns(2)

            with r1:
                st.markdown("""
                <div class="result-card">
                    <div class="result-label">Predicted Category</div>
                    <div class="result-value">{}</div>
                </div>
                """.format(result["prediction"]), unsafe_allow_html=True)

            with r2:
                st.markdown("""
                <div class="result-card">
                    <div class="result-label">Model Confidence</div>
                    <div class="result-value">{:.2f}%</div>
                </div>
                """.format(result["confidence"]), unsafe_allow_html=True)

            st.markdown("#### Class Probability Distribution")

            p1, p2 = st.columns(2)

            with p1:
                st.write("🔴 Tumor")
                st.progress(
                    min(1.0, max(0.0, result["tumor_probability"] / 100))
                )
                st.write(f'{result["tumor_probability"]:.2f}%')

            with p2:
                st.write("🟢 No Tumor")
                st.progress(
                    min(1.0, max(0.0, result["no_tumor_probability"] / 100))
                )
                st.write(f'{result["no_tumor_probability"]:.2f}%')

            st.caption(f'Analyzed file: {result["filename"]}')

            st.warning(
                "This is an educational model prediction, not a medical "
                "diagnosis. Model confidence does not represent clinical certainty."
            )

            # Scroll smoothly to the prediction output once, after analysis.
            if st.session_state.get("scroll_to_results", False):
                components.html(
                    """
                    <script>
                        setTimeout(() => {
                            const target = window.parent.document.getElementById(
                                "prediction-results"
                            );
                            if (target) {
                                target.scrollIntoView({
                                    behavior: "smooth",
                                    block: "start"
                                });
                            }
                        }, 250);
                    </script>
                    """,
                    height=0,
                )
                st.session_state["scroll_to_results"] = False

    else:
        st.info("Upload an MRI image to begin the analysis.")

    st.divider()

    st.markdown("### Quick Model Performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("Test Accuracy", "94.50%")
    m2.metric("Test Loss", "0.1837")
    m3.metric("Input Resolution", "128 × 128")

# ---------------------------------------
# Model performance page
# ---------------------------------------

elif page == "Model Performance":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Model Evaluation</div>
        <h1>Performance Analytics</h1>
        <p>Explore training metrics and evaluation visualizations.</p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c = st.columns(3)
    a.metric("Test Accuracy", "94.50%")
    b.metric("Test Loss", "0.1837")
    c.metric("Model Type", "CNN")

    st.divider()

    st.markdown("### Training and Validation Accuracy")
    show_result_image(
        RESULTS_DIR / "accuracy.png",
        "Accuracy over training epochs"
    )

    st.divider()

    st.markdown("### Training and Validation Loss")
    show_result_image(
        RESULTS_DIR / "loss.png",
        "Loss over training epochs"
    )

    st.divider()

    st.markdown("### Confusion Matrix")
    show_result_image(
        RESULTS_DIR / "confusion_matrix.png",
        "Test dataset confusion matrix"
    )

# ---------------------------------------
# About project page
# ---------------------------------------

elif page == "About Project":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Project Information</div>
        <h1>About NeuroScan AI</h1>
        <p>A deep learning educational project for brain MRI image classification.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project Objective")

    st.write(
        "This project demonstrates how a Convolutional Neural Network "
        "can learn image patterns and classify brain MRI images into "
        "Tumor and No Tumor categories."
    )

    st.markdown("### Technology Stack")

    technologies = [
        ("Python", "Programming language"),
        ("TensorFlow / Keras", "CNN model development"),
        ("Streamlit", "Interactive frontend"),
        ("Pillow", "Image loading and validation"),
        ("Matplotlib", "Performance visualization"),
        ("Podman", "Containerization and deployment")
    ]

    for technology, purpose in technologies:
        st.markdown(f"- **{technology}:** {purpose}")

    st.markdown("### Model Details")

    st.write("""
    - Architecture: Convolutional Neural Network
    - Input image dimensions: 128 × 128
    - Output classes: Tumor and No Tumor
    - Test accuracy: 94.50%
    - Test loss: 0.1837
    """)

    st.markdown("### Important Disclaimer")

    st.warning(
        "This project is strictly for educational and demonstration "
        "purposes. It is not validated for clinical use and must not "
        "be used to diagnose, treat, or make medical decisions."
    )

# ---------------------------------------
# Footer
# ---------------------------------------

st.markdown("""
<hr>
<div class="footer">
    NeuroScan AI · Brain Tumor MRI Classifier<br>
    Built with Python · TensorFlow · CNN · Streamlit · Podman
</div>
""", unsafe_allow_html=True)
