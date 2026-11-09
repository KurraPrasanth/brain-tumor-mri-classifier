# Brain Tumor MRI Classifier

## 1. Project Overview

The Brain Tumor MRI Classifier is a deep learning-based web application that analyzes brain MRI images and classifies them into two categories:

- Tumor
- No Tumor

The project uses a Convolutional Neural Network (CNN) built with TensorFlow and provides an interactive interface using Streamlit. The application is containerized and deployed using Podman.

**This project is intended for educational purposes and is not a medical diagnostic tool.**

## 2. Objectives

- Prepare and preprocess brain MRI image datasets.
- Develop a CNN model for binary image classification.
- Evaluate model performance using test data.
- Build an interactive MRI image prediction application.
- Containerize the application using Podman.
- Demonstrate container images, networking, volumes and lifecycle management.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Programming language |
| TensorFlow / Keras | CNN model development |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Streamlit | Web application |
| Podman | Containerization |
| Docker Hub | Container image registry |
| Ubuntu on WSL2 | Development environment |

## 4. Dataset

Dataset: Brain Tumor MRI Dataset by Masoud Nickparvar (Kaggle).

Dataset link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The original dataset contains four categories:
- Glioma
- Meningioma
- Pituitary
- No Tumor

For binary classification, the tumor categories are combined.

### Processed dataset

| Dataset | Tumor | No Tumor |
|---|---:|---:|
| Training | 4,200 | 1,400 |
| Testing | 1,200 | 400 |

## 5. CNN Model

The CNN architecture consists of:

1. Image input of size 128 × 128.
2. Convolutional layers with 32, 64 and 128 filters.
3. Max pooling layers.
4. Flatten layer.
5. Dense layer with 128 neurons.
6. Dropout layer with 0.5 dropout rate.
7. Sigmoid output layer for binary classification.

Optimizer: Adam  
Loss function: Binary Crossentropy  
Evaluation metric: Accuracy

## 6. Model Results

| Metric | Result |
|---|---:|
| Test Accuracy | 94.50% |
| Test Loss | 0.1837 |

The model is evaluated using a separate testing dataset and classification metrics.

## 7. Project Structure

```text
brain-tumor-mri-classifier/
├── app/
│   ├── app.py
│   └── predictor.py
├── training/
│   └── train.py
├── scripts/
├── tests/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── brain_tumor_cnn.keras
├── results/
│   ├── accuracy.png
│   ├── loss.png
│   └── confusion_matrix.png
├── docs/
├── podman-bind-data/
├── Containerfile
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Installation and Execution

Clone the repository:

```bash
git clone https://github.com/KurraPrasanth/brain-tumor-mri-classifier.git
cd brain-tumor-mri-classifier
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

## 9. Podman Deployment

### Build the application image

```bash
podman build -t brain-tumor-mri:v2 .
```

### Verify the image

```bash
podman images
podman image inspect brain-tumor-mri:v2
```

### Run the application

```bash
podman run -d --name brain-tumor-app -p 8501:8501 brain-tumor-mri:v2
```

### Container management

```bash
podman ps
podman ps -a
podman logs brain-tumor-app
podman port brain-tumor-app
podman stop brain-tumor-app
podman start brain-tumor-app
podman rm brain-tumor-app
```

### Docker Hub deployment

Published image:

```text
docker.io/prasanth2927/brain-tumor-mri:v2
```

Pull the published image:

```bash
podman pull docker.io/prasanth2927/brain-tumor-mri:v2
```

Run the downloaded image:

```bash
podman run -d --name brain-tumor-demo -p 8501:8501 docker.io/prasanth2927/brain-tumor-mri:v2
```

Docker Hub repository:

https://hub.docker.com/r/prasanth2927/brain-tumor-mri

## 10. Podman Networking

Create a custom network:

```bash
podman network create brain-tumor-net
```

Run a separate Nginx demonstration container on the custom network:

```bash
podman run -d --name network-demo --network brain-tumor-net nginx
```
This Nginx container demonstrates custom network attachment. It is separate from the Brain Tumor MRI application container.


Inspect the network:

```bash
podman network inspect brain-tumor-net
```

## 11. Podman Persistent Storage

Create a volume:

```bash
podman volume create brain-tumor-data
```

List volumes:

```bash
podman volume ls
```

Inspect the volume:

```bash
podman volume inspect brain-tumor-data
```

Mount the volume in a container:

```bash
podman run --rm -v brain-tumor-data:/data alpine ls -la /data
```

## 12. Results

The application supports:

- Uploading brain MRI images.
- Preprocessing uploaded images.
- Predicting Tumor or No Tumor categories.
- Displaying prediction confidence and class probabilities.
- Viewing model performance visualizations.

## 13. Future Enhancements

- Improve model performance and generalization.
- Add multi-class tumor classification.
- Implement prediction history.
- Improve the user interface.
- Explore model explainability techniques.

## 14. Disclaimer

This application is developed as an academic project. Its predictions are not intended for clinical diagnosis, treatment decisions or medical use. Always consult qualified medical professionals for healthcare decisions.

## 15. References

1. Brain Tumor MRI Dataset – Kaggle, Masoud Nickparvar.
2. TensorFlow Documentation – https://www.tensorflow.org/
3. Streamlit Documentation – https://docs.streamlit.io/
4. Podman Documentation – https://docs.podman.io/
5. OpenCV Documentation – https://docs.opencv.org/
