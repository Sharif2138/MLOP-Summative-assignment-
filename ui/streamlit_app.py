import streamlit as st
import os
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Skin Diseases Detector", layout="centered")

st.title("Skin Disease Detection App")
st.write("Upload a skin image to know what disease it is.")

# =========================
# PREDICTION SECTION
# =========================
st.header("Detect Skin Disease")

uploaded_file = st.file_uploader(
    "Upload infected image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        files = {
            "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
        }

        response = requests.post(f"{API_URL}/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Class: {result['class_name']}")
            st.info(f"Confidence: {result['confidence']:.4f}")
        else:
            st.error(f"Prediction failed: {response.text}")


# =========================
# MODEL PERFORMANCE PLOTS
# =========================
st.header("Model Performance Dashboard")

plot_files = {
    "Class Distribution": "plots/class_distribution.png",
    "Training Accuracy": "plots/accuracy_curve.png",
    "Training Loss": "plots/loss_curve.png",
    "Confusion Matrix": "plots/confusion_matrix.png",
}

available = {k: v for k, v in plot_files.items() if os.path.exists(v)}

if available:
    col1, col2 = st.columns(2)
    items = list(available.items())

    for i, (title, path) in enumerate(items):
        if i % 2 == 0:
            with col1:
                st.image(path, caption=title, use_container_width=True)
        else:
            with col2:
                st.image(path, caption=title, use_container_width=True)
else:
    st.info("No training plots found. Train the model to generate performance metrics.")


# =========================
# UPLOAD TRAINING DATA
# =========================
st.header("Upload Training Data")

class_name = st.text_input("Class label (ie 'benign', 'malignant')")

uploaded_files = st.file_uploader(
    "Upload images for this class",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Upload Data"):
        if not class_name.strip():
            st.error("Please enter a class label before uploading.")
        else:
            files = [
                ("files", (f"{class_name}/{f.name}", f, f.type))
                for f in uploaded_files
            ]

            response = requests.post(f"{API_URL}/upload", files=files)

            if response.status_code == 200:
                st.success(
                    f"Uploaded {len(uploaded_files)} images under class '{class_name}'!")
                st.json(response.json())
            else:
                st.error(f"Upload failed: {response.text}")


# =========================
# RETRAIN MODEL
# =========================
st.header("Retrain Model")

if st.button("Start Retraining"):
    response = requests.post(f"{API_URL}/retrain")

    if response.status_code == 200:
        st.warning("Retraining started in background!")
        st.json(response.json())
    else:
        st.error(f"Failed to start retraining: {response.text}")
