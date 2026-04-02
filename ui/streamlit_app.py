import streamlit as st
import os
from api.upload_new_data import upload_new_data
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Skin diseases detector", layout="centered")

st.title("Skin Disease Detection App")
st.write("Upload a skin image to know what disease it is .")

# prediction section
st.header("Detect Skin Disease")

uploaded_file = st.file_uploader(
    "Upload infected image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}

        response = requests.post(
            f"{API_URL}/predict",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:
            result = response.json()
            st.success(f"Class: {result['class_name']}")
            st.info(f"Confidence: {result['confidence']:.4f}")
        else:
            st.error("Prediction failed")
            
# plots
st.header("Model Performance Dashboard")

plot_files = {
    "class_distribution": "plots/class_distribution.png",
    "Training Accuracy": "plots/accuracy_curve.png",
    "Training Loss": "plots/loss_curve.png",
    "confusion_matrix": "plots/confusion_matrix.png",
}

# Check if at least one plot exists
available_plots = [p for p in plot_files.values() if os.path.exists(p)]

if available_plots:

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists(plot_files["class_distribution"]):
            st.image(plot_files["class_distribution"],
                     caption="Class Distribution", use_container_width=True)

        if os.path.exists(plot_files["Training Accuracy"]):
            st.image(plot_files["Training Accuracy"],
                     caption="Training Accuracy", use_container_width=True)

    with col2:
        
        if os.path.exists(plot_files["Training Loss"]):
            st.image(plot_files["confusion_matrix"],
                     caption="Confusion Matrix", use_container_width=True)
        
        if os.path.exists(plot_files["confusion_matrix"]):
            st.image(plot_files["confusion_matrix"],
                     caption="Confusion Matrix", use_container_width=True)

else:
    st.info("No training plots found. Train the model to generate performance metrics.")



# upoad data section
st.header("Upload Training Data")

uploaded_files = st.file_uploader(
    "Upload folder of new training images organised in subfolders by class",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Upload Data"):
        files = [("files", (f.name, f.getvalue(), f.type))
                 for f in uploaded_files]

        response = requests.post(
            f"{API_URL}/upload",
            files=files
        )

        if response.status_code == 200:
            st.success("Upload successful!")
            st.json(response.json())
        else:
            st.error("Upload failed")



# rerain section
st.header("Retrain Model")

if st.button("Start Retraining"):
    response = requests.post(f"{API_URL}/retrain")

    if response.status_code == 200:
        st.warning("Retraining started in background!")
        st.json(response.json())
    else:
        st.error("Failed to start retraining")
