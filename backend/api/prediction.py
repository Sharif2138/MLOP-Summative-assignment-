import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
import joblib
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_disease_detection.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "encoder.pkl")

model = None
encoder = None


def load_model():
    global model, encoder

    model = tf.keras.models.load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)



def predict(path):

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = np.max(preds)

    class_name = encoder.inverse_transform([class_id])[0]

    return class_name, float(confidence)
