import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
import joblib
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model('../model/mobilenetv2.h5')
encoder = joblib.load('../model/encoder.pkl')

def predict(path):
    
    
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = preprocess_input(img)

    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = np.max(preds)

    class_name = encoder.inverse_transform([class_id])[0]

    return class_name, float(confidence)
