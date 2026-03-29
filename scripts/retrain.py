from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import pickle
from pathlib import Path
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder


folder_path = "../data/uploads"

def retrain(folder_path):
    train_dataset = Path(folder_path)
    train_images = list(train_dataset.glob('*/*.jpg'))
    train_labels = [path.parent.name for path in train_images]

    # encoding the labels
    with open('../model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    train_labels_encoded = encoder.fit_transform(train_labels)
    num_classes = len(encoder.classes_)

    # forming a tf.data datasets
    train_data = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels_encoded))

    # function to load the images
    def load_images(path, label):
      img = tf.io.read_file(path)
      img = tf.image.decode_jpeg(img, channels=3)
      img = tf.image.resize(img, [128, 128])
      img = preprocess_input(img)

      return img, label

    # preprocess
    Batch_size = 32
    autotune = tf.data.AUTOTUNE

    train_data = train_data.map(load_images, num_parallel_calls=autotune)

    # data augmentation
    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ])

    def augment(img, label):
        return data_aug(img, training=True), label

    train_data = train_data.map(augment, num_parallel_calls=autotune)

    # CPU and GPU optimizatitions
    train_data = train_data.shuffle(len(train_images)).batch(
        Batch_size).prefetch(autotune)
    
    model = tf.keras.models.load_model('../model/mobilenetv2.h5')
    for layer in model.layers[:-5]:
       layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_data,
        epochs=10,
        verbose=1,
    )
    
    model.save('../model/mobilenetv2.h5')
    return history
