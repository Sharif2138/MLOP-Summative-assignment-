import shutil                       # To copy files locally
import os                           # To handle folders
from supabase import create_client  # To connect to Supabase
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import pickle
from pathlib import Path
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from supabase import create_client
import os
from sklearn.model_selection import train_test_split


# Initialize Supabase client
supabase = create_client(
    supabase_url=os.getenv("supabase_project_url"),
    supabase_key=os.getenv("supabase_annon_key")
)


TRAIN_FOLDER = "dataset/train"
if os.path.exists(TRAIN_FOLDER):
    shutil.rmtree(TRAIN_FOLDER)

os.makedirs(TRAIN_FOLDER, exist_ok=True)

# Function to download a folder from Supabase to local machine
def download_from_supabase(bucket, cloud_folder, local_root):
    files = supabase.storage.from_(bucket).list(cloud_folder)
    for f in files:
        # Full path in the cloud
        cloud_path = f"{cloud_folder}/{f['name']}"
        # Local path mirrors cloud structure
        local_path = os.path.join(local_root, cloud_path)
        # Make subfolders if they don't exist locally
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        # Download the file
        data = supabase.storage.from_(bucket).download(cloud_path)
        with open(local_path, "wb") as file:
            file.write(data)


#Download original dataset
download_from_supabase("training-images", "original_dataset", "dataset")

#Download new user uploads
download_from_supabase("training-images", "user_uploads", "dataset")

#Merge original + new uploads into TRAIN_FOLDER
def merge_to_train(source_folder, train_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith((".jpg", ".png")):
                # Get subfolder path relative to source_folder
                relative_path = os.path.relpath(root, source_folder)
                # Create same subfolder in train folder
                dest_folder = os.path.join(train_folder, relative_path)
                os.makedirs(dest_folder, exist_ok=True)
                # Copy image to train folder
                shutil.copy(os.path.join(root, file),
                            os.path.join(dest_folder, file))


# Merge everything
merge_to_train("dataset/original_dataset", TRAIN_FOLDER)
merge_to_train("dataset/user_uploads", TRAIN_FOLDER)

#Check result
for root, dirs, files in os.walk(TRAIN_FOLDER):
    for file in files:
        print(os.path.join(root, file))

def retrain(TRAIN_FOLDER):
    
    all_dataset = Path(TRAIN_FOLDER)
    all_images = list(all_dataset.glob('*/*.jpg')) + list(all_dataset.glob('*/*.png'))
    all_images = [str(p) for p in all_images]
    all_labels = [Path(p).parent.name for p in all_images]

    # encoding the labels
    with open('../model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    all_labels_encoded = encoder.transform(all_labels)
    num_classes = len(encoder.classes_)
    
    train_images, val_images, train_labels_encoded, val_labels_encoded = train_test_split(
        all_images,
        all_labels_encoded,
        test_size=0.2,
        stratify=all_labels_encoded,  # important for class balance
        random_state=42
    )

    # forming a tf.data datasets
    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels_encoded))
    val_data = tf.data.Dataset.from_tensor_slices((val_images, val_labels_encoded))

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
    val_data = val_data.batch(Batch_size).prefetch(autotune)
    
    model = tf.keras.models.load_model('../model/mobilenetv2.h5')
    
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[:-20]:
       layer.trainable = True
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_data,
        epochs=5,
        verbose=1,
    )
    
    model.save('../model/mobilenetv2.h5')
    return history
