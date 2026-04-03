import shutil
import os
from supabase import create_client
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import pickle
from pathlib import Path
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os

load_dotenv()


def retrain_pipeline():

    supabase = create_client(
        supabase_url=os.getenv("supabase_project_url"),
        supabase_key=os.getenv("supabase_anon_key")
    )
    
    print("\n[bucket root contents]")
    root_items = supabase.storage.from_("training_data").list("")
    for item in root_items:
        print(item)

    print("\n[new_data contents]")
    try:
        nd_items = supabase.storage.from_("training_data").list("new_data")
        for item in nd_items:
            print(item)
    except Exception as e:
        print(f"Error listing new_data: {e}")

    TRAIN_FOLDER = "dataset"
    DOWNLOAD_FOLDER = "dataset_raw"

    for folder in [TRAIN_FOLDER, DOWNLOAD_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    def download_from_supabase(bucket, cloud_folder, local_root):
        """Recursively download all files from a Supabase storage folder."""
        try:
            items = supabase.storage.from_(bucket).list(cloud_folder)
        except Exception as e:
            print(f"[error listing] {cloud_folder}: {e}")
            return

        for item in items:
            cloud_path = f"{cloud_folder}/{item['name']}"

            # Supabase folders have no 'id', files always have one
            is_folder = item.get("id") is None

            if is_folder:
                print(f"[folder] {cloud_path} — recursing...")
                download_from_supabase(bucket, cloud_path, local_root)
            else:
                local_path = os.path.join(local_root, cloud_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                try:
                    data = supabase.storage.from_(bucket).download(cloud_path)
                    with open(local_path, "wb") as f:
                        f.write(data)
                    print(f"[downloaded] {cloud_path}")
                except Exception as e:
                    print(f"[error downloading] {cloud_path}: {e}")

    # Download original dataset and new uploads into raw folder
    download_from_supabase(
        "training_data", "training_data/original_data", DOWNLOAD_FOLDER)
    download_from_supabase(
        "training_data", "training_data/new_data", DOWNLOAD_FOLDER)

    def merge_to_train(source_folder, train_folder):
        """
        Walk source_folder, find images, and copy them into train_folder/class_name/.
        Works regardless of how deeply nested the source images are.
        """
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    class_name = os.path.basename(root)
                    dest_folder = os.path.join(train_folder, class_name)
                    os.makedirs(dest_folder, exist_ok=True)
                    src = os.path.join(root, file)
                    dst = os.path.join(dest_folder, file)
                    if os.path.exists(dst):
                        base, ext = os.path.splitext(file)
                        dst = os.path.join(dest_folder, f"{base}_dup{ext}")
                    shutil.copy(src, dst)

    # Merge everything into flat class-based structure: dataset/class_name/img.jpg
    merge_to_train(os.path.join(DOWNLOAD_FOLDER,
                   "training_data", "original_data"), TRAIN_FOLDER)
    merge_to_train(os.path.join(DOWNLOAD_FOLDER,
                   "training_data", "new_data"), TRAIN_FOLDER)

    # Debug: print what ended up in TRAIN_FOLDER
    print("\n[dataset contents]")
    for root, dirs, files in os.walk(TRAIN_FOLDER):
        for file in files:
            print(os.path.join(root, file))

    def retrain(TRAIN_FOLDER):

        all_dataset = Path(TRAIN_FOLDER)
        all_images = [
            str(p)
            for p in list(all_dataset.glob("*/*.jpg"))
            + list(all_dataset.glob("*/*.jpeg"))
            + list(all_dataset.glob("*/*.png"))
        ]

        if not all_images:
            raise ValueError(
                f"No images found in '{TRAIN_FOLDER}'. Aborting retrain.")

        all_labels = [Path(p).parent.name for p in all_images]

        encoder = LabelEncoder()
        all_labels_encoded = encoder.fit_transform(all_labels)
        num_classes = len(encoder.classes_)
        print(f"[encoder] {num_classes} classes: {list(encoder.classes_)}")
        
        BASE_DIR = os.path.dirname(__file__)

        MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "skin_disease_detection.keras")
        ENCODER_PATH = os.path.join(BASE_DIR, "..", "model", "encoder.pkl")

        os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump(encoder, f)

        train_images, val_images, train_labels_encoded, val_labels_encoded = train_test_split(
            all_images,
            all_labels_encoded,
            test_size=0.2,
            stratify=all_labels_encoded,
            random_state=42
        )

        train_data = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels_encoded))
        val_data = tf.data.Dataset.from_tensor_slices(
            (val_images, val_labels_encoded))

        def load_images(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(
                img, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = tf.image.resize(img, [224, 224])
            img = preprocess_input(img)
            return img, label

        Batch_size = 16
        autotune = tf.data.AUTOTUNE

        train_data = train_data.map(load_images, num_parallel_calls=autotune)
        val_data = val_data.map(load_images, num_parallel_calls=autotune)

        data_aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ])

        def augment(img, label):
            return data_aug(img, training=True), label

        train_data = train_data.map(augment, num_parallel_calls=autotune)
        train_data = train_data.shuffle(1000).batch(
            Batch_size).cache().prefetch(autotune)
        val_data = val_data.batch(Batch_size).cache().prefetch(autotune)

        
        base_model = tf.keras.models.load_model(MODEL_PATH)


        # Find the last non-output dense layer by name
        base_model_output = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, layers.Dense):
              base_model_output = layer.input
              break

            if base_model_output is None:
              raise ValueError("Could not find a Dense layer in the base model.")

        new_output = layers.Dense(
        num_classes, activation="softmax", name="predictions")(base_model_output)
        model = tf.keras.Model(inputs=base_model.inputs, outputs=new_output)

        for layer in model.layers[:-40]:
            layer.trainable = False
        for layer in model.layers[-40:]:
            layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=1,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=5,
            callbacks=callbacks,
            verbose=1,
        )

        model.save(MODEL_PATH)
        return history

    history = retrain(TRAIN_FOLDER)

    # Clean up after training
    shutil.rmtree(DOWNLOAD_FOLDER, ignore_errors=True)
    shutil.rmtree(TRAIN_FOLDER, ignore_errors=True)

    return history
