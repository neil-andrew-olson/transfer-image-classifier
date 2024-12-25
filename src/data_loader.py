import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
def load_and_preprocess_data(data_dir, batch_size, image_size):
    """Loads and preprocesses image data from directories."""

    train_data = image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size
    )
    val_data = image_dataset_from_directory(
        os.path.join(data_dir, 'validation'),
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size
    )
    test_data = image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size
    )

    return train_data, val_data, test_data