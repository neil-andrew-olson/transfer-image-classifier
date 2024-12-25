import tensorflow as tf
import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_cifar_data_structure(data_dir, validation_size=0.1):
    """Downloads CIFAR-10, creates the required directory structure, and populates with images."""

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Make our directories if they don't already exist:
    Path(os.path.join(data_dir, 'train')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_dir, 'validation')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(data_dir, 'test')).mkdir(parents=True, exist_ok=True)

    # make our 10 class directories under data/train/, data/validation/, and data/test/
    for i in range(10):
        Path(os.path.join(data_dir, 'train', f'class{i}')).mkdir(exist_ok=True)
        Path(os.path.join(data_dir, 'validation', f'class{i}')).mkdir(exist_ok=True)
        Path(os.path.join(data_dir, 'test', f'class{i}')).mkdir(exist_ok=True)

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, stratify=y_train)

    #Remove any extraneous folders if they are found.
    for split in ["train", "validation", "test"]:
         data_path = os.path.join(data_dir, split)
         folders = os.listdir(data_path)
         if len(folders) > 10:
            for dir in folders:
                try:
                    int(dir[-1])
                except:
                    shutil.rmtree(os.path.join(data_path, dir))

    def save_images(x_data, y_data, path):
        """Saves the images from the dataset."""
        for i, (image, label) in enumerate(zip(x_data, y_data)):
            label_int = int(label[0])
            image_path = os.path.join(path, f'class{label_int}', f'image_{i}.png')
            tf.keras.utils.save_img(image_path, image)

    # save our training images to the train directories:
    save_images(x_train, y_train, os.path.join(data_dir, 'train'))
    # save our validation images to the validation directories:
    save_images(x_val, y_val, os.path.join(data_dir, 'validation'))
    # save our test images to the test directories:
    save_images(x_test, y_test, os.path.join(data_dir, 'test'))

if __name__ == '__main__':
    data_dir = 'data'  # You can change this to your desired data directory path
    create_cifar_data_structure(data_dir)
    print("CIFAR-10 dataset downloaded, structured, and populated!")