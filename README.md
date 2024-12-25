# Image Classification with Transfer Learning

This repository implements an image classification model using transfer learning with TensorFlow and Keras. It utilizes a pre-trained convolutional neural network (CNN) as a feature extractor and fine-tunes it for the CIFAR-10 dataset.

## Project Overview

This project demonstrates the following:

*   **Transfer Learning:** Leveraging pre-trained models (like VGG16, ResNet50, MobileNetV2) trained on large datasets to improve performance on smaller datasets.
*   **Fine-tuning:** Training custom classification layers on top of a pre-trained model to adapt it to the specific classification task.
*   **CIFAR-10 Dataset:** Using the CIFAR-10 dataset (10 classes of images) for experimentation.
*   **Data Preprocessing:** Organizing the dataset for use with the Keras `image_dataset_from_directory` function.
*   **Model Training:** Training the model on the CIFAR-10 dataset.
*   **Model Evaluation:** Evaluating the model's performance on the test set.
*   **Training History Visualization:** Plotting training and validation metrics to monitor training.
*   **Model Saving:** Saving trained models for future use.

## Project Structure

The project directory is structured as follows:
transfer-image-classifier/
├── data/
│ ├── train/
│ │ ├── class0/
│ │ ├── class1/
│ │ ├── ...
│ │ └── class9/
│ ├── validation/
│ │ ├── class0/
│ │ ├── class1/
│ │ ├── ...
│ │ └── class9/
│ └── test/
│ ├── class0/
│ ├── class1/
│ ├── ...
│ └── class9/
├── models/ (Trained model weights will be saved here)
├── notebooks/ (Jupyter notebooks for experimentation)
├── src/
│ ├── init.py (Makes the src directory a Python package)
│ ├── data_loader.py (Loads and preprocesses image data)
│ ├── model.py (Contains model creation, training, and saving code)
│ ├── utils.py (Utility functions e.g. plots)
│ └── train.py (Main training script)
├── data_setup.py (Downloads, structures, and populates the CIFAR-10 data)
├── requirements.txt (Lists project dependencies)
├── .gitignore (Specifies files to be ignored by Git)
└── README.md (This file)

## Dataset

The project uses the CIFAR-10 dataset. The `data_setup.py` script automatically downloads, structures, and splits the dataset into training, validation, and test folders with their respective class folders (`class0`, `class1`, ... `class9`).

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/neil-andrew-olson/transfer-image-classifier.git
    cd transfer-image-classifier
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    # source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Required Packages:**
#   pip freeze > requirements.txt # if using virtual environment
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download, Structure and Prepare the Dataset:**
    ```bash
    python data_setup.py
    ```
    This script will download the CIFAR-10 dataset, and create the class directory structures in `data/`. It will then copy images to the `train`, `validation`, and `test` subdirectories.

## Usage

1.  **Run the training script:**
    ```bash
    python -m src.train
    ```
    This will run the training script with default command line arguments:
       *   `--data_dir data`: Path to the dataset directory.
       *   `--base_model VGG16`: Pre-trained model used.
       *   `--batch_size 32`: Batch size for training.
       *   `--image_size_x 224`: Image width.
       *   `--image_size_y 224`: Image height.
       *   `--num_epochs 10`: Number of training epochs.
       *   `--learning_rate 0.001`: Learning rate for the optimizer.
       *   `--model_save_path models/transfer_model.h5`: Path to save the trained model.

    *   Alternatively, you can run the script with your chosen parameters:
    ```bash
    python -m src.train --data_dir data --base_model VGG16 --batch_size 64 --image_size_x 224 --image_size_y 224 --num_epochs 20 --learning_rate 0.0001 --model_save_path models/my_custom_model.h5
    ```

2.  **Trained Model:** After training, the model will be saved to the path specified via the command line argument (by default `models/transfer_model.h5`).

## Code Explanation

*   **`src/data_loader.py`:**  Contains the `load_and_preprocess_data` function to load and preprocess images using TensorFlow's `image_dataset_from_directory`, which automatically reads data from class directories.
*   **`src/model.py`:** Contains the functions to create the transfer learning model using a pre-trained architecture (VGG16, ResNet50, or MobileNetV2), fine-tune the model, and perform model evaluation using training and testing data. This also includes the save model function.
*   **`src/utils.py`:** Includes the `plot_training_history` function to plot the training and validation metrics.
*   **`src/train.py`:** The main script for training. It handles the setup, data loading, model creation, training, evaluation, and saving. It also uses `argparse` to manage command line arguments.
*   **`data_setup.py`:** This script downloads the CIFAR-10 data, formats it into the appropriate structure, and saves it into the `data` directory.

## Contributing

Feel free to contribute by submitting pull requests or opening issues.

### Example output

Found 9479 files belonging to 10 classes.
Found 10000 files belonging to 10 classes.
Epoch 1/10
2673/2673 ━━━━━━━━━━━━━━━━━━━━ 1528s 571ms/step - accuracy: 0.6314 - loss: 1.2280 - val_accuracy: 0.8074 - val_loss: 0.5564
Epoch 2/10
2673/2673 ━━━━━━━━━━━━━━━━━━━━ 1495s 559ms/step - accuracy: 0.7692 - loss: 0.6719 - val_accuracy: 0.8213 - val_loss: 0.5163
Epoch 3/10
2673/2673 ━━━━━━━━━━━━━━━━━━━━ 1483s 555ms/step - accuracy: 0.7876 - loss: 0.6146 - val_accuracy: 0.8278 - val_loss: 0.4930
Epoch 4/10
2673/2673 ━━━━━━━━━━━━━━━━━━━━ 1484s 555ms/step - accuracy: 0.8022 - loss: 0.5754 - val_accuracy: 0.8383 - val_loss: 0.4695
Epoch 5/10
2673/2673 ━━━━━━━━━━━━━━━━━━━━ 1487s 557ms/step - accuracy: 0.8073 - loss: 0.5547 - val_accuracy: 0.8420 - val_loss: 0.4562
Epoch 6/10
2673/2673 ━━━━━━━━━━━━━━━━━━━━ 1482s 554ms/step - accuracy: 0.8174 - loss: 0.5279 - val_accuracy: 0.8479 - val_loss: 0.4344
Epoch 7/10
2673/2673 ━━━━━━━━━━━━━━━━━━━━ 1457s 545ms/step - accuracy: 0.8259 - loss: 0.5044 - val_accuracy: 0.8513 - val_loss: 0.4352
Epoch 8/10
2673/2673 ━━━━━━━━━━━━━━━━━━━━ 1459s 546ms/step - accuracy: 0.8255 - loss: 0.5004 - val_accuracy: 0.8575 - val_loss: 0.4162
Epoch 9/10
2673/2673 ━━━━━━━━━━━━━━━━━━━━ 33128s 12s/step - accuracy: 0.8322 - loss: 0.4828 - val_accuracy: 0.8599 - val_loss: 0.4024
