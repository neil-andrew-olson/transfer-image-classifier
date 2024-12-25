import argparse
import os
from . import data_loader, model, utils

# Define command-line arguments
parser = argparse.ArgumentParser(description='Train an image classifier with transfer learning.')
parser.add_argument('--data_dir', type=str, default='data', help='Path to the dataset directory.')
parser.add_argument('--base_model', type=str, default='VGG16', help='Name of the base model (e.g., VGG16, ResNet50, MobileNetV2).')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
parser.add_argument('--image_size_x', type=int, default=224, help='Width of image')
parser.add_argument('--image_size_y', type=int, default=224, help='Height of image')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--model_save_path', type=str, default='models/transfer_model.h5', help='Path to save the trained model.')

args = parser.parse_args()

if not os.path.exists(args.model_save_path):
  # Create the directory if not exists
  os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

# Load and preprocess the data
image_size = (args.image_size_x, args.image_size_y)
train_data, val_data, test_data = data_loader.load_and_preprocess_data(args.data_dir, args.batch_size, image_size)

num_classes = len(train_data.class_names)

# Create and train the model
transfer_model = model.create_transfer_model(args.base_model, num_classes, image_size)
history = model.train_model(transfer_model, train_data, val_data, args.num_epochs, args.learning_rate)

# Evaluate the model
model.evaluate_model(transfer_model, test_data)

# Plot the training history
utils.plot_training_history(history)

# Save the trained model
model.save_model(transfer_model, args.model_save_path)