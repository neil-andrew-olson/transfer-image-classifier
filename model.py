import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16 # Or other base models

def create_transfer_model(base_model_name, num_classes, image_size):
  """Creates a model with transfer learning."""

  if base_model_name == 'VGG16':
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
  elif base_model_name == 'ResNet50':
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
  elif base_model_name == 'MobileNetV2':
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))
  else:
    raise ValueError("Unsupported base model name.")

  # Freeze the base model's layers
  base_model.trainable = False

  # Add custom classifier layers
  model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
    ])

  return model

def train_model(model, train_data, val_data, num_epochs, lr=0.001):
  """Trains the model and logs loss and metrics"""

  optimizer = optimizers.Adam(learning_rate=lr)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  history = model.fit(train_data,
                      epochs=num_epochs,
                      validation_data=val_data,
                      )
  return history

def evaluate_model(model, test_data):
  """Evaluates the trained model."""

  loss, accuracy = model.evaluate(test_data)
  print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

def save_model(model, save_path):
  """Saves the entire model (architecture, weights, and optimizer state)."""

  model.save(save_path)
  print(f"Model saved to {save_path}")