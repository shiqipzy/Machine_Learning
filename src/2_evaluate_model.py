
import tensorflow as tf
import os

# Define paths
eval_dir = 'evaluation'

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Load evaluation dataset
eval_ds = tf.keras.utils.image_dataset_from_directory(
    eval_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

# Load the trained model
model = tf.keras.models.load_model('food_classifier_model.keras')

print("Model loaded successfully. Evaluating on the evaluation dataset...")

# Evaluate the model
loss, accuracy = model.evaluate(eval_ds)

print(f'Evaluation accuracy: {accuracy:.4f}')
print(f'Evaluation loss: {loss:.4f}')
