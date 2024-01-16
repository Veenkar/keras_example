# Import necessary modules from Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np

# Load the data
(_, _), (test_images, test_labels) = mnist.load_data()

# Normalize the images.
test_images = (test_images / 255) - 0.5

# Flatten the images.
test_images = test_images.reshape((-1, 784))

# Load the trained model
model = load_model('model.h5')

# Make predictions on the test data
predictions = model.predict(test_images)

# Convert predictions to labels
predicted_labels = np.argmax(predictions, axis=1)

# Print the predicted labels
print(predicted_labels)