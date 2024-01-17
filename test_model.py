# Import necessary modules from Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the data
(_, _), (test_images, test_labels) = mnist.load_data()

# Normalize the images.
test_images = (test_images / 255) - 0.5

# Flatten the images.
test_images = test_images.reshape((-1, 784))


# Display the first image
# plt.imshow(test_images[2].reshape(28, 28), cmap='gray')
# plt.axis('off')
# plt.show()

# Load the trained model
model = load_model('model.h5')

# Make predictions on the test data
predictions = model.predict(test_images)

# Convert predictions to labels
predicted_labels = np.argmax(predictions, axis=1)

# Print the predicted labels
print("predicted:")
print(predicted_labels)
print("actual:")
print(test_labels)

print(type(test_images))
# print(test_images.shape)

# Calculate accuracy
accuracy = np.mean(predicted_labels == test_labels) * 100

# Print the accuracy
print("Accuracy:", accuracy)

