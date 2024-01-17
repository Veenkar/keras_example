# Import necessary modules from Keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
import numpy as np
import imageio
import matplotlib.pyplot as plt
# Load the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images_initial = train_images

# Convert train_images to binary images
# train_images = np.where(train_images > 16.0, 255.0, 0)
# Normalize the binary images.
train_images = (train_images / 255.0) - 0.5
# Flatten the binary images.
train_images = train_images.reshape((-1, 784))

# Convert train_images to binary images
# test_images = np.where(test_images > 16.0, 255.0, 0)
# Normalize the binary images.
test_images = (test_images / 255.0) - 0.5
# Flatten the binary images.
test_images = test_images.reshape((-1, 784))

# Build the model.
# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# Compile the model.
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=20,
  batch_size=128,
)

# Evaluate the model.
model.evaluate(
  test_images,
  to_categorical(test_labels)
)

# Save the model to disk.
model.save('model.h5')

# Load the model from disk later using:
# model.load_weights('model.h5')

np_img = train_images[0]

# image = np_img.reshape(28, 28)
# imageio.imwrite('image.png', image)


# plt.imshow(np_img.reshape(28, 28), cmap='gray')
# plt.axis('off')
# plt.show()


