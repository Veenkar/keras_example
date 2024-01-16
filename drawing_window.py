# Import necessary modules from Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import tkinter as tk
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt

def perform_detection(image):
    # Normalize the image
    normalized_image = (image / 255) - 0.5

    # Reshape the image
    reshaped_image = normalized_image.reshape((-1, 784))

    # Load the trained model
    model = load_model('model.h5')

    # Make prediction on the image
    prediction = model.predict(reshaped_image)

    # Convert prediction to label
    predicted_label = np.argmax(prediction)

    # Update the label with the detected digit
    detected_label.config(text=f"Detected Digit: {predicted_label}")
    return predicted_label

def draw(event):
    x = event.x
    y = event.y
    canvas.create_rectangle(x, y, x + 10, y + 10, fill='black')

def save_drawing():
    canvas.postscript(file="drawing.eps", colormode='color')
    img = Image.open("drawing.eps")
    img = img.convert("L")
    img = img.resize((28, 28))

    # Invert the image
    img = ImageOps.invert(img)
    img.save("drawing.bmp", "BMP")

    np_img = np.array([np.array(img)])
    print(np_img)
    # plt.imshow(np_img.reshape(28, 28), cmap='gray')
    # plt.axis('off')
    # plt.show()

    detected = perform_detection(np_img)
    print(f"detect digit: {detected}")

def clear_canvas():
    canvas.delete("all")

root = tk.Tk()
root.title("Drawing Window")

canvas = tk.Canvas(root, width=280, height=280, bg='white')
canvas.pack()

canvas.bind("<B1-Motion>", draw)

text = tk.Label(root, text="Hello, World!")
text.pack()

additional_text = tk.Label(root, text="Additional Text")
additional_text.pack()

save_button = tk.Button(root, text="Save Drawing", command=save_drawing)
save_button.pack()

clear_button = tk.Button(root, text="Clear Canvas", command=clear_canvas)
clear_button.pack()

detected_label = tk.Label(root, text="Detected Digit: ")
detected_label.pack()

root.mainloop()