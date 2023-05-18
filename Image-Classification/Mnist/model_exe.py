import tensorflow as tf
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt


def show_image(chosen_image):
    plt.figure()
    plt.imshow(chosen_image[0])
    plt.show()


print("Getting datasets...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Concluded\n")

print("Loading the model...")
with open("../Mnist/Model/model.json") as f:
    load_model = f.read()
    f.close()
model = model_from_json(load_model)
model.load_weights("../Mnist/Model/model.h5")
print("Model loaded\n")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("ALL classes...")
print("0, 1, 2, 3, 4, 5, 6, 7, 8, 9\n")


while True:
    index = input("Put a value(1 - 10000) or exit:")

    if index == "exit":
        break

    index = int(index)
    index -= 1
    image = x_test[index].reshape((1, 28, 28))
    show_image(image)

    print(f'Predict: {np.argmax(model.predict(image))}')
    print(f'Result: {int(y_test[index])}\n')
