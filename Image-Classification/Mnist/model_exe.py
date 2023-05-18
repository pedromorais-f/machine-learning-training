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

# Normalizing the pixels
x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test / 255.0

y_test = np.array(y_test)
print("Concluded")

print("Loading the model...")
with open("../Mnist/Model/model.json") as f:
    load_model = f.read()
    f.close()
model = model_from_json(load_model)
model.load_weights("../Mnist/Model/model.h5")
print("Model loaded")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


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
