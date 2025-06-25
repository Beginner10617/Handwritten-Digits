import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from mnistReader import x_test, y_test, x_train, y_train
import os
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

def train_model1():
    model = models.Sequential()

    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    model.save("model1.h5")

def train_model2():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    model.save("model2.h5")

if __name__ == "__main__":
    if os.path.exists("model1.h5"):
        print("Model 1 already exists, skipping training.")
    else:
        print("Training Model 1...")
        train_model1()
    if os.path.exists("model2.h5"):
        print("Model 2 already exists, skipping training.")
    else:
        print("Training Model 2...")
        train_model2()
    print("Training complete.")
    
    model = models.load_model("model1.h5")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Model 1 - Loss: {loss}, Accuracy: {accuracy}")

    model = models.load_model("model2.h5")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Model 2 - Loss: {loss}, Accuracy: {accuracy}")