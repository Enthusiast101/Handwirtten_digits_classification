import json
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


# Define the model-building function for Keras Tuner
def build_model(data):
    model = Sequential()
    for _ in range(data["conv_layers"]):
        model.add(Conv2D(filters=data["conv1_filters"],
                         kernel_size=data["conv1_kernel"],
                         activation='relu',
                         padding='same',
                         input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    for _ in range(data["deep_layers"]):
        model.add(Dropout(rate=data['dropout']))
        model.add(Dense(units=data['dense_units'], activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=data['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


with open(r"tmp/tuned_vals.json", "r") as file:
    data = json.load(file)

model = build_model(data)
model.fit(X_train, y_train, epochs=5)
model.save("digit_classification_model.h5")

