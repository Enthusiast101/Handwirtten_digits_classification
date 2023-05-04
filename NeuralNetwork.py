import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
#import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Visualization
#plt.gray()
#plt.imshow(X_train[0])
#plt.show()

# Size
print(len(X_train))
print(len(X_test))

# Layers / Neurons
print(tf.test.is_gpu_available(cuda_only=True))

X_train = X_train / 255
X_test = X_test / 255
print(tf.__version__)


def train_model():
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1, input_shape=(28, 28, 1)),
        layers.RandomZoom(0.1),
    ])

    model = keras.Sequential([
        data_augmentation,
        # CNN
        layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation="relu"),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation="relu"),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation="relu"),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation="relu"),
        layers.MaxPool2D((2, 2)),
        # Network
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(.2),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dropout(.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="RMSprop",
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=25, batch_size=1000)

    model.save("digit_classification_model.h5")


if tf.test.is_gpu_available(cuda_only=True):
    with tf.device("/GPU:0"):
        train_model()
else:
    train_model()
