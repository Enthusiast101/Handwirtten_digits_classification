import json
import tensorflow as tf
from tensorflow import keras
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
def build_model(hp):
    model = Sequential()
    for _ in range(hp.Int("conv_layers", 1, 3, step=1)):
        model.add(Conv2D(filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
                         kernel_size=hp.Choice('conv1_kernel', values=[3, 5]),
                         activation='relu',
                         padding='same',
                         input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    for _ in range(hp.Int("deep_layers", 1, 5, step=1)):
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(units=hp.Int('dense_units', min_value=128, max_value=512, step=64), activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='keras_tuner_dir',
                     project_name='mnist_cnn_tuning')

datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)

# Perform the hyperparameter search with data augmentation
tuner.search(datagen.flow(X_train, y_train, batch_size=64), epochs=10, validation_data=(X_test, y_test))

best_hps = tuner.get_best_hyperparameters()[0].values
with open(r"tmp/tuned_vals.json", "w") as file:
    json.dump(best_hps, file)

