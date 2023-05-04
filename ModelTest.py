import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

model = tf.keras.models.load_model("digit_classification_model.h5")

y_pred = [np.argmax(val) for val in model.predict(X_test)]

from sklearn.metrics import classification_report

print("Classification Report: \n", classification_report(y_test, y_pred))


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
import seaborn as sns
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

