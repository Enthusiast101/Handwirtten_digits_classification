import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

model = tf.keras.models.load_model("digit_classification_model.h5", compile=False)

y_pred = [np.argmax(val) for val in model.predict(X_test)]

from sklearn.metrics import classification_report

print("Classification Report: \n", classification_report(y_test, y_pred))


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

