import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from autils import *

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# load data
X, y = load_data_code()

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Load the model
model = tf.keras.models.load_model("my_trained_model.h5")
y_pred = model.predict(X_test)

# Plot predictions vs actual values
index = 5
X_random_reshaped = X[index].reshape((20, 20)).T
print(f"y_pred: {y_pred[index]} vs actual: {y[index]}")

# Display the image
plt.imshow(X_random_reshaped, cmap="gray")
plt.show()
