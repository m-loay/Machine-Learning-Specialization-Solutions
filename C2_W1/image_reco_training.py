import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from autils import load_data_code

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# load data
X, y = load_data_code()
length = X.shape[1]
print(f"length: {length}")

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# build model
model = Sequential(
    [
        tf.keras.Input(shape=(length,)),
        Dense(units=25, activation="relu", name="layer1"),
        Dense(units=15, activation="relu", name="layer2"),
        Dense(units=1, activation="sigmoid", name="layer3"),
    ],
    name="my_model",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

# train model
model.fit(X_train, y_train, epochs=10)

# evaluate model
# evaluate model on training data
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print(f"\n \nTraining Loss: {train_loss}, Training Accuracy: {train_accuracy}")

# evaluate model on validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"\n \nValidation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# evaluate model on testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n \nTesting Loss: {test_loss}, Testing Accuracy: {test_accuracy}")

# Save the model
model.save("my_trained_model.h5")
