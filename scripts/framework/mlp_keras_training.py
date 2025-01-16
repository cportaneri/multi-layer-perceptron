import numpy as np
import time
import keras
from keras import layers, regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Parameters
num_classes = 10
input_shape = (28, 28, 1)
learning_rate = 0.0002
batch_size = 32
epochs = 30
l2_lambda = 0.0001

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the model with L2 regularization
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(45, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(0.3),
        layers.Dense(35, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(23, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# Compile the model with custom learning rate
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Measure training time
start_training = time.time()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
end_training = time.time()

training_time = end_training - start_training
print(f"Training time: {training_time:.2f} seconds")

# Evaluate the model
start_testing = time.time()
model.evaluate(x_test, y_test, verbose=0)
end_testing = time.time()

testing_time = end_testing - start_testing
print(f"Testing time: {testing_time:.2f} seconds")

# Predict on test set for additional metrics
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate precision, recall, and F1-score
accuracy = accuracy_score(y_true_classes, y_pred_classes) * 100
precision = precision_score(y_true_classes, y_pred_classes, average='weighted') * 100
recall = recall_score(y_true_classes, y_pred_classes, average='weighted') * 100
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted') * 100

# Output results
print("Results:")
print(f" Accuracy: {accuracy:.2f}%")
print(f" Precision: {precision:.2f}%")
print(f" Recall: {recall:.2f}%")
print(f" F1-Score: {f1:.2f}%")

# Output model configuration
print("Neural Network Configuration:")
print(f" Input dimension: {input_shape[1] * input_shape[2]}")
print(f" Depth: 4")
print(f"  Layer 0 dimension: 45, activation : relu")
print(f"  Layer 1 dimension: 35, activation : relu")
print(f"  Layer 2 dimension: 23, activation : relu")
print(f"  Layer 3 dimension: {num_classes}, activation : softmax")
print(f" Learning Rate: {learning_rate}")
print(f" Max Epochs: {epochs}")
print(f" Batch Size: {batch_size}")
print(f" Optimizer: Adam")
print(f" Error Loss: cross entropy")
print(f" L2 Regularization Lambda: {l2_lambda}")
