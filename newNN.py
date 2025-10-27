import math
import random
import time
import numpy as np
from tensorflow.keras.datasets import mnist

"""NN with numpy, sgd and full dataset"""

def output_function(self):
 linear_output = (self.x @ self.weights + self.bias).item()
 return 1 / (1 + math.exp(-linear_output))  # sigmoid

def cost(self, y_true):
 y_prediction = self.output_function()
 return 0.5 * (y_true - y_prediction) ** 2

def train_step(self, y_true):
 y_prediction = self.output_function()
 error = y_prediction - y_true
 sigmoid_derivative = y_prediction * (1 - y_prediction)
 d_cost_d_linear = error * sigmoid_derivative
 d_weights = np.array([[self.x[0, i] * d_cost_d_linear] for i in range(self.x.shape[1])])
 d_bias = d_cost_d_linear
 self.weights -= self.learning_rate * d_weights
 self.bias -= self.learning_rate * d_bias
 return y_prediction, self.cost(y_true)


# Network
class Network:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05):
        self.learning_rate = learning_rate
        # Initialize weights and biases
        self.W1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.b1 = np.random.uniform(-0.5, 0.5, (hidden_size, 1))
        self.W2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
        self.b2 = np.random.uniform(-0.5, 0.5, (output_size, 1))

def forward(self, X):


def backward(self, X, Y, Z1, A1, Z2, A2):


def train_step(self, input_vector, y_true_vector):


def predict(self, input_vector):



#  Load MNIST
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Combine into list of tuples for shuffling
training_data = list(zip(X_train, Y_train))

#  Training
NN = Network(784, 64, 10)
EPOCHS = 10
BATCH_SIZE = 32  # typical small batch for SGD

for epoch in range(EPOCHS):
    start_time = time.perf_counter()

    random.shuffle(training_data)  # key step for SGD randomness

    # Loop through mini-batches
    for batch_start in range(0, len(training_data), BATCH_SIZE):
        batch = training_data[batch_start:batch_start + BATCH_SIZE]
        for x, y_label in batch:
            y_true = [0] * 10
            y_true[y_label] = 1
            NN.train_step(x, y_true)

    # Evaluate after each epoch
    correct = 0
    for i in range(len(X_test)):  # test on all 10,000
        prediction = NN.predict(list(X_test[i]))
        if prediction == Y_test[i]:
            correct += 1

    accuracy = (correct / len(X_test)) * 100
    elapsed = time.perf_counter() - start_time
    print(f"Epoch {epoch + 1}/{EPOCHS} — Test Accuracy: {accuracy:.2f}% — Time: {elapsed:.2f}s")
