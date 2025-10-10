import math
import random
import time
from Matrix import Matrix
from tensorflow.keras.datasets import mnist

"""NN attempt without NumPY - MNIST dataset"""

# Creates an Object-Oriented Neuron class
class SigmoidNeuron:
    def __init__(self, x, weights, bias, learning_rate=0.1):
        self.x = Matrix([x]) if isinstance(x[0], (int, float)) else Matrix(x)
        self.weights = weights if isinstance(weights, Matrix) else Matrix(weights)
        self.bias = float(bias)
        self.learning_rate = learning_rate

    def output_function(self):
        linear_output = (self.x * self.weights).data[0][0] + self.bias
        return 1 / (1 + math.exp(-linear_output))  # sigmoid

    def cost(self, y_true):
        y_prediction = self.output_function()
        return 0.5 * (y_true - y_prediction) ** 2

    def train_step(self, y_true):
        y_prediction = self.output_function()
        error = y_prediction - y_true
        sigmoid_derivative = y_prediction * (1 - y_prediction)
        d_cost_d_linear = error * sigmoid_derivative

        # gradient shape must match weights
        d_weights = Matrix([[self.x.data[0][i] * d_cost_d_linear] for i in range(self.x.cols)])
        d_bias = d_cost_d_linear

        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias

        return y_prediction, self.cost(y_true)


class Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_layer = [
            SigmoidNeuron(
                x=[0] * self.input_size,
                weights=Matrix([[random.uniform(-0.5, 0.5)] for _ in range(self.input_size)]),
                bias=random.uniform(-0.5, 0.5)
            )
            for _ in range(self.hidden_size)
        ]

        self.output_layer = [
            SigmoidNeuron(
                x=[0] * self.hidden_size,
                weights=Matrix([[random.uniform(-0.5, 0.5)] for _ in range(self.hidden_size)]),
                bias=random.uniform(-0.5, 0.5)
            )
            for _ in range(self.output_size)
        ]

    def forward(self, input_vector):
        hidden_outputs = []
        for neuron in self.hidden_layer:
            neuron.x = Matrix([input_vector])
            hidden_outputs.append(neuron.output_function())

        final_outputs = []
        for neuron in self.output_layer:
            neuron.x = Matrix([hidden_outputs])
            final_outputs.append(neuron.output_function())

        return hidden_outputs, final_outputs

    def train_step(self, input_vector, y_true_vector):
        hidden_outputs, final_outputs = self.forward(input_vector)

        # Output layer
        output_errors = []
        for i, neuron in enumerate(self.output_layer):
            y_prediction, cost = neuron.train_step(y_true_vector[i])
            output_errors.append(y_prediction - y_true_vector[i])

        # Hidden layer
        for j, hidden_neuron in enumerate(self.hidden_layer):
            error_from_outputs = sum(
                output_errors[k] * self.output_layer[k].weights.data[j][0]
                for k in range(self.output_size)
            )
            hidden_neuron.train_step(error_from_outputs)

        return final_outputs

    def predict(self, input_vector):
        _, outputs = self.forward(input_vector)
        return outputs.index(max(outputs))


# MNIST
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# creating the network
NN = Network(784, 15, 10)

# training loop
EPOCHS = 10
SAMPLES = 5000

# stopwatch to test efficiency
start_time = time.perf_counter()

for epoch in range(EPOCHS):
    for i in range(SAMPLES):
        x = list(X_train[i])
        y_true = [0] * 10
        y_true[Y_train[i]] = 1
        NN.train_step(x, y_true)

# testing accuracy
    correct = 0
    for i in range(500):
        prediction = NN.predict(list(X_test[i]))
        if prediction == Y_test[i]:
            correct += 1
    accuracy = (correct / 500) * 100
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Epoch {epoch + 1}/{EPOCHS} — Test Accuracy: {accuracy:.2f}%" + " Elapsed time: " + str(round(duration,3)) + " seconds")
