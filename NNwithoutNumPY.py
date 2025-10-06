import math
import random
from Matrix import Matrix
from tensorflow.keras.datasets import mnist

"""NN attempt without NumPY"""


class SigmoidNeuron:
    def __init__(self, x, weights, bias, learning_rate=0.1):
        self.x = Matrix([x]) if isinstance(x[0], (int, float)) else Matrix(x)
        self.weights = Matrix(weights)
        self.bias = float(bias)
        self.learning_rate = learning_rate

    def output_function(self):
        # matrix product -> scalar
        linear_output = Matrix.multiplication(self.x, self.weights).data[0][0] + self.bias
        return 1 / (1 + math.exp(-linear_output))  # sigmoid

    def cost(self, y_true):
        y_prediction = self.output_function()
        return 0.5 * (y_true - y_prediction) ** 2

    def train_step(self, y_true):
        y_prediction = self.output_function()
        error = y_prediction - y_true
        sigmoid_derivative = y_prediction * (1 - y_prediction)
        d_cost_d_linear = error * sigmoid_derivative

        d_weights = self.x * d_cost_d_linear
        d_bias = d_cost_d_linear

        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias

        return y_prediction, self.cost(y_true)


class Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Input layer → Hidden layer
        self.hidden_layer = [
            SigmoidNeuron(
                x=[0] * self.input_size,  # input layer
                # random small weights instead of zeros
                weights=[[random.uniform(-0.5, 0.5)] for _ in range(self.input_size)],
                bias=random.uniform(-0.5, 0.5)
            )
            for _ in range(self.hidden_size)
        ]

        # Hidden layer → Output layer
        self.output_layer = [
            SigmoidNeuron(
                x=[0] * self.hidden_size,  # hidden layer
                # random small weights instead of zeros
                weights=[[random.uniform(-0.5, 0.5)] for _ in range(self.hidden_size)],
                bias=random.uniform(-0.5, 0.5)
            )
            for _ in range(self.output_size)
        ]

    def forward(self, input_vector):
        """ Forward pass: input → hidden → output"""

        # Feed inputs into hidden layer neurons
        hidden_outputs = []
        for neuron in self.hidden_layer:
            neuron.x = Matrix([input_vector])  # update input
            hidden_outputs.append(neuron.output_function())

        #  Feed hidden outputs into output layer neurons
        final_outputs = []
        for neuron in self.output_layer:
            neuron.x = Matrix([hidden_outputs])  # update input
            final_outputs.append(neuron.output_function())

        return final_outputs


# Load the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Flatten images 28x28 → 784 and normalize to [0,1]
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

NN = Network(784, 15, 10)

# Create the network

# Quick evaluation on test set (random weights → ~10% accuracy)
correct = 0
for i in range(1000):  # test of first 1000 samples
    output = NN.forward(list(X_test[i]))
    prediction = output.index(max(output))  # argmax of outputs
    if prediction == Y_test[i]:
        correct += 1

accuracy = (correct / 1000) * 100
print("Accuracy with random weights (≈10% expected):", accuracy , " %")
