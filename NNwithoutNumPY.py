import math
from Matrix import Matrix

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
                x=[0] * self.input_size,  #  input layer
                weights=[[0] for _ in range(self.input_size)],  # weight column for each input
                bias=0.0
            )
            for _ in range(self.hidden_size)
        ]

        # Hidden layer → Output layer
        self.output_layer = [
            SigmoidNeuron(
                x=[0] * self.hidden_size,  #  hidden layer
                weights=[[0] for _ in range(self.hidden_size)],  # weight column for each hidden node
                bias=0.0
            )
            for _ in range(self.output_size)
        ]


NN = Network(784, 15, 10)
