 import numpy as np

class SigmoidNeuron:
    def __init__(self, x, weights, bias, learning_rate=0.1):
        self.x = np.array(x)
        self.weights = np.array(weights, dtype=float)
        self.bias = float(bias)
        self.learning_rate = learning_rate

    def output_function(self):
        linear_output = np.dot(self.x, self.weights) + self.bias
        return 1 / (1 + np.exp(-linear_output))  # sigmoid activation

    def cost(self, y_true):
        """Mean Squared Error"""
        y_prediction = self.output_function()
        return 0.5 * (y_true - y_prediction) ** 2

    def train_step(self, y_true):
        """
        One step of training using gradient descent.
        """
        # Forward pass
        y_prediction = self.output_function()

        # Compute error derivative w.r.t prediction (MSE derivative)
        error = y_prediction - y_true

        # Derivative of sigmoid w.r.t linear output
        sigmoid_derivative = y_prediction * (1 - y_prediction)

        # Gradient of cost w.r.t linear output
        d_cost_d_linear = error * sigmoid_derivative

        # Gradients w.r.t weights and bias
        d_weights = self.x * d_cost_d_linear
        d_bias = d_cost_d_linear

        # Update weights and bias
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias

        return y_prediction, self.cost(y_true)
