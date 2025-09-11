import numpy as np


# Currently using step function, will change to sigmoid

class Perceptron:

    @staticmethod
    def dotproduct(x, weights, bias):
        """
        Simple perceptron
        x: input vector
        weights: weight vector
        bias: bias term
        """
        linear_output = np.dot(x, weights) + bias
        return 1 if linear_output >= 0 else 0




# Test Code
x = np.array([1, 0])       # input
weights = np.array([-2, -2])
bias = 3

Perceptron = Perceptron()
output = Perceptron.DotProduct(x, weights, bias)
print("Output:", output)
