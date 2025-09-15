import numpy as np


# need to add weights then learning, then layers

"""
 Simple perceptron
 x: input vector
 weights: weight vector
 bias: bias term
 dot product is wrapped in a sigmoid function to give range of output values
"""


class SigmoidNeuron:

    def __init__(self, x, weights, bias):
        # Initialization code
        self.x = x
        self.weights = weights
        self.bias = bias


    def OutputFunction(self):
        linear_output = np.dot(x, weights) + bias
        return 1 / (1 + np.exp(-linear_output))
        #return 1 if linear_output >= 0 else 0




# Test Code
x = np.array([1, 0])       # input
weights = np.array([-2, -2])
bias = 3

Sigmoid = SigmoidNeuron(x, weights, bias)
output = Sigmoid.OutputFunction()
print("Output:", output)
