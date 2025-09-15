import numpy as np


# need to add weights then learning, then layers

""" 
 simple sigmoid neuron 
"""


class SigmoidNeuron:

    def __init__(self, x, weights, bias):
        # Initialization code
        self.x = x
        self.weights = weights
        self.bias = bias


    def outputfunction(self):
        linear_output = np.dot(x, weights) + bias
        return 1 / (1 + np.exp(-linear_output))
        #return 1 if linear_output >= 0 else 0




# Test Code
x = np.array([0.5, 0.04])       # input
weights = np.array([-2, -2])
bias = 3

# make a 8x8 grid of neurons
Layer = [[SigmoidNeuron(x, weights, bias) for i in range(8)] for j in range(8)]

# loop through rows and columns
for i in range(8):
    print("\n")
    for j in range(8):
        output = Layer[i][j].outputfunction()
        print(round(output, 2) , end=' ')



#Sigmoid = SigmoidNeuron(x, weights, bias)
#output = Sigmoid.outputfunction()
#print("Output:", output)
