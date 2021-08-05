"""
network.py

A module to define the artificial neural network.
"""

# Import modules
import numpy as np
from .activation import sigmoid

class Network:

    def __init__(self, size):
        """
        :param size: tuple of integers containing the number of nodes in each layer.
            For example (3, 5, 3) means there are 3 nodes on the first layer (input layer)
            , 5 in the second (hidden layer) and 3 in the third (output layer).
        """
        # Get ANN parameters
        self.size = size
        self.nLayers = len(size)

        # Initialize weights and biases
        self.biases = [ np.random.standard_normal(n) for n in size[1:]]
        self.weights = [ np.random.standard_normal(m) for m in zip(size[1:], size[:-1])]

    def feedforward(self, nodeValues):
        """
        nodeValues are the input values for the network
        """
        for weights, biases in zip(self.weights, self.biases):
            nodeValues = sigmoid(np.dot(weights, nodeValues) + biases)

        return nodeValues

    def training(self, dataset, niter=20, alpha=0.15):
        """
        Training
        :param dataset: a list of tuples in the form (inValues, outValues)
        :param niter: number of iterations
        :param alpha: step
        """
        for iter in niter:
        #   For each row of data pass input and attempted output to backpropagate
            nablaW = [np.zeros(w.shape) for w in self.weights]
            nablaB = [np.zeros(b.shape) for b in self.biases]
            for i in range(len(data)):
                dnablaW, dnablaB = self.backpropagate(self.nLayers, self.weights, self.biases, inValue[i], outValue[i])
                # Updating B and W 
                nablaB = [nb+dnb for nb, dnb in zip(nablaB, dnablaB)]
                nablaW = [nw+dnw for nw, dnw in zip(nablaW, dnablaW)]
            # Updating self.weights and self.biases
            self.weights = [w-alpha*nw for w, nw in zip(self.weights, nablaW)]
            self.biases = [b-alpha*nb for b, nb in zip(self.biases, nablaB)]