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

        # Initialize weights and biaises
        self.biaises = [ np.random.standard_normal(n) for n in size[1:]]
        self.weights = [ np.random.standard_normal(m) for m in zip(size[1:], size[:-1])]

    def feedforward(self, nodeValues):
        """
        nodeValues are the input values for the network
        """
        for weights, biaises in zip(self.weights, self.biaises):
            nodeValues = sigmoid(np.dot(weights, nodeValues) + biaises)

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
            nablaB = [np.zeros(b.shape) for b in self.biaises]
            for i in range(len(data)):
                dnablaW, dnablaB = self.backpropagate(self, inValue[i], outValue[i])
                # Updating B and W 
                nablaB = [nb+dnb for nb, dnb in zip(nablaB, dnablaB)]
                nablaW = [nw+dnw for nw, dnw in zip(nablaW, dnablaW)]
            # Updating self.weights and self.biaises
            self.weights = [w-alpha*nw for w, nw in zip(self.weights, nablaW)]
            self.biaises = [b-alpha*nb for b, nb in zip(self.biaises, nablaB)]

    def backpropagate(self, inValues, outValues):
        """
        :param invalues: input values
        :param outValues: attempted output values
        :return : updated weights and biaises
        """
        # Initialize
        nablaW = [np.zeros(w.shape) for w in self.weights]
        nablaB = [np.zeros(b.shape) for b in self.biaises]

        # Feedforward
        activation = inValues
        activations = [inValues]
        wsvectors = [] # wsvectors are the weighted sums of inputs calculated for each layer
        for w, b in zip(self.weights, self.biaises):
            wsvector = np.dot(w, activation) + b
            wsvectors.append(wsvector)
            activation = sigmoid(wsvector)
            activations.append(activation)
        
        # Backward
        delta = (activations[-1] - outValues) * sigmoid(wsvectors[-1], deriv=True)
        nablaW[-1] = np.dot(delta, activations[-2].transpose())
        nablaB[-1] = delta
        for iLayer in range(2, self.nLayers):
            wsvector = wsvectors[-iLayer]
            delta = (self.weights[-iLayer+1].transpose(), delta) * sigmoid(wsvector, deriv=True)
            nablaW[-iLayer] = np.dot(delta, activations[-iLayer-1].transpose())
            nablaB[-iLayer] = delta

        return nablaW, nablaB