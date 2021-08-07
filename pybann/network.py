"""
network.py

A module to define the artificial neural network.
"""

# Import modules
import numpy as np
from .activation import sigmoid
from .backpropagation import backpropagate
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
        # Vector with size (n, 1) needed for dot product with numpy
        self.biases = [ np.random.randn(n, 1) for n in size[1:]]
        self.weights = [ np.random.randn(m, n) for n, m in zip(size[:-1], size[1:])]

    def load(self):
        """
        To load ANN weights and biases (of a previously trained ANN) from files
        """
        pass

    def feedforward(self, nodeValues):
        """
        nodeValues are the input values for the network
        """
        nodeValues = np.atleast_2d(nodeValues).transpose()
        for weights, biases in zip(self.weights, self.biases):
            nodeValues = sigmoid(np.dot(weights, nodeValues) + biases)

        # Return the result as a 1D vector
        return nodeValues.transpose().flatten()

    def train(self, dataset, niter=250, alpha=0.02, decay=0.0):
        """
        Train
        :param dataset: a list of tuples in the form (inValues, outValues)
        :param niter: number of iterations
        :param alpha: step
        """
        
        diffW = [np.zeros(w.shape) for w in self.weights]
        diffB = [np.zeros(b.shape) for b in self.biases]

        for iter in range(niter):
        #   For each row of data pass input and attempted output to backpropagate
            nablaW = [np.zeros(w.shape) for w in self.weights]
            nablaB = [np.zeros(b.shape) for b in self.biases]
            for i in range(len(dataset)):
                dnablaW, dnablaB = backpropagate(self.nLayers, self.weights, self.biases, dataset[i][0], dataset[i][1])
                # Updating B and W 
                nablaB = [nb+dnb for nb, dnb in zip(nablaB, dnablaB)]
                nablaW = [nw+dnw for nw, dnw in zip(nablaW, dnablaW)]
            # Updating self.weights and self.biases
            #self.weights = [w-alpha*nw for w, nw in zip(self.weights, nablaW)]
            #self.biases = [b-alpha*nb for b, nb in zip(self.biases, nablaB)]

            diffW = [decay*snw-alpha*nw for snw, nw in zip(diffW, nablaW)]
            diffB = [decay*snb-alpha*nb for snb, nb in zip(diffB, nablaB)]

            self.weights = [w+dw for w, dw in zip(self.weights, diffW)]
            self.biases = [b+db for b, db in zip(self.biases, diffB)]


    def test(self, dataset):
        pass