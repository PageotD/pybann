"""
network.py

A module to define the artificial neural network.
"""

# Import modules
import numpy as np

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
        self.weights = [ np.random.standard_normal(n) for n in size[1:]]
        self.biaises = [ np.random.standard_normal((n, m)) for n, m in zip(size[1:], size[:-1])]