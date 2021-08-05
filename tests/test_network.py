import unittest
import numpy as np
from pybann import Network

class test_activation_parameters(unittest.TestCase):

    def test_initialization(self):

        # Define the size of the network
        size = (3, 5, 4)

        # Initalize
        network = Network(size)

        # Check
        self.assertTupleEqual(network.size, size) 
        self.assertEqual(network.nLayers, 3)
        
    def test_initialization_biases(self):

        # Define the size of the network
        size = (3, 5, 4)

        # Initalize
        network = Network(size)

        # Check weights
        self.assertEqual(len(network.biases), 2)
        self.assertEqual(len(network.biases[0]), 5)
        self.assertEqual(len(network.biases[1]), 4)

    def test_initialization_weights(self):

        # Define the size of the network
        size = (3, 5, 4)

        # Initalize
        network = Network(size)

        # Check biases
        self.assertEqual(len(network.weights), 2)
        self.assertTupleEqual(np.shape(network.weights[0]), (5, 3))
        self.assertTupleEqual(np.shape(network.weights[1]), (4, 5))
    
    def test_feedforward(self):

        # Define the size of the network
        size = (3, 5, 4)

        # Initalize
        network = Network(size)

        inValues = [3., 1., -1.]

        outValues = network.feedforward(inValues)
        
        self.assertEqual(len(outValues), 4)