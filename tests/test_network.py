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
        
    def test_initialization_weights(self):

        # Define the size of the network
        size = (3, 5, 4)

        # Initalize
        network = Network(size)

        # Check weights
        self.assertEqual(len(network.weights), 2)
        self.assertEqual(len(network.weights[0]), 5)
        self.assertEqual(len(network.weights[1]), 4)

    def test_initialization_biaises(self):

        # Define the size of the network
        size = (3, 5, 4)

        # Initalize
        network = Network(size)

        # Check biaises
        self.assertEqual(len(network.biaises), 2)
        self.assertTupleEqual(np.shape(network.biaises[0]), (5, 3))
        self.assertTupleEqual(np.shape(network.biaises[1]), (4, 5))
        