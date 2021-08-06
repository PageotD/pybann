import unittest
import numpy as np
from pybann import Network
from pybann.backpropagation import backpropagate


class test_activation(unittest.TestCase):

    def test_backpropagate_null(self):

        # Initialize network
        size = (3, 5, 4)
        network = Network(size)

        # Initialize input values
        inValues = (1., 1., 1.)

        # Calculate output values
        outValues = network.feedforward(inValues)

        # Test that backpropagation return zero values
        W, B = backpropagate(network.nLayers, network.weights, network.biases, inValues, outValues)

        self.assertEqual(np.shape(W[0]), (5, 3))
        self.assertEqual(np.shape(W[1]), (4, 5))
        for i in range(5):
            self.assertListEqual(list(W[0][i]), [0., 0., 0.])
        for i in range(4):
            self.assertEqual(list(W[1][i]), [0., 0., 0., 0., 0.])

        self.assertEqual(len(B[0]), 5)
        self.assertEqual(len(B[1]), 4)
        self.assertListEqual(list(B[0]), [0., 0., 0., 0., 0.])
        self.assertListEqual(list(B[1]), [0., 0., 0., 0.])