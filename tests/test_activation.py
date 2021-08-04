import unittest
import numpy as np
from pybann.activation import sigmoid, dsigmoid

class test_activation(unittest.TestCase):

    def test_sigmoid(self):

        # Input
        a = [-1., 0., 1.]

        # Attempted output
        o = [0.26894142, 0.5, 0.73105858]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(sigmoid(a[i]), o[i], delta=1.e-7)

    def test_sigmoid_derivative(self):

        # Input
        a = [-1., 0., 1.]

        # Attempted output
        o = [0.19661193, 0.25, 0.19661193]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(dsigmoid(a[i]), o[i], delta=1.e-7)