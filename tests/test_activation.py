import unittest
from pybann.activation import sigmoid, relu, leakyrelu, tanhyp, softplus, gaussian

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
            self.assertAlmostEqual(sigmoid(a[i], deriv=True), o[i], delta=1.e-7)

    def test_relu(self):

        # Input
        a = [-1., 0., 2.]

        # Attempted output
        o = [0., 0., 2.]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(relu(a[i]), o[i], delta=1.e-7)

    def test_relu_derivative(self):

        # Input
        a = [-1., 0., 1.]

        # Attempted output
        o = [0., 1., 1.]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(relu(a[i], deriv=True), o[i], delta=1.e-7)

    def test_leakyrelu(self):

        # Input
        a = [-1., 0., 2.]

        # Attempted output
        o = [-0.01, 0., 2.]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(leakyrelu(a[i]), o[i], delta=1.e-7)

    def test_leayrelu_derivative(self):

        # Input
        a = [-1., 0., 1.]

        # Attempted output
        o = [-0.01, 1., 1.]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(leakyrelu(a[i], deriv=True), o[i], delta=1.e-7)

    def test_tanhyp(self):

        # Input
        a = [-1., 0., 1.]

        # Attempted output
        o = [-0.76159415, 0.0, 0.76159415]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(tanhyp(a[i]), o[i], delta=1.e-7)

    def test_tanhyp_derivative(self):

        # Input
        a = [-1., 0., 1.]

        # Attempted output
        o = [0.41997434, 1.0, 0.41997434]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(tanhyp(a[i], deriv=True), o[i], delta=1.e-7)

    def test_softplus(self):

        # Input
        a = [-1., 0., 1.]

        # Attempted output
        o = [0.31326168, 0.69314718, 1.31326168]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(softplus(a[i]), o[i], delta=1.e-7)

    def test_softplus_derivative(self):

        # Input
        a = [-1., 0., 1.]

        # Attempted output
        o = [0.26894142, 0.5, 0.73105858]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(softplus(a[i], deriv=True), o[i], delta=1.e-7)

    def test_gaussian(self):

        # Input
        a = [-1., 0., 1.]

        # Attempted output
        o = [0.36787944, 1.00000000, 0.36787944]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(gaussian(a[i]), o[i], delta=1.e-7)

    def test_gaussian_derivative(self):

        # Input
        a = [-1., 0., 1.]

        # Attempted output
        o = [0.73575888, 0.00000000, -0.73575888]

        # Output
        for i in range(len(a)):
            self.assertAlmostEqual(gaussian(a[i], deriv=True), o[i], delta=1.e-7)