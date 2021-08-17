"""
activation.py

A module containing some activation functions for the
artificial neural network.
"""

# Import modules
import numpy as np

class Activation:

    def __init__(self):
        pass

    @staticmethod
    def sigmoid(x, deriv=False):
        """
        The sigmoid function and its derivative.
        """
        f = 1.0 / (1.0 + np.exp(-x))
        if not deriv:
            return f
        else:
            return f * (1.0 - f)

    @staticmethod
    def tanhyp(x, deriv=False):
        """
        Hyperbolic tangent
        """
        f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        if not deriv:
            return f
        else:
            return 1. - f**2

    @staticmethod
    def relu(x, a=0., deriv=False):
        """
        Leaky Rectified Linear Unit
        """
        # Add an epsilon value to ensure there is no strict zero values
        if deriv: x += np.finfo(float).eps

        if np.ndim(x) != 0:
            if not deriv:
                f = [a * x[i] if x[i] <=0. else x[i] for i in range(len(x))]
                #f = np.maximum(x, a * x)
                return np.array(f)
            else:
                #f = [a if x[i] <0. else 1. for i in range(len(x))]
                f = np.copy(x)
                f[f < 0.] = a
                f[f >=0.] = 1.
                return f
        else:
            if not deriv:
                return a * x if x <= 0. else x
            else:
                return a if x < 0. else 1.

    @staticmethod
    def softplus(x, deriv=False):
        """
        Softplus 
        """
        if not deriv:
            return np.log(1. + np.exp(x))
        else:
            return Activation.sigmoid(x)

    @staticmethod
    def gaussian(x, deriv=False):
        """
        Gaussian
        """
        f = np.exp(-(x**2))
        if not deriv:
            return f
        else:
            return -2 * x * f