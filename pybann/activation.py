"""
activation.py

A module containing some activation functions for the
artificial neural network.
"""

# Import modules
import numpy as np


def sigmoid(x, deriv=False):
    """
    The sigmoid function and its derivative.
    """
    sig = 1.0 / (1.0 + np.exp(-x))
    if not deriv:
        return sig
    else:
        return sig * (1.0 - sig)

def tanhyp(x, deriv=False):
    """
    Hyperbolic tangent
    """
    tan = 2. / (1. + np.exp(-2. * x))
    if not deriv:
        return tan
    else:
        return 1 - (tan * tan)

def softplus(x, deriv=False):
    """
    Softplus 
    """
    splus = np.log(1. + np.exp(x))
    if not deriv:
        return splus
    else:
        return sigmoid(x)

def gaussian(x, deriv=False):
    """
    Gaussian
    """
    gauss = np.exp(-x**2)
    if not deriv:
        return gauss
    else:
        return -2 * x * gauss