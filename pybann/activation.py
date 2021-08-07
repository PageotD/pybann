"""
activation.py

A module containing some activation functions for the
artificial neural network.
"""

# Import modules
import numpy as np

def identity(x, deriv=False):
    """
    The sigmoid function and its derivative.
    """
    idt = x
    if not deriv:
        return idt
    else:
        return np.ones(len(x))

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
    tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    if not deriv:
        return tanh
    else:
        return 1. - tanh**2

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