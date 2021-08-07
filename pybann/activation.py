"""
activation.py

A module containing some activation functions for the
artificial neural network.
"""

# Import modules
import numpy as np

def identity(x, deriv=False):
    """
    The linear identity function and its derivative.
    """
    if not deriv:
        return x
    else:
        return 1.

def heaviside(x, deriv=False):
    """
    """
    # Add an epsilon value to ensure there is no strict zero values
    if deriv: x += np.finfo(float).eps

    if not deriv and x < 0.:
        return 0.
    if not deriv and x >= 0.:
        return 1.
    if deriv: return 0.

def sigmoid(x, deriv=False):
    """
    The sigmoid function and its derivative.
    """
    f = 1.0 / (1.0 + np.exp(-x))
    if not deriv:
        return f
    else:
        return f * (1.0 - f)

def relu(x, deriv=False):
    """
    Rectified Linear Unit

    Nair, V., & Hinton, G. E. (2010, January). Rectified linear units improve
    restricted boltzmann machines. In Icml.
    """
    # Add an epsilon value to ensure there is no strict zero values
    if deriv: x += np.finfo(float).eps
    
    if not deriv and x <= 0.:
        return 0.
    if not deriv and x > 0.:
        return x
    if deriv and x < 0.:
        return 0.
    if deriv and x > 0.:
        return 1.
     

def tanhyp(x, deriv=False):
    """
    Hyperbolic tangent
    """
    f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    if not deriv:
        return f
    else:
        return 1. - f**2

def softplus(x, deriv=False):
    """
    Softplus 
    """
    if not deriv:
        return np.log(1. + np.exp(x))
    else:
        return sigmoid(x)

def gaussian(x, deriv=False):
    """
    Gaussian
    """
    f = np.exp(-x**2)
    if not deriv:
        return f
    else:
        return -2 * x * f