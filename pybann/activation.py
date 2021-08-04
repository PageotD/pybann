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
