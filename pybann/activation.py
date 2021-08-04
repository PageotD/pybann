"""
activation.py

A module containing some activation functions for the
artificial neural network.
"""

# Import modules
import math as m


def sigmoid(x):
    """
    The sigmoid function.
    """
    return 1.0 / (1.0 + m.exp(-x))

def dsigmoid(x):
    """
    The sigmoid derivative.
    """
    return sigmoid(x) * (1.0- sigmoid(x))
