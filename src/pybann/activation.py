"""
activation.py

This module provides activation functions and their derivatives.

"""
import numpy as np

class Activation:
    """
    This class provides activation functions and their derivatives.
    
    Attributes
    ----------
    sigmoid : function
        The sigmoid activation function.
    sigmoid_derivative : function
        The derivative of the sigmoid activation function.
    relu : function
        The ReLU activation function.
    relu_derivative : function
        The derivative of the ReLU activation function.
    tanh : function
        The hyperbolic tangent activation function.
    tanh_derivative : function
        The derivative of the hyperbolic tangent activation function.
    softmax : function
        The softmax activation function.
    softmax_derivative : function
        The derivative of the softmax activation function.
    """

    @staticmethod
    def sigmoid(x):
        """
        Apply the sigmoid activation functions to the input values.
        
        Parameters
        ----------
        x : float or numpy array
            input value(s)
        
        Returns
        -------
        result : float or np.array
            the value of the function at x
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """
        Apply the derivative of the sigmoid activation functions to the input values.
        
        Parameters
        ----------
        x : float or numpy array
            input value(s)
        
        Returns
        -------
        result : float or np.array
            the value of the function at x
        """
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x):
        """
        Apply the ReLU activation functions to the input values.
        
        Parameters
        ----------
        x : float or numpy array
            input value(s)
        
        Returns
        -------
        result : float or np.array
            the value of the function at x

        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """
        Apply the derivative of the ReLU activation functions to the input values.
        
        Parameters
        ----------
        x : float or numpy array
            input value(s)
        
        Returns
        -------
        result : float or np.array
            the value of the function at x
        """
        x = np.array(x)  # Convert x to a NumPy array
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x):
        """
        Apply the hyperbolic tangent activation functions to the input values.
        
        Parameters
        ----------
        x : float or numpy array
            input value(s)
        
        Returns
        -------
        result : float or np.array
            the value of the function at x
        """
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        """
        Apply the derivative of the hyperbolic tangent activation functions to the input values.
        
        Parameters
        ----------
        x : float or numpy array
            input value(s)
        
        Returns
        -------
        result : float or np.array
            the value of the function at x
        """
        return 1 - np.tanh(x)**2

    @staticmethod
    def softmax(x):
        """
        Apply the softmax activation functions to the input values.
        
        Parameters
        ----------
        x : float or numpy array
            input value(s)
        
        Returns
        -------
        result : float or np.array
            the value of the function at x
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def softmax_derivative(x):
        """
        Apply the derivative of the softmax activation functions to the input values.
        
        Parameters
        ----------
        x : float or numpy array
            input value(s)
        
        Returns
        -------
        result : float or np.array
            the value of the function at x
        """
        s = Activation.softmax(x)
        return s * (np.eye(s.shape[-1]) - s[..., np.newaxis])