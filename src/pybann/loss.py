import numpy as np

class Loss:
    """
    Class for loss functions.
    
    Attributes
    ----------
    mse : function
        The mean squared error loss function.
    mse_derivative : function
        The derivative of the mean squared error loss function.
    mae : function
        The mean absolute error loss function.
    mae_derivative : function
        The derivative of the mean absolute error loss function.
    binary_cross_entropy : function
        The binary cross-entropy loss function.
    binary_cross_entropy_derivative : function
        The derivative of the binary cross-entropy loss function.
    categorical_cross_entropy : function
        The categorical cross-entropy loss function.
    categorical_cross_entropy_derivative : function
        The derivative of the categorical cross-entropy loss function.
    """
    @staticmethod
    def mse(y_true, y_pred):
        """
        Mean squared error loss function.
        
        Parameters
        ----------
        y_true : float or numpy array
            true value(s)
        y_pred : float or numpy array
            predicted value(s)
        
        Returns
        -------
        result : float or np.array
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        """
        Derivative of the mean squared error loss function.
        
        Parameters
        ----------
        y_true : float or numpy array
            true value(s)
        y_pred : float or numpy array
            predicted value(s)
        
        Returns
        -------
        result : float or np.array
        """
        return 2 * (y_pred - y_true) / y_true.size
    
    @staticmethod
    def mae(y_true, y_pred):
        """
        Mean absolute error loss function.
        
        Parameters
        ----------
        y_true : float or numpy array
            true value(s)
        y_pred : float or numpy array
            predicted value(s)
        
        Returns
        -------
        result : float or np.array
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mae_derivative(y_true, y_pred):
        """
        Derivative of the mean absolute error loss function.
        
        Parameters
        ----------
        y_true : float or numpy array
            true value(s)
        y_pred : float or numpy array
            predicted value(s)
        
        Returns
        -------
        result : float or np.array
        """
        return np.sign(y_pred - y_true) / y_true.size
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """
        Binary cross-entropy loss function.
        
        Parameters
        ----------
        y_true : float or numpy array
            true value(s)
        y_pred : float or numpy array
            predicted value(s)
        
        Returns
        -------
        result : float or np.array
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        """
        Derivative of the binary cross-entropy loss function.
        
        Parameters
        ----------
        y_true : float or numpy array
            true value(s)
        y_pred : float or numpy array
            predicted value(s)
        
        Returns
        -------
        result : float or np.array
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.size
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        """
        Categorical cross-entropy loss function.
        
        Parameters
        ----------
        y_true : float or numpy array
            true value(s)
        y_pred : float or numpy array
            predicted value(s)
        
        Returns
        -------
        result : float or np.array
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    @staticmethod
    def categorical_cross_entropy_derivative(y_true, y_pred):
        """
        Derivative of the categorical cross-entropy loss function.
        
        Parameters
        ----------
        y_true : float or numpy array
            true value(s)
        y_pred : float or numpy array
            predicted value(s)
        
        Returns
        -------
        result : float or np.array
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / y_true.shape[0]
