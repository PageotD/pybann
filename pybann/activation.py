"""
activation.py
"""

# Import modules
from typing import Union
import numpy as np


class Activation:
    """
    Collection of activation functions commonly used in the design of
    artificial neural networks.

    An activation function, in artificial neural networks, defines how the weighted sum
    of the input is transformed into an output.

    All methods are statics which means they can be called without creating an instance.
    """
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(wsum: Union[float, np.array], deriv: bool = False) -> Union[float, np.array]:
        """sigmoid

        Apply the sigmoid activation functions or its derivative to the input values.

        The sigmoid activation function returns near 0 values for input values <-5 and
        near 1 for input values >5

        The sigmoid derivative is a zero-centered Gaussian-like function returning values
        between 0 and 0.25.

        Parameters
        ----------
        wsum: float or numpy array
            input value(s)
        deriv: bool, default: False
            when `True`, returns the derivative

        Returns
        -------
        result: float or np.array
            the value of the function (or its derivative).

        Examples
        --------
        >>> a = 0.
        >>> sigmoid(a)
        0.5

        >>> import numpy as np
        >>> a = np.array([-1., 0., 1.])
        >>> sigmoid(a)
        array([0.26894142, 0.5, 0.73105858])
        """
        result = 1.0 / (1.0 + np.exp(-wsum))
        if not deriv:
            return result

        return result * (1.0 - result)

    @staticmethod
    def tanhyp(wsum, deriv=False):
        """
        Returns the value of the Hyperbolic tangent function
        (or its derivative).

        Parameters
        ----------
        wsum: float or numpy array
            input value(s)
        deriv: bool, default: False
            when `True`, returns the derivative

        Returns
        -------
        result: float or np.array
            the value of the function (or its derivative).

        Examples
        --------
        >>> x = 0.
        >>> tanhyp(x)
        0.

        >>> import numpy as np
        >>> x = np.array([-1., 0., 1.])
        >>> tanhyp(x)
        array([-0.76159415, 0.0, 0.76159415])
        """
        result = (np.exp(wsum) - np.exp(-wsum)) / (np.exp(wsum) + np.exp(-wsum))
        if not deriv:
            return result

        return 1. - result**2

    @staticmethod
    def relu(wsum, leak=0., deriv=False):
        """
        Returns the value of the Rectified Linear Unit function
        (or its derivative).

        Parameters
        ----------
        wsum: float or numpy array
            input value(s)
        deriv: bool, default: False
            when `True`, returns the derivative

        Returns
        -------
        result: float or np.array
            the value of the function (or its derivative).

        Examples
        --------
        >>> x = 0.
        >>> relu(x)
        0.

        >>> import numpy as np
        >>> x = np.array([-1., 0., 2.])
        >>> relu(x)
        array([0., 0., 2.])
        """
        if not deriv:
            result = np.where(wsum > 0, wsum, wsum * leak)
            return result

        result = np.where(wsum >= 0, 1., leak)
        return result

    @staticmethod
    def softplus(wsum, deriv=False):
        """
        Returns the value of the Softplus function (or its derivative).

        Parameters
        ----------
        wsum: float or numpy array
            input value(s)
        deriv: bool, default: False
            when `True`, returns the derivative

        Returns
        -------
        result: float or np.array
            the value of the function (or its derivative).

        Examples
        --------
        >>> x = 0.
        >>> softplus(x)
        0.69314718

        >>> import numpy as np
        >>> x = np.array([-1., 0., 1.])
        >>> softplus(x)
        array([0.31326168, 0.69314718, 1.31326168])
        """
        if not deriv:
            return np.log(1. + np.exp(wsum))

        return Activation.sigmoid(wsum)

    @staticmethod
    def gaussian(wsum, deriv=False):
        """
        Returns the value of the Gaussian function (or its derivative).

        Parameters
        ----------
        wsum: float or numpy array
            input value(s)
        deriv: bool, default: False
            when `True`, returns the derivative

        Returns
        -------
        result: float or np.array
            the value of the function (or its derivative).

        Examples
        --------
        >>> x = 0.
        >>> gaussian(x)
        1.

        >>> import numpy as np
        >>> x = np.array([-1., 0., 1.])
        >>> gaussian(x)
        array([0.36787944, 1.00000000, 0.36787944])
        """
        result = np.exp(-(wsum**2))
        if not deriv:
            return result

        return -2 * wsum * result
