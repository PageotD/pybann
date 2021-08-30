"""
activation.py

A module containing some activation functions for the
artificial neural network.
"""

# Import modules
import numpy as np
from typing import Union


class Activation:
    """
    Collection of activation functions commonly used in the design of
    artificial neural networks. All methods are statics which means they
    can be called without creating an instance.
    """
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(a: Union[float, np.array], deriv: bool = False) -> Union[float, np.array]:
        r"""
        Returns the value of the sigmoid function
        (or its derivative).


        Parameters
        ----------
        a: float or numpy array
            input value(s)
        deriv: bool, default: False
            when `True`, returns the derivative

        Returns
        -------
        f: float or np.array
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
        f = 1.0 / (1.0 + np.exp(-a))
        if not deriv:
            return f
        else:
            return f * (1.0 - f)

    @staticmethod
    def tanhyp(x, deriv=False):
        """
        Returns the value of the Hyperbolic tangent function
        (or its derivative).

        Parameters
        ----------
        a: float or numpy array
            input value(s)
        deriv: bool, default: False
            when `True`, returns the derivative

        Returns
        -------
        f: float or np.array
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
        f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        if not deriv:
            return f
        else:
            return 1. - f**2

    @staticmethod
    def relu(x, a=0., deriv=False):
        """
        Returns the value of the Rectified Linear Unit function
        (or its derivative).

        Parameters
        ----------
        a: float or numpy array
            input value(s)
        deriv: bool, default: False
            when `True`, returns the derivative

        Returns
        -------
        f: float or np.array
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
        # Add an epsilon value to ensure there is no strict zero values
        if deriv:
            x += np.finfo(float).eps

        if np.ndim(x) != 0:
            if not deriv:
                f = [a * x[i] if x[i] <= 0. else x[i] for i in range(len(x))]
                return np.array(f)
            else:
                f = np.copy(x)
                f[f < 0.] = a
                f[f >= 0.] = 1.
                return f
        else:
            if not deriv:
                return a * x if x <= 0. else x
            else:
                return a if x < 0. else 1.

    @staticmethod
    def softplus(x, deriv=False):
        """
        Returns the value of the Softplus function (or its derivative).

        Parameters
        ----------
        a: float or numpy array
            input value(s)
        deriv: bool, default: False
            when `True`, returns the derivative

        Returns
        -------
        f: float or np.array
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
            return np.log(1. + np.exp(x))
        else:
            return Activation.sigmoid(x)

    @staticmethod
    def gaussian(x, deriv=False):
        """
        Returns the value of the Gaussian function (or its derivative).

        Parameters
        ----------
        a: float or numpy array
            input value(s)
        deriv: bool, default: False
            when `True`, returns the derivative

        Returns
        -------
        f: float or np.array
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
        f = np.exp(-(x**2))
        if not deriv:
            return f
        else:
            return -2 * x * f
