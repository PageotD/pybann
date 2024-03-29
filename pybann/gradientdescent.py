"""gradientdescent.py
"""

from typing import Union
import numpy as np
from tqdm import tqdm


class GradientDescent:
    """
    Class to train the neural network using the gradient descent
    method.
    """

    def __init__(self, dataset, batchsize: int, alpha: float,
                 nepoch: int, momentum: float, layers) -> None:
        """
        Initialize the gradient descent class
        
        Parameters
        ----------
        dataset: list or np.array
            a list of tuples in the form (inValues, outValues)
        batchsize: int, optional
            size of minibatches for training
        nepoch: int, optional
            maximum number of iterations (default: 1000)
        alpha: float, optional
            step for gradient descent (default: 0.05)
        momentum: float, optional
            step for the momentum (default: 0.5)

        """
        self.dataset = dataset
        self.alpha = alpha
        self.nepoch = nepoch
        self.momentum = momentum
        self.layers = layers
        self.batchsize = batchsize

    def init_update(self):
        """
        Initialize weights and biases update array to zero.
        """
        for ilayer in range(1, len(self.layers)):
            self.layers[ilayer].weightsUpdate[:, :] = 0.
            self.layers[ilayer].biasesUpdate[:] = 0.

    def datasplit(self, dataset):
        """
        Split the data in input and output array.
        """
        # Get input and output
        invalues = np.atleast_2d(dataset[0])
        outvalues = np.atleast_2d(dataset[1])
        return invalues, outvalues

    def forward(self, invalues) -> Union[list, list]:
        """
        Return the output of the neural network for a given
        input dataset and stores the intermediate results
        (transfer and activation results).

        Parameters
        ----------
        invalues: np.array
            vector containing the input values for the neural network model

        Returns
        -------
        activation: np.array
            activation value for each node
        transfer: np.array
            transfer value for each node

        """
        # Activation
        activation = [invalues.transpose()]
        transfer = []

        # Feed Forward
        for ilayer in range(1, len(self.layers)):
            transfer.append(np.dot(
                self.layers[ilayer].weights, activation[ilayer-1])
                + self.layers[ilayer].biases)
            activation.append(self.layers[ilayer].activation(transfer[-1]))

        return activation, transfer

    def backward(self, activation, transfer, outvalues) -> None:
        """
        Calculates the gradient for each weight matrix and biase vector
        of the neural network.
        """
        # Backward
        delta = (activation[-1] -
            outvalues.transpose()) * self.layers[-1].activation(transfer[-1], deriv=True)
        self.layers[-1].weightsUpdate += np.dot(
            delta, activation[-2].transpose())
        self.layers[-1].biasesUpdate += delta
        for ilayer in range(2, len(self.layers)):
            wsvector = transfer[-ilayer]
            delta = self.layers[-ilayer].activation(
                wsvector, deriv=True) * np.dot(
                    self.layers[-ilayer+1].weights.transpose(), delta)
            self.layers[-ilayer].weightsUpdate += np.dot(
                delta, activation[-ilayer-1].transpose())
            self.layers[-ilayer].biasesUpdate += delta

    def update(self) -> None:
        """
        Update the weight matrix and biase vector of the neural network.
        """
        # Update weights and biases
        for ilayer in range(1, len(self.layers)):

            # Increment
            self.layers[ilayer].weights += (
                -self.alpha * self.layers[ilayer].weightsUpdate
                + self.momentum * self.layers[ilayer].weightsUpdateSave)
            self.layers[ilayer].biases += (
                -self.alpha * self.layers[ilayer].biasesUpdate
                + self.momentum * self.layers[ilayer].biasesUpdateSave)

            # Store update
            self.layers[ilayer].weightsUpdateSave = (
                -self.alpha * self.layers[ilayer].weightsUpdate
                + self.momentum * self.layers[ilayer].weightsUpdateSave)
            self.layers[ilayer].biasesUpdateSave = (
                -self.alpha * self.layers[ilayer].biasesUpdate
                + self.momentum * self.layers[ilayer].biasesUpdateSave)

    def run(self) -> None:
        """
        Train
        """

        for epoch in tqdm(range(self.nepoch),
                         bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                         desc="Training..."):
            # Re-initialize update arrays
            self.init_update()

            if self.batchsize != 0:
                # Shuffle dataset
                np.random.shuffle(self.dataset)
            else:
                self.batchsize = len(self.dataset)

            # Loop over datasets
            for idata in range(self.batchsize):

                # Split dataset
                invalues, outvalues = self.datasplit(self.dataset[idata])

                # Feed forward
                activation, transfer = self.forward(invalues)

                # Feed backward
                self.backward(activation, transfer, outvalues)

            # Update
            self.update()
