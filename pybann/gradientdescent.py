import numpy as np
import copy as cp
from tqdm import tqdm

class GradientDescent:

    def __init__(self, dataset, alpha:float, niter:int, momentum:float, layers)->None:
        """
        Constructor
        """
        self.dataset = dataset
        self.alpha = alpha
        self.niter = niter
        self.momentum = momentum
        self.layers = layers

    def initializeUpdate(self):
        """
        Initialize weights and biases update array to zero
        """
        for ilayer in range(1, len(self.layers)):
            self.layers[ilayer].weightsUpdate[:] = 0.
            self.layers[ilayer].biasesUpdate[:] = 0.

    def dataSplit(self, dataset):
        """
        Split the data in input and output array
        """
        # Get input and output
        inValues = np.atleast_2d(dataset[0])
        outValues = np.atleast_2d(dataset[1])
        return inValues, outValues

    def forward(self, inValues)->None:
        """
        Return the output of the neural network for a given 
        input dataset and stores the intermediate results
        (transfer and activation results).
        """
        # Activation
        activation = [inValues.transpose()]
        transfer = []

        # Feed Forward
        for ilayer in range(1, len(self.layers)):
            transfer.append(np.dot(self.layers[ilayer].weights, activation[ilayer-1]) + self.layers[ilayer].biases)
            activation.append(self.layers[ilayer].activation(transfer[-1]))

        return activation, transfer

    def backward(self, activation, transfer, outValues)->None:
        """
        Calculates the gradient for each weight matrix and biase vector
        of the neural network.
        """
        # Backward
        delta = (activation[-1] - outValues.transpose()) * self.layers[-1].activation(transfer[-1], deriv=True)
        self.layers[-1].weightsUpdate += np.dot(delta, activation[-2].transpose())
        self.layers[-1].biasesUpdate += delta
        for ilayer in range(2, len(self.layers)):
            wsvector = transfer[-ilayer]
            delta =  self.layers[-ilayer].activation(wsvector, deriv=True) * np.dot(self.layers[-ilayer+1].weights.transpose(), delta)
            self.layers[-ilayer].weightsUpdate += np.dot(delta, activation[-ilayer-1].transpose())
            self.layers[-ilayer].biasesUpdate += delta

    def update(self)->None:
        """
        Update the weight matrix and biase vector of the neural network.
        """
        # Update weights and biases
        for ilayer in range(1, len(self.layers)):
            
            # Increment
            self.layers[ilayer].weights += (-self.alpha * self.layers[ilayer].weightsUpdate + self.momentum * self.layers[ilayer].weightsUpdateSave)
            self.layers[ilayer].biases += (-self.alpha * self.layers[ilayer].biasesUpdate + self.momentum * self.layers[ilayer].biasesUpdateSave)

            # Store update
            self.layers[ilayer].weightsUpdateSave = (-self.alpha * self.layers[ilayer].weightsUpdate + self.momentum * self.layers[ilayer].weightsUpdateSave)
            self.layers[ilayer].biasesUpdateSave = (-self.alpha * self.layers[ilayer].biasesUpdate + self.momentum * self.layers[ilayer].biasesUpdateSave)


    def run(self)->None:
        """
        Train
        :param dataset: a list of tuples in the form (inValues, outValues)
        :param niter: number of iterations
        :param alpha: step
        """

        for iter in tqdm(range(self.niter), bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}', desc="Training..."):
            # Re-initialize update arrays
            self.initializeUpdate()

            # Loop over datasets
            for idata in range(len(self.dataset)):

                # Split dataset
                inValues, outValues = self.dataSplit(self.dataset[idata])

                # Feed forward
                activation, transfer = self.forward(inValues)

                # Feed backward
                self.backward(activation, transfer, outValues)

            # Update
            self.update()
