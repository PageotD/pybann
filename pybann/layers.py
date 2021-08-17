# Import modules
import numpy as np
from pybann import Activation

class Layer:
    def __init__(self, neurons: int, label: str="")->None:

        try:
            if neurons >= 1:        
                self.neurons = neurons
        except ValueError():
            print("The number of neuron(s) must be greater or equal to 1.")
        
        self.label = label

    def addActivation(self, activation: str="sigmoid"):
        """
        Add an activation function
        """
        self.__setattr__('activation', getattr(Activation, activation))

    def addWeights(self, inputNeurons:int)->None:
        """
        Add a matrice containing random weights

        :param inputNeurons: number of neurons in the previous layer if exist
        """
        self.__setattr__('weights', np.random.randn(self.neurons, inputNeurons))
        self.__setattr__('weightsUpdate', np.zeros((self.neurons, inputNeurons)))

    def addBiases(self)->None:
        """
        Add a vector containing random biases
        """
        self.__setattr__('biases', np.random.randn(self.neurons, 1))
        self.__setattr__('biasesUpdate', np.zeros((self.neurons,1)))

    def __repr__(self)->str:
        return "Layer({}, {}, {})".format(self.neurons, self.activation, self.label)

    #def __str__(self)->None:
    #    print("Layer {}: \n - neurons: {}\n - weights: {}\n - biases: {}\n - activation: {}".format(self.index, self.n, self.weights.shape(), self.biases.shape(), self.activation))
