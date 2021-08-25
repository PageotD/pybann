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

    def addActivation(self, activation: str="sigmoid")->None:
        """
        Add an activation function to the Layer object

        Parameters
        ----------
        activation: str (optional)
            name of the activation function (default `sigmoid`)
        
        Examples
        --------

        >>> from pybann import Layer
        >>> layer1 = Layer(neurons=4)
        >>> layer1.addActivation()
        >>> layer1.activation.__name__
        sigmoid

        >>> layer2 = Layer(neurons=4)
        >>> layer2.addActivation(activation="relu")
        >>> layer2.activation.__name__
        relu

        """
        self.__setattr__('activation', getattr(Activation, activation))

    def addWeights(self, inputNeurons:int)->None:
        """
        Add weight matrix for the feed forward, and updated weight matrices
        for the gradient descent.

        Parameters
        ----------
        inputNeurons: int
            number of neurons in the layer

        Examples
        --------

        >>> import numpy as np
        >>> from pybann import Layer
        >>> layer1 = Layer(neurons=4)
        >>> layer1.addWeights(inputNeurons=8)
        >>> np.shape(layer1.weights)
        (4, 8)
        >>> np.shape(layer1.weightsUpdate)
        (4, 8)
        >>> np.shape(layer1.weightsUpdateSave)
        (4, 8)
        
        """
        self.__setattr__('weights', np.random.randn(self.neurons, inputNeurons))
        self.__setattr__('weightsUpdate', np.zeros((self.neurons, inputNeurons)))
        self.__setattr__('weightsUpdateSave', np.zeros((self.neurons, inputNeurons)))

    def addBiases(self)->None:
        """
        Add a vector containing random biases
        """
        self.__setattr__('biases', np.random.randn(self.neurons, 1))
        self.__setattr__('biasesUpdate', np.zeros((self.neurons,1)))
        self.__setattr__('biasesUpdateSave', np.zeros((self.neurons,1)))

    def __repr__(self)->str:
        return "Layer({}, {}, {})".format(self.neurons, self.activation, self.label)

    #def __str__(self)->None:
    #    print("Layer {}: \n - neurons: {}\n - weights: {}\n - biases: {}\n - activation: {}".format(self.index, self.n, self.weights.shape(), self.biases.shape(), self.activation))
