# Import modules
import numpy as np
import pickle
from tqdm import tqdm
from pybann import Layer
from pybann import GradientDescent


class Model:

    def __init__(self, name: str = "New model") -> None:
        self.name = name
        self.layers = []

    def __repr__(self) -> None:
        return "Model(name={})".format(self.name)

    def __str__(self) -> None:
        pass

    def addInput(self, neurons: int, label: str = "") -> None:
        """
        Add an input layer to the network model.

        Parameters
        ----------
        neurons: int
            number of neurons in the input layer
        label: str, optional
            label (name) of the layer

        Examples
        --------

        >>> from pybann import Model
        >>> network = Model()
        >>> network.addInput(neurons=4, label="Input layer")

        """

        try:
            # Assert the number of neurons is at least 1
            assert neurons >= 1
            self.layers.append(Layer(neurons, label))

        except AssertionError():
            print("The layer must have at least 1 neuron.")

    def addLayer(self, neurons: int, activation: str = "sigmoid",
                 label: str = "") -> None:
        """
        Add a layer to the network model.

        Parameters
        ----------
        neurons: int
            number of neurons in the layer
        activation: str, optional
            activation function for the layer (default: 'sigmoid')
        label: str, optional
            label (name) of the layer

        Examples
        --------

        >>> from pybann import Model
        >>> network = Model()
        >>> network.addInput(neurons=4, label="Input layer")
        >>> network.addLayer(neurons=8, activation="relu", label="1st hidden layer")

        """
        try:
            # Assert the number of neurons is at least 1
            assert neurons >= 1
            self.layers.append(Layer(neurons, label))
            self.layers[-1].add_activation(activation)

        except AssertionError():
            print("The layer must have at least 1 neuron.")

    def build(self) -> None:
        """
        Build the network model

        Example
        -------

        >>> from pybann import Model
        >>> network = Model()
        >>> network.addInput(neurons=4, label="Input layer")
        >>> network.addLayer(neurons=8, activation="relu", label="Hidden layer")
        >>> network.addLayer(neurons=3, activation="sigmoid", label="Output layer")
        >>> network.build()

        """
        # build the model
        # Add weights, biaises
        for i in tqdm(range(1, len(self.layers)),
                      bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                      desc="Building..."):
            # Add biases
            self.layers[i].add_biases()
            # Add weights
            self.layers[i].add_weights(self.layers[i-1].neurons)

    def forward(self, inValues) -> np.array:
        """
        Feed forward the network model

        Parameters
        ----------
        inValues: np.array
            vector containing the input values for the neural network model

        Example
        -------

        >>> import numpy as np
        >>> from pybann import Model
        >>> network = Model()
        >>> network.addInput(neurons=4, label="Input layer")
        >>> network.addLayer(neurons=8, activation="relu", label="Hidden layer")
        >>> network.addLayer(neurons=3, activation="sigmoid", label="Output layer")
        >>> network.build()
        >>> inData = np.array([0., 1., 2., 3.])
        >>> output = network.forward(inData)

        """
        # Simple feeed forward

        # Convert to 2D (n, 1) and transpose for dot product
        inValues = np.atleast_2d(inValues).transpose()
        for i in range(1, len(self.layers)):
            transfer = np.dot(self.layers[i].weights, inValues)
            inValues = self.layers[i].activation(transfer+self.layers[i].biases)

        # Flatten output (convert to 1D vector)
        return inValues.flatten()

    def SGD(self, dataset, batchsize=0, alpha: float = 0.05,
            nepoch: int = 1000, momentum: float = 0.5) -> None:
        """
        Train the neural network model

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
        SGDescent = GradientDescent(dataset, batchsize,
                                    alpha, niter, momentum, self.layers)
        SGDescent.run()

    def PSO(self):
        # particle swarm optimization
        # Need PSO class with options
        pass

    def save(self, filename: str = "network.bann") -> None:
        """
        Save the network model using Pickle

        Parameters
        ----------
        filename: str 
            filename where to save the network model (default: 'network.bann')
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename: str):
        """
        Load a network model using Pickle

        Parameters
        ----------
        filename: str 
            name of the saved network model (default: 'network.bann')
        """
        with open(filename, 'rb') as f:
            self = pickle.load(f)

    def show(self):
        # print network structure
        pass
