# Import modules
import numpy as np

from pybann import Layer
from pybann import GradientDescent

class Model:

    def __init__(self, name:str="New model")->None:
        self.name = name
        self.layers = []

    def __repr__(self)->None:
        return "Model(name={})".format(self.name)

    def __str__(self)->None:
        pass

    def addInput(self, neurons: int, label: str="")->None:

        try:
            # Assert the number of neurons is at least 1
            assert neurons >= 1
            self.layers.append(Layer(neurons, label))

        except AssertionError():
            print("The layer must have at least 1 neuron.")

    def addLayer(self, neurons: int, activation: str="sigmoid", label: str="")->None:

        try:
            # Assert the number of neurons is at least 1
            assert neurons >= 1
            self.layers.append(Layer(neurons, label))
            self.layers[-1].addActivation(activation)

        except AssertionError():
            print("The layer must have at least 1 neuron.")

    def build(self):
        # build the model
        # Add weights, biaises
        
        for i in range(1, len(self.layers)):
            # Add biases
            self.layers[i].addBiases()
            # Add weights
            self.layers[i].addWeights(self.layers[i-1].neurons)

    def forward(self, inValues):
        # Simple feeed forward
        
        # Convert to 2D (n, 1) and transpose for dot product
        inValues = np.atleast_2d(inValues).transpose()
        for i in range(1, len(self.layers)):
            transfer = np.dot(self.layers[i].weights, inValues)
            inValues = self.layers[i].activation(transfer+self.layers[i].biases)
        
        # Flatten output (convert to 1D vector)
        return inValues.flatten()

    def SGD(self, dataset, alpha:float=0.05, niter:int=1000)->None:
        """
        Train
        :param dataset: a list of tuples in the form (inValues, outValues)
        :param niter: number of iterations
        :param alpha: step
        """
        SGDescent = GradientDescent(dataset, alpha, niter, self.layers)
        SGDescent.run()

    def PSO(self):
        # particle swarm optimization
        # Need PSO class with options
        pass

    
    def save(self):
        # Save to HDF5 or NetCDF
        pass

    def load(self):
        # Load a saved train network model
        # HDF5 or NetCDF
        pass

    def show(self):
        # print network structure
        pass

class Backpropagation:
    # Can set attributes for Layers (gradient weights and biases)
    pass