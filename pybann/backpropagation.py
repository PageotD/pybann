import numpy as np
from .activation import sigmoid

def backpropagate(nLayers, weights, biases, inValues, outValues):
    """
    :param invalues: input values
    :param outValues: attempted output values
    :return : updated weights and biaises
    """
    # Initialize
    nablaW = [np.zeros(w.shape) for w in weights]
    nablaB = [np.zeros(b.shape) for b in biases]

    # Feedforward
    activation = np.array(inValues)
    activations = [activation]
    wsvectors = [] # wsvectors are the weighted sums of inputs calculated for each layer
    for b, w in zip(biases, weights):
        wsvector = np.dot(w, activation) + b
        wsvectors.append(wsvector)
        activation = sigmoid(wsvector)
        activations.append(activation)
        
    # Backward
    delta = (activations[-1] - outValues) * sigmoid(wsvectors[-1], deriv=True)
    print(delta)
    nablaW[-1] = np.dot(delta, activations[-1].transpose())
    nablaB[-1] = delta
    for iLayer in range(2, nLayers):
        wsvector = wsvectors[-iLayer]
        delta = np.dot(weights[-iLayer+1].transpose(), delta) * sigmoid(wsvector, deriv=True)
        nablaW[-iLayer] = np.dot(delta, activations[-iLayer].transpose())
        nablaB[-iLayer] = delta

    return nablaW, nablaB