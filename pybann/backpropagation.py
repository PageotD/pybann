import numpy as np
from .activation import sigmoid

def backpropagate(nLayers, weights, biases, injected, predicted):
    """
    :param injected: input values
    :param predicted: attempted output values
    :return : updated weights and biaises
    """
    # Input/output vectors must be at least 2D (n, 1) to be used with numpy dot product
    injected = np.atleast_2d(injected)
    predicted = np.atleast_2d(predicted)

    # Initialize
    nablaW = [np.zeros(w.shape) for w in weights]
    nablaB = [np.zeros(b.shape) for b in biases]

    # Feedforward
    activation = injected.transpose()
    activations = [activation]
    wsvectors = [] # wsvectors are the weighted sums of inputs calculated for each layer
    for b, w in zip(biases, weights):
        wsvector = np.dot(w, activation) + b
        wsvectors.append(wsvector)
        activation = sigmoid(wsvector)
        activations.append(activation)

    # Backward
    delta = (activations[-1] - predicted.transpose()) * sigmoid(wsvectors[-1], deriv=True)
    nablaW[-1] = np.dot(delta, activations[-2].transpose())
    nablaB[-1] = delta
    for iLayer in range(2, nLayers):
        wsvector = wsvectors[-iLayer]
        delta =  sigmoid(wsvector, deriv=True) * np.dot(weights[-iLayer+1].transpose(), delta)
        nablaW[-iLayer] = np.dot(delta, activations[-iLayer-1].transpose())
        nablaB[-iLayer] = delta

    return nablaW, nablaB