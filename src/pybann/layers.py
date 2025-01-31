import numpy as np
from pybann.activation import Activation

class Layer:

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass
    
    def backward(self, output_gradient, learning_rate):
        pass
    
class Dense(Layer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size,))

    def forward(self, input):
        return np.dot(self.weights, input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient
    
class ActivationLayer(Layer):

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation.derivative(self.input))
    
