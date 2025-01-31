import pytest
import numpy as np
from pybann.layers import Dense, ActivationLayer
from pybann.activation import Activation

def test_dense_layer_initialization():
    input_size, output_size = 2, 3
    layer = Dense(input_size, output_size)
    assert layer.weights.shape == (output_size, input_size)
    assert layer.bias.shape == (output_size,)

def test_dense_layer_forward():
    input_size, output_size = 2, 3
    layer = Dense(input_size, output_size)
    input_data = np.array([1, 2])
    output = layer.forward(input_data)
    assert output.shape == (output_size,)

def test_dense_layer_backward():
    input_size, output_size = 3, 2
    layer = Dense(input_size, output_size)
    input_data = np.array([[1, 2, 3]])
    layer.forward(input_data)
    output_gradient = np.array([[0.1, 0.2]])
    input_gradient = layer.backward(output_gradient, learning_rate=0.01)
    assert input_gradient.shape == input_data.shape

def test_activation_layer_sigmoid():
    layer = ActivationLayer("sigmoid")
    input_data = np.array([-1, 0, 1])
    output = layer.forward(input_data)
    assert np.allclose(output, Activation.sigmoid(input_data))

def test_activation_layer_relu():
    layer = ActivationLayer(Activation.relu)
    input_data = np.array([-1, 0, 1])
    output = layer.forward(input_data)
    assert np.allclose(output, Activation.relu(input_data))

def test_activation_layer_backward():
    layer = ActivationLayer(Activation.sigmoid)
    input_data = np.array([-1, 0, 1])
    layer.forward(input_data)
    output_gradient = np.array([0.1, 0.2, 0.3])
    input_gradient = layer.backward(output_gradient, learning_rate=0.01)
    expected_gradient = output_gradient * Activation.sigmoid_derivative(input_data)
    assert np.allclose(input_gradient, expected_gradient)

def test_dense_activation_combination():
    dense = Dense(3, 2)
    activation = ActivationLayer(Activation.sigmoid)
    input_data = np.array([[1, 2, 3]])
    dense_output = dense.forward(input_data)
    final_output = activation.forward(dense_output)
    assert final_output.shape == (1, 2)
    assert np.all((final_output >= 0) & (final_output <= 1))

