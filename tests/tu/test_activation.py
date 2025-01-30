import pytest
import numpy as np
from pybann.activation import Activation

def test_sigmoid():
    assert np.isclose(Activation.sigmoid(0), 0.5)
    assert np.isclose(Activation.sigmoid(100), 1)
    assert np.isclose(Activation.sigmoid(-100), 0)

def test_sigmoid_derivative():
    x = np.array([-1, 0, 1])
    expected = Activation.sigmoid(x) * (1 - Activation.sigmoid(x))
    assert np.allclose(Activation.sigmoid_derivative(x), expected)

def test_relu():
    assert np.array_equal(Activation.relu([-1, 0, 1]), [0, 0, 1])

def test_relu_derivative():
    assert np.array_equal(Activation.relu_derivative([-1, 0, 1]), [0, 0, 1])

def test_tanh():
    assert np.isclose(Activation.tanh(0), 0)
    assert np.isclose(Activation.tanh(100), 1)
    assert np.isclose(Activation.tanh(-100), -1)

def test_tanh_derivative():
    x = np.array([-1, 0, 1])
    expected = 1 - np.tanh(x)**2
    assert np.allclose(Activation.tanh_derivative(x), expected)

def test_softmax():
    x = np.array([1, 2, 3])
    result = Activation.softmax(x)
    assert np.isclose(np.sum(result), 1)
    assert np.all(result > 0)

def test_softmax_derivative():
    x = np.array([1.0, 2.0, 3.0])
    
    # Compute the Jacobian numerically
    epsilon = 1e-7
    numerical_jacobian = np.zeros((3, 3))
    for i in range(3):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_minus = x.copy()
        x_minus[i] -= epsilon
        numerical_jacobian[:, i] = (Activation.softmax(x_plus) - Activation.softmax(x_minus)) / (2 * epsilon)
    
    # Compute the Jacobian using our implementation
    analytical_jacobian = Activation.softmax_derivative(x)
    
    # Check if they are close
    assert np.allclose(numerical_jacobian, analytical_jacobian, atol=1e-5)
