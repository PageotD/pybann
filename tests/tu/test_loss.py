import pytest
import numpy as np
from pybann.loss import Loss

def test_mse():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 2.9])
    expected_mse = 0.01
    assert np.isclose(Loss.mse(y_true, y_pred), expected_mse, atol=1e-4)

def test_mse_derivative():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 2.9])
    expected_derivative = np.array([0.0667, 0.0667, -0.0667])
    assert np.allclose(Loss.mse_derivative(y_true, y_pred), expected_derivative, atol=1e-4)

def test_mae():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 2.9])
    expected_mae = 0.1
    assert np.isclose(Loss.mae(y_true, y_pred), expected_mae, atol=1e-4)

def test_mae_derivative():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 2.9])
    expected_derivative = np.array([0.3333, 0.3333, -0.3333])
    assert np.allclose(Loss.mae_derivative(y_true, y_pred), expected_derivative, atol=1e-4)

def test_binary_cross_entropy():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.1, 0.9, 0.8])
    expected_bce = 0.1446
    assert np.isclose(Loss.binary_cross_entropy(y_true, y_pred), expected_bce, atol=1e-4)

def test_binary_cross_entropy_derivative():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.1, 0.9, 0.8])
    expected_derivative = np.array([0.3704, -0.3704, -0.4166])
    assert np.allclose(Loss.binary_cross_entropy_derivative(y_true, y_pred), expected_derivative, atol=1e-4)

def test_categorical_cross_entropy():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
    expected_cce = 0.2283
    assert np.isclose(Loss.categorical_cross_entropy(y_true, y_pred), expected_cce, atol=1e-4)

def test_categorical_cross_entropy_derivative():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
    expected_derivative = np.array([[-0.0333, 0.0167, 0.0167],
                                    [0.0333, -0.0667, 0.0333],
                                    [0.0333, 0.0667, -0.1000]])
    assert np.allclose(Loss.categorical_cross_entropy_derivative(y_true, y_pred), expected_derivative, atol=1e-4)
