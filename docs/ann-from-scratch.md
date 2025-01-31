# ANN from Scratch

#### Table of Contents
- [1. Layers](#1.-Layers)
- [2. Weights](#2.-Weights)
- [3. Biases](#3.-Biases)
- [4. Feed Forward](#4.-Feed-Forward)
- [5. Activation Functions](#5.-Activation-Functions)
- [6. Inputs and Outputs](#6.-Inputs-and-Outputs)

## 1. Layers

There is three main types of layers in a neural network model:

1. **Input Layer**: the first layer of the neural network which receives the input data.
2. **Output Layer**: the last layer of the neural network which produces the output (predictions).
3. **Hidden Layer**: the layers in between the input and the output layers.

Layers are generally numbered starting at the first hidden layer (1) and ending at the last output layer. The input layer is excluded since it contains the raw input. However, for mathematical and algorithmic reasons, the input layer is often numbered at 0.

The number of nodes in a layer will be noted as $n^{(l)}$ where $l$ is the layer number and $L$ is the total number of layers.

## 2. Weights

Weights are the values used to calculate the values in the $l+1$ layer from the values of the $l$ layer. The total number of weights is equal to the number of connections between the layers, _i.e._ $n^{(l)} \times n^{(l+1)}$.

Weights are represented as a matrix $W^{(l)}$ with dimensions $n^{(l)} \times n^{(l-1)}$.

## 3. Biases

Each neuron in a layer has a bias which is added to the weighted sum of the input values. Biases are represented as a vector $b^{(l)}$ with dimensions $n^{(l)} \times 1$.

## 4. Feed Forward

Feed forward is the process of calculating the values in a layer from the values in the previous layer. The process is repeated for all layers in the model.

Let represent the matrix of nodes in a layer as $z^{l}$. The values for $z^{l}$ can be calculated as:

$$z^{l} = W^{l}z^{l-1} + b^{l}$$

## 5. Activation Functions

Once a value is calculated for a neuron, the value needs to be transformed into another value. This transformation is called an **activation function**. There are many activation functions, but the most common one is the sigmoid function.:

$$g(x) = \frac{1}{1 + e^{-x}}$$

Activations functions must be:
- nonlinear
- differentiable
- compress the the input to a predetermined range (like 0 to 1)

For each neuron in a layer, the activation function is applied:

$$a^{l} = g(z^{l})$$

## 6. Inputs and Outputs

The input data is the first layer of the model and will be noted as $x^{i}$ where $i$ is the training sample. The output data is the last layer of the model and will be noted as $y^{i}$ (the true value) where $i$ is the training sample. The prediction will be noted as $\^y^{i}$ where $i$ is the training sample.



