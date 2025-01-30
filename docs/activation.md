# Activation Functions

This document describes the common activation functions used in PyBANN.

## Sigmoid

The sigmoid function maps any input value to a value between 0 and 1.

### Function:

$f(x) = 1 / (1 + e^(-x))$


### Derivative:

$f'(x) = f(x) * (1 - f(x))$


### Characteristics:
- Output range: (0, 1)
- Smooth gradient
- Can cause vanishing gradient problem

## ReLU (Rectified Linear Unit)

ReLU returns 0 for negative inputs and the input value for positive inputs.

### Function:

$f(x) = max(0, x)$


### Derivative:

$f'(x) = 1 if x > 0, else 0$


### Characteristics:
- Output range: [0, ∞)
- Computationally efficient
- Can suffer from "dying ReLU" problem

## Tanh (Hyperbolic Tangent)

Tanh is similar to sigmoid but maps inputs to values between -1 and 1.

### Function:

$f(x) = (e^x - e^(-x)) / (e^x + e^(-x))$


### Derivative:

$f'(x) = 1 - tanh^2(x)$

### Characteristics:
- Output range: (-1, 1)
- Zero-centered
- Can still suffer from vanishing gradient problem

## Softmax

Softmax is often used in the output layer for multi-class classification problems.

### Function:

$f(x_i) = e^(x_i) / Σ(e^(x_j))$

where Σ is the sum over all output nodes.

### Derivative:
The derivative of softmax is more complex as it depends on all outputs. A description is provided here after.

### Characteristics:
- Outputs sum to 1
- Emphasizes the largest values while suppressing lower ones
- Often used with cross-entropy loss


## Softmax Derivative

The derivative of the softmax function is more complex than other activation functions because the output of each unit depends on all the inputs. The Jacobian matrix of the softmax function is given by:

### Formula:
For i = j:
$∂S_i / ∂x_j = S_i * (1 - S_i)$

For i ≠ j:
$∂S_i / ∂x_j = -S_i * S_j$


Where S_i is the i-th output of the softmax function.

### Characteristics:
- The result is a Jacobian matrix, not a vector.
- Each element (i,j) of the Jacobian represents how a change in the j-th input affects the i-th output of the softmax function.
- The diagonal elements represent how each output is affected by its own input.
- The off-diagonal elements represent how each output is affected by other inputs.

### Implementation:
In matrix form, the Jacobian can be computed as:

$J = diag(S) - S * S^T$

Where S is the output of the softmax function, diag(S) is a diagonal matrix with the elements of S on the diagonal, and S^T is the transpose of S.

### Usage:
The softmax derivative is primarily used in the backpropagation algorithm for neural networks, especially in the output layer for multi-class classification problems. It's often combined with cross-entropy loss, which simplifies the gradient calculation.
