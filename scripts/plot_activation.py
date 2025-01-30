import sys
import numpy as np
import matplotlib.pyplot as plt
from pybann.activation import Activation

# Ensure valid command-line arguments
if len(sys.argv) < 2:
    print("Usage: python plot_activation.py <activation_function> [save]")
    sys.exit(1)

# Define activation function
activation_name = sys.argv[1].lower()
x = np.linspace(-6, 6, 1201)

# Set up figure with publication-quality style
plt.rcParams.update({
    "font.family": "serif",  # Use serif fonts like Times New Roman
    "font.size": 9,          # Default font size
    "axes.titlesize": 10,    # Title size
    "axes.labelsize": 9,     # Axis label size
    "xtick.labelsize": 8,    # X-axis tick size
    "ytick.labelsize": 8,    # Y-axis tick size
    "legend.fontsize": 8,    # Legend size
    "lines.linewidth": 1.2,  # Standard line width
})

# Set up figure
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)  # Publication quality

# Plot activation functions
if activation_name == 'sigmoid':
    y = Activation.sigmoid(x)
    y_deriv = Activation.sigmoid_derivative(x)
    title = "Sigmoid Activation Function"

elif activation_name == 'relu':
    y = Activation.relu(x)
    y_deriv = Activation.relu_derivative(x)
    title = "ReLU Activation Function"

elif activation_name == 'tanh':
    y = Activation.tanh(x)
    y_deriv = Activation.tanh_derivative(x)
    title = "Tanh Activation Function"

elif activation_name == 'softmax':
    y = Activation.softmax(x)
    y_deriv = Activation.softmax_derivative(x)
    title = "Softmax Activation Function"

else:
    print("Invalid activation function. Choose from: sigmoid, relu, tanh, softmax")
    sys.exit(1)

# Plot primary function (activation)
ax.plot(x, y, color='black', linewidth=1.5, label=f"{activation_name.capitalize()}")

# Plot derivative
ax.plot(x, y_deriv, color='gray', linewidth=1, linestyle='--', label=f"{activation_name.capitalize()} Derivative")

# Formatting
ax.set_title(title, fontsize=10, fontweight="bold")
ax.set_xlabel("Input (x)", fontsize=9)
ax.set_ylabel("Output", fontsize=9)
ax.legend(fontsize=8, loc="best", frameon=False)  # Legend without border
ax.grid(True, linestyle="--", alpha=0.5)  # Subtle grid lines
ax.spines["top"].set_visible(False)  # Remove top border
ax.spines["right"].set_visible(False)  # Remove right border

# Adjust axis ticks
ax.xaxis.set_major_locator(plt.MultipleLocator(2))  # Major ticks every 2 units
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))  # Minor ticks every 1 unit
ax.tick_params(axis="both", which="major", labelsize=8, width=0.8)

# Show plot or save as PNG
if len(sys.argv) > 2 and sys.argv[2].lower() == "save":
    filename = f"{activation_name}_plot.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Plot saved as {filename}")
else:
    plt.show()
