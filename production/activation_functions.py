import numpy as np
import matplotlib.pyplot as plt
from pybann import Activation

x = np.linspace(-6, 6, 1201)

figsize = (6, 4)
rows = 3
cols = 2

axs = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)

axList = axs.flatten()

for ax in axList:
    ax.set_xlim(min(x), max(x))
    ax.set_xticks(np.linspace(min(x), max(x), 7))
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks(np.linspace(-1.5, 1.5, 7))
    ax.grid(which='major', axis='both', color='gray', linestyle=':')
    ax.hlines(0, min(x), max(x), linewidth=0.5, color='gray')
    ax.vlines(0,-10, 10, linewidth=0.5, color='gray')

ax = axList[0]
ax.set_ylim(-0.5, 1.5)
ax.set_yticks(np.linspace(-0.5, 1.5, 5))
ax.plot(x, Activation.sigmoid(x), color='blue', linewidth=1)
ax.plot(x, Activation.sigmoid(x, deriv=True), color='red', linewidth=1, linestyle='--')

ax = axList[1]
ax.set_ylim(-1.0, 1.5)
ax.set_yticks(np.linspace(-1.0, 1.5, 6))
ax.plot(x, Activation.gaussian(x), color='blue', linewidth=1)
ax.plot(x, Activation.gaussian(x, deriv=True), color='red', linewidth=1, linestyle='--')

ax = axList[2]
ax.set_ylim(-1.5, 1.5)
ax.set_yticks(np.linspace(-1.5, 1.5, 7))
ax.plot(x, Activation.tanhyp(x), color='blue', linewidth=1)
ax.plot(x, Activation.tanhyp(x, deriv=True), color='red', linewidth=1, linestyle='--')

ax = axList[3]
ax.set_ylim(-2.0, 6.0)
ax.set_yticks(np.linspace(-2.0, 6.0, 5))
ax.plot(x, Activation.softplus(x), color='blue', linewidth=1)
ax.plot(x, Activation.softplus(x, deriv=True), color='red', linewidth=1, linestyle='--')

ax = axList[4]
ax.set_ylim(-1.0, 6.0)
ax.set_yticks(np.linspace(-2.0, 6.0, 5))
ax.plot(x, Activation.relu(x), color='blue', linewidth=1)
ax.plot(x, Activation.relu(x, deriv=True), color='red', linewidth=1, linestyle='--')

ax = axList[5]
ax.set_ylim(-1.0, 6.0)
ax.set_yticks(np.linspace(-2.0, 6.0, 5))
ax.plot(x, Activation.relu(x, a=0.1), color='blue', linewidth=1)
ax.plot(x, Activation.relu(x, a=0.1, deriv=True), color='red', linewidth=1, linestyle='--')


plt.show()