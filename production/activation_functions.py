import numpy as np
import matplotlib.pyplot as plt
from pybann.activation import identity, sigmoid, tanhyp, gaussian, softplus

def plot_identity(x):
    idt = identity(x)
    didt = identity(x, deriv=True)
    return idt, didt

def plot_sigmoid(x):
    sig = sigmoid(x)
    dsig = sigmoid(x, deriv=True)
    return sig, dsig

def plot_tanhyp(x):
    thp = tanhyp(x)
    dthp = tanhyp(x, deriv=True)
    return thp, dthp

def plot_gaussian(x):
    gss = gaussian(x)
    dgss = gaussian(x, deriv=True)
    return gss, dgss

def plot_softplus(x):
    sps = softplus(x)
    dsps = softplus(x, deriv=True)
    return sps, dsps

x = np.linspace(-6, 6, 1201)

idt, didt = plot_identity(x)
sig, dsig = plot_sigmoid(x)
thp, dthp = plot_tanhyp(x)
gss, dgss = plot_gaussian(x)
sps, dsps = plot_softplus(x)

figsize = (7, 8)
rows = 4
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
ax.plot(x, sigmoid(x), color='blue', linewidth=1)
ax.plot(x, sigmoid(x, deriv=True), color='red', linewidth=1, linestyle='--')

ax = axList[1]
ax.set_ylim(-1.0, 1.5)
ax.set_yticks(np.linspace(-1.0, 1.5, 6))
ax.plot(x, gaussian(x), color='blue', linewidth=1)
ax.plot(x, gaussian(x, deriv=True), color='red', linewidth=1, linestyle='--')

ax = axList[2]
ax.set_ylim(-1.5, 1.5)
ax.set_yticks(np.linspace(-1.5, 1.5, 7))
ax.plot(x, tanhyp(x), color='blue', linewidth=1)
ax.plot(x, tanhyp(x, deriv=True), color='red', linewidth=1, linestyle='--')

ax = axList[3]
ax.set_ylim(-2.0, 6.0)
ax.set_yticks(np.linspace(-2.0, 6.0, 5))
ax.plot(x, softplus(x), color='blue', linewidth=1)
ax.plot(x, softplus(x, deriv=True), color='red', linewidth=1, linestyle='--')

ax = axList[4]
ax.set_ylim(-6.0, 6.0)
ax.set_yticks(np.linspace(-6.0, 6.0, 7))
ax.plot(x, identity(x), color='blue', linewidth=1)
ax.plot(x, identity(x, deriv=True), color='red', linewidth=1, linestyle='--')

# plt.subplot(3, 2, 1)
# plt.xlim(min(x), max(x))
# plt.ylim(-0.5, 1.5)
# plt.xticks(np.linspace(min(x), max(x), 7))
# plt.yticks(np.linspace(-0.5, 1.5, 5))
# plt.grid(which='major', axis='both', color='gray', linestyle=':')
# plt.plot(x, sig, color='blue', linewidth=1)
# plt.plot(x, dsig, color='red', linewidth=1, linestyle='--')

# plt.subplot(3, 2, 2)
# plt.xlim(min(x), max(x))
# plt.plot(x, gss, color='blue', linewidth=1)
# plt.plot(x, dgss, color='red', linewidth=1, linestyle='--')

# plt.subplot(3, 2, 3)
# plt.xlim(min(x), max(x))
# plt.plot(x, sps, color='blue', linewidth=1)
# plt.plot(x, dsps, color='red', linewidth=1, linestyle='--')


# plt.subplot(3, 2, 5)
# plt.xlim(min(x), max(x))
# plt.plot(x, idt, color='blue', linewidth=1)
# plt.plot(x, didt, color='red', linewidth=1, linestyle='--')

# plt.subplot(3, 2, 6)
# plt.xlim(min(x), max(x))
# plt.plot(x, thp, color='blue', linewidth=1)
# plt.plot(x, dthp, color='red', linewidth=1, linestyle='--')

plt.show()