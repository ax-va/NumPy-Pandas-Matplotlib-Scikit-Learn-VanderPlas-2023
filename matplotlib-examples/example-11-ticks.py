import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

# Each Figure can contain one or more Axes instances.
# Each axes instance has attributes xaxis and yaxis
# that have a locator and formatter.

# # # major and minor tickmarks

# logarithmic plot
ax = plt.axes(xscale='log', yscale='log')
ax.set(xlim=(1, 1e3), ylim=(1, 1e3))
ax.grid(True, linestyle=':')

ax.xaxis.get_major_locator()
# <matplotlib.ticker.LogLocator at 0x7f8025cff790>
ax.xaxis.get_minor_locator()
# <matplotlib.ticker.LogLocator at 0x7f8025cfec20>
ax.xaxis.get_major_formatter()
# <matplotlib.ticker.LogFormatterSciNotation at 0x7f8025cff430>
ax.xaxis.get_minor_formatter()
# <matplotlib.ticker.LogFormatterSciNotation at 0x7f8025cfef20>

plt.savefig('../matplotlib-examples-figures/ticks-1--logarithmic-plot.svg')
plt.close()

# # # hiding ticks or labels

# plt.NullLocator and plt.NullFormatter
ax = plt.axes()
rng = np.random.default_rng(42)
ax.plot(rng.random(50))
ax.grid(True, linestyle=':')
# Formatter: remove the labels (but keep the ticks/gridlines) from the x-axis
ax.xaxis.set_major_formatter(plt.NullFormatter())
# Locator: remove the ticks (and thus, the labels and gridlines) from the y-axis
ax.yaxis.set_major_locator(plt.NullLocator())

plt.savefig('../matplotlib-examples-figures/ticks-2--NullFormatter--NullLocator.svg')
plt.close()

# Show different faces without ticks
nrows = 5
ncols = 5
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 6))
fig.subplots_adjust(hspace=0, wspace=0)
# Get some face data from Scikit-Learn
faces = fetch_olivetti_faces().images
for i in range(nrows):
    for j in range(ncols):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        ax[i, j].imshow(faces[10 * i + j], cmap='binary_r')
plt.savefig('../matplotlib-examples-figures/ticks-3--faces-without-ticks.svg')
plt.close()

# # # reducing or increasing the number of ticks

# plt.MaxNLocator specifies the maximum number of ticks
fig, ax = plt.subplots(4, 4, sharex="all", sharey="all")
# For every axis, set the x and y major locator
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(3))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))
plt.savefig('../matplotlib-examples-figures/ticks-4--MaxNLocator.svg')
plt.close()

# # # fancy tick formats

# Plot a sine and cosine curve
fig, ax = plt.subplots()
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=2, label='Sine')
ax.plot(x, np.cos(x), lw=2, label='Cosine')
# Set up grid, legend, and limits
ax.grid(True, linestyle=':')
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(-1.5 * np.pi, 1.5 * np.pi)
# Space the ticks and gridlines in multiples of pi with plt.MultipleLocator
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
plt.savefig('../matplotlib-examples-figures/ticks-5--MultipleLocator.svg')
plt.close()

# Use plt.FuncFormatter to replace numbers with symbols


def format_func(value, tick_number):
    # find number of multiples of pi/2
    n = int(np.round(2 * value / np.pi))
    if n == 0:
        return "0"
    elif n == 1:
        return r"$\pi/2$"
    elif n == -1:
        return r"$-\pi/2$"
    elif n == 2:
        return r"$\pi$"
    elif n == -2:
        return r"$-\pi$"
    elif n % 2 != 0:
        return rf"${n}\pi/2$"
    else:
        return rf"${n // 2}\pi$"


fig, ax = plt.subplots()
x = np.linspace(-3 * np.pi, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=2, label='Sine')
ax.plot(x, np.cos(x), lw=2, label='Cosine')
# Set up grid, legend, and limits
ax.grid(True, linestyle=':')
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(-2 * np.pi, 2 * np.pi)
# Space the ticks and gridlines in multiples of pi with plt.MultipleLocator
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.savefig('../matplotlib-examples-figures/ticks-6--FuncFormatter.svg')
plt.close()

# Locator class         Description

# NullLocator           No ticks
# FixedLocator          Tick locations are fixed
# IndexLocator          Locator for index plots (e.g., where x = range(len(y)))
# LinearLocator         Evenly spaced ticks from min to max
# LogLocator            Logarithmically spaced ticks from min to max
# MultipleLocator       Ticks and range are a multiple of base
# MaxNLocator           Finds up to a max number of ticks at nice locations
# AutoLocator           (Default) MaxNLocator with simple defaults
# AutoMinorLocator      Locator for minor ticks


# Formatter class       Description

# NullFormatter         No labels on the ticks
# IndexFormatter        Set the strings from a list of labels
# FixedFormatter        Set the strings manually for the labels
# FuncFormatter         User-defined function sets the labels
# FormatStrFormatter    Use a format string for each value
# ScalarFormatter       Default formatter for scalar values
# LogFormatter          Default formatter for log axes
