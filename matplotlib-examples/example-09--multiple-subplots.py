import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

# plt.axes: subplots by hand

# standard axes
ax1 = plt.axes()
# top-right corner
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])  # [left, bottom, width, height]
plt.savefig('../matplotlib-examples-figures/multiple-subplots-1--axes.svg')
plt.close()

# object-oriented interface by fig.add_axes
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], ylim=(-1.2, 1.2))
# 0.5 (bottom of ax1) = 0.1 (bottom of ax2) + 0.4 (height of ax2)
x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))
plt.savefig('../matplotlib-examples-figures/multiple-subplots-2--add_axes.svg')
plt.close()

# plt.subplot: simple grids of subplots

for i in range(1, 7):
    # args: # number of rows, number of columns, index from the upper left to the bottom right
    plt.subplot(2, 3, i)  # one-based indexing
    plt.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')  # ha = horizontal alignment
plt.savefig('../matplotlib-examples-figures/multiple-subplots-3--subplot.svg')
plt.close()

fig = plt.figure()
# hspace and wspace specify the spacing along the height and width of the figure,
# in units of the subplot size
fig.subplots_adjust(hspace=0.75, wspace=0.25)
for i in range(1, 7):
    # args: # number of rows, number of columns, index from the upper left to the bottom right
    ax = fig.add_subplot(2, 3, i)  # one-based indexing
    ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')  # ha = horizontal alignment
plt.savefig('../matplotlib-examples-figures/multiple-subplots-4--subplots_adjust--add_subplot.svg')
plt.close()

# plt.subplots: a full grid of subplots in a single line

# Create a 2x3 grid of subplots, where all axes in the same row share their y-axis scale,
# and all axes in the same column share their x-axis scale.

fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
# axes are in a two-dimensional array, indexed by [row, col]
for i in range(2):
    for j in range(3):
        # ax is a NumPy array
        ax[i, j].text(0.5, 0.5, str((i, j)), fontsize=18, ha='center')  # zero-based indexing
plt.savefig('../matplotlib-examples-figures/multiple-subplots-5--subplots.svg')
plt.close()

# plt.GridSpec: more complicated arrangements

grid = plt.GridSpec(nrows=2, ncols=3, wspace=0.75, hspace=0.25)
plt.subplot(grid[0, 0])  # top left
plt.subplot(grid[0, 1:])  # top right
plt.subplot(grid[1, :2])  # bottom left
plt.subplot(grid[1, 2])  # bottom right
plt.savefig('../matplotlib-examples-figures/multiple-subplots-6--GridSpec.svg')
plt.close()

# multiaxes histogram plot with GridSpec, add_subplot, hist

# Create some normally distributed data
mean = [0, 0]
cov = [[1, 1], [1, 2]]
rng = np.random.default_rng(1701)
x, y = rng.multivariate_normal(mean, cov, 3000).T

# Set up the axes with GridSpec
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(nrows=4, ncols=4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# Scatter points on the main axes
main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)

# Histogram on the attached axes
x_hist.hist(x, 40, histtype='stepfilled', orientation='vertical', color='gray')
x_hist.invert_yaxis()  # Flip the orientation
y_hist.hist(y, 40, histtype='stepfilled', orientation='horizontal', color='gray')
y_hist.invert_xaxis()  # Flip the orientation
plt.savefig('../matplotlib-examples-figures/multiple-subplots-7--multiaxes-histogram-plot.svg')
plt.close()
