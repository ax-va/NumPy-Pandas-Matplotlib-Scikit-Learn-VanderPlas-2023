import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.style.use('seaborn-v0_8-whitegrid')

x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])
plt.imshow(I)
plt.colorbar()
plt.savefig('../matplotlib-examples-figures/colorbars-1.svg')
plt.close()

plt.imshow(I, cmap='Blues')
plt.colorbar()
plt.savefig('../matplotlib-examples-figures/colorbars-2--cmap.svg')
plt.close()

# choosing the colormap


def grayscale_cmap(cmap):
    """ Return a grayscale version of the given colormap. """
    cmap = matplotlib.colormaps.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    # Convert RGBA to perceived grayscale luminance
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(colors[:, :3] ** 2 @ RGB_weight)
    colors[:, :3] = luminance[:, np.newaxis]
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)


def view_colormap(cmap):
    """ Plot a colormap with its grayscale equivalent. """
    cmap = matplotlib.colormaps.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))
    fig, ax = plt.subplots(
        2,
        figsize=(6, 2),
        subplot_kw=dict(xticks=[], yticks=[])
    )
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])


# View the jet colormap and its uneven luminance scale
view_colormap('jet')
plt.savefig('../matplotlib-examples-figures/colorbars-3--jet-colormap-with-uneven-luminance-scale.svg')
plt.close()

# View the viridis colormap and its even luminance scale
view_colormap('viridis')
plt.savefig('../matplotlib-examples-figures/colorbars-4--viridis-colormap-with-even-luminance-scale.svg')
plt.close()

view_colormap('RdBu')
plt.savefig('../matplotlib-examples-figures/colorbars-5--RdBu-colormap-with-luminance-scale.svg')
plt.close()
# The positive/negative information will be lost upon translation to grayscale

# color limits and extensions

# Make noise in 1% of the image pixels
speckles = (np.random.random(I.shape) < 0.01)
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))

plt.figure(figsize=(10, 3.5))
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='RdBu')
plt.colorbar()

# Use "extend" to see the noisy pixels.
plt.subplot(1, 2, 2)
plt.imshow(I, cmap='RdBu')
# Indicate values that are above or below those limits.
plt.colorbar(extend='both')
plt.clim(-1, 1)

plt.savefig('../matplotlib-examples-figures/colorbars-6--extend.svg')
plt.close()

# discrete colorbars

# Pass a number of desired bins
cmap = matplotlib.colormaps.get_cmap("Blues")
colors = cmap(np.arange(cmap.N))
cmap = LinearSegmentedColormap.from_list("Blues", colors, N=6)
plt.imshow(I, cmap=cmap)
plt.colorbar(extend='both')
plt.clim(-1, 1)
plt.savefig('../matplotlib-examples-figures/colorbars-7--discrete-colorbar.svg')
plt.close()
