import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cycler

x = np.random.randn(1000)
plt.hist(x)
plt.savefig('../matplotlib-examples-figures/configurations-and-stylesheets-1--hist.svg')
plt.close()

# Adjust by hand
fig = plt.figure(facecolor='white')
ax = plt.axes(facecolor='#E6E6E6')
ax.set_axisbelow(True)

# Draw solid white gridlines
plt.grid(color='w', linestyle='solid')

# Hide axis spines
for spine in ax.spines.values():
    spine.set_visible(False)
# An axis spine -- the line noting the data area boundaries.

# Hide top and right ticks
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# ticks and labels
ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')

# Control face and edge color of histogram
ax.hist(x, edgecolor='#E6E6E6', color='#EE6666')
plt.savefig('../matplotlib-examples-figures/configurations-and-stylesheets-2--hist-customization.svg')
plt.close()

# # # changing the defaults: rcParams

# plt.rc

colors = cycler(
    'color',
    ['#EE6666', '#3388BB', '#9988DD',
     '#EECC55', '#88BB44', '#FFBBBB']
)
plt.rc(
    'figure',
    facecolor='white'
)
plt.rc(
    'axes',
    facecolor='#E6E6E6',
    edgecolor='none',
    axisbelow=True,
    grid=True,
    prop_cycle=colors
)
plt.rc(
    'grid',
    color='w',
    linestyle='solid'
)
plt.rc(
    'xtick',
    direction='out',
    color='gray'
)
plt.rc(
    'ytick',
    direction='out',
    color='gray'
)
plt.rc(
    'patch',
    edgecolor='#E6E6E6',
    force_edgecolor=True
)
plt.rc(
    'lines',
    linewidth=2
)
plt.hist(x)
plt.savefig('../matplotlib-examples-figures/configurations-and-stylesheets-3--rc--hist.svg')
plt.close()

for i in range(4):
    plt.plot(np.random.rand(10))
plt.savefig('../matplotlib-examples-figures/configurations-and-stylesheets-4--rc--lines.svg')
plt.close()

# Settings can be saved in a .matplotlibrc file

# # # stylesheets

# The stylesheets are formatted similarly to the .matplotlibrc,
# but must be named with a .mplstyle extension.

# built-in stylesheets
plt.style.available
# ['Solarize_Light2',
#  '_classic_test_patch',
#  '_mpl-gallery',
#  '_mpl-gallery-nogrid',
#  'bmh',
#  'classic',
#  'dark_background',
#  'fast',
#  'fivethirtyeight',
#  'ggplot',
#  'grayscale',
#  'seaborn-v0_8',
#  'seaborn-v0_8-bright',
#  'seaborn-v0_8-colorblind',
#  'seaborn-v0_8-dark',
#  'seaborn-v0_8-dark-palette',
#  'seaborn-v0_8-darkgrid',
#  'seaborn-v0_8-deep',
#  'seaborn-v0_8-muted',
#  'seaborn-v0_8-notebook',
#  'seaborn-v0_8-paper',
#  'seaborn-v0_8-pastel',
#  'seaborn-v0_8-poster',
#  'seaborn-v0_8-talk',
#  'seaborn-v0_8-ticks',
#  'seaborn-v0_8-white',
#  'seaborn-v0_8-whitegrid',
#  'tableau-colorblind10']

# Use either:
# plt.style.use('<stylename>')
# or in the 'with' context manager:
# with plt.style.context('<stylename>'):
#    # plots


def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))
    # one hist in one ax
    ax[0].hist(np.random.randn(1000))
    # three lines in one ax
    for i in range(3):
        ax[1].plot(np.random.rand(10))
        ax[1].legend(['a', 'b', 'c'], loc='lower left')


with plt.style.context('default'):
    hist_and_lines()
plt.savefig('../matplotlib-examples-figures/configurations-and-stylesheets-5--available-styles-1--default.svg')
plt.close()

# FiveThirtyEight website https://fivethirtyeight.com/
with plt.style.context('fivethirtyeight'):
    hist_and_lines()
plt.savefig('../matplotlib-examples-figures/configurations-and-stylesheets-6--available-styles-2--fivethirtyeight.svg')
plt.close()

# the ggplot package in the R language
with plt.style.context('ggplot'):
    hist_and_lines()
plt.savefig('../matplotlib-examples-figures/configurations-and-stylesheets-7--available-styles-3--ggplot.svg')
plt.close()

# Bayesian Methods for Hackers Style
with plt.style.context('bmh'):
    hist_and_lines()
plt.savefig('../matplotlib-examples-figures/configurations-and-stylesheets-8--available-styles-4--bmh.svg')
plt.close()

with plt.style.context('dark_background'):
    hist_and_lines()
plt.savefig('../matplotlib-examples-figures/configurations-and-stylesheets-9--available-styles-5--dark_background.svg')
plt.close()

with plt.style.context('grayscale'):
    hist_and_lines()
plt.savefig('../matplotlib-examples-figures/configurations-and-stylesheets-10--available-styles-6--grayscale.svg')
plt.close()

with plt.style.context('seaborn-v0_8'):
    hist_and_lines()
plt.savefig('../matplotlib-examples-figures/configurations-and-stylesheets-11--available-styles-7--seaborn-v0_8.svg')
plt.close()
