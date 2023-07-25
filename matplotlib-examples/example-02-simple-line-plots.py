import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.available
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

plt.style.use('seaborn-v0_8-whitegrid')

x = np.linspace(0, 10, 1000)

fig = plt.figure()
ax = plt.axes()

ax.plot(x, np.sin(x))
plt.savefig('../matplotlib-examples-figures/simple-line-plots-01--sin-1.svg')
plt.close()

# alternatively
plt.plot(x, np.sin(x))
plt.savefig('../matplotlib-examples-figures/simple-line-plots-02--sin-2.svg')
plt.close()

# Create a single figure with multiple lines
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.savefig('../matplotlib-examples-figures/simple-line-plots-03--multiple-lines.svg')
plt.close()

# colors and linestyles

# colored lines

# color by name
plt.plot(x, np.sin(x - 0), color='blue')
# short color code (rgbcmyk)
plt.plot(x, np.sin(x - 1), color='g')
# grayscale between 0 and 1
plt.plot(x, np.sin(x - 2), color='0.75')
# hex code (RRGGBB, 00 to FF)
plt.plot(x, np.sin(x - 3), color='#FFDD44')
# RGB tuple, values 0 to 1
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3))
# HTML color names supported
plt.plot(x, np.sin(x - 5), color='chartreuse')
plt.savefig('../matplotlib-examples-figures/simple-line-plots-04--colored-lines.svg')
plt.close()

# linestyles

plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted')
plt.savefig('../matplotlib-examples-figures/simple-line-plots-05--linestyles-1.svg')
plt.close()

# For short, you can use the following codes:
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--')  # dashed
plt.plot(x, x + 6, linestyle='-.')  # dashdot
plt.plot(x, x + 7, linestyle=':')  # dotted
plt.savefig('../matplotlib-examples-figures/simple-line-plots-06--linestyles-2.svg')
plt.close()

# combined the linestyle and color codes

plt.plot(x, x + 8, '-g')  # solid green
plt.plot(x, x + 9, '--c')  # dashed cyan
plt.plot(x, x + 10, '-.k')  # dashdot black
plt.plot(x, x + 11, ':r')  # dotted red
plt.savefig('../matplotlib-examples-figures/simple-line-plots-07--colored-linestyles.svg')
plt.close()

# axes limits

plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)
plt.savefig('../matplotlib-examples-figures/simple-line-plots-08--axes-limits.svg')
plt.close()

plt.plot(x, np.sin(x))
plt.xlim(10, 0)
plt.ylim(1.2, -1.2)
plt.savefig('../matplotlib-examples-figures/simple-line-plots-09--reversed-axis.svg')
plt.close()

plt.plot(x, np.sin(x))
plt.axis('tight')
plt.savefig('../matplotlib-examples-figures/simple-line-plots-10--tight-axis.svg')
plt.close()

plt.plot(x, np.sin(x))
plt.axis('equal')  # equal axis ratio
plt.savefig('../matplotlib-examples-figures/simple-line-plots-11--equal-axis-ratio.svg')
plt.close()

# Other axis options: 'on', 'off', 'square', 'image', and more

# labeling plots

plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("$x$")
plt.ylabel("$\sin(x)$")
plt.savefig('../matplotlib-examples-figures/simple-line-plots-12--labels.svg')
plt.close()

plt.plot(x, np.sin(x), '-g', label='$\sin(x)$')
plt.plot(x, np.cos(x), ':b', label='$\cos(x)$')
plt.axis('equal')
plt.legend()
plt.savefig('../matplotlib-examples-figures/simple-line-plots-13--legend.svg')
plt.close()

# Matplotlib gotchas
# - plt.plot -> ax.plot
# - plt.legend -> ax.legend
# etc., BUT
# - plt.xlabel -> ax.set_xlabel
# - plt.ylabel -> ax.set_ylabel
# - plt.xlim -> ax.set_xlim
# - plt.ylim -> ax.set_ylim
# - plt.title -> ax.set_title

# object-oriented approach

ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(
    xlim=(0, 10),
    ylim=(-2, 2),
    xlabel='$x$',
    ylabel='$\sin(x)$',
    title='A Simple Plot'
)
plt.show()
