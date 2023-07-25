import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)

fig = plt.figure()
ax = plt.axes()

ax.plot(x, np.sin(x))
plt.savefig('../matplotlib-examples-figures/simple-line-plots--sin-1.svg')
plt.close()

# alternatively
plt.plot(x, np.sin(x))
plt.savefig('../matplotlib-examples-figures/simple-line-plots--sin-2.svg')
plt.close()

# Create a single figure with multiple lines
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.savefig('../matplotlib-examples-figures/simple-line-plots--multiple-lines.svg')
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
plt.savefig('../matplotlib-examples-figures/simple-line-plots--colored-lines.svg')
plt.close()

# linestyles

plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted')
plt.savefig('../matplotlib-examples-figures/simple-line-plots--linestyles-1.svg')
plt.close()

# For short, you can use the following codes:
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--')  # dashed
plt.plot(x, x + 6, linestyle='-.')  # dashdot
plt.plot(x, x + 7, linestyle=':')  # dotted
plt.savefig('../matplotlib-examples-figures/simple-line-plots--linestyles-2.svg')
plt.close()

# combined the linestyle and color codes

plt.plot(x, x + 8, '-g')  # solid green
plt.plot(x, x + 9, '--c')  # dashed cyan
plt.plot(x, x + 10, '-.k')  # dashdot black
plt.plot(x, x + 11, ':r')  # dotted red
plt.savefig('../matplotlib-examples-figures/simple-line-plots--colored-linestyles.svg')
plt.close()

# axes limits

plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)
plt.savefig('../matplotlib-examples-figures/simple-line-plots--axes-limits.svg')
plt.close()

plt.plot(x, np.sin(x))
plt.xlim(10, 0)
plt.ylim(1.2, -1.2)
plt.savefig('../matplotlib-examples-figures/simple-line-plots--reversed-axis.svg')
plt.close()

plt.plot(x, np.sin(x))
plt.axis('tight')
plt.savefig('../matplotlib-examples-figures/simple-line-plots--tight-axis.svg')
plt.close()

plt.plot(x, np.sin(x))
plt.axis('equal')  # equal axis ratio
plt.savefig('../matplotlib-examples-figures/simple-line-plots--equal-axis-ratio.svg')
plt.close()

# Other axis options: 'on', 'off', 'square', 'image', and more

# labeling plots

plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("$x$")
plt.ylabel("$\sin(x)$")
plt.savefig('../matplotlib-examples-figures/simple-line-plots--labels.svg')
plt.close()

plt.plot(x, np.sin(x), '-g', label='$\sin(x)$')
plt.plot(x, np.cos(x), ':b', label='$\cos(x)$')
plt.axis('equal')
plt.legend()
plt.savefig('../matplotlib-examples-figures/simple-line-plots--legend.svg')
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
