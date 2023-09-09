from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid')

x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend()
plt.savefig('../matplotlib-examples-figures/legends-1.svg')

ax.legend(loc='upper left', frameon=True)
plt.savefig('../matplotlib-examples-figures/legends-2--upper-left.svg')

# Specify the number of columns in the legend by ncol
ax.legend(loc='lower center', ncol=2)
plt.savefig('../matplotlib-examples-figures/legends-3--lower-center.svg')

# rounded box (fancybox),
# shadow,
# frame transparency (alpha value),
# padding around the text
ax.legend(
    frameon=True,
    fancybox=True,
    framealpha=0.5,
    shadow=True,
    borderpad=5
)
plt.savefig('../matplotlib-examples-figures/legends-4--more-attributes.svg')
plt.close()

# Choose elements for the legend

x[:, np.newaxis]  # offsets
# array([[ 0.        ],
#        [ 0.01001001],
#        [ 0.02002002],
# ...
#        [ 9.97997998],
#        [ 9.98998999],
#        [10.        ]])

x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5)
# array([[0.00000000e+00, 1.57079633e+00, 3.14159265e+00, 4.71238898e+00],
#        [1.00100100e-02, 1.58080634e+00, 3.15160266e+00, 4.72239899e+00],
#        [2.00200200e-02, 1.59081635e+00, 3.16161267e+00, 4.73240900e+00],
#        ...,
#        [9.97997998e+00, 1.15507763e+01, 1.31215726e+01, 1.46923690e+01],
#        [9.98998999e+00, 1.15607863e+01, 1.31315826e+01, 1.47023790e+01],
#        [1.00000000e+01, 1.15707963e+01, 1.31415927e+01, 1.47123890e+01]])

y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
lines = plt.plot(x, y)

# "lines" is a list of plt.Line2D instances
plt.legend(lines[:2], ['first', 'second'], frameon=True)
plt.savefig('../matplotlib-examples-figures/legends-5.svg')
plt.close()

# The legend ignores all elements without a label attribute set
plt.plot(x, y[:, 0], label='first')
plt.plot(x, y[:, 1], label='second')
plt.plot(x, y[:, 2:])
plt.plot(x, y[:, 3:], label="forth")
plt.legend(frameon=True)
plt.savefig('../matplotlib-examples-figures/legends-6.svg')
plt.close()

# Legend for size of points

# data from:
# https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks_v1/data/california_cities.csv
cities = pd.read_csv('../matplotlib-examples-data/california_cities.csv')

# Extract the data we're interested in
latitude, longitude = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

# Scatter the points, using size and color but no label
plt.scatter(
    longitude, latitude,
    label=None,
    c=np.log10(population),
    cmap='viridis',
    s=area,
    linewidth=0,
    alpha=0.5
)
plt.axis('equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

# Create a legend:
# plot empty lists with the desired size and label
for area in [10, 50, 100, 500, 1000]:
    plt.scatter(
        [], [],
        c='k',
        alpha=0.3,
        s=area,
        label=str(area) + ' km$^2$'
    )
plt.legend(
    scatterpoints=1,
    frameon=False,
    labelspacing=2,
    title='City Area'
)
plt.title('California Cities: Area and Population')
plt.savefig('../matplotlib-examples-figures/legends-7--point-size-in-legend.svg')
plt.close()

# multiple legends using the Artist class

fig, ax = plt.subplots()
lines = []
styles = ['-', '--', '-.', ':']

x = np.linspace(0, 10, 1000)

for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2), styles[i], color='black')
ax.axis('equal')
# Specify the lines and labels of the first legend
ax.legend(lines[:2], ['line A', 'line B'], loc='upper right')
# Create the second legend and add the artist manually
leg = Legend(ax, lines[2:], ['line C', 'line D'], loc='lower right')
ax.add_artist(leg)
plt.savefig('../matplotlib-examples-figures/legends-8--multiple-legends.svg')
plt.close()
