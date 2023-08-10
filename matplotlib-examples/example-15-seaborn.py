import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# seaborn's method to set its chart style
sns.set()

# # # histograms, KDE, and densities

data = np.random.multivariate_normal(
    mean=[0, 0],
    cov=[[5, 2], [2, 2]],
    size=2000
)
data = pd.DataFrame(data, columns=['x', 'y'])

# hist in Matplotlib
for col in 'xy':
    plt.hist(data[col], density=True, alpha=0.5)
plt.savefig('../matplotlib-examples-figures/seaborn-01--matplotlib-hist.svg')
plt.close()

# Get a smooth estimate of the distribution using kernel density estimation (KDE)
sns.kdeplot(data=data, fill=True)
plt.savefig('../matplotlib-examples-figures/seaborn-02--kdeplot-1--kde.svg')
plt.close()

# Get a two-dimensional visualization of the joint density
sns.kdeplot(data=data, x='x', y='y')
plt.savefig('../matplotlib-examples-figures/seaborn-03--kdeplot-2--joint-density.svg')
plt.close()

# # # pair plots

# petals and sepals of three Iris species
iris = sns.load_dataset("iris")
iris.head(3)
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa

iris.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 5 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   sepal_length  150 non-null    float64
#  1   sepal_width   150 non-null    float64
#  2   petal_length  150 non-null    float64
#  3   petal_width   150 non-null    float64
#  4   species       150 non-null    object
# dtypes: float64(4), object(1)
# memory usage: 6.0+ KB

# Plot the relationships between four variables:
# sepal_length, sepal_width, petal_length, and petal_width
sns.pairplot(iris, hue='species', height=2.5)
plt.savefig('../matplotlib-examples-figures/seaborn-04--pairplot.svg')
plt.close()

# # # faceted histograms

# Trinkgeld
tips = sns.load_dataset('tips')
tips.head(3)
#    total_bill   tip     sex smoker  day    time  size
# 0       16.99  1.01  Female     No  Sun  Dinner     2
# 1       10.34  1.66    Male     No  Sun  Dinner     3
# 2       21.01  3.50    Male     No  Sun  Dinner     3

tips.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 244 entries, 0 to 243
# Data columns (total 7 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   total_bill  244 non-null    float64
#  1   tip         244 non-null    float64
#  2   sex         244 non-null    category
#  3   smoker      244 non-null    category
#  4   day         244 non-null    category
#  5   time        244 non-null    category
#  6   size        244 non-null    int64
# dtypes: category(4), float64(2), int64(1)
# memory usage: 7.4 KB

tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']
tips.head()
#    total_bill   tip     sex  ...    time size    tip_pct
# 0       16.99  1.01  Female  ...  Dinner    2   5.944673
# 1       10.34  1.66    Male  ...  Dinner    3  16.054159
# 2       21.01  3.50    Male  ...  Dinner    3  16.658734
# 3       23.68  3.31    Male  ...  Dinner    2  13.978041
# 4       24.59  3.61  Female  ...  Dinner    4  14.680765

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15))
plt.savefig('../matplotlib-examples-figures/seaborn-05--FacetGrid.svg')
plt.close()

# # # categorical plots

with sns.axes_style(style='ticks'):
    g = sns.catplot(
        x="day",
        y="total_bill",
        hue="sex",  # Farbe
        data=tips,
        kind="box"
    )
    g.set_axis_labels("Day", "Total Bill")
plt.savefig('../matplotlib-examples-figures/seaborn-06--catplot-1--box.svg')
plt.close()

# # # joint distributions

# Show the joint distribution between different datasets,
# along with the associated marginal distributions
with sns.axes_style('white'):
    g = sns.jointplot(
        x="total_bill",
        y="tip",
        data=tips,
        kind='hex'
    )
    g.set_axis_labels("Total Bill", "Tip")
plt.savefig('../matplotlib-examples-figures/seaborn-07--jointplot-1--hex.svg')
plt.close()

# automatic kernel density estimation and regression
g = sns.jointplot(
    x="total_bill",
    y="tip",
    data=tips,
    kind='reg'
)
g.set_axis_labels("Total Bill", "Tip")
plt.savefig('../matplotlib-examples-figures/seaborn-08--jointplot-2--reg.svg')
plt.close()

# # # bar plots

planets = sns.load_dataset('planets')
planets.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1035 entries, 0 to 1034
# Data columns (total 6 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   method          1035 non-null   object
#  1   number          1035 non-null   int64
#  2   orbital_period  992 non-null    float64
#  3   mass            513 non-null    float64
#  4   distance        808 non-null    float64
#  5   year            1035 non-null   int64
# dtypes: float64(3), int64(2), object(1)
# memory usage: 48.6+ KB

with sns.axes_style('white'):
    g = sns.catplot(
        x="year",
        data=planets,
        aspect=2,
        kind="count",
        color='steelblue'
    )
    g.set_xticklabels(step=5)
plt.savefig('../matplotlib-examples-figures/seaborn-09--catplot-2--count-1.svg')
plt.close()

with sns.axes_style('white'):
    g = sns.catplot(
        x="year",
        data=planets,
        aspect=4.0,
        kind='count',
        hue='method',
        order=range(2001, 2015)
    )
    g.set_ylabels('Number of Planets Discovered')
plt.savefig('../matplotlib-examples-figures/seaborn-10--catplot-3--count-2.svg')
plt.close()





