import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# data from: https://github.com/jakevdp/marathon-data
data = pd.read_csv('../matplotlib-examples-data/marathon-data.csv')
data.head()
#    age gender     split     final
# 0   33      M  01:05:38  02:08:51
# 1   32      M  01:06:26  02:09:28
# 2   31      M  01:06:49  02:10:42
# 3   38      M  01:06:16  02:13:45
# 4   31      M  01:06:32  02:13:59

# The distance is splitted in two halves: "split" and "final"

data.dtypes


# age        int64
# gender    object
# split     object
# final     object
# dtype: object


def convert_time(s):
    h, m, s = map(int, s.split(':'))
    return datetime.timedelta(hours=h, minutes=m, seconds=s)


data = pd.read_csv(
    '../matplotlib-examples-data/marathon-data.csv',
    converters={
        'split': convert_time,
        'final': convert_time
    }
)
data.head()
#    age gender           split           final
# 0   33      M 0 days 01:05:38 0 days 02:08:51
# 1   32      M 0 days 01:06:26 0 days 02:09:28
# 2   31      M 0 days 01:06:49 0 days 02:10:42
# 3   38      M 0 days 01:06:16 0 days 02:13:45
# 4   31      M 0 days 01:06:32 0 days 02:13:59

data.dtypes
# age                 int64
# gender             object
# split     timedelta64[ns]
# final     timedelta64[ns]
# dtype: object

# Get the times in seconds in temporal data
data['split_sec'] = data['split'].view(int) / 1E9
data['final_sec'] = data['final'].view(int) / 1E9
data.head()
#    age gender  ... split_sec final_sec
# 0   33      M  ...    3938.0    7731.0
# 1   32      M  ...    3986.0    7768.0
# 2   31      M  ...    4009.0    7842.0
# 3   38      M  ...    3976.0    8025.0
# 4   31      M  ...    3992.0    8039.0
#
# [5 rows x 6 columns]

# Set seaborn's chart style
sns.set()

# jointplot over the data
with sns.axes_style('white'):
    g = sns.jointplot(
        x='split_sec',
        y='final_sec',
        data=data,
        kind='hex',
    )
    g.ax_joint.plot(
        np.linspace(4000, 14000),
        np.linspace(8000, 28000),
        ':k'
    )
plt.tight_layout()
plt.savefig('../matplotlib-examples-figures/marathon-finishing-times-1--joint-distribution.svg')
plt.close()

data['split_frac'] = 1 - 2 * data['split_sec'] / data['final_sec']
data.head()
#    age gender  ... final_sec split_frac
# 0   33      M  ...    7731.0  -0.018756
# 1   32      M  ...    7768.0  -0.026262
# 2   31      M  ...    7842.0  -0.022443
# 3   38      M  ...    8025.0   0.009097
# 4   31      M  ...    8039.0   0.006842
#
# [5 rows x 7 columns]

sns.displot(data['split_frac'], kde=False)
plt.axvline(0, color="k", linestyle="--")
plt.savefig('../matplotlib-examples-figures/marathon-finishing-times-2--split-fraction.svg')
plt.close()
# 0.0 indicates a runner who completed the first and second halves in identical times

sum(data.split_frac < 0), sum(data.split_frac > 0)
# (251, 36995)

# correlation between the split fractions and other variables
g = sns.PairGrid(
    data,
    vars=['age', 'split_sec', 'final_sec', 'split_frac'],
    hue='gender',
    palette='RdBu_r'
)
g.map(plt.scatter, alpha=0.8)
g.add_legend()
plt.savefig('../matplotlib-examples-figures/marathon-finishing-times-3--correlations.png')
plt.close()
# Faster runners tend to have closer to even splits on their marathon time (i.e. the final time)

# split fractions by gender
sns.kdeplot(data.split_frac[data.gender == 'M'], label='men', fill=True)
sns.kdeplot(data.split_frac[data.gender == 'W'], label='women', fill=True)
plt.xlabel('split_frac')
plt.legend()
plt.savefig('../matplotlib-examples-figures/marathon-finishing-times-4--split-fractions-by-gender.svg')
plt.close()
# There are many more men than women who are running close to an even split

# the split fraction by gender
sns.violinplot(
    x="gender",
    y="split_frac",
    data=data,
    palette=["lightblue", "lightpink"]
)
plt.savefig('../matplotlib-examples-figures/marathon-finishing-times-5--violinplot-1.svg')
plt.close()

data['age_dec'] = data.age.map(lambda age: 10 * (age // 10))
data.head()
#    age gender  ... split_frac age_dec
# 0   33      M  ...  -0.018756      30
# 1   32      M  ...  -0.026262      30
# 2   31      M  ...  -0.022443      30
# 3   38      M  ...   0.009097      30
# 4   31      M  ...   0.006842      30
#
# [5 rows x 8 columns]

# the split fraction by gender and age
with sns.axes_style(style=None):
    sns.violinplot(
        x="age_dec",
        y="split_frac",
        hue="gender",
        data=data,
        split=True,
        inner="quartile",
        palette=["lightblue", "lightpink"]
    )
plt.savefig('../matplotlib-examples-figures/marathon-finishing-times-6--violinplot-2.svg')
plt.close()
# The split distributions of men in their 20s to 50s show
# a pronounced overdensity toward lower splits
# when compared to women of the same age.

# the small effect for 80-year-old women
(data.age > 80).sum()
# 7

len(data.query("age > 80 & gender == 'M'"))
# 5
len(data.query("age > 80 & gender == 'W'"))
# 2

# Fit a linear regression model to the data
g = sns.lmplot(
    x='final_sec',
    y='split_frac',
    col='gender',
    data=data,
    markers=".",
    scatter_kws=dict(color='c')
)
g.map(plt.axhline, y=0.0, color="k", ls=":")
plt.savefig('../matplotlib-examples-figures/marathon-finishing-times-7--linear-regression.png')
plt.close()
