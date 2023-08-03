import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from datetime import datetime

plt.style.use('seaborn-v0_8-whitegrid')

# motivation: effect of holidays on US births

# data on births in the US, provided by the Centers for Disease Control (CDC)
births = pd.read_csv('../pandas-examples-data/births.csv')
quartiles = np.percentile(births['births'], [25, 50, 75])
# array([4358. , 4814. , 5289.5])
mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
# (4814.0, 689.31)
# Drop not plausible values
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
#        year  month  ...  gender births
# 0      1969      1  ...       F   4046
# 1      1969      1  ...       M   4440
# 2      1969      1  ...       F   4454
# 3      1969      1  ...       M   4548
# 4      1969      1  ...       F   4548
# ...     ...    ...  ...     ...    ...
# 15062  1988     12  ...       M   5944
# 15063  1988     12  ...       F   5742
# 15064  1988     12  ...       M   6095
# 15065  1988     12  ...       F   4435
# 15066  1988     12  ...       M   4698
#
# [14610 rows x 5 columns]

births.info()
# <class 'pandas.core.frame.DataFrame'>
# Index: 14610 entries, 0 to 15066
# Data columns (total 5 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   year    14610 non-null  int64
#  1   month   14610 non-null  int64
#  2   day     14610 non-null  float64
#  3   gender  14610 non-null  object
#  4   births  14610 non-null  int64
# dtypes: float64(1), int64(3), object(1)
# memory usage: 684.8+ KB

births['day'] = births['day'].astype(int)
births.info()
# <class 'pandas.core.frame.DataFrame'>
# Index: 14610 entries, 0 to 15066
# Data columns (total 5 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   year    14610 non-null  int64
#  1   month   14610 non-null  int64
#  2   day     14610 non-null  int64
#  3   gender  14610 non-null  object
#  4   births  14610 non-null  int64
# dtypes: int64(4), object(1)
# memory usage: 684.8+ KB

births.index = pd.to_datetime(
    10000 * births.year + 100 * births.month + births.day,
    format='%Y%m%d'
)
#             year  month  day gender  births
# 1969-01-01  1969      1    1      F    4046
# 1969-01-01  1969      1    1      M    4440
# 1969-01-02  1969      1    2      F    4454
# 1969-01-02  1969      1    2      M    4548
# 1969-01-03  1969      1    3      F    4548
# ...          ...    ...  ...    ...     ...
# 1988-12-29  1988     12   29      M    5944
# 1988-12-30  1988     12   30      F    5742
# 1988-12-30  1988     12   30      M    6095
# 1988-12-31  1988     12   31      F    4435
# 1988-12-31  1988     12   31      M    4698
#
# [14610 rows x 5 columns]

births_by_date = births.pivot_table(values='births', index=[births.index.month, births.index.day])
#          births
# 1  1   4009.225
#    2   4247.400
#    3   4500.900
#    4   4571.350
#    5   4603.625
# ...         ...
# 12 27  4850.150
#    28  5044.200
#    29  5120.150
#    30  5172.350
#    31  4859.200
#
# [366 rows x 1 columns]

births_by_date.index = [datetime(2012, month, day) for (month, day) in births_by_date.index]
#               births
# 2012-01-01  4009.225
# 2012-01-02  4247.400
# 2012-01-03  4500.900
# 2012-01-04  4571.350
# 2012-01-05  4603.625
# ...              ...
# 2012-12-27  4850.150
# 2012-12-28  5044.200
# 2012-12-29  5120.150
# 2012-12-30  5172.350
# 2012-12-31  4859.200
#
# [366 rows x 1 columns]

fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
plt.savefig('../matplotlib-examples-figures/text--annotation--arrows-1--motivation-1.svg')
plt.close()

# Use plt.text or ax.text
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

# Add labels to the plot
style = dict(size=10, color='gray')
ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)  # ha = horizontal alignment
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)

# Label the axes
ax.set(
    title='USA births by day of year (1969-1988)',
    ylabel='average daily births'
)

# Format the x-axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))
plt.savefig('../matplotlib-examples-figures/text--annotation--arrows-2--motivation-2.svg')
plt.close()
# See plt.text and mpl.text.Text for more information

# transforms and text position

# ax.transData      Transform associated with data coordinates
# ax.transAxes      Transform associated with the axes (in units of axes dimensions: from 0 to 1)
# fig.transFigure   Transform associated with the figure (in units of figure dimensions: : from 0 to 1)

fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])

# transform=ax.transData is the default
ax.text(4, 6, "Data", transform=ax.transData)
ax.text(0, 0, "Axes 1", transform=ax.transAxes)
ax.text(0.4, 0.4, "Axes 2", transform=ax.transAxes)
ax.text(1, 1, "Axes 3", transform=ax.transAxes)
ax.text(0, 0, "Figure 1", transform=fig.transFigure)
ax.text(0.5, 0.98, "Figure 2", transform=fig.transFigure)
plt.savefig('../matplotlib-examples-figures/text--annotation--arrows-3--text-position-1.svg')
plt.close()

fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])

ax.set_xlim(0, 2)
ax.set_ylim(-6, 6)
ax.text(2, 2, "Data", transform=ax.transData)
ax.text(0, 0, "Axes 1", transform=ax.transAxes)
ax.text(0.4, 0.4, "Axes 2", transform=ax.transAxes)
ax.text(1, 1, "Axes 3", transform=ax.transAxes)
ax.text(0, 0, "Figure 1", transform=fig.transFigure)
ax.text(0.5, 0.98, "Figure 2", transform=fig.transFigure)
plt.savefig('../matplotlib-examples-figures/text--annotation--arrows-3--text-position-2.svg')
plt.close()

# arrows and annotation

fig, ax = plt.subplots()
x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')
ax.annotate(
    'local maximum',
    xy=(6.28, 1),
    xytext=(10, 4),
    arrowprops=dict(
        facecolor='black',
        shrink=0.05
    )
)
ax.annotate(
    'local minimum',
    xy=(5 * np.pi, -1),
    xytext=(2, -6),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="angle3,angleA=0,angleB=-90"
    )
)
plt.savefig('../matplotlib-examples-figures/text--annotation--arrows-4--arrows-1.svg')
plt.close()

# birthrate plot
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

# Add labels to the plot
ax.annotate(
    "New Year's Day",
    xy=('2012-1-1', 4100),
    xycoords='data',
    xytext=(50, -30),
    textcoords='offset points',
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=-0.2"
    )
)
ax.annotate(
    "Independence Day",
    xy=('2012-7-4', 4250),
    xycoords='data',
    bbox=dict(boxstyle="round", fc="none", ec="gray"),
    xytext=(10, -40),
    textcoords='offset points',
    ha='center',
    arrowprops=dict(arrowstyle="->")
)
ax.annotate(
    'Labor Day Weekend',
    xy=('2012-9-4', 4850),
    xycoords='data',
    ha='center',
    xytext=(0, -20),
    textcoords='offset points'
)
ax.annotate(
    '',
    xy=('2012-9-1', 4850),
    xytext=('2012-9-7', 4850),
    xycoords='data',
    textcoords='data',
    arrowprops={
        'arrowstyle': '|-|,widthA=0.2,widthB=0.2',
    }
)
ax.annotate(
    'Halloween',
    xy=('2012-10-31', 4600),
    xycoords='data',
    xytext=(-80, -40),
    textcoords='offset points',
    arrowprops=dict(
        arrowstyle="fancy",
        fc="0.6",
        ec="none",
        connectionstyle="angle3,angleA=0,angleB=-90"
    )
)
ax.annotate(
    'Thanksgiving',
    xy=('2012-11-25', 4500),
    xycoords='data',
    xytext=(-120, -60),
    textcoords='offset points',
    bbox=dict(
        boxstyle="round4,pad=.5",
        fc="0.9"
    ),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=80,rad=20"
    )
)
ax.annotate(
    'Christmas',
    xy=('2012-12-25', 3850),
    xycoords='data',
    xytext=(-30, 0),
    textcoords='offset points',
    size=13,
    ha='right',
    va="center",
    bbox=dict(
        boxstyle="round",
        alpha=0.1
    ),
    arrowprops=dict(
        arrowstyle="wedge,tail_width=0.5",
        alpha=0.1
    )
)
# Label the axes
ax.set(
    title='USA births by day of year (1969-1988)',
    ylabel='average daily births'
)
# Format the x-axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))
ax.set_ylim(3600, 5400)
plt.savefig('../matplotlib-examples-figures/text--annotation--arrows-4--arrows-2.svg')
plt.close()
