from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data on births in the US, provided by the Centers for Disease Control (CDC)
births = pd.read_csv('../pandas-examples-data/births.csv')
births.head()
#    year  month  day gender  births
# 0  1969      1  1.0      F    4046
# 1  1969      1  1.0      M    4440
# 2  1969      1  2.0      F    4454
# 3  1969      1  2.0      M    4548
# 4  1969      1  3.0      F    4548
births.shape  # (15547, 5)

births['decade'] = 10 * (births['year'] // 10)
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')
# gender         F         M
# decade
# 1960     1753634   1846572
# 1970    16263075  17121550
# 1980    18310351  19243452
# 1990    19479454  20420553
# 2000    18229309  19106428

births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()
plt.ylabel('total births per year')
plt.savefig('../pandas-examples-figures/birthrate-data-1.svg')
plt.close()

# Clean the data

quartiles = np.percentile(births['births'], [25, 50, 75])
# array([4358. , 4814. , 5289.5])
mu = quartiles[1]  # 4814.0
# Get a robust estimate of the sample standard deviation, where the 0.74 comes
# from the interquartile range of a Gaussian distribution (see sigma-clipping
# operations).
sig = 0.74 * (quartiles[2] - quartiles[0])  # 689.31

births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
births.head()
#    year  month  day gender  births  decade
# 0  1969      1  1.0      F    4046    1960
# 1  1969      1  1.0      M    4440    1960
# 2  1969      1  2.0      F    4454    1960
# 3  1969      1  2.0      M    4548    1960
# 4  1969      1  3.0      F    4548    1960
births.shape  # (14610, 6)

# Set the 'day' column to integer; it originally was a string due to nulls.
# Nulls were together with anomaly numbers of births.
births['day'] = births['day'].astype(int)
#    year  month  day gender  births  decade
# 0  1969      1    1      F    4046    1960
# 1  1969      1    1      M    4440    1960
# 2  1969      1    2      F    4454    1960
# 3  1969      1    2      M    4548    1960
# 4  1969      1    3      F    4548    1960
births.shape  # (14610, 6)

# Create a datetime index from the year, month, day
births.index = pd.to_datetime(10000 * births.year + 100 * births.month + births.day, format='%Y%m%d')
births.head()
#             year  month  day gender  births  decade
# 1969-01-01  1969      1    1      F    4046    1960
# 1969-01-01  1969      1    1      M    4440    1960
# 1969-01-02  1969      1    2      F    4454    1960
# 1969-01-02  1969      1    2      M    4548    1960
# 1969-01-03  1969      1    3      F    4548    1960

births['dayofweek'] = births.index.dayofweek
births.head()
#             year  month  day gender  births  decade  dayofweek
# 1969-01-01  1969      1    1      F    4046    1960          2
# 1969-01-01  1969      1    1      M    4440    1960          2
# 1969-01-02  1969      1    2      F    4454    1960          3
# 1969-01-02  1969      1    2      M    4548    1960          3
# 1969-01-03  1969      1    3      F    4548    1960          4
births.pivot_table('births', index='dayofweek', columns='decade', aggfunc='mean')
# decade            1960         1970         1980
# dayofweek
# 0          5063.826923  4689.097701  5276.907249
# 1          5286.096154  4885.252399  5503.842553
# 2          5074.622642  4750.376200  5367.642553
# 3          4978.288462  4696.923372  5333.485106
# 4          5107.884615  4782.095785  5393.087234
# 5          4651.057692  4207.784483  4483.901064
# 6          4342.346154  3979.278736  4308.120469

# Plot births by weekday for several decades

births.pivot_table('births', index='dayofweek', columns='decade', aggfunc='mean').plot()
plt.gca().set(xticks=range(7), xticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day')
plt.savefig('../pandas-examples-figures/birthrate-data-2.svg')
plt.close()
# The 1990s and 2000s are missing because starting in 1989,
# the CDC data contains only the month of birth.

births_by_date = births.pivot_table('births', index=[births.index.month, births.index.day])
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

# Link with a dummy year variable (2012)
# (choose a leap year so February 29th is correctly handled)
births_by_date.index = [datetime(2012, month, day) for (month, day) in births_by_date.index]
births_by_date.head()
#               births
# 2012-01-01  4009.225
# 2012-01-02  4247.400
# 2012-01-03  4500.900
# 2012-01-04  4571.350
# 2012-01-05  4603.625

# Plot the mean number of births by the day of the year

fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
plt.ylabel('births by the day of the year')
plt.savefig('../pandas-examples-figures/birthrate-data-3.svg')
plt.close()
