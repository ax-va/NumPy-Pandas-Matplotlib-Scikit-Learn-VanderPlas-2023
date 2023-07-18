import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# https://github.com/jakevdp/bicycle-data/blob/main/FremontBridge.csv
data = pd.read_csv('../pandas-examples-data/FremontBridge.csv')
data.info()
# RangeIndex: 147278 entries, 0 to 147277
# Data columns (total 4 columns):
#  #   Column                        Non-Null Count   Dtype
# ---  ------                        --------------   -----
#  0   Date                          147278 non-null  object
#  1   Fremont Bridge Total          147256 non-null  float64
#  2   Fremont Bridge East Sidewalk  147256 non-null  float64
#  3   Fremont Bridge West Sidewalk  147255 non-null  float64
# dtypes: float64(3), object(1)
# memory usage: 4.5+ MB
data = pd.read_csv('../pandas-examples-data/FremontBridge.csv',
                   index_col='Date', parse_dates=True, date_format="%m/%d/%Y %I:%M:%S %p")
data.head(20)
#                      Fremont Bridge Total  Fremont Bridge East Sidewalk  Fremont Bridge West Sidewalk
# Date
# 2019-11-01 00:00:00                  12.0                           7.0                           5.0
# 2019-11-01 01:00:00                   7.0                           0.0                           7.0
# 2019-11-01 02:00:00                   1.0                           0.0                           1.0
# 2019-11-01 03:00:00                   6.0                           6.0                           0.0
# 2019-11-01 04:00:00                   6.0                           5.0                           1.0
# 2019-11-01 05:00:00                  20.0                           9.0                          11.0
# 2019-11-01 06:00:00                  97.0                          43.0                          54.0
# 2019-11-01 07:00:00                 299.0                         120.0                         179.0
# 2019-11-01 08:00:00                 583.0                         261.0                         322.0
# 2019-11-01 09:00:00                 332.0                         130.0                         202.0
# 2019-11-01 10:00:00                 124.0                          56.0                          68.0
# 2019-11-01 11:00:00                  94.0                          46.0                          48.0
# 2019-11-01 12:00:00                 104.0                          40.0                          64.0
# 2019-11-01 13:00:00                 110.0                          46.0                          64.0
# 2019-11-01 14:00:00                 131.0                          46.0                          85.0
# 2019-11-01 15:00:00                 214.0                          55.0                         159.0
# 2019-11-01 16:00:00                 420.0                          92.0                         328.0
# 2019-11-01 17:00:00                 637.0                         137.0                         500.0
# 2019-11-01 18:00:00                 320.0                          84.0                         236.0
# 2019-11-01 19:00:00                 115.0                          38.0                          77.0

data.columns = ['Total', 'East', 'West']
data.head()
#                      Total  East  West
# Date
# 2019-11-01 00:00:00   12.0   7.0   5.0
# 2019-11-01 01:00:00    7.0   0.0   7.0
# 2019-11-01 02:00:00    1.0   0.0   1.0
# 2019-11-01 03:00:00    6.0   6.0   0.0
# 2019-11-01 04:00:00    6.0   5.0   1.0

# summary statistics
data.dropna().describe()
#                Total           East           West
# count  147255.000000  147255.000000  147255.000000
# mean      110.341462      50.077763      60.263699
# std       140.422051      64.634038      87.252147
# min         0.000000       0.000000       0.000000
# 25%        14.000000       6.000000       7.000000
# 50%        60.000000      28.000000      30.000000
# 75%       145.000000      68.000000      74.000000
# max      1097.000000     698.000000     850.000000

# Visualize the data
data.plot()
plt.ylabel('Hourly Bicycle Count')
# plt.show()
plt.savefig('../pandas-examples-figures/seattle-bicycle-counts-1.svg')
plt.close()

# Resample by week
weekly = data.resample('W').sum()  # Sum out inside every week
weekly.plot(style=['-', ':', '--'])
plt.ylabel('Weekly Bicycle Count')
# plt.show()
plt.savefig('../pandas-examples-figures/seattle-bicycle-counts-2.svg')
plt.close()

# Use a 30-day rolling mean
daily = data.resample('D').sum()  # Sum out inside every day
daily.rolling(30, center=True).sum().plot(style=['-', ':', '--'])
plt.ylabel('30-Day Centered Rolling Mean Count')
# plt.show()
plt.savefig('../pandas-examples-figures/seattle-bicycle-counts-3.svg')
plt.close()

# Use a Gaussian window - a smoother version of a rolling mean
daily.rolling(50, center=True, win_type='gaussian').sum(std=10).plot(style=['-', ':', '--'])
plt.ylabel('Gaussian Mean Count')
# plt.show()
plt.savefig('../pandas-examples-figures/seattle-bicycle-counts-4.svg')
plt.close()

# Look at the average traffic as a function of the time of day
by_time = data.groupby(data.index.time).mean()
hourly_ticks = 4 * 60 * 60 * np.arange(6)
by_time.plot(xticks=hourly_ticks, style=['-', ':', '--'])
plt.ylabel('Day-Time Average Traffic')
# plt.show()
plt.savefig('../pandas-examples-figures/seattle-bicycle-counts-5.svg')
plt.close()

# Look at the average traffic as a function of the weekdays
by_weekday = data.groupby(data.index.dayofweek).mean()
by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
by_weekday.plot(style=['-', ':', '--'])
plt.ylabel('Weekday Average Traffic')
# plt.show()
plt.savefig('../pandas-examples-figures/seattle-bicycle-counts-6.svg')
plt.close()

# weekdays versus weekends
weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
# array(['Weekday', 'Weekday', 'Weekday', ..., 'Weekday', 'Weekday',
#        'Weekday'], dtype='<U7')
by_time = data.groupby([weekend, data.index.time]).mean()
#                        Total        East        West
# Weekday 00:00:00    9.104423    3.873461    5.230962
#         01:00:00    4.518468    2.015960    2.502508
#         02:00:00    3.006612    1.470360    1.536252
#         03:00:00    2.588919    1.328773    1.260146
#         04:00:00    7.445280    4.037620    3.408438
#         05:00:00   31.860237   19.444596   12.415641
#         06:00:00  114.374373   68.647743   45.726630
#         07:00:00  290.439580  179.658003  110.781578
#         08:00:00  405.313041  237.831509  167.481532
#         09:00:00  220.941834  119.039690  101.902144
#         10:00:00   98.266651   50.345803   47.920849
#         11:00:00   76.499316   39.138686   37.360630
#         12:00:00   80.184535   39.861314   40.323221
#         13:00:00   86.534884   42.034200   44.500684
#         14:00:00   98.322919   44.775599   53.547320
#         15:00:00  140.413683   58.562372   81.851311
#         16:00:00  266.840365   88.795439  178.044926
#         17:00:00  480.419840  131.787685  348.632155
#         18:00:00  324.128620  105.235120  218.893501
#         19:00:00  155.417560   56.771266   98.646294
#         20:00:00   86.181984   34.070696   52.111288
#         21:00:00   53.831471   23.174002   30.657469
#         22:00:00   33.950285   14.420753   19.529532
#         23:00:00   21.197948    8.607298   12.590650
# Weekend 00:00:00   15.535694    6.405483    9.130211
#         01:00:00    8.942890    3.965163    4.977727
#         02:00:00    5.795625    2.666091    3.129534
#         03:00:00    3.349515    1.661336    1.688178
#         04:00:00    3.696745    1.524272    2.172473
#         05:00:00    7.258709    3.820103    3.438607
#         06:00:00   16.862935    7.820674    9.042262
#         07:00:00   32.439178   16.469446   15.969732
#         08:00:00   59.833238   30.877213   28.956025
#         09:00:00   82.635066   42.523130   40.111936
#         10:00:00  101.755568   53.312964   48.442604
#         11:00:00  123.560251   63.515134   60.045117
#         12:00:00  137.237007   69.778412   67.458595
#         13:00:00  145.027984   73.206739   71.821245
#         14:00:00  148.699029   75.055397   73.643632
#         15:00:00  146.740148   73.898915   72.841234
#         16:00:00  134.378641   67.231868   67.146773
#         17:00:00  110.735580   54.743004   55.992576
#         18:00:00   84.940605   42.105654   42.834951
#         19:00:00   57.801256   28.009138   29.792119
#         20:00:00   41.281553   19.373501   21.908053
#         21:00:00   29.432895   13.455740   15.977156
#         22:00:00   21.700171    9.684180   12.015991
#         23:00:00   16.897773    6.960023    9.937750
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
by_time.loc['Weekday'].plot(
    ax=ax[0],
    title='Weekdays',
    xticks=hourly_ticks,
    style=['-', ':', '--']
)
by_time.loc['Weekend'].plot(
    ax=ax[1],
    title='Weekends',
    xticks=hourly_ticks,
    style=['-', ':', '--']
)
# plt.show()
plt.savefig('../pandas-examples-figures/seattle-bicycle-counts-7.svg')
plt.close()
