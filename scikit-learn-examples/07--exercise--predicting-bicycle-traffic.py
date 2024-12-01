import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# data from: https://github.com/jakevdp/bicycle-data

counts = pd.read_csv(
    '../pandas-examples-data/FremontBridge.csv',
    index_col='Date',
    parse_dates=True,
    date_format="%m/%d/%Y %I:%M:%S %p"
)

counts.shape
# (147278, 3)

counts.info()
# <class 'pandas.core.frame.DataFrame'>
# DatetimeIndex: 147278 entries, 2019-11-01 00:00:00 to 2021-12-31 23:00:00
# Data columns (total 3 columns):
#  #   Column                        Non-Null Count   Dtype
# ---  ------                        --------------   -----
#  0   Fremont Bridge Total          147256 non-null  float64
#  1   Fremont Bridge East Sidewalk  147256 non-null  float64
#  2   Fremont Bridge West Sidewalk  147255 non-null  float64
# dtypes: float64(3)
# memory usage: 4.5 MB

weather = pd.read_csv(
    '../scikit-learn-examples-data/SeattleWeather.csv',
    index_col='DATE',
    parse_dates=True
)

weather.shape
# (3653, 28)

weather.info()
# <class 'pandas.core.frame.DataFrame'>
# DatetimeIndex: 3653 entries, 2012-01-01 to 2021-12-31
# Data columns (total 28 columns):
#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   STATION  3653 non-null   object
#  1   NAME     3653 non-null   object
#  2   AWND     3653 non-null   float64
#  3   FMTM     31 non-null     float64
#  4   PGTM     83 non-null     float64
#  5   PRCP     3650 non-null   float64
#  6   SNOW     3653 non-null   float64
#  7   SNWD     3653 non-null   float64
#  8   TAVG     3197 non-null   float64
#  9   TMAX     3653 non-null   int64
#  10  TMIN     3653 non-null   int64
#  11  WDF2     3653 non-null   int64
#  12  WDF5     3615 non-null   float64
#  13  WSF2     3653 non-null   float64
#  14  WSF5     3615 non-null   float64
#  15  WT01     1630 non-null   float64
#  16  WT02     187 non-null    float64
#  17  WT03     38 non-null     float64
#  18  WT04     10 non-null     float64
#  19  WT05     4 non-null      float64
#  20  WT08     215 non-null    float64
#  21  WT09     1 non-null      float64
#  22  WT13     195 non-null    float64
#  23  WT14     53 non-null     float64
#  24  WT16     268 non-null    float64
#  25  WT17     1 non-null      float64
#  26  WT18     20 non-null     float64
#  27  WT22     9 non-null      float64
# dtypes: float64(23), int64(3), object(2)
# memory usage: 827.6+ KB

counts = counts[counts.index < "2020-01-01"]
counts.shape
# (127008, 3)
counts.head(3)
#                      Fremont Bridge Total  Fremont Bridge East Sidewalk  Fremont Bridge West Sidewalk
# Date
# 2019-11-01 00:00:00                  12.0                           7.0                           5.0
# 2019-11-01 01:00:00                   7.0                           0.0                           7.0
# 2019-11-01 02:00:00                   1.0                           0.0                           1.0

weather = weather[weather.index < "2020-01-01"]
weather.shape
# (3653, 28)
weather.head(3)
#                 STATION                           NAME   AWND  FMTM  PGTM  ...  WT14  WT16  WT17  WT18  WT22
# DATE                                                                       ...
# 2012-01-01  USW00024233  SEATTLE TACOMA AIRPORT, WA US  10.51   NaN   NaN  ...   1.0   NaN   NaN   NaN   NaN
# 2012-01-02  USW00024233  SEATTLE TACOMA AIRPORT, WA US  10.07   NaN   NaN  ...   NaN   1.0   NaN   NaN   NaN
# 2012-01-03  USW00024233  SEATTLE TACOMA AIRPORT, WA US   5.14   NaN   NaN  ...   NaN   1.0   NaN   NaN   NaN
#
# [3 rows x 28 columns]

daily = counts.resample('d').sum()
daily.head(3)
#             Fremont Bridge Total  Fremont Bridge East Sidewalk  Fremont Bridge West Sidewalk
# Date
# 2012-10-03                7042.0                        3520.0                        3522.0
# 2012-10-04                6950.0                        3416.0                        3534.0
# 2012-10-05                6296.0                        3116.0                        3180.0

daily['Total'] = daily.sum(axis=1)
daily.head(3)
#             Fremont Bridge Total  Fremont Bridge East Sidewalk  Fremont Bridge West Sidewalk    Total
# Date
# 2012-10-03                7042.0                        3520.0                        3522.0  14084.0
# 2012-10-04                6950.0                        3416.0                        3534.0  13900.0
# 2012-10-05                6296.0                        3116.0                        3180.0  12592.0

daily = daily[['Total']]  # Remove other columns
daily.head(3)
#               Total
# Date
# 2012-10-03  14084.0
# 2012-10-04  13900.0
# 2012-10-05  12592.0

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
    # Add a new column with hot one
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)

daily.head(3)
#               Total  Mon  Tue  Wed  Thu  Fri  Sat  Sun
# Date
# 2012-10-03  14084.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
# 2012-10-04  13900.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
# 2012-10-05  12592.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0

cal = USFederalHolidayCalendar()
# <pandas.tseries.holiday.USFederalHolidayCalendar at 0x7f52ae15fe20>
holidays = cal.holidays('2012', '2020')
# DatetimeIndex(['2012-01-02', '2012-01-16', '2012-02-20', '2012-05-28',
#                '2012-07-04', '2012-09-03', '2012-10-08', '2012-11-12',
#                '2012-11-22', '2012-12-25', '2013-01-01', '2013-01-21',
#                '2013-02-18', '2013-05-27', '2013-07-04', '2013-09-02',
#                '2013-10-14', '2013-11-11', '2013-11-28', '2013-12-25',
#                '2014-01-01', '2014-01-20', '2014-02-17', '2014-05-26',
#                '2014-07-04', '2014-09-01', '2014-10-13', '2014-11-11',
#                '2014-11-27', '2014-12-25', '2015-01-01', '2015-01-19',
#                '2015-02-16', '2015-05-25', '2015-07-03', '2015-09-07',
#                '2015-10-12', '2015-11-11', '2015-11-26', '2015-12-25',
#                '2016-01-01', '2016-01-18', '2016-02-15', '2016-05-30',
#                '2016-07-04', '2016-09-05', '2016-10-10', '2016-11-11',
#                '2016-11-24', '2016-12-26', '2017-01-02', '2017-01-16',
#                '2017-02-20', '2017-05-29', '2017-07-04', '2017-09-04',
#                '2017-10-09', '2017-11-10', '2017-11-23', '2017-12-25',
#                '2018-01-01', '2018-01-15', '2018-02-19', '2018-05-28',
#                '2018-07-04', '2018-09-03', '2018-10-08', '2018-11-12',
#                '2018-11-22', '2018-12-25', '2019-01-01', '2019-01-21',
#                '2019-02-18', '2019-05-27', '2019-07-04', '2019-09-02',
#                '2019-10-14', '2019-11-11', '2019-11-28', '2019-12-25',
#                '2020-01-01'],
#               dtype='datetime64[ns]', freq=None)
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily.head(3)
#
#               Total  Mon  Tue  Wed  Thu  Fri  Sat  Sun  holiday
# Date
# 2012-10-03  14084.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0      NaN
# 2012-10-04  13900.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0      NaN
# 2012-10-05  12592.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0      NaN

daily['holiday'].fillna(0, inplace=True)
daily.head(3)
#               Total  Mon  Tue  Wed  Thu  Fri  Sat  Sun  holiday
# Date
# 2012-10-03  14084.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0      0.0
# 2012-10-04  13900.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0      0.0
# 2012-10-05  12592.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0      0.0


def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """ Compute the hours of daylight for the given date """
    days = (date - pd.to_datetime("2000-12-21")).days
    m = (1. - np.tan(np.radians(latitude)) * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.


daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily.head(3)
#               Total  Mon  Tue  Wed  ...  Sat  Sun  holiday  daylight_hrs
# Date                                ...
# 2012-10-03  14084.0  0.0  0.0  1.0  ...  0.0  0.0      0.0     11.277359
# 2012-10-04  13900.0  0.0  0.0  0.0  ...  0.0  0.0      0.0     11.219142
# 2012-10-05  12592.0  0.0  0.0  0.0  ...  0.0  0.0      0.0     11.161038
#
# [3 rows x 10 columns]

daily[['daylight_hrs']].plot()
plt.ylim(8, 17)
plt.savefig('../scikit-learn-examples-figures/predicting-bicycle-traffic-1--daylight-hours.svg')
plt.close()

# Add the average temperature and total precipitation
weather['Temp (F)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])
weather['Rainfall (in)'] = weather['PRCP']
weather['dry day'] = (weather['PRCP'] == 0).astype(int)
weather.head(3)
#                 STATION                           NAME   AWND  FMTM  ...  WT22  Temp (F)  Rainfall (in)  dry day
# DATE                                                                 ...
# 2012-01-01  USW00024233  SEATTLE TACOMA AIRPORT, WA US  10.51   NaN  ...   NaN      48.0           0.00        1
# 2012-01-02  USW00024233  SEATTLE TACOMA AIRPORT, WA US  10.07   NaN  ...   NaN      44.0           0.43        0
# 2012-01-03  USW00024233  SEATTLE TACOMA AIRPORT, WA US   5.14   NaN  ...   NaN      49.0           0.03        0
#
# [3 rows x 31 columns]

daily = daily.join(weather[['Rainfall (in)', 'Temp (F)', 'dry day']])
#               Total  Mon  Tue  Wed  Thu  ...  holiday  daylight_hrs  Rainfall (in)  Temp (F)  dry day
# Date                                     ...
# 2012-10-03  14084.0  0.0  0.0  1.0  0.0  ...      0.0     11.277359            0.0      56.0        1
# 2012-10-04  13900.0  0.0  0.0  0.0  1.0  ...      0.0     11.219142            0.0      56.5        1
# 2012-10-05  12592.0  0.0  0.0  0.0  0.0  ...      0.0     11.161038            0.0      59.5        1
#
# [3 rows x 13 columns]

# Add the information how many years have passed
daily['annual'] = (daily.index - daily.index[0]).days / 365.
daily.head(3)
#               Total  Mon  Tue  Wed  Thu  ...  daylight_hrs  Rainfall (in)  Temp (F)  dry day    annual
# Date                                     ...
# 2012-10-03  14084.0  0.0  0.0  1.0  0.0  ...     11.277359            0.0      56.0        1  0.000000
# 2012-10-04  13900.0  0.0  0.0  0.0  1.0  ...     11.219142            0.0      56.5        1  0.002740
# 2012-10-05  12592.0  0.0  0.0  0.0  0.0  ...     11.161038            0.0      59.5        1  0.005479
#
# [3 rows x 14 columns]

daily.shape
# (2646, 14)

# Drop any rows with null values
daily.dropna(axis=0, how='any', inplace=True)

daily.shape
# (2646, 14)

column_names = [
    'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun',
    'holiday', 'daylight_hrs', 'Rainfall (in)',
    'dry day', 'Temp (F)', 'annual'
]

# Prepare data for the linear regression
X = daily[column_names]
y = daily['Total']

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
daily['predicted'] = model.predict(X)

# Compare the total and predicted bicycle traffic visually
daily[['Total', 'predicted']].plot(alpha=0.5)
plt.savefig('../scikit-learn-examples-figures/predicting-bicycle-traffic-2--total-and-predicted-bicycle-traffic.svg')
plt.close()

# 1) Either our features are not complete or 2) there are some
# nonlinear relationships that we have failed to take into account:

# 1) E.g., people decide whether to ride to work based on more than just these features;
# 2) E.g., people ride less at both high and low temperatures.

params = pd.Series(model.coef_, index=X.columns)
# Mon              -3309.953439
# Tue              -2860.625060
# Wed              -2962.889892
# Thu              -3480.656444
# Fri              -4836.064503
# Sat             -10436.802843
# Sun             -10795.195718
# holiday          -5006.995232
# daylight_hrs       409.146368
# Rainfall (in)    -2789.860745
# dry day           2111.069565
# Temp (F)           179.026296
# annual             324.437749
# dtype: float64

# Compute uncertainty
np.random.seed(1)
err = np.std(
    [model.fit(*resample(X, y)).coef_ for i in range(1000)],
    axis=0
)

# How to interpret:
# the changed number of riders depending on a changed unit of a feature
with_uncertainty = pd.DataFrame(
    {
        'effect': params.round(0),
        'uncertainty': err.round(0)
    }
)
#                 effect  uncertainty
# Mon            -3310.0        265.0
# Tue            -2861.0        274.0
# Wed            -2963.0        268.0
# Thu            -3481.0        268.0
# Fri            -4836.0        261.0
# Sat           -10437.0        259.0
# Sun           -10795.0        267.0
# holiday        -5007.0        401.0
# daylight_hrs     409.0         26.0
# Rainfall (in)  -2790.0        186.0
# dry day         2111.0        101.0
# Temp (F)         179.0          7.0
# annual           324.0         22.0
