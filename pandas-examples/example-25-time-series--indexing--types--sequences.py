from datetime import datetime

import pandas as pd
from pandas.tseries.offsets import BDay

index = pd.DatetimeIndex(
    [
        '2020-07-04',
        '2020-08-04',
        '2021-07-04',
        '2021-08-04'
    ]
)
data = pd.Series([0, 1, 2, 3], index=index)
# 2020-07-04    0
# 2020-08-04    1
# 2021-07-04    2
# 2021-08-04    3
# dtype: int64

data['2020-07-04':'2021-07-04']
# 2020-07-04    0
# 2020-08-04    1
# 2021-07-04    2
# dtype: int64

# only year to index
data['2021']
# 2021-07-04    2
# 2021-08-04    3
# dtype: int64

data['2020':'2021']
# 2020-07-04    0
# 2020-08-04    1
# 2021-07-04    2
# 2021-08-04    3
# dtype: int64

# Pandas datetime types:
# based on numpy.datetime64: Timestamp, DatetimeIndex, Period, PeriodIndex
# based on numpy.timedelta64: Timedelta, TimedeltaIndex

dates = pd.to_datetime(
    [
        '2015-07-03',
        datetime(2021, 7, 3),
        '4th of July, 2021',
        '2021-Jul-6',
        '07-07-2021',  # MM-DD-YYYY
        '20210708',
        '09-07-2021',  # MM-DD-YYYY
    ], format='mixed'
)
# DatetimeIndex(['2015-07-03', '2021-07-03', '2021-07-04', '2021-07-06',
#                '2021-07-07', '2021-07-08', '2021-09-07'],
#               dtype='datetime64[ns]', freq=None)

dates.to_period('D')
# PeriodIndex(['2015-07-03', '2021-07-03', '2021-07-04', '2021-07-06',
#              '2021-07-07', '2021-07-08', '2021-09-07'],
#             dtype='period[D]')

dates - dates[0]
# TimedeltaIndex([   '0 days', '2192 days', '2193 days', '2195 days',
#                 '2196 days', '2197 days', '2258 days'],
#                dtype='timedelta64[ns]', freq=None)

# pd.date_range for timestamps
# pd.period_range for periods
# pd.timedelta_range for time deltas

pd.date_range('2015-07-03', '2015-07-10')
# DatetimeIndex(['2015-07-03', '2015-07-04', '2015-07-05', '2015-07-06',
#                '2015-07-07', '2015-07-08', '2015-07-09', '2015-07-10'],
#               dtype='datetime64[ns]', freq='D')

pd.date_range('2015-07-03', periods=8)
# DatetimeIndex(['2015-07-03', '2015-07-04', '2015-07-05', '2015-07-06',
#                '2015-07-07', '2015-07-08', '2015-07-09', '2015-07-10'],
#               dtype='datetime64[ns]', freq='D')

pd.date_range('2015-07-03', periods=8, freq='H')
# DatetimeIndex(['2015-07-03 00:00:00', '2015-07-03 01:00:00',
#                '2015-07-03 02:00:00', '2015-07-03 03:00:00',
#                '2015-07-03 04:00:00', '2015-07-03 05:00:00',
#                '2015-07-03 06:00:00', '2015-07-03 07:00:00'],
#               dtype='datetime64[ns]', freq='H')

pd.period_range('2015-07', periods=8, freq='M')
# PeriodIndex(['2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12',
#              '2016-01', '2016-02'],
#             dtype='period[M]')

pd.timedelta_range(0, periods=6, freq='H')
# TimedeltaIndex(['0 days 00:00:00', '0 days 01:00:00', '0 days 02:00:00',
#                 '0 days 03:00:00', '0 days 04:00:00', '0 days 05:00:00'],
#                dtype='timedelta64[ns]', freq='H')

# frequency or date offset:
# D         calendar day
# W         weekly
# M         month end
# Q         quarter end
# A         year end
# H         hours
# T         minutes
# S         seconds
# L         milliseconds
# U         microseconds
# N         nanoseconds
# B         business day
# BM        business month end
# BQ        business quarter end
# BA        business year end
# BH        business hours
# MS        month start
# QS        quarter start
# AS        year start
# BMS       business month start
# BQS       business quarter start
# BAS       business year start

# Change the month used to mark any quarterly or annual code
# Q-JAN, BQ-FEB, QS-MAR, BQS-APR, etc.
# A-JAN, BA-FEB, AS-MAR, BAS-APR, etc.

# Modify the split point of the weekly frequency
# W-SUN, W-MON, W-TUE, W-WED, etc.

# Specify a frequency of 2 hours and 30 minutes

pd.timedelta_range(0, periods=6, freq="2H30T")
# TimedeltaIndex(['0 days 00:00:00', '0 days 02:30:00', '0 days 05:00:00',
#                 '0 days 07:30:00', '0 days 10:00:00', '0 days 12:30:00'],
#                dtype='timedelta64[ns]', freq='150T')

pd.date_range('2015-07-01', periods=10, freq=BDay())
# DatetimeIndex(['2015-07-01', '2015-07-02', '2015-07-03', '2015-07-06',
#                '2015-07-07', '2015-07-08', '2015-07-09', '2015-07-10',
#                '2015-07-13', '2015-07-14'],
#               dtype='datetime64[ns]', freq='B')
