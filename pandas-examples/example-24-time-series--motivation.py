# Python and third-party packages:
from datetime import datetime
from dateutil import parser

datetime(year=2021, month=7, day=4)
# datetime.datetime(2021, 7, 4, 0, 0)

date = parser.parse("4th of July, 2021")
# datetime.datetime(2021, 7, 4, 0, 0)

date.strftime('%A')
# 'Sunday'

# NumPy:
import numpy as np

date = np.array('2021-07-04', dtype=np.datetime64)
# array('2021-07-04', dtype='datetime64[D]')

np.arange(12)
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

date + np.arange(12)
# array(['2021-07-04', '2021-07-05', '2021-07-06', '2021-07-07',
#        '2021-07-08', '2021-07-09', '2021-07-10', '2021-07-11',
#        '2021-07-12', '2021-07-13', '2021-07-14', '2021-07-15'],
#       dtype='datetime64[D]')

# datetime64 is restricted by 2⁶⁴ fundamental time units in time spans

# day-based datetime
np.datetime64('2021-07-04')
# numpy.datetime64('2021-07-04')

# minute-based datetime
np.datetime64('2021-07-04 12:00')
# numpy.datetime64('2021-07-04T12:00')

# nanosecond-based time
np.datetime64('2021-07-04 12:59:59.50', 'ns')
# numpy.datetime64('2021-07-04T12:59:59.500000000')

# Pandas:
import pandas as pd

date = pd.to_datetime("4th of July, 2021")
# Timestamp('2021-07-04 00:00:00')

date.strftime('%A')
# 'Sunday'

date + pd.to_timedelta(np.arange(12), 'D')
# DatetimeIndex(['2021-07-04', '2021-07-05', '2021-07-06', '2021-07-07',
#                '2021-07-08', '2021-07-09', '2021-07-10', '2021-07-11',
#                '2021-07-12', '2021-07-13', '2021-07-14', '2021-07-15'],
#               dtype='datetime64[ns]', freq=None)
