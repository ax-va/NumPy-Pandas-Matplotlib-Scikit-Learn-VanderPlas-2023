import numpy as np
import pandas as pd
import seaborn as sns


# More than one thousand extrasolar planets are discovered up to 2014
planets = sns.load_dataset('planets')
planets.shape
# (1035, 6)
planets.head()
#             method  number  orbital_period   mass  distance  year
# 0  Radial Velocity       1         269.300   7.10     77.40  2006
# 1  Radial Velocity       1         874.774   2.21     56.95  2008
# 2  Radial Velocity       1         763.000   2.60     19.84  2011
# 3  Radial Velocity       1         326.030  19.40    110.62  2007
# 4  Radial Velocity       1         516.220  10.50    119.47  2009

rng = np.random.RandomState(42)
data = rng.rand(5)
# array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864])

ser = pd.Series(data)
# 0    0.374540
# 1    0.950714
# 2    0.731994
# 3    0.598658
# 4    0.156019
# dtype: float64

ser.sum()
#  2.811925491708157

ser.mean()
# 0.5623850983416314

data_a = rng.rand(5)
# array([0.15599452, 0.05808361, 0.86617615, 0.60111501, 0.70807258])

data_b = rng.rand(5)
# array([0.02058449, 0.96990985, 0.83244264, 0.21233911, 0.18182497])

df = pd.DataFrame({'A': data_a, 'B': data_b})
#           A         B
# 0  0.155995  0.020584
# 1  0.058084  0.969910
# 2  0.866176  0.832443
# 3  0.601115  0.212339
# 4  0.708073  0.181825

df.mean()
# A    0.477888
# B    0.443420
# dtype: float64

df.mean(axis='columns')
# 0    0.088290
# 1    0.513997
# 2    0.849309
# 3    0.406727
# 4    0.444949
# dtype: float64

planets.dropna().shape
# (498, 6)
planets.dropna().describe()
#           number  orbital_period        mass    distance         year
# count  498.00000      498.000000  498.000000  498.000000   498.000000
# mean     1.73494      835.778671    2.509320   52.068213  2007.377510
# std      1.17572     1469.128259    3.636274   46.596041     4.167284
# min      1.00000        1.328300    0.003600    1.350000  1989.000000
# 25%      1.00000       38.272250    0.212500   24.497500  2005.000000
# 50%      1.00000      357.000000    1.245000   39.940000  2009.000000
# 75%      2.00000      999.600000    2.867500   59.332500  2011.000000
# max      6.00000    17337.500000   25.000000  354.000000  2014.000000

# These are all methods of DataFrame and Series objects:
# Aggregation       Returns
# count             Total number of items
# first, last       First and last item
# mean, median      Mean and median
# min, max          Minimum and maximum
# std, var          Standard deviation and variance
# mad               Mean absolute deviation
# prod              Product of all items
# sum               Sum of all items
