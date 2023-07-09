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
planets.describe()
#             number  orbital_period        mass     distance         year
# count  1035.000000      992.000000  513.000000   808.000000  1035.000000
# mean      1.785507     2002.917596    2.638161   264.069282  2009.070531
# std       1.240976    26014.728304    3.818617   733.116493     3.972567
# min       1.000000        0.090706    0.003600     1.350000  1989.000000
# 25%       1.000000        5.442540    0.229000    32.560000  2007.000000
# 50%       1.000000       39.979500    1.260000    55.250000  2010.000000
# 75%       2.000000      526.005000    3.040000   178.500000  2012.000000
# max       7.000000   730000.000000   25.000000  8500.000000  2014.000000

# Count discovered planets by method and by decade

decade = 10 * (planets['year'] // 10)
# 0       2000
# 1       2000
# 2       2010
# 3       2000
# 4       2000
#         ...
# 1030    2000
# 1031    2000
# 1032    2000
# 1033    2000
# 1034    2000
# Name: year, Length: 1035, dtype: int64

decade = decade.astype(str) + 's'
# 0       2000s
# 1       2000s
# 2       2010s
# 3       2000s
# 4       2000s
#         ...
# 1030    2000s
# 1031    2000s
# 1032    2000s
# 1033    2000s
# 1034    2000s
# Name: year, Length: 1035, dtype: object

decade.name = 'decade'
# 0       2000s
# 1       2000s
# 2       2010s
# 3       2000s
# 4       2000s
#         ...
# 1030    2000s
# 1031    2000s
# 1032    2000s
# 1033    2000s
# 1034    2000s
# Name: decade, Length: 1035, dtype: object

planets.groupby(['method', decade])['number'].sum()
# method                         decade
# Astrometry                     2010s       2
# Eclipse Timing Variations      2000s       5
#                                2010s      10
# Imaging                        2000s      29
#                                2010s      21
# Microlensing                   2000s      12
#                                2010s      15
# Orbital Brightness Modulation  2010s       5
# Pulsar Timing                  1990s       9
#                                2000s       1
#                                2010s       1
# Pulsation Timing Variations    2000s       1
# Radial Velocity                1980s       1
#                                1990s      52
#                                2000s     475
#                                2010s     424
# Transit                        2000s      64
#                                2010s     712
# Transit Timing Variations      2010s       9
# Name: number, dtype: int64

planets.groupby(['method', decade])['number'].sum().unstack()
# decade                         1980s  1990s  2000s  2010s
# method
# Astrometry                       NaN    NaN    NaN    2.0
# Eclipse Timing Variations        NaN    NaN    5.0   10.0
# Imaging                          NaN    NaN   29.0   21.0
# Microlensing                     NaN    NaN   12.0   15.0
# Orbital Brightness Modulation    NaN    NaN    NaN    5.0
# Pulsar Timing                    NaN    9.0    1.0    1.0
# Pulsation Timing Variations      NaN    NaN    1.0    NaN
# Radial Velocity                  1.0   52.0  475.0  424.0
# Transit                          NaN    NaN   64.0  712.0
# Transit Timing Variations        NaN    NaN    NaN    9.0

planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)
# decade                         1980s  1990s  2000s  2010s
# method
# Astrometry                       0.0    0.0    0.0    2.0
# Eclipse Timing Variations        0.0    0.0    5.0   10.0
# Imaging                          0.0    0.0   29.0   21.0
# Microlensing                     0.0    0.0   12.0   15.0
# Orbital Brightness Modulation    0.0    0.0    0.0    5.0
# Pulsar Timing                    0.0    9.0    1.0    1.0
# Pulsation Timing Variations      0.0    0.0    1.0    0.0
# Radial Velocity                  1.0   52.0  475.0  424.0
# Transit                          0.0    0.0   64.0  712.0
# Transit Timing Variations        0.0    0.0    0.0    9.0
