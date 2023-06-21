import pandas as pd
import seaborn as sns

# split-apply-combine operations by using groupby
df = pd.DataFrame(
    {
        'key': ['A', 'B', 'C', 'A', 'B', 'C'],
        'data': range(6)
    },
    columns=['key', 'data']
)
#   key  data
# 0   A     0
# 1   B     1
# 2   C     2
# 3   A     3
# 4   B     4
# 5   C     5

# split operation
df.groupby('key')
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f02a5c1a0e0>

# apply-combine operations
df.groupby('key').sum()
#      data
# key
# A       3
# B       5
# C       7

planets = sns.load_dataset('planets')
planets.head()
#             method  number  orbital_period   mass  distance  year
# 0  Radial Velocity       1         269.300   7.10     77.40  2006
# 1  Radial Velocity       1         874.774   2.21     56.95  2008
# 2  Radial Velocity       1         763.000   2.60     19.84  2011
# 3  Radial Velocity       1         326.030  19.40    110.62  2007
# 4  Radial Velocity       1         516.220  10.50    119.47  2009

planets.groupby('method')
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f02a519e170>
planets.groupby('orbital_period')
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f02a5af43a0>

# column indexing
planets.groupby('method')['orbital_period']
# <pandas.core.groupby.generic.SeriesGroupBy object at 0x7f02a5cbc1f0>
planets.groupby('orbital_period')['orbital_period']
# <pandas.core.groupby.generic.SeriesGroupBy object at 0x7f02a519de40>
planets.groupby('method')['orbital_period'].median()
# method
# Astrometry                         631.180000
# Eclipse Timing Variations         4343.500000
# Imaging                          27500.000000
# Microlensing                      3300.000000
# Orbital Brightness Modulation        0.342887
# Pulsar Timing                       66.541900
# Pulsation Timing Variations       1170.000000
# Radial Velocity                    360.200000
# Transit                              5.714932
# Transit Timing Variations           57.011000
# Name: orbital_period, dtype: float64
planets.groupby('method')['orbital_period'].mean()
# method
# Astrometry                          631.180000
# Eclipse Timing Variations          4751.644444
# Imaging                          118247.737500
# Microlensing                       3153.571429
# Orbital Brightness Modulation         0.709307
# Pulsar Timing                      7343.021201
# Pulsation Timing Variations        1170.000000
# Radial Velocity                     823.354680
# Transit                              21.102073
# Transit Timing Variations            79.783500
# Name: orbital_period, dtype: float64
planets.groupby('orbital_period')['orbital_period'].median()
# orbital_period
# 0.090706              0.090706
# 0.240104              0.240104
# 0.342887              0.342887
# 0.355000              0.355000
# 0.453285              0.453285
#                      ...
# 40000.000000      40000.000000
# 69000.000000      69000.000000
# 170000.000000    170000.000000
# 318280.000000    318280.000000
# 730000.000000    730000.000000
# Name: orbital_period, Length: 988, dtype: float64

# iteration over groups
for (method, group) in planets.groupby('method'):
    print(f"{method:30s} shape={group.shape}")
# Astrometry                     shape=(2, 6)
# Eclipse Timing Variations      shape=(9, 6)
# Imaging                        shape=(38, 6)
# Microlensing                   shape=(23, 6)
# Orbital Brightness Modulation  shape=(3, 6)
# Pulsar Timing                  shape=(5, 6)
# Pulsation Timing Variations    shape=(1, 6)
# Radial Velocity                shape=(553, 6)
# Transit                        shape=(397, 6)
# Transit Timing Variations      shape=(4, 6)

# Dispatch methods:
# Any method not explicitly implemented by the GroupBy object will be passed
# through and called on the groups, whether they are DataFrame or Series objects.
planets.groupby('method')['year'].describe()
#                                count         mean       std     min      25%     50%      75%     max
# method
# Astrometry                       2.0  2011.500000  2.121320  2010.0  2010.75  2011.5  2012.25  2013.0
# Eclipse Timing Variations        9.0  2010.000000  1.414214  2008.0  2009.00  2010.0  2011.00  2012.0
# Imaging                         38.0  2009.131579  2.781901  2004.0  2008.00  2009.0  2011.00  2013.0
# Microlensing                    23.0  2009.782609  2.859697  2004.0  2008.00  2010.0  2012.00  2013.0
# Orbital Brightness Modulation    3.0  2011.666667  1.154701  2011.0  2011.00  2011.0  2012.00  2013.0
# Pulsar Timing                    5.0  1998.400000  8.384510  1992.0  1992.00  1994.0  2003.00  2011.0
# Pulsation Timing Variations      1.0  2007.000000       NaN  2007.0  2007.00  2007.0  2007.00  2007.0
# Radial Velocity                553.0  2007.518987  4.249052  1989.0  2005.00  2009.0  2011.00  2014.0
# Transit                        397.0  2011.236776  2.077867  2002.0  2010.00  2012.0  2013.00  2014.0
# Transit Timing Variations        4.0  2012.500000  1.290994  2011.0  2011.75  2012.5  2013.25  2014.0
planets.groupby('method')['year'].describe().unstack()
#        method
# count  Astrometry                          2.0
#        Eclipse Timing Variations           9.0
#        Imaging                            38.0
#        Microlensing                       23.0
#        Orbital Brightness Modulation       3.0
#                                          ...
# max    Pulsar Timing                    2011.0
#        Pulsation Timing Variations      2007.0
#        Radial Velocity                  2014.0
#        Transit                          2014.0
#        Transit Timing Variations        2014.0
# Length: 80, dtype: float64
