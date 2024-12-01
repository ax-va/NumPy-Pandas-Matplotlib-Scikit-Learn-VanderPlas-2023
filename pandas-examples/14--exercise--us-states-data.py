"""
Data sources:
https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv
https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-areas.csv
https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-abbrevs.csv
"""
import pandas as pd

pop = pd.read_csv('../pandas-examples-data/state-population.csv')
print(pop.tail())
#      state/region     ages  year   population
# 2539          USA    total  2010  309326295.0
# 2540          USA  under18  2011   73902222.0
# 2541          USA    total  2011  311582564.0
# 2542          USA  under18  2012   73708179.0
# 2543          USA    total  2012  313873685.0
areas = pd.read_csv('../pandas-examples-data/state-areas.csv')
print(areas.tail())
#                   state  area (sq. mi)
# 47         West Virginia          24231
# 48             Wisconsin          65503
# 49               Wyoming          97818
# 50  District of Columbia             68
# 51           Puerto Rico           3515
abbrevs = pd.read_csv('../pandas-examples-data/state-abbrevs.csv')
print(abbrevs.tail())
#             state abbreviation
# 46       Virginia           VA
# 47     Washington           WA
# 48  West Virginia           WV
# 49      Wisconsin           WI
# 50        Wyoming           WY

# Issue: rank US states and territories by their 2010 population density

merged = pd.merge(pop, abbrevs, how='outer', left_on='state/region', right_on='abbreviation')
print(merged.tail())
#      state/region     ages  year   population state abbreviation
# 2539          USA    total  2010  309326295.0   NaN          NaN
# 2540          USA  under18  2011   73902222.0   NaN          NaN
# 2541          USA    total  2011  311582564.0   NaN          NaN
# 2542          USA  under18  2012   73708179.0   NaN          NaN
# 2543          USA    total  2012  313873685.0   NaN          NaN

# Drop the abbreviation column:
merged = merged.drop('abbreviation', axis=1)
print(merged.tail())
#      state/region     ages  year   population state
# 2539          USA    total  2010  309326295.0   NaN
# 2540          USA  under18  2011   73902222.0   NaN
# 2541          USA    total  2011  311582564.0   NaN
# 2542          USA  under18  2012   73708179.0   NaN
# 2543          USA    total  2012  313873685.0   NaN

print(merged.isnull().any())
# state/region    False
# ages            False
# year            False
# population       True
# state            True
# dtype: bool

print(merged[merged['population'].isnull()].tail())
#      state/region     ages  year  population state
# 2463           PR    total  1998         NaN   NaN
# 2464           PR    total  1997         NaN   NaN
# 2465           PR  under18  1997         NaN   NaN
# 2466           PR    total  1999         NaN   NaN
# 2467           PR  under18  1999         NaN   NaN

# Tle lines are equivalent.
# The second line combines masking and indexing in loc.
print(merged.loc[merged['state'].isnull()]['state/region'])
print(merged.loc[merged['state'].isnull(), 'state/region'])
# 2448     PR
# 2449     PR
# 2450     PR
# 2451     PR
# 2452     PR
#        ...
# 2539    USA
# 2540    USA
# 2541    USA
# 2542    USA
# 2543    USA
# Name: state/region, Length: 96, dtype: object

# unique values
print(merged.loc[merged['state'].isnull(), 'state/region'].unique())
# ['PR' 'USA']

merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
print(merged.isnull().any())
# state/region    False
# ages            False
# year            False
# population       True
# state           False
# dtype: bool

final = pd.merge(merged, areas, on='state', how='left')
print(final.tail())
#      state/region     ages  year   population          state  area (sq. mi)
# 2539          USA    total  2010  309326295.0  United States            NaN
# 2540          USA  under18  2011   73902222.0  United States            NaN
# 2541          USA    total  2011  311582564.0  United States            NaN
# 2542          USA  under18  2012   73708179.0  United States            NaN
# 2543          USA    total  2012  313873685.0  United States            NaN

print(final.isnull().any())
# state/region     False
# ages             False
# year             False
# population        True
# state            False
# area (sq. mi)     True
# dtype: bool

print(final['state'][final['area (sq. mi)'].isnull()].unique())
# ['United States']

final.dropna(inplace=True)
print(final.tail())
#      state/region     ages  year  population        state  area (sq. mi)
# 2491           PR  under18  2010    896945.0  Puerto Rico         3515.0
# 2492           PR  under18  2011    869327.0  Puerto Rico         3515.0
# 2493           PR    total  2011   3686580.0  Puerto Rico         3515.0
# 2494           PR  under18  2012    841740.0  Puerto Rico         3515.0
# 2495           PR    total  2012   3651545.0  Puerto Rico         3515.0

data2010 = final.query("year == 2010 & ages == 'total'")  # NumExpr installed required
print(data2010.tail())
#      state/region   ages  year  population          state  area (sq. mi)
# 2298           WA  total  2010   6742256.0     Washington        71303.0
# 2309           WV  total  2010   1854146.0  West Virginia        24231.0
# 2394           WI  total  2010   5689060.0      Wisconsin        65503.0
# 2405           WY  total  2010    564222.0        Wyoming        97818.0
# 2490           PR  total  2010   3721208.0    Puerto Rico         3515.0

data2010.set_index('state', inplace=True)
print(data2010.tail())
# state
# Washington              WA  total  2010   6742256.0        71303.0
# West Virginia           WV  total  2010   1854146.0        24231.0
# Wisconsin               WI  total  2010   5689060.0        65503.0
# Wyoming                 WY  total  2010    564222.0        97818.0
# Puerto Rico             PR  total  2010   3721208.0         3515.0

density = data2010['population'] / data2010['area (sq. mi)']
print(density.tail())
# state
# Washington         94.557817
# West Virginia      76.519582
# Wisconsin          86.851900
# Wyoming             5.768079
# Puerto Rico      1058.665149
# dtype: float64

density.sort_values(ascending=False, inplace=True)
print(density.head())
# state
# District of Columbia    8898.897059
# Puerto Rico             1058.665149
# New Jersey              1009.253268
# Rhode Island             681.339159
# Connecticut              645.600649
# dtype: float64
