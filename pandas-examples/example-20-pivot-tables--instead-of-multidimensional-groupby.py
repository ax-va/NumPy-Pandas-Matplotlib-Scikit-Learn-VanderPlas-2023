import numpy as np
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')

titanic.head()
#    survived  pclass     sex   age  sibsp  parch     fare embarked  class    who  adult_male deck  embark_town alive  alone
# 0         0       3    male  22.0      1      0   7.2500        S  Third    man        True  NaN  Southampton    no  False
# 1         1       1  female  38.0      1      0  71.2833        C  First  woman       False    C    Cherbourg   yes  False
# 2         1       3  female  26.0      0      0   7.9250        S  Third  woman       False  NaN  Southampton   yes   True
# 3         1       1  female  35.0      1      0  53.1000        S  First  woman       False    C  Southampton   yes  False
# 4         0       3    male  35.0      0      0   8.0500        S  Third    man        True  NaN  Southampton    no   True

titanic.groupby('sex')[['survived']].mean()
#         survived
# sex
# female  0.742038
# male    0.188908

# Approximately, 3/4 of females and 1/5 of males were survived.

titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
# class      First    Second     Third
# sex
# female  0.968085  0.921053  0.500000
# male    0.368852  0.157407  0.135447

# Get an equivalent result, but by a more readable code

# Two lines of code are equivalent
titanic.pivot_table('survived', index='sex', columns='class', aggfunc='mean')
titanic.pivot_table('survived', index='sex', columns='class')  # default: aggfunc='mean'
# class      First    Second     Third
# sex
# female  0.968085  0.921053  0.500000
# male    0.368852  0.157407  0.135447

# multilevel pivot tables

age = pd.cut(titanic['age'], [0, 18, 80])
# 0      (18.0, 80.0]
# 1      (18.0, 80.0]
# 2      (18.0, 80.0]
# 3      (18.0, 80.0]
# 4      (18.0, 80.0]
#            ...
# 886    (18.0, 80.0]
# 887    (18.0, 80.0]
# 888             NaN
# 889    (18.0, 80.0]
# 890    (18.0, 80.0]
# Name: age, Length: 891, dtype: category
# Categories (2, interval[int64, right]): [(0, 18] < (18, 80]]

titanic.pivot_table('survived', ['sex', age], 'class')
# class               First    Second     Third
# sex    age
# female (0, 18]   0.909091  1.000000  0.511628
#        (18, 80]  0.972973  0.900000  0.423729
# male   (0, 18]   0.800000  0.600000  0.215686
#        (18, 80]  0.375000  0.071429  0.133663

fare = pd.qcut(titanic['fare'], 2)
# 0       (-0.001, 14.454]
# 1      (14.454, 512.329]
# 2       (-0.001, 14.454]
# 3      (14.454, 512.329]
# 4       (-0.001, 14.454]
#              ...
# 886     (-0.001, 14.454]
# 887    (14.454, 512.329]
# 888    (14.454, 512.329]
# 889    (14.454, 512.329]
# 890     (-0.001, 14.454]
# Name: fare, Length: 891, dtype: category
# Categories (2, interval[float64, right]): [(-0.001, 14.454] < (14.454, 512.329]]

titanic.pivot_table('survived', ['sex', age], [fare, 'class'])
# fare            (-0.001, 14.454]                     (14.454, 512.329]
# class                      First    Second     Third             First    Second     Third
# sex    age
# female (0, 18]               NaN  1.000000  0.714286          0.909091  1.000000  0.318182
#        (18, 80]              NaN  0.880000  0.444444          0.972973  0.914286  0.391304
# male   (0, 18]               NaN  0.000000  0.260870          0.800000  0.818182  0.178571
#        (18, 80]              0.0  0.098039  0.125000          0.391304  0.030303  0.192308

# Additional pivot table options

# call signature as of Pandas 2.0.1
# def pivot_table(self,
#                 values: Any = None,
#                 index: Any = None,
#                 columns: Any = None,
#                 aggfunc: (...) -> Any | str | list[(...) -> Any | str] | dict[Hashable, (...) -> Any | str | list[(...) -> Any | str]] = "mean",
#                 fill_value: Any = None,
#                 margins: bool = False,
#                 dropna: bool = True,
#                 margins_name: Hashable = "All",
#                 observed: bool = False,
#                 sort: bool = True) -> DataFrame

# When specifying a mapping for aggfunc, the 'values' keyword is determined automatically.

titanic.pivot_table(index='sex', columns='class', aggfunc={'survived': sum, 'fare': 'mean'})
#               fare                       survived
# class        First     Second      Third    First Second Third
# sex
# female  106.125798  21.970121  16.118810       91     70    72
# male     67.226127  19.741782  12.661633       45     17    47

# Compute totals along each grouping.

titanic.pivot_table('survived', index='sex', columns='class', margins=True)
# class      First    Second     Third       All
# sex
# female  0.968085  0.921053  0.500000  0.742038
# male    0.368852  0.157407  0.135447  0.188908
# All     0.629630  0.472826  0.242363  0.383838


