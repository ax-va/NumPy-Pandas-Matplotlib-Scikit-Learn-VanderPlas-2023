import pandas as pd


class display:
    """ Display HTML representation of multiple objects """
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}{1}"""

    def __init__(self, *args):
        self.args = args

    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_()) for a in self.args)

    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a)) for a in self.args)


# one-to-one joins
df1 = pd.DataFrame(
    {
        'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
        'group': ['Accounting', 'Engineering', 'Engineering', 'HR']
    }
)
df2 = pd.DataFrame(
    {
        'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
        'hire_year': [2004, 2008, 2012, 2014]
    }
)
display('df1', 'df2')
# df1
#   employee        group
# 0      Bob   Accounting
# 1     Jake  Engineering
# 2     Lisa  Engineering
# 3      Sue           HR
#
# df2
#   employee  hire_year
# 0     Lisa       2004
# 1      Bob       2008
# 2     Jake       2012
# 3      Sue       2014

df3 = pd.merge(df1, df2)
#   employee        group  hire_year
# 0      Bob   Accounting       2008
# 1     Jake  Engineering       2012
# 2     Lisa  Engineering       2004
# 3      Sue           HR       2014

# many-to-one joins
df4 = pd.DataFrame(
    {
        'group': ['Accounting', 'Engineering', 'HR'],
        'supervisor': ['Carly', 'Guido', 'Steve']
    }
)
display('df3', 'df4', 'pd.merge(df3, df4)')
# df3
#   employee        group  hire_year
# 0      Bob   Accounting       2008
# 1     Jake  Engineering       2012
# 2     Lisa  Engineering       2004
# 3      Sue           HR       2014
#
# df4
#          group supervisor
# 0   Accounting      Carly
# 1  Engineering      Guido
# 2           HR      Steve
#
# pd.merge(df3, df4)
#   employee        group  hire_year supervisor
# 0      Bob   Accounting       2008      Carly
# 1     Jake  Engineering       2012      Guido
# 2     Lisa  Engineering       2004      Guido
# 3      Sue           HR       2014      Steve

# many-to-many joins
df5 = pd.DataFrame(
    {
        'group': ['Accounting', 'Accounting', 'Engineering', 'Engineering', 'HR', 'HR'],
        'skills': ['math', 'spreadsheets', 'software', 'math', 'spreadsheets', 'organization']
    }
)
display('df1', 'df5', "pd.merge(df1, df5)")
# df1
#   employee        group
# 0      Bob   Accounting
# 1     Jake  Engineering
# 2     Lisa  Engineering
# 3      Sue           HR
#
# df5
#          group        skills
# 0   Accounting          math
# 1   Accounting  spreadsheets
# 2  Engineering      software
# 3  Engineering          math
# 4           HR  spreadsheets
# 5           HR  organization
#
# pd.merge(df1, df5)
#   employee        group        skills
# 0      Bob   Accounting          math
# 1      Bob   Accounting  spreadsheets
# 2     Jake  Engineering      software
# 3     Jake  Engineering          math
# 4     Lisa  Engineering      software
# 5     Lisa  Engineering          math
# 6      Sue           HR  spreadsheets
# 7      Sue           HR  organization

# on
display('df1', 'df2', "pd.merge(df1, df2, on='employee')")
# df1
#   employee        group
# 0      Bob   Accounting
# 1     Jake  Engineering
# 2     Lisa  Engineering
# 3      Sue           HR
#
# df2
#   employee  hire_year
# 0     Lisa       2004
# 1      Bob       2008
# 2     Jake       2012
# 3      Sue       2014
#
# pd.merge(df1, df2, on='employee')
#   employee        group  hire_year
# 0      Bob   Accounting       2008
# 1     Jake  Engineering       2012
# 2     Lisa  Engineering       2004
# 3      Sue           HR       2014

# left_on, right_on
df3 = pd.DataFrame(
    {
        'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
        'salary': [70000, 80000, 120000, 90000]
    }
)
# with redundant columns 'employee' and 'name'
display('df1', 'df3', 'pd.merge(df1, df3, left_on="employee", right_on="name")')
# df1
#   employee        group
# 0      Bob   Accounting
# 1     Jake  Engineering
# 2     Lisa  Engineering
# 3      Sue           HR
#
# df3
#    name  salary
# 0   Bob   70000
# 1  Jake   80000
# 2  Lisa  120000
# 3   Sue   90000
#
# pd.merge(df1, df3, left_on="employee", right_on="name")
#   employee        group  name  salary
# 0      Bob   Accounting   Bob   70000
# 1     Jake  Engineering  Jake   80000
# 2     Lisa  Engineering  Lisa  120000
# 3      Sue           HR   Sue   90000

# without redundant columns
pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)
#   employee        group  salary
# 0      Bob   Accounting   70000
# 1     Jake  Engineering   80000
# 2     Lisa  Engineering  120000
# 3      Sue           HR   90000

# left_index, right_index
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
display('df1a', 'df2a')
# df1a
#                 group
# employee
# Bob        Accounting
# Jake      Engineering
# Lisa      Engineering
# Sue                HR
#
# df2a
#           hire_year
# employee
# Lisa           2004
# Bob            2008
# Jake           2012
# Sue            2014

# These lines are equivalent
pd.merge(df1a, df2a, left_index=True, right_index=True)
df1a.join(df2a)
#                 group  hire_year
# employee
# Bob        Accounting       2008
# Jake      Engineering       2012
# Lisa      Engineering       2004
# Sue                HR       2014

# Combine left_index and right_on
display('df1a', 'df3', "pd.merge(df1a, df3, left_index=True, right_on='name')")
# df1a
#                 group
# employee
# Bob        Accounting
# Jake      Engineering
# Lisa      Engineering
# Sue                HR
#
# df3
#    name  salary
# 0   Bob   70000
# 1  Jake   80000
# 2  Lisa  120000
# 3   Sue   90000
#
# pd.merge(df1a, df3, left_index=True, right_on='name')
#          group  name  salary
# 0   Accounting   Bob   70000
# 1  Engineering  Jake   80000
# 2  Engineering  Lisa  120000
# 3           HR   Sue   90000

df6 = pd.DataFrame(
    {
        'name': ['Peter', 'Paul', 'Mary'],
        'food': ['fish', 'beans', 'bread']
    },
    columns=['name', 'food']
)
df7 = pd.DataFrame(
    {
        'name': ['Mary', 'Joseph'],
        'drink': ['wine', 'beer']
    },
    columns=['name', 'drink']
)
display('df6', 'df7', 'pd.merge(df6, df7)')
# df6
#     name   food
# 0  Peter   fish
# 1   Paul  beans
# 2   Mary  bread
#
# df7
#      name drink
# 0    Mary  wine
# 1  Joseph  beer
#
# pd.merge(df6, df7)
#    name   food drink
# 0  Mary  bread  wine

pd.merge(df6, df7, how='inner')  # default: how='inner'
#    name   food drink
# 0  Mary  bread  wine

display('df6', 'df7', "pd.merge(df6, df7, how='outer')")
# df6
#     name   food
# 0  Peter   fish
# 1   Paul  beans
# 2   Mary  bread
#
# df7
#      name drink
# 0    Mary  wine
# 1  Joseph  beer
#
# pd.merge(df6, df7, how='outer')
#      name   food drink
# 0   Peter   fish   NaN
# 1    Paul  beans   NaN
# 2    Mary  bread  wine
# 3  Joseph    NaN  beer

display('df6', 'df7', "pd.merge(df6, df7, how='left')")
# df6
#     name   food
# 0  Peter   fish
# 1   Paul  beans
# 2   Mary  bread
#
# df7
#      name drink
# 0    Mary  wine
# 1  Joseph  beer
#
# pd.merge(df6, df7, how='left')
#     name   food drink
# 0  Peter   fish   NaN
# 1   Paul  beans   NaN
# 2   Mary  bread  wine

display('df6', 'df7', "pd.merge(df6, df7, how='right')")
# df6
#     name   food
# 0  Peter   fish
# 1   Paul  beans
# 2   Mary  bread
#
# df7
#      name drink
# 0    Mary  wine
# 1  Joseph  beer
#
# pd.merge(df6, df7, how='right')
#      name   food drink
# 0    Mary  bread  wine
# 1  Joseph    NaN  beer

# overlapping column names

df8 = pd.DataFrame(
    {
        'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
        'rank': [1, 2, 3, 4]
    }
)
df9 = pd.DataFrame(
    {
        'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
        'rank': [3, 1, 4, 2]
    }
)
display('df8', 'df9', 'pd.merge(df8, df9, on="name")')
# df8
#    name  rank
# 0   Bob     1
# 1  Jake     2
# 2  Lisa     3
# 3   Sue     4
#
# df9
#    name  rank
# 0   Bob     3
# 1  Jake     1
# 2  Lisa     4
# 3   Sue     2
#
# pd.merge(df8, df9, on="name")
#    name  rank_x  rank_y
# 0   Bob       1       3
# 1  Jake       2       1
# 2  Lisa       3       4
# 3   Sue       4       2

pd.merge(df8, df9, on="name", suffixes=("_L", "_R"))
#    name  rank_L  rank_R
# 0   Bob       1       3
# 1  Jake       2       1
# 2  Lisa       3       4
# 3   Sue       4       2
