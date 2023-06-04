import numpy as np
import pandas as pd

area = pd.Series(
    {
        'California': 423967,
        'Texas': 695662,
        'Florida': 170312,
        'New York': 141297,
        'Pennsylvania': 119280,
    }
)
pop = pd.Series(
    {
        'California': 39538223,
        'Texas': 29145505,
        'Florida': 21538187,
        'New York': 20201249,
        'Pennsylvania': 13002700,
    }
)
data = pd.DataFrame({'area': area, 'pop': pop})
#                 area       pop
# California    423967  39538223
# Texas         695662  29145505
# Florida       170312  21538187
# New York      141297  20201249
# Pennsylvania  119280  13002700

# individual Series via dictionary-style indexing:
data['area']
# California      423967
# Texas           695662
# Florida         170312
# New York        141297
# Pennsylvania    119280
# Name: area, dtype: int64

# individual Series via a column name (does not work for all cases):
data.area
# California      423967
# Texas           695662
# Florida         170312
# New York        141297
# Pennsylvania    119280
# Name: area, dtype: int64

data.pop is data["pop"]
# False
# Because DataFrame has the method "pop"

# adding a new column
data['density'] = data['pop'] / data['area']
data
#                 area       pop     density
# California    423967  39538223   93.257784
# Texas         695662  29145505   41.896072
# Florida       170312  21538187  126.463121
# New York      141297  20201249  142.970120
# Pennsylvania  119280  13002700  109.009893

data.values
# array([[4.23967000e+05, 3.95382230e+07, 9.32577842e+01],
#        [6.95662000e+05, 2.91455050e+07, 4.18960717e+01],
#        [1.70312000e+05, 2.15381870e+07, 1.26463121e+02],
#        [1.41297000e+05, 2.02012490e+07, 1.42970120e+02],
#        [1.19280000e+05, 1.30027000e+07, 1.09009893e+02]])

data.values[0]
# array([4.23967000e+05, 3.95382230e+07, 9.32577842e+01])

# Transpose the matrix to swap the rows and columns
data.T
#            California         Texas       Florida      New York  Pennsylvania
# area     4.239670e+05  6.956620e+05  1.703120e+05  1.412970e+05  1.192800e+05
# pop      3.953822e+07  2.914550e+07  2.153819e+07  2.020125e+07  1.300270e+07
# density  9.325778e+01  4.189607e+01  1.264631e+02  1.429701e+02  1.090099e+02

data.iloc[:3, :2]
#               area       pop
# California  423967  39538223
# Texas       695662  29145505
# Florida     170312  21538187

data.loc[:'Florida', :'pop']
#               area       pop
# California  423967  39538223
# Texas       695662  29145505
# Florida     170312  21538187

# Combine masking and fancy indenxing in the loc indexer:
data.loc[data.density > 120, ['pop', 'density']]
#                pop     density
# Florida   21538187  126.463121
# New York  20201249  142.970120

# Modify values:
data.iloc[0, 2] = 90
data
#                 area       pop     density
# California    423967  39538223   90.000000
# Texas         695662  29145505   41.896072
# Florida       170312  21538187  126.463121
# New York      141297  20201249  142.970120
# Pennsylvania  119280  13002700  109.009893

# Indexing refers to columns, slicing refers to rows:

data['Florida':'New York']
#             area       pop     density
# Florida   170312  21538187  126.463121
# New York  141297  20201249  142.970120

data[1:3]
#            area       pop     density
# Texas    695662  29145505   41.896072
# Florida  170312  21538187  126.463121

# Direct masking operations are interpreted row-wise:

data[data.density > 120]
#             area       pop     density
# Florida   170312  21538187  126.463121
# New York  141297  20201249  142.970120
