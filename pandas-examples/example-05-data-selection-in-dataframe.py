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
