import numpy as np
from vega_datasets import data

# Use DataFrame operations to extract rainfall as a NumPy array
rainfall_mm = np.array(
    data.seattle_weather().set_index('date')['precipitation']['2015']
)
print(len(rainfall_mm))  # 365

# Construct a mask of all rainy days
rainy = (rainfall_mm > 0)
# Construct a mask of all summer days (June 21st is the 172nd day)
days = np.arange(365)
summer = (days > 172) & (days < 262)

print("Median precip on rainy days in 2015 (mm):", np.median(rainfall_mm[rainy]))
# Median precip on rainy days in 2015 (mm): 3.8
print("Median precip on summer days in 2015 (mm): ", np.median(rainfall_mm[summer]))
# Median precip on summer days in 2015 (mm):  0.0
print("Maximum precip on summer days in 2015 (mm): ", np.max(rainfall_mm[summer]))
# Maximum precip on summer days in 2015 (mm):  32.5
print("Median precip on non-summer rainy days (mm):", np.median(rainfall_mm[rainy & ~summer]))
# Median precip on non-summer rainy days (mm): 4.1
