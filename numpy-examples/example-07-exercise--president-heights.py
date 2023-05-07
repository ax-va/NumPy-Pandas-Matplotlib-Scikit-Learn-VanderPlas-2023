import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)
# [189 170 189 163 183 171 185 168 173 183 173 173 175 178 183 193 178 173
#  174 183 183 168 170 178 182 180 183 178 182 188 175 179 183 193 182 183
#  177 185 188 188 182 185]
print("Mean height:", heights.mean())  # Mean height: 179.73809523809524
print("Standard deviation:", heights.std())  # Standard deviation: 6.931843442745892
print("Minimum height:", heights.min())  # Minimum height: 163
print("Maximum height:", heights.max())  # Maximum height: 193

# Compute quantiles
print("0th percentile:", np.percentile(heights, 0))  # 0th percentile: 163.0
print("25th percentile:", np.percentile(heights, 25))  # 25th percentile: 174.25
# Next two lines are equivalent:
print("Median:", np.median(heights))  # Median: 182.0
print("50th percentile:", np.percentile(heights, 50))  # 50th percentile: 182.0
print("75th percentile:", np.percentile(heights, 75))  # 75th percentile: 183.0
print("100th percentile:", np.percentile(heights, 100))  # 100th percentile: 93.0

# Visualize distribution
plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number')
plt.savefig('../figures/president_heights.svg')
plt.close()



