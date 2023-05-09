# NumPy-VanderPlas-2023

## IPython useful "magic" commands:
```ipython
from my_module import my_func
%timeit my_func(1, 2)
# 45.7 µs ± 1.67 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```

## Install line_profiler:
```
pip install line_profiler
```

## line_profiler in IPython:
```ipython
from my_module import my_func
%load_ext line_profiler
%lprun -f my_func my_func(1, 2)
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#      1                                           def my_func(x, y):
#      2      1000    1901731.0   1901.7     31.5      for i in range(1000):
#      3      1000    1994973.0   1995.0     33.0          x += 1
#      4      1000    2140892.0   2140.9     35.4          y += 1
#      5         1       3353.0   3353.0      0.1      return x + y
```

## Install memory_profiler:
```
pip install memory_profiler
```

## memory_profiler in IPython:
```ipython
from my_module import my_func
%load_ext memory_profiler
%memit my_func(1, 2)
# peak memory: 74.48 MiB, increment: 0.00 MiB
peak memory: 74.48 MiB, increment: 0.00 MiB
%mprun -f my_func my_func(1, 2)
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
#      1     74.5 MiB     74.5 MiB           1   def my_func(x, y):
#      2     74.5 MiB      0.0 MiB        1001       for i in range(1000):
#      3     74.5 MiB      0.0 MiB        1000           x += 1
#      4     74.5 MiB      0.0 MiB        1000           y += 1
#      5     74.5 MiB      0.0 MiB           1       return x + y
```

## Matplotlib in IPython:
```ipython
%matplotlib inline
```

## NumPy Documentation:
https://numpy.org/
