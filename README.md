# NumPy-Pandas-VanderPlas-2023

## Use %timeit command in IPython:
```ipython
from my_module import my_func
%timeit my_func(1, 2)
# 45.7 µs ± 1.67 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```

## Get attributes, methods, and functions in IPython:
```ipython
import my_module
my_module.<TAB>
```

## Get the source code in IPython:
```ipython
from my_module import my_func
my_func??
# Signature: my_func(x, y)
# Source:   
# def my_func(x, y):
#     """
#     It is my function
#     """
#     for i in range(1000):
#         x += 1
#         y += 1
#     return x + y
# File:      ~/PycharmProjects/NumPy-Pandas-Matplotlib-Scikit-learn-VanderPlas-2023/my_module.py
# Type:      function
```

## Get the description in IPython:
```ipython
my_func?
# Signature: my_func(x, y)
# Docstring: It is my function
# File:      ~/PycharmProjects/NumPy-Pandas-Matplotlib-Scikit-learn-VanderPlas-2023/my_module.py
# Type:      function
```

## Install line_profiler:
```
pip install line_profiler
```

## Use line_profiler in IPython:
```ipython
from my_module import my_func
%load_ext line_profiler
%lprun -f my_func my_func(1, 2)
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#      6                                           def my_func(x, y):
#      7                                               """
#      8                                               It is my function
#      9                                               """
#     10      1000    1799776.0   1799.8     31.6      for i in range(1000):
#     11      1000    1879619.0   1879.6     33.0          x += 1
#     12      1000    2012540.0   2012.5     35.3          y += 1
#     13         1       2375.0   2375.0      0.0      return x + y
```

## Install memory_profiler:
```
pip install memory_profiler
```

## Use memory_profiler in IPython:
```ipython
from my_module import my_func
%load_ext memory_profiler
%memit my_func(1, 2)
# peak memory: 74.48 MiB, increment: 0.00 MiB
peak memory: 74.48 MiB, increment: 0.00 MiB
%mprun -f my_func my_func(1, 2)
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
#      6     82.1 MiB     82.1 MiB           1   def my_func(x, y):
#      7                                             """
#      8                                             It is my function
#      9                                             """
#     10     82.1 MiB      0.0 MiB        1001       for i in range(1000):
#     11     82.1 MiB      0.0 MiB        1000           x += 1
#     12     82.1 MiB      0.0 MiB        1000           y += 1
#     13     82.1 MiB      0.0 MiB           1       return x + y
```

## IPython documentation:
https://ipython.readthedocs.io/en/st

## NumPy website:
https://numpy.org/

## Pandas website:
https://pandas.pydata.org/

## Time series in Pandas:
https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

## Effective Pandas
https://leanpub.com/effective-pandas

## PyVideo (NumPy, Pandas, scikit-learn, Matplotlib, Plotly, etc)
https://pyvideo.org/search?q=pandas

## Xarray website: 
Pandas-style N-dimensional arrays https://xarray.pydata.org/

## Matplotlib website
https://matplotlib.org/stable/index.html

## Matplotlib gallery
https://matplotlib.org/stable/gallery/

## "Ten Simple Rules for Better Figures" by Nicholas Rougier, Michael Droettboom, and Philip Bourne, 2014
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833

## Choosing Colormaps in Matplotlib
https://matplotlib.org/stable/tutorials/colors/colormaps.html

## Annotations in Matplotlib
https://matplotlib.org/stable/tutorials/text/annotations.html

## Customizing Matplotlib with style sheets and rcParams
https://matplotlib.org/stable/tutorials/introductory/customizing.html

## Seaborn website
http://seaborn.pydata.org/

## Seaborn example gallery
https://seaborn.pydata.org/examples/index.html

# PyViz: Python tools for data visualization
https://pyviz.org/
