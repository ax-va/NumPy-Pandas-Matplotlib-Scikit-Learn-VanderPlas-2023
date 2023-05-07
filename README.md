# NumPy-VanderPlas-2023

## IPython useful "magic" commands:
```ipython
%timeit my_func(1, 2)
```

## Install line_profiler:
pip install line_profiler

## line_profiler in IPython:
%load_ext line_profiler
%lprun -f my_func my_func(1, 2)

## Install memory_profiler:
pip install memory_profiler

## memory_profiler in IPython:
%load_ext memory_profiler
%memit my_func(1, 2)
from my_module import my_func
%mprun -f my_func my_func(1, 2)

## Matplotlib in IPython:
%matplotlib inline

## NumPy Documentation:
https://numpy.org/
