# NumPy-VanderPlas-2023

## IPython magic commands:
%time
%timeit

## line_profiler in IPython:
%load_ext line_profiler
%lprun -f sum_of_lists sum_of_lists(5000)

## memory_profiler in IPython:
%load_ext memory_profiler
%memit sum_of_lists(1000000)
%mprun -f sum_of_lists sum_of_lists(1000000)

## Documentation:
https://numpy.org/