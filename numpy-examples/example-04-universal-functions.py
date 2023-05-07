import numpy as np
from scipy import special

# Create:
# array([0, 1, 2, 3])
x = np.arange(4)
# Add 5:
# array([5, 6, 7, 8])
x + 5
# Subtract 5:
# array([-5, -4, -3, -2])
x - 5
# Multiply by 2:
# array([0, 2, 4, 6])
x * 2
# Divide by 2:
# array([0. , 0.5, 1. , 1.5])
x / 2
# Divide to get integers (floor divide):
# array([0, 0, 1, 1], dtype=int32)
x // 2
# Negative:
# array([ 0, -1, -2, -3])
-x
# Power:
# array([0, 1, 4, 9])
x ** 2
# Modulo:
# array([0, 1, 0, 1], dtype=int32)
x % 2
# Combine operations:
# array([-1.  , -2.25, -4.  , -6.25])
-(0.5*x + 1) ** 2

# Create:
# array([-2, -1,  0,  1,  2])
x = np.array([-2, -1, 0, 1, 2])
# Absolute:
# array([2, 1, 0, 1, 2])
abs(x)

# Create:
# array([3.-4.j, 4.-3.j, 2.+0.j, 0.+1.j])
x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
# Absolute complex numbers:
# array([5., 5., 2., 1.])
abs(x)

rng = np.random.default_rng(seed=42)
big_array = rng.random(1_000_000)
small_array = rng.random(1_000)
# Compare abs and np.abs
# In IPython:
# %timeit abs(big_array)
# 4.48 ms ± 520 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# %timeit np.abs(big_array)
# 3.27 ms ± 617 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# %timeit abs(small_array)
# 946 ns ± 13.3 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
# %timeit np.abs(small_array)
# 933 ns ± 9.16 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

# Create:
# array([0.        , 1.57079633, 3.14159265])
theta = np.linspace(0, np.pi, 3)
# Sine:
# array([0.0000000e+00, 1.0000000e+00, 1.2246468e-16])
np.sin(theta)
# Cosine:
# array([ 1.000000e+00,  6.123234e-17, -1.000000e+00])
np.cos(theta)
# Tangent:
# array([ 0.00000000e+00,  1.63312394e+16, -1.22464680e-16])
np.tan(theta)

x = [-1, 0, 1]
# Acrsin:
# array([-1.57079633,  0.        ,  1.57079633])
np.arcsin(x)
# Arccos:
# array([3.14159265, 1.57079633, 0.        ])
np.arccos(x)
# Arctan:
# array([-0.78539816,  0.        ,  0.78539816])
np.arctan(x)

x = [1, 2, 3]
# e^x:
# array([ 2.71828183,  7.3890561 , 20.08553692])
np.exp(x)
# 2^x:
# array([2., 4., 8.])
np.exp2(x)
# 3^x:
# array([ 3.,  9., 27.])
np.power(3., x)

x = [1, 2, 4, 10]
# ln(x):
# array([0.        , 0.69314718, 1.38629436, 2.30258509])
np.log(x)
# log2(x):
# array([0.        , 1.        , 2.        , 3.32192809])
np.log2(x)
# log10(x)
# array([0.        , 0.30103   , 0.60205999, 1.        ])
np.log10(x)

x = [0, 0.0001, 0.001, 0.01, 0.1]
# exp(x) - 1:
# array([0.00000000e+00, 1.00005000e-04, 1.00050017e-03, 1.00501671e-02,
#        1.05170918e-01])
np.expm1(x)
# array([0.00000000e+00, 1.00005000e-04, 1.00050017e-03, 1.00501671e-02,
#        1.05170918e-01])
np.exp(x) - 1
# log(x + 1):
# array([0.00000000e+00, 9.99950003e-05, 9.99500333e-04, 9.95033085e-03,
#        9.53101798e-02])
np.log1p(x)
# array([0.00000000e+00, 9.99950003e-05, 9.99500333e-04, 9.95033085e-03,
#        9.53101798e-02])
np.log(np.array(x) + 1)

x = [1, 5, 10]
# gamma(x):
# array([1.0000e+00, 2.4000e+01, 3.6288e+05])
special.gamma(x)
# ln|gamma(x)|:
# array([ 0.        ,  3.17805383, 12.80182748])
special.gammaln(x)
# beta(x, 2):
# array([0.5       , 0.03333333, 0.00909091])
special.beta(x, 2)

x = np.array([0, 0.3, 0.7, 1.0])
# error function (integral of Gaussian):
# array([0.        , 0.32862676, 0.67780119, 0.84270079])
special.erf(x)
# its complement:
# array([1.        , 0.67137324, 0.32219881, 0.15729921])
special.erfc(x)
# its inverse:
# array([0.        , 0.27246271, 0.73286908,        inf])
special.erfinv(x)
# array([0. , 0.3, 0.7, 1. ])
special.erfinv(special.erf(x))
