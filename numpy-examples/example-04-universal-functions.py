import numpy as np
from scipy import special

# Create:
x = np.arange(4)
# array([0, 1, 2, 3])
# Add 5:
x + 5
# array([5, 6, 7, 8])
# Subtract 5:
x - 5
# array([-5, -4, -3, -2])
# Multiply by 2:
x * 2
# array([0, 2, 4, 6])
# Divide by 2:
x / 2
# array([0. , 0.5, 1. , 1.5])
# Divide to get integers (floor divide):
x // 2
# array([0, 0, 1, 1])
# Negative:
-x
# array([ 0, -1, -2, -3])
# Power:
x ** 2
# array([0, 1, 4, 9])
# Modulo:
x % 2
# array([0, 1, 0, 1])
# Combine operations:
-(0.5*x + 1) ** 2
# array([-1.  , -2.25, -4.  , -6.25])

# Create:
x = np.array([-2, -1, 0, 1, 2])
# array([-2, -1,  0,  1,  2])
# Absolute:
abs(x)
# array([2, 1, 0, 1, 2])

# Create:
x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
# array([3.-4.j, 4.-3.j, 2.+0.j, 0.+1.j])
# Absolute complex numbers:
abs(x)
# array([5., 5., 2., 1.])

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
theta = np.linspace(0, np.pi, 3)
# array([0.        , 1.57079633, 3.14159265])
# Sine:
np.sin(theta)
# array([0.0000000e+00, 1.0000000e+00, 1.2246468e-16])
# Cosine:
np.cos(theta)
# array([ 1.000000e+00,  6.123234e-17, -1.000000e+00])
# Tangent:
np.tan(theta)
# array([ 0.00000000e+00,  1.63312394e+16, -1.22464680e-16])

x = [-1, 0, 1]
# Acrsin:
np.arcsin(x)
# array([-1.57079633,  0.        ,  1.57079633])
# Arccos:
np.arccos(x)
# array([3.14159265, 1.57079633, 0.        ])
# Arctan:
np.arctan(x)
# array([-0.78539816,  0.        ,  0.78539816])

x = [1, 2, 3]
# e^x:
np.exp(x)
# array([ 2.71828183,  7.3890561 , 20.08553692])
# 2^x:
np.exp2(x)
# array([2., 4., 8.])
# 3^x:
np.power(3., x)
# array([ 3.,  9., 27.])

x = [1, 2, 4, 10]
# ln(x):
np.log(x)
# array([0.        , 0.69314718, 1.38629436, 2.30258509])
# log2(x):
np.log2(x)
# array([0.        , 1.        , 2.        , 3.32192809])
# log10(x)
np.log10(x)
# array([0.        , 0.30103   , 0.60205999, 1.        ])

x = [0, 0.0001, 0.001, 0.01, 0.1]
# exp(x) - 1:
np.expm1(x)
# array([0.00000000e+00, 1.00005000e-04, 1.00050017e-03, 1.00501671e-02,
#        1.05170918e-01])
np.exp(x) - 1
# array([0.00000000e+00, 1.00005000e-04, 1.00050017e-03, 1.00501671e-02,
#        1.05170918e-01])
# log(x + 1):
np.log1p(x)
# array([0.00000000e+00, 9.99950003e-05, 9.99500333e-04, 9.95033085e-03,
#        9.53101798e-02])
np.log(np.array(x) + 1)
# array([0.00000000e+00, 9.99950003e-05, 9.99500333e-04, 9.95033085e-03,
#        9.53101798e-02])

x = [1, 5, 10]
# gamma(x):
special.gamma(x)
# array([1.0000e+00, 2.4000e+01, 3.6288e+05])
# ln|gamma(x)|:
special.gammaln(x)
# array([ 0.        ,  3.17805383, 12.80182748])
# beta(x, 2):
special.beta(x, 2)
# array([0.5       , 0.03333333, 0.00909091])

x = np.array([0, 0.3, 0.7, 1.0])
# error function (integral of Gaussian):
special.erf(x)
# array([0.        , 0.32862676, 0.67780119, 0.84270079])
# its complement:
special.erfc(x)
# array([1.        , 0.67137324, 0.32219881, 0.15729921])
# its inverse:
special.erfinv(x)
# array([0.        , 0.27246271, 0.73286908,        inf])
special.erfinv(special.erf(x))
# array([0. , 0.3, 0.7, 1. ])
