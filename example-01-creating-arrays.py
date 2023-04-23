import numpy as np

print(np.__version__)  # 1.23.5

# int32 array
np_array = np.array([1, 4, 2, 5, 3])
print(np_array.dtype)  # int32

# float64 array
np_array = np.array([1.1, 2, 3, 4])
print(np_array.dtype)  # float64

# float32 array
np_array = np.array([1, 2, 3, 4], dtype=np.float32)
print(np_array.dtype)  # float32
# array([1., 2., 3., 4.], dtype=float32)

# multidimensional array from nested lists
np_array = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
# array([[1, 2, 3],
#       [4, 5, 6],
#       [7, 8, 9]])

np_array = np.zeros(10, dtype=int)
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

np_array = np.ones((2, 3), dtype=float)
# array([[1., 1., 1.],
#        [1., 1., 1.]])

np_array = np.full((2, 3), 3.14)
# array([[3.14, 3.14, 3.14],
#        [3.14, 3.14, 3.14]])

np_array = np.arange(0, 20, 2)
# array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])

# Evently spaced values
np_array = np.linspace(0, 1, 5)
# array([0.  , 0.25, 0.5 , 0.75, 1.  ])

# create uniformly distributed array with values between 0 and 1
np_array = np.random.random((3, 3))
# array([[0.89659217, 0.98819377, 0.16615932],
#        [0.60721296, 0.96274164, 0.17307982],
#        [0.23210178, 0.22946746, 0.81018554]])

# Create normaly distributed array with mean = 0, standard deviation = 1
np_array = np.random.normal(0, 1, (3, 3))
# array([[ 1.09252596, -0.86352121,  0.76450169],
#        [ 0.62465109,  0.45006704, -1.35394571],
#        [-1.00243169, -0.63577864,  0.2391272 ]])

# Create array with random integers from 0 to 9
np_array = np.random.randint(0, 10, (3, 3))
# array([[9, 8, 5],
#        [7, 6, 2],
#        [5, 9, 8]])

# Create the identaty matrix
np_array = np.eye(3)
# array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])

# Create array witout initialisation, values are taken from memory
np_array = np.empty(3)
# array([1., 1., 1.])
np_array = np.empty((3,4))
# array([[7.32510748e-312, 3.16202013e-322, 0.00000000e+000, 0.00000000e+000],
#        [1.06099790e-312, 1.04015305e-042, 1.86255286e+160, 2.67784644e+184],
#        [1.60529780e-051, 3.14254102e+179, 1.04855743e-042, 1.03012321e-071]])
