import numpy as np
import timeit

arr1 = np.random.random(100000)
arr2 = np.random.random(100000)

def convolve_arrays():
    return np.convolve(arr1, arr2)

time_taken = timeit.timeit(convolve_arrays, number=1)
print(f"Time taken: {time_taken:.6f} seconds")

from scipy import signal

def convolve_faster():
    return signal.fftconvolve(arr1,arr2)

time_taken = timeit.timeit(convolve_faster, number=1) *1000
print(f"Time taken: {time_taken:.6f} milliseconds")