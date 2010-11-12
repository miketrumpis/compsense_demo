""" -*- python -*- file
"""

import numpy as np
cimport numpy as cnp
cimport cython

def is_power_of_two(m):
    if m <= 1:
        return False
    while (m & 1) == 0:
        m = m >> 1
    return m <= 1

@cython.boundscheck(False)
def noiselet_apply_matrix(x):

    oshape = x.shape
    if len(x.shape) < 2:
        x.shape = oshape + (1,)
    m, n = x.shape

    if not ((m==1 or is_power_of_two(m)) and \
            (n==1 or is_power_of_two(n))):
        raise ValueError('Matrix dimensions must be powers of 2')

    cdef cnp.ndarray[double, ndim=2] y = np.zeros((m,n), 'd')
    print 'dimensions:', m, n
    cdef Py_ssize_t i, j, k, c, d
    cdef double temp
    j = 0
    while j < n:
        # apply noiselet vector to x[:,j]

        c = m - 1
        for i in xrange(m>>1):
            k = i ^ c
            y[i,j] = x[i,j] + x[k,j]
            y[k,j] = x[i,j] - x[k,j]
        d = c >> 1
        while d > 0:
            for i in xrange(m>>1):
                k = i ^ c ^ d
                temp = y[i,j]
                y[i,j] = y[i,j] - y[k,j]
                y[k,j] = temp + y[k,j]
            d = d >> 1
        j += 1
    
    # The following clause is not allowed, due to the strict Cython construction
##     if x.shape != oshape:
##         y.shape = oshape
##         x.shape = oshape
    return y
                
