""" -*- python -*- file
"""

import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
def jpeg_zigzag(nrow, ncol):
    """
    Find the indexing in an (nrow, ncol) matrix that simulate jpeg's
    zigzag indexing.

    Parameters
    ----------

    nrow, ncol : int
      Matrix dimensions

    Returns
    -------

    z : (vector) ndarray
      Vector of indices in the flattened, row-major representation of
      the matrix
    coords : ndarray
      Matrix coordinates of the indices, eg: list of (i,j) matrix coordinates
    """

    cdef cnp.ndarray[int, ndim=2] pairs = np.zeros((nrow*ncol, 2), 'i')
    cdef cnp.ndarray[int, ndim=1] z = np.zeros((nrow*ncol,), 'i')

    cdef Py_ssize_t cur, ir, ic
    cur = 0; ir = 0; ic = 0
    state = 1
    while cur < nrow*ncol:
        pairs[cur,0] = ir
        pairs[cur,1] = ic
##         z[cur] = nrow*ic + ir
        # let's count row-major
        z[cur] = ir*ncol + ic
        if state==1:
            if ic < ncol-1:
                ic += 1
            else:
                ir += 1
            state = 2
            cur += 1
        elif state==2:
            ir += 1
            ic -= 1
            state = 3
            cur += 1
        elif state==3:
            if ic > 0 and ir < nrow-1:
                ir += 1
                ic -= 1
                cur += 1
            else:
                state = 4
        elif state==4:
            if ir < nrow-1:
                ir += 1
            else:
                ic += 1
            state = 5
            cur += 1
        elif state==5:
            ir -= 1
            ic += 1
            state = 6
            cur += 1
        else:
            if ir > 0 and ic < ncol-1:
                ir -= 1
                ic += 1
                cur += 1
            else:
                state = 1

    return z, pairs
