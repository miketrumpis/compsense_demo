import numpy as np
import scipy.fftpack as fftpack

def A_dct2(x, n, omega):
    """
    Take the 2-dimensional type II DCT of the flattened (n,n) image
    contained in vector x.

    Parameters
    ----------

    x : ndarray, shape (n*n,)
      flattened image vector
    n : int
      image column/row size
    omega : ndarray
      support of the output (ie, indices at which output vector is sampled)

    Returns
    -------

    y = dct2(x.reshape(n,n))[omega]
    """
    x.shape = (n,n)
    y = fftpack.dct(x, type=2, axis=0, norm='ortho')
    y = fftpack.dct(y, type=2, axis=1, norm='ortho')
    return y.flat[omega]

def At_dct2(y, n, omega):
    """
    Take the 2-dimensional type III DCT of the flattened (n,n) matrix
    contained in vector y. This is the adjoint operator to the A_dct2
    operator defined above
    """

    y2 = np.zeros((n,n), 'd')
    y2.flat[omega] = y
    w = fftpack.dct(y2, type=3, axis=0, norm='ortho')
    w = fftpack.dct(w, type=3, axis=1, norm='ortho')
    return w.flatten()
    
