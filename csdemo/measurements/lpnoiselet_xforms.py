import numpy as np
from .realnoiselet import noiselet_apply_matrix
from .dct2_xforms import A_dct2, At_dct2

def A_lpnlet(x, n, omega1, omega2):

    lp_dct = A_dct2(x, n, omega1)
    noiselet = A_noiselet(x, omega2)
    return np.concatenate( (lp_dct, noiselet) )

def At_lpnlet(y, n, omega1, omega2):
    k1 = len(omega1)
    k2 = k1 + len(omega2)

    x_dct = At_dct2(y[:k1], n, omega1)
    x_nlet = At_noiselet(y[k1:k2], omega2, n*n)
    return x_dct + x_nlet

def A_noiselet(x, omega):
    """
    Make noiselet measurements of x, and return measurements over
    the support of omega
    """
    oshape = x.shape
    if len(x.shape) > 1:
        x.shape = -1
    n = len(x)
    w = noiselet_apply_matrix(x).reshape(n)
    y = w[omega]
    y /= np.sqrt(n)
    x.shape = oshape
    return y

def At_noiselet(y, omega, n):
    vn = np.zeros((n,), 'd')
    vn[omega] = y/np.sqrt(n)
    return noiselet_apply_matrix(vn).squeeze()
