from realnoiselet import noiselet_apply_matrix

def A_noiselet(x, omega):
    n = len(x)
    w = noiselet_apply_matrix(x)
    y = w[omega]
    y /= np.sqrt(n)

def At_noiselet(y, omega, n):
    vn = np.zeros((n,), 'i')
    vn[omega] = y
    vn /= np.sqrt(n)
    return noiselet_apply_matrix(vn)
