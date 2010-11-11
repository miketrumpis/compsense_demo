import numpy as np

def jpeg_zigzag(nrow, ncol):

    map = np.zeros((nrow, ncol), 'i')
    pairs = np.zeros((nrow*ncol, 2), 'i')
    z = np.zeros((nrow*ncol,), 'i')

    cur = 0; ir = 0; ic = 0; state = 1
    while cur < nrow*ncol:
        map[ir,ic] = cur
        pairs[cur] = ir, ic
        z[cur] = nr*ic + ir
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

    return z, pairs, map
