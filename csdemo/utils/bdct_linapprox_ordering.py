import numpy as np
from .jpgzzind import jpeg_zigzag

def bdct_linapprox_ordering(n, blocksize):
    """
    Returns the index ordering for taking linear approximations using a
    blocked DCT.  DCT coefficients within a block are taken in the standard
    JPEG zigzag order (see jpgzzind.m).  "Extra" coefficients are assigned to
    DCT blocks block-columnwise (there is never a difference of more than 1
    terms taken between each of the blocks).

    Usage: order = bdct_linapprox_ordering(n, blocksize)
    n - sidelength (image is nxn)
    blocksize - sidelength of blocks
    
    (originally) Written by: Justin Romberg, Georgia Tech
    Created: April 2007
    """
    if n % blocksize:
        raise ValueError(
            'blocksize %d must divide the edge length %d'%(blocksize, n)
            )

    b = n/blocksize
    bsq = b*b
    
    ind, coords = jpeg_zigzag(blocksize, blocksize)
    bigind = np.arange(n*n).reshape(n,n)
    order = np.zeros((n*n,), 'i')
    for i in xrange(blocksize*blocksize):
        order[i*bsq : (i+1)*bsq] = bigind[coords[i,0]:n:blocksize,
                                          coords[i,1]:n:blocksize].flatten()
    return order
