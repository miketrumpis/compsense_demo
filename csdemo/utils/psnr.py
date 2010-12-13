import numpy as np

def psnr(orig, recon):
    """
    Calculate the PSNR between the original and reconstructed image
    """

    err = orig - recon
    l2_sq_err = np.dot(err.ravel(), err.ravel())
    ly, lx = orig.shape
    snr = 10*np.log10(255*255*lx*ly/l2_err)
    return snr
    
