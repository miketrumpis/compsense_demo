import numpy as np

def psnr(orig, recon):
    """
    Calculate the PSNR between the original and reconstructed image
    """

    m, n = orig.shape
    err = orig - recon
    err_sq = np.dot(err.flatten(), err.flatten())
    snr = 10*np.log10(255*255*m*n/err_sq)
    return snr
    
