import numpy as np

def psnr(orig, recon):
    """
    Calculate the PSNR between the original and reconstructed image
    """

    err = orig - recon
    l2_err = np.dot(err.ravel(), err.ravel())**0.5
    snr = 20*np.log10(256*255/l2_err)
    return snr
    
