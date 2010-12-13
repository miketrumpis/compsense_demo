#!/usr/bin/env python
from time import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from csdemo.utils.bdct_linapprox_ordering import bdct_linapprox_ordering
from csdemo.utils.psnr import psnr
from csdemo.measurements.dct2_xforms import A_dct2, At_dct2
from csdemo.measurements.lpnoiselet_xforms import A_lpnlet, At_lpnlet

import csdemo.optimization.tvqc as tvqc

def _find_image(name):
    import os.path as p
    
    name, ext = p.splitext(name)
    name += '.npy'
    data_path = p.join(p.split(__file__)[0], p.join('imagedata', name))
    a = np.load(data_path)
    return a

def compare_recons(n_coefs, image_name='cameraman',
                   return_images=False, be_loud=False):
    """
    Run demo as a function of the number of coefficients used in
    reconstructing the image.

    Returns
    -------

    DCT psnr, LPTV psnr, CS psnr

    if return_images is True, then the 3 corresponding reconstructions
    are returned following the 3-tuple of psnr measurements

    """
    
    pic = _find_image(image_name)

    n = pic.shape[0]
    x = pic.flatten().astype('d')
    N = len(x)

    # for repeatable experiments
    np.random.seed(1981)

    # for linear approximation
    lporder = bdct_linapprox_ordering(n, n)
    # low pass the DCT to contain K1 coefficients
    K1 = 1000
    omega1 = lporder[:K1]

    # "% for random projection, avoid mean" -- I think avoid sampling DCT at k=0
    q = np.arange(N)
    np.random.shuffle(q)

    # K2 = number of auxiliary measurements
    # (either noiselet or more dct2)
    K2 = n_coefs

    # --- Measurement functions ----------------------------------------------
    omega2 = q[:K2]

    # for DCT + noiselet approximation
    Phi = lambda z: A_lpnlet(z, n, omega1, omega2)
    Phit = lambda z: At_lpnlet(z, n, omega1, omega2)

    # for linear and tvlp approximations
    om_lin = lporder[:(K1+K2)]
    Phi2 = lambda z: A_dct2(z, n, om_lin)
    Phi2t = lambda z: At_dct2(z, n, om_lin)


    # take measurements
    y = Phi(x)
    y2 = Phi2(x)

    # linear DCT reconstruction
    xlin = Phi2t(y2)

    # optimal l2 solution for compressed sensing, use this
    # image as a starting point for CS optimization
    PPt = lambda z: Phi(Phit(z))
    A = LinearOperator( (K1+K2, K1+K2), matvec=PPt, dtype=y.dtype )
    y0, i = cg(A, y, tol=1e-8, maxiter=200)
    if i != 0:
        if i < 0:
            raise ValueError('bad inputs to CG algorithm')
        else:
            print 'Warning, CG did not converge after', i, 'iterations'
    x0 = Phit(y0)

    # parameters for optimization
    lb_tol = 918
    ## lb_tol = 7
    mu = 5
    cg_tol = 1e-8
    cg_maxiter = 800

    # lowpass tv recovery
    eps2 = 1e-3 * np.dot(y2,y2)**0.5
    # make LinearOperators from Phi2, Phi2t
    # Phi2 is (K1+K2, N)
    A = LinearOperator( (K1+K2, N), matvec=Phi2, dtype=y2.dtype )
    # Phi2t is (N, K1+K2)
    At = LinearOperator( (N, K1+K2), matvec=Phi2t, dtype=y2.dtype )
    print 'finding LPTV solution'
    xlptv, tlptv = tvqc.logbarrier(
        xlin, A, At, y2, eps2, lb_tol, mu, cg_tol, cg_maxiter, be_loud=be_loud
        )
    xlptv.shape = (n,n)

    # CS recovery
    eps = 1e-3 * np.dot(y,y)**0.5
    A = LinearOperator( (K1+K2, N), matvec=Phi, dtype=y.dtype )
    At = LinearOperator( (N, K1+K2), matvec=Phit, dtype=y.dtype )
    xp, tp = tvqc.logbarrier(
        x0, A, At, y, eps, lb_tol, mu, cg_tol, cg_maxiter, be_loud=be_loud
        )
    xp.shape = (n, n)

    xlin.shape = (n,n)

    dct_psnr = psnr(pic, xlin)
    lptv_psnr = psnr(pic, xlptv)
    cs_psnr = psnr(pic, xp)
    r_tuple = (dct_psnr, lptv_psnr, cs_psnr)
    if return_images:
        r_tuple += (xlin, xlptv, xp)
    return r_tuple

def compare_at(n_coefs_trials, be_loud=False, plot=False):
    """
    Run a number of trials of the comparison demo, using the the given
    sequence as the number of coefficients at each step
    """

    dct_psnrs = []
    lptv_psnrs = []
    cs_psnrs = []
    for nc in n_coefs_trials:
        dct, lptv, cs = compare_recons(nc, be_loud=be_loud)
        dct_psnrs.append(dct)
        lptv_psnrs.append(lptv)
        cs_psnrs.append(cs)
    if plot:
        import matplotlib.pyplot as pp
        f = pp.figure()
        ax = f.add_subplot(111)
        n_coefs = np.array(n_coefs_trials)
        ax.plot(n_coefs, dct)
        ax.plot(n_coefs, lptv)
        ax.plot(n_coefs, cs)
        x_min = n_coefs.min()
        x_max = n_coefs.max()
        x_width = x_max - x_min
        ax.set_xlim( (x_min - 0.05*width, x_max + 0.05*width) )
        pp.show()
    return dct_psnrs, lptv_psnrs, cs_psnrs

def show_comparison(n_coefs):
    dct, lptv, cs = compare_recons(n_coefs, be_loud=True)
    print 'K = 1000 +', n_coefs, '=', 1000+n_coefs
    print 'DCT PSNR = %5.2f'%dct
    print 'LPTV PSNR = %5.2f'%lptv
    print 'CS PSNR = %5.2f'%cs

if __name__=='__main__':
    show_comparison(20000)
    
