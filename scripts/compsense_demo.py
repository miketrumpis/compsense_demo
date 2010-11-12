#!/usr/bin/env python

import numpy as np
import PIL.Image as Image
import scipy.optimize as opt

from csdemo.utils.bdct_linapprox_ordering import bdct_linapprox_ordering
from csdemo.utils.psnr import psnr
from csdemo.measurements.dct2_xforms import A_dct2, At_dct2
from csdemo.measurements.lpnoiselet_xforms import A_lpnlet, At_lpnlet
from csdemo.optimization.cgsolve import cgsolve

pic = np.array( Image.open(open('man.tiff')) )

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
q = np.arange(1,N)
np.random.shuffle(q)

# K2 = number of auxiliary measurements
# (either noiselet or more dct2)
K2 = 20000

# --- Measurement functions --------------------------------------------------
omega2 = q[:K2]

# for DCT + noiselet approximation
Phi = lambda z: A_lpnlet(z, n, omega1, omega2)
Phit = lambda z: At_lpnlet(z, n, omega2, omega2)

# for linear and tvlp approximations
om_lin = lporder[:(K1+K2)]
Phi2 = lambda z: A_dct2(z, n, om_lin)
Phi2t = lambda z: At_dct2(z, n, om_lin)


# take measurements
y = Phi(x)
y2 = Phi2(x)

# optimal l2 solution for compressed sensing
PPt = lambda z: Phi(Phit(z))
## foo = opt.fmin_cg(PPt, y, gtol=1e-8, norm=2, maxiter=200)
foo = cgsolve(PPt, y, 1e-8, 200, verbose=True)
x0 = Phit(foo)

# linear reconstruction
xlin = Phi2t(y2).reshape(n, n)

print 'K =', K1, '+', K2, '=', K1+K2
print 'DCT PSNR = %5.2f'%psnr(pic, xlin)
