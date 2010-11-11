#!/usr/bin/env python

import numpy as np
import PIL.Image as Image

from csdemo.utils.bdct_linapprox_ordering import bdct_linapprox_ordering
from csdemo.utils.psnr import psnr
from csdemo.measurements.dct2_xforms import A_dct2, At_dct2

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

# measurement functions
omega2 = q[:K2]

# for linear and tvlp approximations
om_lin = lporder[:(K1+K2)]
Phi2 = lambda z: A_dct2(z, n, om_lin)
Phi2t = lambda z: At_dct2(z, n, om_lin)


# take measurements
y2 = Phi2(x)

# linear reconstruction
xlin = Phi2t(y2).reshape(n, n)

print 'DCT PSNR = %5.2f'%psnr(pic, xlin)
