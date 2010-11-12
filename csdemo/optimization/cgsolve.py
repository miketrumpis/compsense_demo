import numpy as np

def cgsolve(A, b, tol, maxiter, verbose=False):

    if isinstance(A, np.ndarray):
        call = lambda x: np.dot(A,x)
    else:
        call = A

    n = len(b)
    x = np.zeros_like(b)
    r = b
    d = r
    delta = np.dot(r,r)
    delta0 = delta

    numiter = 0
    bestx = x
    bestres = np.sqrt(delta/delta0)
    crit = tol**2 * delta0
    while numiter < maxiter and delta > crit:
        q = call(d)

        alpha = delta / np.dot(d, q)
        x += alpha*d

        if not (numiter+1)%50:
            r = b - call(x)
        else:
            r -= alpha*q

        deltaold = delta
        delta = np.dot(r,r)
        beta = delta/deltaold
        d = r + beta*d

        numiter += 1

        if np.sqrt(delta/delta0) < bestres:
            bestx = x
            bestres = np.sqrt(delta/delta0)

        if not numiter%50 and verbose:
            print 'cg: Iter', numiter,
            print 'Best resid = %8.3f, Current resid = %8.3f'%(bestres, np.sqrt(delta/delta0))

if verbose:
    print 'cg: Iterations =', numiter, 'Best resid = %14.8f'%bestres

return bestx, bestres, numiter
    
    
