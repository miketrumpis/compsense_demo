import numpy as np
import scipy.sparse.linalg as sp_la


def cgsolve2(Q, b, tol, maxiter, verbose=False):

    if isinstance(Q, np.ndarray):
        Qdot = lambda x: np.dot(Q,x)
    elif isinstance(Q, sp_la.LinearOperator):
        Qdot = Q.matvec
    elif callable(Q):
        Qdot = Q
    else:
        raise ValueError('unknown operator type %s'%type(Q))

    x = np.zeros_like(b)
    d = b
    g = -b
    num_iter = 0
    crit = tol**2 * np.dot(b,b) # ???
    best_res = np.dot(b,b)**0.5
    bestx = x
    res_err_sq = 2 * crit
    while res_err_sq > crit and num_iter < maxiter:

        gtd = np.dot(g,d)
        Qd = Qdot(d)
        dQd = np.dot(d, Qd)
        alpha = -gtd/dQd

        x = x + alpha*d
        g = Qdot(x) - b

        gQd = np.dot(g, Qd)
        beta = gQd/dQd

        d = beta*d - g
        num_iter += 1

        res_err_sq = np.dot(g,g)
        if res_err_sq < best_res:
            best_res = res_err_sq**0.5
            bestx = x
        if not num_iter%50 and verbose:
            print 'cg: Iter', num_iter,
            print 'Best resid = %8.3f, Current resid = %8.3f'%(
                best_res, res_err_sq**0.5
                )

    if verbose:
        print 'cg: Iterations =', num_iter, 'Best resid = %14.8f'%best_res
        print ''


    return bestx, best_res, num_iter

def cgsolve(A, b, tol, maxiter, verbose=False):


    if isinstance(A, np.ndarray):
        Adot = lambda x: np.dot(A,x)
    elif isinstance(A, sp_la.LinearOperator):
        Adot = A.matvec
    elif callable(A):
        Adot = A
    else:
        raise ValueError('unknown operator type %s'%type(A))

    n = len(b)
    x = np.zeros_like(b)
    r = b
    d = r
    delta = np.dot(r,r)
    delta0 = np.dot(b,b) # wtf, it's the same

    numiter = 0
    bestx = x
    bestres = np.sqrt(delta/delta0)
    crit = tol**2 * delta0
    while numiter < maxiter and delta > crit:
        q = Adot(d)

        alpha = delta / np.dot(d, q)
        x = x + alpha*d

        # explicitly compute residual every 50 iterations
        if not (numiter+1)%50:
            r = b - Adot(x)
        else:
            r = r - alpha*q

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
        print ''

    return bestx, bestres, numiter
    
    
