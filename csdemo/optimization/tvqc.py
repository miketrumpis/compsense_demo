import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_la

def _make_TV_operators(n):
    # these are the sparse horizontal and vertical differencing
    # operators for calculating total total variation (TV)

    # these are constructed under the assumption that the 2D image
    # has been flattened from a row-major storage -- IE, two adjacent
    # pixels in a row are can be found at locations x[i] and x[i+1],
    # and two adjacent pixels in a column can be found at locations
    # x[k] and x[k+ncol]

    # horizontal difference
    # want a pattern of [-1]*n-1 + [0] repeated n times on the main diagonal
    k0 = np.concatenate( (-np.ones((n,n-1)), np.zeros((n,1))), axis=1).ravel()
    k1 = np.concatenate( (np.zeros((n,1)), np.ones((n,n-1))), axis=1).ravel()
    Dh = sparse.spdiags(np.array([k0,k1]), (0,1), n*n, n*n)

    # vertical difference
    # want a pattern of [-1]*n*(n-1) + [0]*n on the diagonal
    # and a pattern of 1s on the nth super-diagonal
    k0 = np.zeros(n*n)
    k0[:n*n-n] = -1
    k1 = np.ones(n*n)
    k1[:n] = 0
    Dv = sparse.spdiags(np.array([k0,k1]), (0,n), n*n, n*n)

    return Dh, Dv

def logbarrier(x0, A, At, b, epsilon,
               lb_tol=1e-3, mu=10, cg_tol=1e-8, cg_maxiter=200,
               be_loud=False
               ):

    newton_tol = lb_tol
    newton_maxiter = 50

    N = len(x0)
    n = np.round(np.sqrt(N))

    Dh, Dv = _make_TV_operators(n)

    x = x0
    Dhx = Dh*x; Dvx = Dv*x

    LTV = np.sqrt(Dhx**2 + Dvx**2)
    t = (0.95 * LTV) + (0.1 * LTV.max())

    # choose initial value of tau so that the duality gap after the first
    # step will be about the origial TV
    tau = (N+1)/LTV.sum()

    lb_iter = np.ceil((np.log(N+1) - np.log(lb_tol) - np.log(tau)) / np.log(mu))

    print 'Number of log barrier iterations =', lb_iter

    total_iter = 0
    for i in xrange(int(lb_iter)):
        xp, tp, n_iter = newton(
            x, t, A, At, b, epsilon, tau,
            newton_tol, newton_maxiter, cg_tol, cg_maxiter,
            be_loud=be_loud
            )
        total_iter += n_iter

        Dhxp = Dh*xp; Dvxp = Dv*xp
        tvxp = (Dhxp**2 + Dvxp**2) ** 0.5
        tvxp = tvxp.sum()

        if be_loud:
            print 'Log barrier iter =', i,
            print 'TV= %1.3f,'%tvxp,
            print 'functional = %8.3f,'%tp.sum(),
            print 'tau = %8.3f,'%tau,
            print 'total newton iter =', total_iter
            print ''

        x = xp
        t = tp
        tau = mu*tau
    print ''
    return x, t

def newton(x0, t0, A, At, b, eps, tau,
           newton_tol, newton_maxiter,
           cg_tol, cg_maxiter,
           be_loud=False):

    alpha = 0.01
    beta = 0.5

    # set up a consistent call function in case A is a
    # LinearOperator or ndarray -- also determine whether to use
    # conjugate gradient solution or LU inverse solution
    if isinstance(A, np.ndarray):
        # just force the At issue in this case
        At = A.T
        Adot = lambda z: np.dot(A,z)
        Atdot = lambda z: np.dot(At,z)
        use_cg = False
        AtA = np.dot(A.T,A)
    elif isinstance(A, sp_la.LinearOperator):
        assert isinstance(At, sp_la.LinearOperator), \
               'inconsistent arguments for A, At'
        Adot = lambda z: A.matvec(z)
        Atdot = lambda z: At.matvec(z)
        use_cg = True
    elif callable(A):
        assert callable(At), 'inconsistent arguments for A, At'
        Adot = A
        Atdot = At
        use_cg = True

    N = len(x0)
    n = np.round(np.sqrt(N))

    Dh, Dv = _make_TV_operators(n)

    # initial point
    x = x0
    t = t0
    r = Adot(x) - b
    Dhx = Dh*x
    Dvx = Dv*x
    ft = (Dhx**2 + Dvx**2 - t**2)/2.0
    fe = (np.dot(r,r) - eps**2)/2.
    f = t.sum() - (np.log(-ft).sum() + np.log(-fe).sum())/tau

    n_iter = 0
    done = False
    while not done:
        
        Atr = Atdot(r)
        ntgx = Dh.T*( Dhx/ft ) + Dv.T*( Dvx/ft ) + Atr/fe
        ntgt = -tau - t/ft
        gradf = -np.r_[ntgx, ntgt]/tau

        sig22 = 1/ft + (t**2)/(ft**2)
        sig12 = -t/(ft**2)
        sigb = 1/(ft**2) - (sig12**2)/sig22

        w1p = ntgx - \
              Dh.T*( Dhx*(sig12/sig22)*ntgt ) - \
              Dv.T*( Dvx*(sig12/sig22)*ntgt )
        if use_cg:
            
            h11pfun = lambda z: H11p(
                z, Adot, Atdot, Dh, Dv, Dhx, Dvx, sigb, ft, fe, Atr
                )
            L = sp_la.LinearOperator( (N,N), matvec=h11pfun, dtype=w1p.dtype )
            dx, i = sp_la.cg(L, w1p, tol=cg_tol, maxiter=cg_maxiter)
            if i != 0:
                if i < 0:
                    raise ValueError('Input errors to conj grad solver')
                if i > 0:
                    print 'CG not convergent after', i, 'iterations.',
                    print 'Returning previous iterate'
                    return x, t, n_iter
            # The residual is measured as the proportion of
            # the residual error energy to the energy of b (w1p, here)
            # IE, ||Ax^ - b|| / ||b||
            err = L*dx - w1p
            nrg_err = np.dot(err, err)
            nrg_b = np.dot(w1p, w1p)
            cg_res = (nrg_err / nrg_b)**0.5
            Adx = Adot(dx)
        else:
            raise NotImplemented('this clause not written')
        Dhdx = Dh*dx
        Dvdx = Dv*dx
        dt = (1.0/sig22) * (ntgt - sig12*(Dhx*Dhdx + Dvx*Dvdx))

        # minimum step size that stays in the interior
        s = 1
        xp = x + dx; tp = t + dt; rp = r + Adx
        Dhxp = Dhx + Dhdx; Dvxp = Dvx + Dvdx
        cone_iter = 0
        while ( np.sqrt(Dhxp**2 + Dvxp**2) - tp ).max() > 0 or \
              ( np.dot(rp,rp) > eps**2 ):
            s = beta*s
            xp = x + s*dx; tp = t + s*dt; rp = r + s*Adx
            Dhxp = Dhx + s*Dhdx; Dvxp = Dvx + s*Dvdx
            cone_iter += 1
            if cone_iter > 32:
                print 'Stuck on cone iterations, returning previous iterate'
                return x, t, n_iter
        # backtracking line search
        ftp = (Dhxp**2 + Dvxp**2 - tp**2)/2.0
        fep = (np.dot(rp, rp) - eps**2)/2.0
        fp = tp.sum() - (np.log(-ftp).sum() + np.log(-fep))/tau
        flin = f + alpha*s*np.dot(gradf, np.r_[dx, dt])
        back_iter = 0
        while fp > flin:
            s = beta*s
            xp = x + s*dx; tp = t + s*dt; rp = r + s*Adx
            Dhxp = Dhx + s*Dhdx; Dvxp = Dvx + s*Dvdx

            ftp = (Dhxp**2 + Dvxp**2 - tp**2)/2.0
            fep = (np.dot(rp, rp) - eps**2)/2.0
            fp = tp.sum() - (np.log(-ftp).sum() + np.log(-fep))/tau
            flin = f + alpha*s*np.dot(gradf, np.r_[dx, dt])
            back_iter += 1
            if back_iter > 32:
                print 'Stuck on backtracking line search, ',
                print 'returning previous iterate'
                return x, t, n_iter

        x = xp; t = tp; r = rp; Dvx = Dvxp; Dhx = Dhxp
        ft = ftp; fe = fep; f = fp

        dx_dt = np.r_[dx, dt]
        lambda2 = -np.dot(gradf, dx_dt)
        stepsize = s*np.linalg.norm(dx_dt, ord=2)
        n_iter += 1
        done = lambda2/2.0 < newton_tol or n_iter >= newton_maxiter

        if be_loud:
            print 'Newton iter =', n_iter, 'Functional = %8.3f'%f,
            print 'Newton decrement = %8.3f'%(lambda2/2.0),
            print 'Stepsize = %8.3e'%stepsize,
            print 'Cone iterations =', cone_iter,
            print 'Backtrack iterations =', back_iter
            if use_cg:
                print 'CG Res = %8.3e,'%cg_res
            print ''
    
    return x, t, n_iter

def H11p(v, A, At, Dh, Dv, Dhx, Dvx, sigb, ft, fe, atr):
    # A, At are callable operators on v

    Dhv = Dh*v
    Dvv = Dv*v

    a1 = (-1/ft + sigb*(Dhx**2))*Dhv + sigb*Dhx*Dvx*Dvv
    a2 = (-1/ft + sigb*(Dvx**2))*Dvv + sigb*Dhx*Dvx*Dhv

    a3 = (1/fe * At(A(v))) - (1/(fe**2) * np.dot(atr, v) * atr) 

    y = Dh.T*a1 + Dv.T*a2 - a3
    return y
        
