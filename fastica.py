'''
Pure python FastICA code; both deflation and parallel extraction are implemented.  Currently
only fixed-point extraction (no gradient calculations).

Created on Mar 2, 2011

@author: Kevin S. Brown, University of Connecticut

This source code is provided under the BSD-3 license.

Copyright (c) 2013, Kevin S. Brown
All rights reserved.
'''


from numpy import tanh,exp,sqrt
from numpy import max as npmax
from numpy import abs as npabs
from numpy import dot,diag,newaxis,zeros,array
from numpy.random import randn
from scipy.linalg import svd,qr,pinv2

from .linalg import whitening_matrix,row_center

"""Nonlinearities/contrast functions."""
def lc(x,alpha=1.0):
    """
    Tanh nonlinearity (tanh(alpha*x)).
    """
    return tanh(alpha*x)

def lcp(x,alpha=1.0):
    """
    Derivative of tanh nonlinearity (see lc(x,alpha)).
    """
    return alpha*(1.0 - tanh(alpha*x)**2)

def gauss(x,alpha=1.0):
    """
    Gaussian nonlinearity.
    """
    return x*exp(-(alpha*x**2)/2.0)

def gaussp(x,alpha=1.0):
    """
    Derivative of gaussian nonlinearity (see gauss(x,alpha)).
    """
    return (1.0 - alpha*x**2)*exp(-(alpha*x**2)/2.0)

def cube(x,alpha=1.0):
    """
    Cubic nonlinearity.
    """
    return x**3

def cubep(x,alpha=0.0):
    """
    Derivative of cubic nonlinearity (see cube(x,alpha)).
    """
    return 3*x**2

def skew(x,alpha=0.0):
    """
    Skew (quadratic) nonlinearity.
    """
    return x**2

def skewp(x,alpha=0.0):
    """
    Derivative of skew(quadratic) nonlinearity (see skew(x,alpha)).
    """
    return 2*x


"""Decorrelation methods."""
def decorrelation_gs(w,W,p):
    """
    Gram-schmidt orthogonalization of w against the first p rows of W.
    """
    w = w - (W[0:p,:]*dot(W[0:p,:],w.T)[:,newaxis]).sum(axis=0)
    w = w/sqrt(dot(w,w))
    return w

def decorrelation_witer(W):
    """
    Iterative MDUM decorrelation that avoids matrix inversion.
    """
    lim = 1.0
    tol = 1.0e-05
    W = W/(W**2).sum()
    while lim > tol:
        W1 = (3.0/2.0)*W - 0.5*dot(dot(W,W.T),W)
        lim = npmax(npabs(npabs(diag(dot(W1,W.T))) - 1.0))
        W = W1
    return W

def decorrelation_mdum(W):
    """
    Minimum distance unitary mapping decorrelation.
    """
    U,D,VT = svd(W)
    Y = dot(dot(U,diag(1.0/D)),U.T)
    return dot(Y,W)


"""FastICA algorithms."""
def ica_def(X, tolerance, g, g_prime, orthog, alpha, max_iter, W_init):
    """Deflationary FastICA using Gram-Schmidt decorrelation at each step. This
    function is not meant to be directly called; it is wrapped by fastica()."""
    n,p = X.shape
    W = W_init
    # j is the index of the extracted component
    for j in xrange(n):
        w = W_init[j, :]
        it = 1
        lim = tolerance + 1
        while ((lim > tolerance) & (it < max_iter)):
            wtx = dot(w, X)
            gwtx = g(wtx, alpha)
            gpwtx = g_prime(wtx, alpha)
            w1 = (X * gwtx).mean(axis=1) - gpwtx.mean() * w
            w1 = decorrelation_gs(w1, W, j)
            lim = npabs(npabs((w1 * w).sum()) - 1.0)
            w = w1
            it = it + 1
        W[j, :] = w
    return W

def ica_par_fp(X, tolerance, g, g_prime, orthog, alpha, max_iter, W_init):
    """Parallel FastICA; orthog sets the method of unmixing vector decorrelation. This
    fucntion is not meant to be directly called; it is wrapped by fastica()."""
    n,p = X.shape
    W = orthog(W_init)
    lim = tolerance + 1
    it = 1
    while ((lim > tolerance) and (it < max_iter)):
        wtx = dot(W,X)
        gwtx = g(wtx,alpha)
        gpwtx = g_prime(wtx,alpha)
        W1 = dot(gwtx,X.T)/p - dot(diag(gpwtx.mean(axis=1)),W)
        W1 = orthog(W1)
        lim = npmax(npabs(npabs(diag(dot(W1,W.T))) - 1.0))
        W = W1
        it = it + 1
    return W



def fastica(X, n_sources=None, algorithm="parallel fp", decorrelation="mdum", nonlinearity="logcosh", alpha=1.0, max_iter=500, tolerance=1e-05, W_init=None, scaled=True):
    """Perform FastICA on data matrix X.  All algorithms currently implemented are fixed point iteration
    (as opposed to gradient descent) methods.  For default usage (good for many applications), simply call:

        A,W,S = fastica(X,nSources)

    Parameters:
    ------------
    X: numpy array, size n x p
       array of data to unmix - n variables, p observations

    n_sources: int, optional
        number of sources to estimate.  if nSources < n, dimensionality reduction (via pca)
        is performed.  If nSources = None, full-rank (nSources = n) decomposition is performed.

    algorithm: string, optional
        'deflation'         : use deflational (one-at-a-time) component extraction
        'parallel fp'       : parallel extraction, fixed point iteration

    decorrelation: string, optional
        decorrelation method for parallel extraction.  (Not adjustable for deflationary ICA.)
            'mdum'  : minimum distance unitary mapping
            'witer' : like mdum, but without matrix inversion

    nonlinearity: string,optional
        function used in the approximate negentropy.  Should be one of:
            'logcosh'   : G(u) = (1/a) log[cosh[a*u]]
            'exp'        : G(u) = -(1/a) exp(-a*u^2/2)
            'skew'        : G(u) = u^3/3
            'kurtosis'  : G(u) = u^4/4

    alpha : float,optional
        parameter for 'logcosh' and 'exp' nonlinearities.  Should be in [1,2]

    max_iter: int, optional
        maximum number of iterations

    tolerance: float, optional
        tolerance at which the unmixing matrix is considered to have converged

    W_init: numpy array, size nSources x n
        initial guess for the unmixing matrix

    scaled : bool, optional
        if scaled == True, the output sources are rescaled to have unit standard deviation
        and the data units will be in the mixing matrix

    Output:
    -----------
    A : (n x nSources)
        mixing matrix (X = A*S)

    W : (nSources x n)
        unmixing matrix (W*X = S)

    S : (nSources x p)
        matrix of independent sources

    """
    algorithm_funcs = {'parallel fp':ica_par_fp, 'deflation':ica_def}
    orthog_funcs = {'mdum': decorrelation_mdum, 'witer': decorrelation_witer}

    if (alpha < 1) or (alpha > 2):
        raise ValueError("alpha must be in [1,2]")

    if nonlinearity == 'logcosh':
        g = lc
        g_prime = lcp
    elif nonlinearity == 'exp':
        g = gauss
        g_prime = gaussp
    elif nonlinearity == 'skew':
        g = skew
        g_prime = skewp
    else:
        g = cube
        g_prime = cubep

    n_mix,n_samp = X.shape

    # default is to do full-rank decomposition
    if n_sources is None:
        n_sources = n_mix
    if W_init is None:
        # start with a random orthogonal decomposition
        W_init = randn(n_sources,n_sources)

    # preprocessing (centering/whitening/pca)
    rowmeansX,X = row_center(X)
    Kw,Kd = whitening_matrix(X,n_sources)
    X = dot(Kw,X)

    # pass through kwargs
    kwargs = {'tolerance': tolerance,'g': g,'g_prime': g_prime,'orthog':orthog_funcs[decorrelation],'alpha': alpha,'max_iter': max_iter,'W_init': W_init}
    func = algorithm_funcs[algorithm]

    # run ICA
    W = func(X, **kwargs)

    # consruct the sources - means are not restored
    S = dot(W,X)

    # mixing matrix
    A = pinv2(dot(W,Kw))

    if scaled == True:
        S = S/S.std(axis=-1)[:,newaxis]
        A = A*S.std(axis=-1)[newaxis,:]

    return A,W,S
