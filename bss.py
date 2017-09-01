from numpy import dot,diag,eye,zeros
from numpy.linalg import svd,pinv,multi_dot,norm,inv,cholesky
from scipy.linalg import expm

# remove later
import numpy as np

import linalg


def ffdiag_update(R_tau,ortho):
    '''
    Single update for the non-orthogonal FFDIAG algorithm.  Set ortho = True to
    do the proper update for the orthogonal version of the algorithm.
    '''
    Dk = {}
    Ek = {}
    dim = len(R_tau[0])
    n_lags = len(R_tau.keys())
    for tau in R_tau.keys():
        Dk[tau] = diag(diag(R_tau[tau]))
        Ek[tau] = R_tau[tau] - Dk[tau]
    W = zeros((dim,dim))
    if ortho is False:
        z = zeros(W.shape)
        y = zeros(W.shape)
        for i in range(0,dim):
            for j in range(0,dim):
                for tau in range(0,n_lags):
                    z[i,j] += Dk[tau][i,i]*Dk[tau][j,j]
                    y[i,j] += Dk[tau][j,j]*Ek[tau][i,j]
        # compute W
        for i in range(0,dim):
            for j in range(i+1,dim):
                W[i][j] = (z[i,j]*y[j,i]-z[i,i]*y[i,j])/(z[j,j]*z[i,i]-z[i,j]*z[i,j])
                W[j][i] = (z[i,j]*y[i,j]-z[j,j]*y[j,i])/(z[j,j]*z[i,i]-z[i,j]*z[i,j])
    else:
        num = zeros((dim,dim))
        den = zeros((dim,dim))
        for i in range(0,dim):
            for j in range(i+1,dim):
                for tau in range(0,n_lags):
                    num[i,j] +=  Ek[tau][i,j]*(Dk[tau][i,i] - Dk[tau][j,j])
                    den[i,j] += (Dk[tau][i,i]-Dk[tau][j,j])**2
                if i != j:
                    W[i,j] = num[i,j]/den[i,j]
                    # W must be skew-symmetric (W = -W^T)
                    W[j,i] = -W[i,j]
    return W



def amuse(X, tau = 1):
    '''
    Runs the AMUSE algorithm on the signal matrix X; extracts a full
    set of X.shape[0] sources

    INPUT:
    ------
    X : array, required
        N_sig x t matrix of signal mixtures

    tau : integer, optional
        sets the lag used for cross-correlation matrix
        computation

    OUTPUT:
    ------
    A : array
        n_sig x n_sig mixing matrix

    W : array
        n_sig x n_sig unmixing matrix

    S : array
        n_sig x t array of extracted sources
    '''

    Rx = dot(X,X.T)
    ux,sx,vx = svd(Rx, full_matrices = False)
    psi = sx**0.5
    C = diag(1/psi)
    Y = dot(C,X)
    t = X.shape[1]

    Ry = linalg.lagged_covariance(Y,tau)[tau]

    uy,sy,vy = svd((Ry + Ry.T)/2, full_matrices=False)

    S = dot(vy.T,Y)
    A = dot(dot(uy, diag(psi)), vy)
    W = pinv(A)

    return A,W,S


def sobi(X, max_lag = 15):
    '''
    Blind source separation using SOBI (second-order blind identification)
    algorithm.

    INPUT:
    ------
    X : array, required
        N_sig x t matrix of signal mixtures

    max_lag : int, optional
        maximum lag (in samples) for covariance matrix calculation
    '''
    Kw,Kd = linalg.whitening_matrix(X, len(X[:,0]))
    Z = dot(Kw,X)
    R_tau = linalg.lagged_covariance(Z,max_lag)
    D = linalg.joint_diagonalizer(R_tau)
    S  = dot(D.T,Z)
    A = dot(Kd,D)
    W = dot(D.T,Kw)
    return A,W,S


def ffdiag(X, max_lag = 10, eps = 1.0e-10, max_iter = 100):
    '''
    Blind source separation using FFDIAG.  This version does not require that
    the estimated mixing matrix be orthogonal.

    INPUT:
    ------
    X : array, required
        N_sig x t matrix of signal mixtures

    max_lag : int, optional
        maximum lag (in samples) for covariance matrix calculation

    eps : double, optional
        convergence criterion for matrix updates

    max_iter : int, optional
        maximum number of iterations/updates
    '''
    R_tau = linalg.lagged_covariance(X,max_lag)
    dim = len(R_tau[0])
    n_lags = len(R_tau.keys())
    W = zeros((dim,dim))
    V = eye(dim)
    C = R_tau
    niter = 0
    theta = 0.9
    iter_eps = 1.0

    while iter_eps > eps and niter < max_iter:
        niter += 1
        Vn1 = V

        for tau in xrange(0,n_lags):
            C[tau] = multi_dot([eye(dim) + W,C[tau],(eye(dim)+W).T])

        # update term
        W = ffdiag_update(C,False)
        if norm(W) > theta:
            W = (W*theta)/norm(W)

        # update V
        V = dot(eye(dim) + W,V)

        delta = 0
        for i in xrange(0,dim):
            for j in xrange(0,dim):
                if i == j:
                    pass
                else:
                    delta += (V[i][j]-Vn1[i][j])**2
        iter_eps = (delta/(dim*(dim-1)))

    ut = dot(V,X)
    return inv(V),V, ut


def ortho_ffdiag(X, max_lag = 10, eps = 1.0e-08, max_iter = 100):
    '''
    Blind source separation using FFDIAG.  This version (like SOBI, AMUSE, etc.)
    finds an orthogonal mixing matrix.

    INPUT:
    ------
    X : array, required
        N_sig x t matrix of signal mixtures

    max_lag : int, optional
        maximum lag (in samples) for covariance matrix calculation

    eps : double, optional
        convergence criterion for matrix updates

    max_iter : int, optional
        maximum number of iterations/updates
    '''
    R_tau = linalg.lagged_covariance(X,max_lag)
    dim = len(R_tau[0]) # formerly N
    n_lags = len(R_tau.keys()) # formerly K
    W = zeros((dim,dim))
    V = eye(dim)
    C = R_tau
    n_iter = 0
    theta = 0.9
    iter_eps = 1.0

    while iter_eps > eps and n_iter < max_iter:
        n_iter += 1
        Vn1 = V
        for tau in xrange(0,n_lags):
            C[tau] = multi_dot([eye(dim) + W,C[tau],(eye(dim)+W).T])
        W = ffdiag_update(C,True)
        if norm(W) > theta:
            W = (W*theta)/norm(W)
        # update V
        V = dot(expm(W),V)
        delta = 0
        for i in xrange(0,dim):
            for j in xrange(0,dim):
                if i != j:
                    delta += (V[i][j]-Vn1[i][j])**2
        eps = (delta/(dim*(dim-1)))

    ut = dot(V,X)
    return inv(V),V, ut


def fobi(X):
    '''
    Blind source separation via the FOBI (fourth order blind identification)
    algorithm.
    '''
    R_x = linalg.lagged_covariance(X,0)
    C = cholesky(R_x)
    Y = dot(inv(C),X)

    R_y = (norm(Y)**2)*(dot(Y,Y.T))
    u,s,Y_i = svd(R_y)

    alpha = dot(Y_i,Y)   # messages (extracted sources)
    X_i = dot(C,Y_i)     # signatures (mixing matrix)
    W_i = pinv(X_i)      # unmixing matrix
    # return things in A,W,S order to conform to other bss methods
    return X_i,W_i,alpha
