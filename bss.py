from numpy import dot,diag
from numpy.linalg import svd,pinv

import linalg

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

    Ry = linalg.lagged_covariance(Y,0)
    Ry = Ry[0]
    ''' DEPRECATED
    # calculate lagged covariance
    for i in xrange(tau,t):
        Y_t = Y[:,0:t-tau]
        Y_ttau = Y[:,tau:t]
    Ry = dot(Y_t,Y_ttau.T)
    '''
    uy,sy,vy = svd((Ry + Ry.T)/2, full_matrices= False)

    S = dot(vy.T,Y)
    A = dot(dot(uy, diag(psi)), vy)
    W = pinv(A)

    return A,W,S


def sobi(X, max_lag = 15):
    '''
    Blind source separation using SOBI (second-order blind independence)
    algorithm.

    INPUT:
    ------
    X : array, required
        N_sig x t matrix of signal mixtures

    max_lag : int, optional
        maximum lag (in samples) for covariance matrix calculation
    '''
    Kw,Kd = linalg.whitening_matrix(X, len(X[:,0]))
    Z = dot(kw,X)
    R_tau = linalg.lagged_covariance(Z,max_lag)
    D = linalg.joint_diagonalizer(R_tau)
    S  = dot(D.T,Z)
    A = dot(Kd,D)
    W = dot(D.T,Kw)
    return A,W,S
