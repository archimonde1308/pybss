from numpy import dot,diag,eye,zeros,cov,newaxis,sqrt
from numpy.linalg import svd

def gs_proj(u, v):
    '''
    Projection for Gram-Schimdt orthogonalization. Assumes that
    input vector U has nonzero norm; dot(u,u) cannot be zero.
    '''
    assert np.dot(u,u) > 0
    return u * np.dot(v,u) / np.dot(u,u)


def gram_schmidt(V):
    '''
    Perfoms Gram-Schmidt orthogonalization on the vectors in V.
    Returns an orthogonalized copy of V.
    '''
    V = 1.0 * V
    U = np.copy(V)
    for i in xrange(1, V.shape[1]):
        for j in xrange(i):
            U[:,i] -= gs_proj(U[:,j], V[:,i])
    # normalize column
    den=(U**2).sum(axis=0) **0.5
    E = U/den
    return E


def whitening_matrix(X,n):
    '''
    Performs dimensionality reduction (via PCA) and returns the whitening/dewhitening
    matrices for the data in matrix X.

    +Whitening matrix will be of size n x X.shape[0]
    +Dewhitening matrix will be of size X.shape[0] x n

    INPUT:
    ------
    X : array, required
        N x t data matrix

    n : integer, required
        resulting dimension of whitened matrix (n <= N)

    OUTPUT:
    ------
    Kw : array
        Whitening matrix; will be size n x X.shape[0]

    Kd : array
        Dewhitening matrix; will be size X.shape[0] x n

    Usage : Kw,Kd = whitening_matrix(X,n)
    '''
    if(n > X.shape[0]):
        n = X.shape[0]
    U,D,Vt = svd(dot(X,X.T)/X.shape[1],full_matrices=False)
    return dot(diag(1.0/sqrt(D[0:n])),U[:,0:n].T),dot(U[:,0:n],diag(sqrt(D[0:n])))


# NEED TO CENTER MATRICES FIRST
def lagged_covariance(X, max_lag):
    '''
    Generates a dictionary of lagged covariance matrices of matrix X, for
    lags 0,1,2,...,max_lag.

    INPUT:
    ------
    X : array, required
        N x t data matrix

    max_lag : integer, required
        covariance matrices will be computed for lags 0,1,...,max_lag

    OUTPUT:
    ------
    R_tau : dictionary, keyed on the lag, with values equal to the lagged
        covariance of X with the lagged version of itself.
    '''
    R_tau = {}
    R0 = dot(X,X.T)
    R_tau[0] = R0
    t = len(X[0,:])
    dim = X.shape[0]
    for tau in range(1,max_lag):
        for i in range(tau,t):
            X_t = X[:,0:t-tau]
            X_ttau = X[:,tau:t]
            # center the lag matrices
            X_t = X_t - X_t.mean(axis=1)[:,newaxis]
            X_ttau = X_ttau - X_ttau.mean(axis=1)[:,newaxis]
            # replace with np.cov(X_t,X_ttau)[0:dim,dim::]
            #Rt = dot(X_t,X_ttau.T)
            R_tau[tau] = cov(X_t,X_ttau)[0:dim,dim::]
    return R_tau


def joint_diagonalizer(X_dict):
    '''
    Computes a joint diagonalizing matrix, via Givens rotations, for a dictionary
    of matrices in X_dict.  Returns the diagonalizing matrix, NOT the diagonalized
    versions of the matrices in X_dict
    '''
    M_ijcsr={}
    n_matrices = len(X_dict.keys()) # formerly K
    dim = len(X_dict[0])    # formerly N
    D = eye(dim)
    for i in range(0,dim):
        G_ij = zeros((2,2))
        for j in range(i+1,dim):
            for k in xrange(0,n_matrices):
                h = zeros((1,2))
                h[0,0] = X_dict[k][i,i]-X_dict[k][j,j]
                h[0,1] = X_dict[k][i,j]+X_dict[k][j,i]
                # G_ij is a 2x2 symmetric matrix for each i,j pair
                G_ij = G_ij + dot(h.T,h)
            # now compute the rotational parameters
            u_ij, s_ij, v_ij = svd(G_ij)
            x,y = v_ij[s_ij.argmax()]
            r = (x**2 + y**2)**0.5
            c = ((x+r)/(2*r))**0.5
            s = y/((2*r*(x+r))**0.5)
            M = eye(dim)
            M[i,i] = c
            M[j,j] = c
            M[i,j] = -s
            M[j,i] = s
            M_ijcsr[i,j] = M
            D = dot(D,M)
    return D
