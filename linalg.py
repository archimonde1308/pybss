import numpy as np

def proj(u, v):
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
            U[:,i] -= proj(U[:,j], V[:,i])
    # normalize column
    den=(U**2).sum(axis=0) **0.5
    E = U/den
    return E
