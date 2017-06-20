from numpy import corrcoef,zeros,diag
import munkres

def Q(s_true,s_est):
    '''
    Computes a measure of source reconstruction quality when true sources
    are known. Q is basically 1 minus the mean of the absolute values of
    the correlations between the true and estimated sources, once they have
    been paired by best (correlational) match.  The matching is done using
    the linear assignment problem.

    Q = 0 indicates perfect reconstruction.

    INPUT:
    ------
    s_true : array, required
        N x t matrix of known (true) sources

    s_est : array, required
        N x t matrix of estimated sources

    OUTPUT:
    ------
    Q : float
        Quality factor

    perm_indices : list
        list of integers giving the row permutations to put the estimated
        sources into row correspondence with the true sources
    '''
    # sizes
    n,t = s_true.shape
    # used for the LAP
    m = munkres.Munkres()
    # cross correlation matrix
    C = corrcoef(s_true,s_est)[0:n,n::]
    # compute the indices to match
    row_pairs = m.compute(-abs(C))
    # generate the permutation indices
    perm_indices = [0]*n
    for pair in row_pairs:
        perm_indices[pair[0]] = pair[1]
    # now compute Q using the permutation indices
    Q = 1.0 - abs(diag(C[:,perm_indices])).sum()/n
    # this gives the Q factor
    return Q,perm_indices
