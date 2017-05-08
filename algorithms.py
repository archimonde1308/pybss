
# coding: utf-8

# In[1]:

import numpy as np
from aautil import sources as ss
from aautil import gram_schmidt as gs
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import munkres
import cPickle
import pyica.fastica as ica


# In[2]:

def ma_signal(nSources = 1,nSamples = 1000,p = [0.7],mu = 0.0):

    """generates a moving average model given the mean, lag parameters and number of sample points
       t(output, 1-D array) : array of time points
       X(output, 1-D array) : array of sample points
       nSources(integer, optional) : number of source signals; default is 1
       nSamples(integer, optional) : number of sample points; default is 1000
       p(1-D array, optional) : set of lag parameters; default is [0.7]
       mu(float, optional) : mean value of the sample distribution; default is 0.0
    """

    wt = ss.unitsources(nSources = (0,0,nSources), nSamples = nSamples)
    X = np.zeros(wt.shape)
    t = np.zeros((1,len(wt[0,:])))

    for i in xrange(0,nSamples):
        c = np.zeros((nSources,1))
        for j in xrange(0,len(p)):
            if i-j <1:
                pass
            else:
                c[:,0] = c[:,0] + p[j]*wt[:,i-j-1]
        X[:,i] = mu + wt[:,i] + c[:,0]
        t[0,i] = 1.0*i/nSamples

    return t,X


# In[3]:

def ar_signal(nSources =1,nSamples = 1000,p = [0.7],mu = 0.0):

    """generates an autoregressive model given the mean, lag parameters and number of sample points
       t(output, 1-D array) : array of time points
       X(output, 1-D array) : array of sample points
       N(integer, optional) : number of sample points; default is 1000
       p(1-D array, optional) : set of lag parameters; default is [0.7]
       mu(float, optional) : mean value of the sample distribution; default is 0.0
    """

    wt = ss.unitsources(nSources = (0,0,nSources), nSamples = nSamples)
    X = np.zeros(wt.shape)
    t = np.zeros((1,len(wt[0,:])))

    for i in xrange(0,nSamples):
        c = np.zeros((nSources,1))
        for j in xrange(0,len(p)):
            if i-j <1:
                pass
            else:
                c[:,0] = c[:,0] + p[j]*X[:,i-j-1]
        X[:,i] = mu + wt[:,i] + c[:,0]
        t[0,i] = 1.0*i/nSamples

    return t,X


# In[4]:

def speech_sources(nSources = 3,nSamples = 1000):
    tfile = open("AliceInWonderland.pydb", 'rb')
    alice = cPickle.load(tfile)

    tfile = open("ConfessionsOfAugustine.pydb", 'rb')
    augustine = cPickle.load(tfile)

    tfile = open("Flatland.pydb", 'rb')
    flatland = cPickle.load(tfile)

    tfile = open("HuckFinn.pydb", 'rb')
    huckfinn = cPickle.load(tfile)

    tfile = open("MobyDick.pydb", 'rb')
    mobydick = cPickle.load(tfile)
    S_main = np.zeros((5,nSamples))

    int_random = np.random.randint(0,nSamples)
    alice_standard = alice[int_random:int_random+nSamples]
    alice_standard = (alice_standard-np.mean(alice_standard))/np.std(alice_standard)
    S_main[0,:] = alice_standard

    int_random = np.random.randint(0,nSamples)
    augustine_standard = augustine[int_random:int_random+nSamples]
    augustine_standard = (augustine_standard-np.mean(augustine_standard))/np.std(augustine_standard)
    S_main[1,:] = augustine_standard

    int_random = np.random.randint(0,nSamples)
    flatland_standard = flatland[int_random:int_random+nSamples]
    flatland_standard = (flatland_standard-np.mean(flatland_standard))/np.std(flatland_standard)
    S_main[2,:] = flatland_standard

    int_random = np.random.randint(0,nSamples)
    huckfinn_standard = huckfinn[int_random:int_random+nSamples]
    huckfinn_standard = (huckfinn_standard-np.mean(huckfinn_standard))/np.std(huckfinn_standard)
    S_main[3,:] = huckfinn_standard

    int_random = np.random.randint(0,nSamples)
    mobydick_standard = mobydick[int_random:int_random+nSamples]
    mobydick_standard = (mobydick_standard-np.mean(mobydick_standard))/np.std(mobydick_standard)
    S_main[4,:] = mobydick_standard

    S = S_main[0:nSources,:]
    return S


# # AMUSE

# In[5]:

def amuse(X1, tau = 1):
    """ Performs the amuse algorithm on the data matrix
    X1(required, N*p array) : Data matrix for decomposition
    S(output, 1*N array) : Extracted sources
    A(output, p*p array) : Extracted mixing matrix
    """
    Rx = np.dot(X1,X1.T)
    u,s,v = np.linalg.svd(Rx, full_matrices = False)
    psi = s**0.5
    C = np.diag(1/psi)
    Y = np.dot(C,X1)
    N = len(X1[0,:])
    #choose arbitrary lag value
    for i in xrange(tau,N):
        Y_t = Y[:,0:N-tau]
        Y_ttau = Y[:,tau:N]
    Ry = np.dot(Y_t,Y_ttau.T)
    u1,s1,v1 = np.linalg.svd((Ry + Ry.T)/2, full_matrices= False)

    S = np.dot(v1.T,Y)
    A = np.dot(np.dot(u1, np.diag(psi)), v1)
    W = []
    return A,W,S

def quality_metric(S_main,S, A):
    """Measures the performance of the algorithm by comparing input and extracted signals
    S_main(required, ndarray N*p) : input source signals
    S(required, ndarray N*p) : extracted source signals
    A(required, ndarray p*p) : extracted mixing matrix
    Q(output, ndarray p*p) : covariance matrix of S_main and S
    S_sorted(output, ndarray N*p) : Extracted source signals sorted by the Munkres algorithm which pairs highly correlated signals
    A_sorted(output, ndarray N*p) : Extracted mixing matrix sorted by the Munkres algorithm
    """
    m = munkres.Munkres()
    Q = np.corrcoef(S_main,S)[0:len(S[:,0]),len(S[:,0])::]
    indexes = m.compute(-abs(Q))
    qual = 1-(sum([abs(Q)[x[0],x[1]] for x in indexes])/len(indexes))
    S_sorted = np.zeros(S.shape)
    A_sorted = np.zeros(A.shape)
    for keys,values in  indexes:
        S_sorted[keys,:] = S[values,:]
        A_sorted[:,keys] = A[:,values]
    return qual, A_sorted, S_sorted


# In[6]:

def pca(X):
    u,s,v = np.linalg.svd(X, full_matrices= False)
    A = u
    S = np.zeros(X.shape)
    for i in xrange(0,len(s)):
        S[i,:] = s[i]*v[i,:]
    W = A.T
    return A,W,S


# # SOBI

# In[7]:

import numpy as np
from pyica.fastica import whiteningmatrix
def generate_covariance_matrices(X, tau_len):
#1. Generating the covariance matrices for K lags
    R_tau = {}
    R0 = np.dot(X,X.T)
    R_tau[0] = R0
    u,s,v = np.linalg.svd(R0, full_matrices = False)
    kw,kd = whiteningmatrix(R0, len(X[:,0]))
    N = len(X[0,:])
    for k in xrange(0,tau_len):
        tau = k
        for i in xrange(tau,N):
            X_t = X[:,0:N-tau]
            X_ttau = X[:,tau:N]
            Rt = np.dot(X_t,X_ttau.T)
            R_tau[k] = Rt
    return R_tau

#2. Compute approximate joint diagonalizer for a set of matrices
def estimate_givens_matrix(R_tau):
    M_ijcsr={}
    K = len(R_tau.keys())
    N = len(R_tau[0])
    D = np.eye(N)
    for i in xrange(0,N):
        G_ij = np.zeros((2,2))
        for j in xrange(i+1,N):
            for k in xrange(0,K):
                h = np.zeros((1,2))
                h[0,0] = R_tau[k][i,i]-R_tau[k][j,j]
                h[0,1] = R_tau[k][i,j]+R_tau[k][j,i]
# h is a vector corresponding to the 'i'th and 'j' th row and column elements in each covariance matrix

                G_ij = G_ij + np.dot(h.T,h)
#Gij is a 2x2 symmetrix matrix for each i,j pair
            u_ij, s_ij, v_ij = np.linalg.svd(G_ij)
            x,y = v_ij[s_ij.argmax()]
            r = (x**2 + y**2)**0.5
            c = ((x+r)/(2*r))**0.5
            s = y/((2*r*(x+r))**0.5)
            M = np.eye(N)
            M[i,i] = c
            M[j,j] = c
            M[i,j] = -s
            M[j,i] = s
            M_ijcsr[i,j] = M
            D = np.dot(D,M)
    return D

#3. Compute components using the approximate joint diagonalizer and whitening/dewhitening matrices
def run_SOBI(X, tau_len = 15):
    kw,kd = whiteningmatrix(X, len(X[:,0]))
    Z = np.dot(kw,X)
    R_tau = generate_covariance_matrices(Z,tau_len)
    D = estimate_givens_matrix(R_tau)
    S_extracted = np.dot(D.T,Z)
    A_extracted = np.dot(kd,D)
    W_extracted = np.dot(D.T,kw)
    return A_extracted,W_extracted, S_extracted


# # FFDIAG

# In[8]:

import numpy as np
def generate_covariance_matrices(X, tau_len):
    #1. Generating the covariance matrices for "tau_len" lags
    R_tau = {}
    R0 = np.dot(X,X.T)
    R_tau[0] = R0
    u,s,v = np.linalg.svd(R0, full_matrices = False)
    N = len(X[0,:])
    for k in xrange(0,tau_len):
        tau = k
        for i in xrange(tau,N):
            X_t = X[:,0:N-tau]
            X_ttau = X[:,tau:N]
            Rt = np.dot(X_t,X_ttau.T)
            R_tau[k] = Rt
    return R_tau

def generate_update_term(R_tau):
    Dk = {}
    Ek = {}
    N = len(R_tau[0])
    K = len(R_tau.keys())
    for keys in R_tau.keys():
        Dk[keys] = np.diag(np.diag(R_tau[keys]))
        Ek[keys] = R_tau[keys] - Dk[keys]
    W = np.zeros((N,N))
    for i in xrange(0,N):
        for j in xrange(0,N):
            z = np.zeros(W.shape)
            y = np.zeros(W.shape)
            for k in xrange(0,K):
                z[i,j] += Dk[k][i,j]
                y[i,j] += Dk[k][j,j]*Ek[k][i,j]
            if i == j:
                pass
            else:
                W[i][j] = (z[i,j]*y[j,i]-z[i,i]*y[i,j])/(z[j,j]*z[i,i]-z[i,j]*z[i,j])
    return W

def FFDIAG(X, tau_len = 10):
    if tau_len == None:
        tau_len = 10
    R_tau = generate_covariance_matrices(X,tau_len)
    N = len(R_tau[0])
    K = len(R_tau.keys())
    W = np.zeros((N,N))
    V = np.zeros((N,N))
    C = R_tau
    niter = 0
    theta = 0.9
    eps = 1.0
    while eps>1.0e-10 or niter<100:
        niter+=1
        Vn1 = V
        V = np.dot((np.eye(N) + W), V)
        for k in xrange(0,K):
            C[k] = np.dot(np.dot((np.eye(N) + W),C[k]),(np.eye(N) + W).T)
        W = generate_update_term(C)
        W = W*theta/np.linalg.norm(W)
        delta = 0
        for i in xrange(0,N):
            for j in xrange(0,N):
                if i == j:
                    pass
                else:
                    delta += (V[i][j]-Vn1[i][j])**2
        eps = delta/(N*(N-1))

    ut = np.dot(V,X)
    return C,V,ut


# # Orthogonal FFDIAG

# In[9]:

import numpy as np
from pyica.fastica import whiteningmatrix
from scipy.linalg import expm
def ortho_generate_covariance_matrices(X, tau_len):
#1. Generating the covariance matrices for "tau_len" lags
    R_tau = {}
    R0 = np.dot(X,X.T)
    R_tau[0] = R0
    u,s,v = np.linalg.svd(R0, full_matrices = False)
    N = len(X[0,:])
    for k in xrange(0,tau_len):
        tau = k
        for i in xrange(tau,N):
            X_t = X[:,0:N-tau]
            X_ttau = X[:,tau:N]
            Rt = np.dot(X_t,X_ttau.T)
            R_tau[k] = Rt
    return R_tau

def ortho_generate_update_term(R_tau):
    Dk = {}
    Ek = {}
    N = len(R_tau[0])
    K = len(R_tau.keys())
    for keys in R_tau.keys():
        Dk[keys] = np.diag(np.diag(R_tau[keys]))
        Ek[keys] = R_tau[keys] - Dk[keys]
    W = np.zeros((N,N))
    numer = np.zeros((N,N))
    denom = np.zeros((N,N))
    # compute W
    for i in xrange(0,N):
        for j in xrange(0,N):
            for keys in R_tau.keys():
                numer[i,j] +=  Ek[keys][i,j]*(Dk[keys][i,i] - Dk[keys][j,j])
                denom[i,j] += (Dk[keys][i,i]-Dk[keys][j,j])**2
    for i in xrange(0,N):
        for j in xrange(0,N):
            if i == j:
                W[i,j] = 0.0
            else:
                W[i,j] = numer[i,j]/denom[i,j]
    return W

def ortho_FFDIAG(X, tau_len = 10):
    R_tau = ortho_generate_covariance_matrices(X,tau_len)
    N = len(R_tau[0])
    K = len(R_tau.keys())
    W = np.zeros((N,N))
    #V = np.zeros((N,N))
    V = np.eye(N)
    C = R_tau
    niter = 0
    theta = 0.9
    eps = 1.0
    while eps>1.0e-10 and niter<100:
        niter+=1
        Vn1 = V
        #V = np.dot((np.eye(N) + W), V)
        for k in xrange(0,K):
            C[k] = np.dot(np.dot((np.eye(N) + W),C[k]),(np.eye(N) + W).T)
        W = ortho_generate_update_term(C)
        if np.linalg.norm(W) > theta:
            W = (W*theta)/np.linalg.norm(W)
        # update V
        V = np.dot(expm(W),V)
        #print W
        #print np.dot((np.eye(N) + W),V)
        delta = 0
        #print V
        #print Vn1
        for i in xrange(0,N):
            for j in xrange(0,N):
                if i == j:
                    pass
                else:
                    delta += (V[i][j]-Vn1[i][j])**2
        eps = (delta/(N*(N-1)))
        #print eps
    #print niter
    #print V
    ut = np.dot(V,X)
    return np.linalg.inv(V),V, ut
