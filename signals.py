from numpy import zeros
from numpy.random import choice,randint
import pickle

import sources

def moving_average(dim = 1,n_samples = 1000,p = [0.7],mu = 0.0):
    '''
    Generates an array of 1D moving average models (not a vector model!) given the mean,
    lag parameters and desired number of samples.

    INPUT:
    ------
        dim : integer,optional
            number of MA signals

        n_samples : integer,optional
            number of sample points for each signal

        p : array-like, optional
            array of lag parameters; determines model order

        mu : float, optional
            mean value of the sample distribution

    OUTPUT:
    ------
        t : numpy array
            1D array of sample times, normalized to [0,1]

        X : numpy array
            array of MA models
    '''
    wt = sources.unitsources(nSources = (0,0,dim), nSamples = n_samples)
    X = zeros(wt.shape)
    t = zeros((1,len(wt[0,:])))

    for i in xrange(0,n_samples):
        c = zeros((dim,1))
        for j in xrange(0,len(p)):
            if i-j <1:
                pass
            else:
                c[:,0] = c[:,0] + p[j]*wt[:,i-j-1]
        X[:,i] = mu + wt[:,i] + c[:,0]
        t[0,i] = 1.0*i/n_samples

    return t,X


def autoregressive(dim = 1,n_samples = 1000,p = [0.7],mu = 0.0):
    '''
    Generates an array of 1D autoregressive models (not a vector model!) given the
    mean, lag parameters and desired number of samples.

    INPUT:
    ------
        dim : integer,optional
            number of MA signals

        n_samples : integer,optional
            number of sample points for each signal

        p : array-like, optional
            array of lag parameters; determines model order

        mu : float, optional
            mean value of the sample distribution

    OUTPUT:
    ------
        t : numpy array
            1D array of sample times, normalized to [0,1]

        X : numpy array
            array of MA models
    '''

    wt = sources.unitsources(nSources = (0,0,dim), nSamples = n_samples)
    X = zeros(wt.shape)
    t = zeros((1,len(wt[0,:])))

    for i in xrange(0,n_samples):
        c = zeros((dim,1))
        for j in xrange(0,len(p)):
            if i-j <1:
                pass
            else:
                c[:,0] = c[:,0] + p[j]*X[:,i-j-1]
        X[:,i] = mu + wt[:,i] + c[:,0]
        t[0,i] = 1.0*i/n_samples

    return t,X


def audiobooks(dim = 3,n_samples = 1000):
    '''
    Returns an array of dim speech signals of block length n_samples.  The
    audiobooks are as follows:
        +Alice In Wonderland
        +The Confessions of St. Augustine
        +Flatland
        +Huckleberry Finn
        +Moby Dick
    Each signal is chosen randomly from one of these audiobooks (with replacement),
    and each block begins at a random starting location within that audiobook.
    Each signal is standardized.
    '''
    S = zeros((dim,n_samples))
    books = ('AliceInWonderland','ConfessionsOfAugustine','Flatland','HuckFinn','MobyDick')
    for i in range(dim):
        audio_file = 'data/'+choice(books)+'.pydb'
        audio_data = pickle.load(open(audio_file,'rb'))
        # generate a random integer between 0 and len(data) - n_samples
        start_loc = randint(0,len(audio_data)-n_samples)
        S[i,:] = audio_data[start_loc:start_loc + n_samples]
        S[i,:] = (S[i,:] - S[i,:].mean())/S[i,:].std()
    return S
