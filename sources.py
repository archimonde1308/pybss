from numpy.random import randn,rand,randint,gamma,laplace
from numpy import pi
from numpy import tan,arctanh,sinh,arcsinh,power,sqrt,log
from numpy import shape,newaxis,vstack

def randbit(size=None):
    '''
    Generates an array of shape size of random {0,1} bits.
    '''
    if size is None:
        return randint(2)
    else:
        return randint(2,size=size)

def randspin(size=None):
    '''
    Generates an array of shape size of random {-1,1} spin variables.
    '''
    return 2*randbit(size=size) - 1;


def unitsources(n_sources=(1,1,1),n_samples=1024,sub_type='dblcosh',sup_type='invcosh'):
    '''
    Generates a sum(n_sources) x n_samples array of sources; each row is a source, and they all have
    zero mean and unit variance (up to sampling errors).

    Parameters:
    ----------
    n_sources : tuple, optional
        n_sources[0] : number of SubGaussian sources (zero is allowed)
        n_sources[1] : number of SuperGaussian sources (zero is allowed)
        n_sources[2] : number of Gaussian sources (zero is allowed)

    n_samples : number of samples (length of each row), optional

    sub_type : string, optional
	type of subgaussian distribution from which to sample; unrecognized
	distribution types default to 'dblcosh'

    sup_type : string, optional
	type of supergaussian distribution from which to sample; unrecognized
	distribution types default to 'invcosh'


    Output:
    ----------
    S : numpy array
        first n_sources[0] rows are SubGaussian (starting at 0)
        next n_sources[1] rows are SuperGaussian
        next n_sources[2] rows are Gaussian

    '''
    # for function dispatching
    super_gauss_table = {'invcosh': sinvcosh, 'laplace': slaplace, 'logistic': slogistic,
	'exparcsinh': sexparcsinh}
    sub_gauss_table = {'dblcosh': sdblcosh, 'expsinh': sexpsinh, 'gg' : sgg}
    source_list = []
    # subgaussians
    try:
        s = sub_gauss_table[sub_type](n_sources[0],n_samples)
    except KeyError:
	# default to dblcosh
        s = sdblcosh(n_sources[0],n_samples)
    source_list.append(s)
    # supergaussians
    try:
        s = super_gauss_table[sup_type](n_sources[1],n_samples)
    except KeyError:
	# default to invcosh
        s = sinvcosh(n_sources[1],n_samples)
    source_list.append(s)
    # gaussians
    source_list.append(sgauss(n_sources[2],n_samples))

    return vstack(list(filter(lambda x : len(x) > 0, source_list)))



""" Gaussian distribution """

def sgauss(n_sources,n_samples):
    '''
    Generates an n_sources x n_samples array of Gaussian sources with mean zero and unit variance.
    Simply wraps the appropriate function from numpy.rand.
    '''
    return randn(n_sources,n_samples)


""" SuperGaussian distributions """

def sinvcosh(n_sources,n_samples):
    '''
    Generates an n_sources x n_samples array of sources distributed according to:

        p(x) = (2*cosh(pi*x/2))^-1

    This yields a set of superGaussian sources (more peaked than Gaussian), and
    each source will have zero mean and unit variance.
    '''
    return (4/pi)*arctanh(tan((pi/2)*(rand(n_sources,n_samples) - 0.5)))


def slaplace(n_sources,n_samples):
    '''
    Generates an n_sources x n_samples array of sources which are Laplace distributed.

        p(x) = (1/sqrt(2))*exp(-sqrt(2)*x)
    '''
    s = laplace(size=(n_sources,n_samples))
    return s/s.std(axis=1)[:,newaxis]


def slogistic(n_sources,n_samples):
    '''
    Generates an n_sources x n_samples array of logistically distributed random
    variates with mean zero and unit variance:

        p(x) = pi*sech^2(pi*x/(2*sqrt(3)))/(4*sqrt(3))

    '''
    return -(sqrt(3)/pi)*log(1.0/rand(n_sources,n_samples) - 1.0)


def sexparcsinh(n_sources,n_samples):
    '''
    Generates and n_sources by n_samples array of sources distributed according to:

        p(x) = (1/sqrt(2*pi*a^2*(1+x/a)^2))*exp(-(arcsinh(y/a)^2)/2)

    This yields a set of superGaussian sources; each source is standardized to
    have unit variance (p(x) above does not).
    '''
    s = sinh(randn(n_sources,n_samples))
    return s/s.std(axis=1)[:,newaxis]


""" SubGaussian distributions """

def sdblcosh(n_sources,n_samples):
    '''
    Generates an n_sources x n_samples array of sources distributed according to:

        p(x) = exp(-x^2)*cosh(x*sqrt(2))/sqrt(pi*e)

    This yields a set of subGaussian sources (flatter top that Gaussian), and
    each source will have zero mean and unit variance.
    '''
    return (.5**.5)*(randn(n_sources,n_samples) + randspin(size=(n_sources,n_samples)))


def sexpsinh(n_sources,n_samples):
    '''
    Generates an n_sources x n_samples set of sources distributed according to:

        p(x) = sqrt((1 + sinh(x)^2)/(2*pi))*exp(-sinh^2(x)/2)

    Each source is standardize to have unit variance.
    '''
    s = arcsinh(randn(n_sources,n_samples))
    return s/s.std(axis=1)[:,newaxis]


def sgg(n_sources,n_samples):
    '''
    Generates an n_sources x n_samples set of sources distributed according to:

        p(x) = (8/(2*Gamma[1/8]))*exp(-x^8),

    i.e. a generalized Gaussian distribution with shape parameter = 8.
    '''
    gg = randspin(size=(n_sources,n_samples))*power(abs(gamma(shape=1.0/8.0,scale=1.0,size=(n_sources,n_samples))),1.0/8.0)
    return gg/gg.std(axis=1)[:,newaxis]
