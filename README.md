# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

*Blind source separation algorithms.

A pure python package for Blind Source Separation, including FastICA, AMUSE, SOBI, and FFDIAG.  Currently, only fixed-point FastICA is supported.
See the documentation in the modules for detailed usage and function arguments.  As an example, if you've installed the package and want to use
FastICA to extract n <= N sources from a data matrix X of size N x T, you can do:

>>import pybss.fastica as ica
>>A,W,S = ica.fastica(X,n)

All source code is made available under the BSD-3 license (see LICENSE).  I am indebted to the FastICA code written by Pierre Lafaye de Micheaux,
Stefan van der Walt, and Gael Varoquaux. (Their original code had a "do whatever you want with this" license).  I humbly suggest you use this code
rather than theirs (if you can find theirs); I believe mine to be actively maintained and more extensively tested.

### Who do I talk to? ###

* Repo owner Ameya Akkalkotkar, Kevin Brown (kevin.s.brown@uconn.edu)
