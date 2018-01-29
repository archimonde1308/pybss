# README #

### What is this repository for? ###

*Blind source separation algorithms.

A pure python package for Blind Source Separation, currently including the following algorithms:

+FastICA (parallel extraction and deflation, "A Fast Fixed-Point Algorithm for Independent Component Analysis",
  A. Hyvarinen and E. Oja, Neural Comput. 9(7) 1997)
+AMUSE ("AMUSE: A New Blind Identification Algorithm", L. Tong, V.C. Soon, Y.F. Huang, and R. Liu.
+SOBI ("A Blind Source Separation Technique Using Second-Order Statistics", A. Belouchrani, IEEE Trans. Sig. Proc., 45 (2) 1997)
+FFDIAG (orthogonal and unrestricted, "A Fast Algorithm for Joint Diagonalization with Non-orthogonal Transformations and its Application to
  Blind Source Separation", A. Ziehe, P. Laskov, G. Nolte, and K.-R. Mueller, JMLR 5 2004)
+FOBI ("Source Separation Using Higher Order Moments", J.-F. Cardoso)

Currently, only fixed-point FastICA is supported.  See the documentation in the modules for detailed usage and function arguments.  
As an example, if you've installed the package and want to use FastICA to extract n <= N sources from a data matrix X of size N x T, you can do:

>>import pybss.fastica as ica
>>A,W,S = ica.fastica(X,n)

All source code is made available under the BSD-3 license (see LICENSE).  We are indebted to the FastICA code written by Pierre Lafaye de Micheaux,
Stefan van der Walt, and Gael Varoquaux. (Their original code had a "do whatever you want with this" license).  We suggest you use this FastICA code
rather than theirs (if you can find theirs); mine is actively maintained and, we believe, more extensively tested.

### Who do I talk to? ###

* Repo owners Ameya Akkalkotkar, Kevin Brown (kevin.s.brown@uconn.edu)
