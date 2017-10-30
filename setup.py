#!/usr/bin/env python

from distutils.core import setup,Command

setup(name='pybss',
      version='0.1.0',
      description='Python package for blind source separation',
      author='Kevin Brown and Ameya Akkalkotkar',
      author_email='kevin.s.brown@uconn.edu',
      url='https://thelahunginjeet@bitbucket.org/Archimonde/pybss.git',
      packages=['pybss'],
      package_dir={'pybss': ''},
      package_data={'pybss' : ['data/*.pydb']},
      install_requires = ['munkres'],
      license='BSD-3',
      classifiers = [
          'License :: OSI Approved :: BSD-3 License',
          'Intended Audience :: Developers',
          'Intended Audience :: Scientists',
          'Progamming Language :: Python',
        ],
     )
