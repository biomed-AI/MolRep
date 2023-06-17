#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Copyright (c) 2023, Sun Yat-sen Univeristy.
All rights reserved.

@author:   Jiahua Rao
@license:  BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact:  jiahua.rao@gmail.com
'''


import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(
    ext_modules = cythonize("molrep/dataset/datasets/algos.pyx"),
    include_dirs=[numpy.get_include()]
)