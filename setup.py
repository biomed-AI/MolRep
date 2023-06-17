#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Copyright (c) 2023, Sun Yat-sen Univeristy.
All rights reserved.

@author:   Jiahua Rao
@license:  BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact:  jiahua.rao@gmail.com
'''


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="molrep", 
    version="2.0.0",
    author="Jiahua Rao, Shuangjia Zheng",
    author_email="raojh6@mail2.sysu.edu.cn",
    description="MolRep: Benchmarking Representation Learning Models for Molecular Property Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jh-SYSU/MolRep",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Creative Commons Zero v1.0 Universal",
        "Operating System :: OS Independent",
    ],
)