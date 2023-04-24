#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy, inc.
 All rights reserved.

 @author: Jiahua Rao, Jiancong Xie, Junjie Xie
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""

import os

from molrep.models import *
from molrep.common.registry import registry

root_dir = os.path.dirname(os.path.abspath(__file__))
registry.register_path("library_root", root_dir)

repo_root = os.path.dirname(root_dir)
registry.register_path("repo_root", repo_root)

cache_root = os.path.join(repo_root, "cache")
registry.register_path("cache_root", cache_root)

feature_root = os.path.join(repo_root, "features")
registry.register_path("features_root", feature_root)

split_root = os.path.join(repo_root, "splits")
registry.register_path("split_root", split_root)
