#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 Copyright (c) 2022, Sun Yat-sen Univeristy.
 All rights reserved.

 @author: Jiahua Rao
 @license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 @contact: jiahua.rao@gmail.com
"""


from molrep.common.registry import registry

from molrep.models.graph_learning.gin import GIN
from molrep.models.graph_learning.gat import GAT
from molrep.models.graph_learning.diffpool import DiffPool
from molrep.models.graph_learning.graphsage import GraphSAGE
from molrep.models.graph_learning.graphnet import GraphNet
from molrep.models.graph_learning.graphformer import Graphformer

from molrep.models.graph_learning.mpnn import MPNN
from molrep.models.graph_learning.cmpnn import CMPNN
from molrep.models.graph_learning.dmpnn import DMPNN

from molrep.models.interactions.PLNLP.PLNLP import PLNLP
from molrep.models.interactions.CFLP.CFLP import CFLP

from molrep.models.pretraining.attr_masking import AttrMasking
from molrep.models.pretraining.edge_pred import EdgePred
from molrep.models.pretraining.context_pred import ContextPred
from molrep.models.pretraining.miga import MIGA

from molrep.models.optims import LinearWarmupStepLRScheduler, LinearWarmupCosineLRScheduler

__all__ = [
    "MPNN", "DMPNN", "CMPNN",
    "GIN", "GAT", "DiffPool", 
    "GraphSAGE", "GraphNet",
    "Graphformer",
    'PLNLP', 'CFLP',
    'AttrMasking', 'EdgePred', 'ContextPred', 'MIGA'
]
