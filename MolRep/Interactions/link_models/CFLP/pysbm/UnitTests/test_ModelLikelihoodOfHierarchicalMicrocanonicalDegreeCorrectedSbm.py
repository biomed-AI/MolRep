from unittest import TestCase

import networkx as nx

from pysbm import sbm
from pysbm.sbm.nxpartitiongraphbased import NxHierarchicalPartition
from pysbm.sbm.peixotos_hierarchical_sbm_full import ModelLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbm


class TestModelLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmUndirected(TestCase):
    def setUp(self):
        self.graphs = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
        nx.add_path(self.graphs[0], [0, 0, 1, 2, 3])
        nx.add_path(self.graphs[1], [0, 1, 2, 3, 0])
        nx.add_path(self.graphs[2], [0, 1, 2, 3, 0, 0])
        nx.add_path(self.graphs[4], [0, 1, 2, 3, 0, 4])
        self.graphs[3] = self.graphs[2].copy()
        self.graphs[3].add_edge(2, 2)

        self.partitions = []
        for graph in self.graphs:
            partition = sbm.NxPartition(graph=graph, number_of_blocks=2)
            self.partitions.append(partition)
            partition.set_from_representation({node: node % partition.B for node in graph})

        # simple hierarchies with only one level
        self.hierarchical_partitions = []
        for graph in self.graphs:
            partition = NxHierarchicalPartition(graph=graph, number_of_blocks=2)
            self.hierarchical_partitions.append(partition)
            partition.set_from_representation([{node: node % partition.B for node in graph}])

        # add more complex partition
        partition = NxHierarchicalPartition(graph=self.graphs[4], number_of_blocks=5)
        self.hierarchical_partitions.append(partition)
        partition.set_from_representation([{node: node % 4 for node in self.graphs[4]},
                                           {0: 0, 1: 0, 2: 1, 3: 1}])

        partition = NxHierarchicalPartition(graph=self.graphs[4], number_of_blocks=5)
        self.hierarchical_partitions.append(partition)
        partition.set_from_representation([{0: 0, 1: 0, 2: 1, 3: 1, 4: 2},
                                           {0: 0, 1: 1, 2: 0}])

        partition = NxHierarchicalPartition(graph=self.graphs[4], number_of_blocks=5)
        self.hierarchical_partitions.append(partition)
        partition.set_from_representation([{node: node for node in self.graphs[4]},
                                           {0: 0, 1: 0, 2: 1, 3: 1, 4: 2},
                                           {0: 0, 1: 1, 2: 0}])

        # information about graphs below
        self.likelihood = ModelLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbm()

        # Hierarchical partitions

        # Information about graphs
        # --------------------------
        #  Graph 0
        # --------------------------
        # Adjacency Matrix
        # 2 1 0 0
        # 1 0 1 0
        # 0 1 0 1
        # 0 0 1 0
        #
        # degree sequence
        # 3, 2, 2, 1
        #
        # partition is b_0 = 0, 2
        #              b_1 = 1, 3
        # Edge count matrix
        # 2 3
        # 3 0
        # Summed = e_0:5, e_1:3
        #
        # node counts 2, 2

        # --------------------------
        #  Graph 1
        # --------------------------
        # 0 1 0 1
        # 1 0 1 0
        # 0 1 0 1
        # 1 0 1 0
        #
        # degree sequence
        # 2, 2, 2, 2
        #
        # partition is b_0 = 0, 2
        #              b_1 = 1, 3
        # Edge count matrix
        # 0 4
        # 4 0
        # Summed = e_0:4, e_1:4

        # --------------------------
        #  Graph 2
        # --------------------------
        # 2 1 0 1
        # 1 0 1 0
        # 0 1 0 1
        # 1 0 1 0
        #
        # degree sequence
        # 4, 2, 2, 2
        #
        # partition is b_0 = 0, 2
        #              b_1 = 1, 3
        # Edge count matrix
        # 2 4
        # 4 0
        # Summed = e_0:6, e_1:4

        # --------------------------
        #  Graph 3
        # --------------------------
        # 2 1 0 1
        # 1 0 1 0
        # 0 1 2 1
        # 1 0 1 0
        #
        # degree sequence
        # 4, 2, 4, 2
        #
        # partition is b_0 = 0, 2
        #              b_1 = 1, 3
        # Edge count matrix
        # 4 4
        # 4 0
        # Summed = e_0:8, e_1:4

        # --------------------------
        #  Graph 4
        # --------------------------
        # 0 1 0 1 1
        # 1 0 1 0 0
        # 0 1 0 1 0
        # 1 0 1 0 0
        # 1 0 0 0 0
        # degree sequence
        # 3, 2, 2, 2, 1
        #
        # partition is b_0 = 0, 2, 4
        #              b_1 = 1, 3
        # Edge count matrix
        # 2 4
        # 4 0
        # Summed = e_0:6, e_1:4

    def test_calculate_p_adjacency_undirected(self):

        # Omega = (5!*3!)/(3!*2^1*1!) = 60
        # Xi = (3!*2!*2!*1!)/(1!*1!*1!*1!*2^1*1!) = 12
        # result  = 12/60=.2
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.partitions[0]), .2)
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.hierarchical_partitions[0]), .2)

        # Omega = (4!*4!)/(4!) = 24
        # Xi = (2!*2!*2!*2!)/(1) = 16
        # result  = 16/24=.8
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.partitions[1]), 16 / 24)
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.hierarchical_partitions[1]), 16 / 24)

        # Omega = (6!*4!)/(4!*2^1*1!) = 360
        # Xi = (4!*2!*2!*2!)/(2) = 96
        # result  = 96/360
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.partitions[2]), 96 / 360)
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.hierarchical_partitions[2]), 96 / 360)

        # Omega = (8!*4!)/(4!*2^2*2!) = 5040
        # Xi = (4!*2!*4!*2!)/(2*2) = 576
        # result  = 576/5040
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.partitions[3]), 576 / 5040)
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.hierarchical_partitions[3]), 576 / 5040)

        # Omega = (6!*4!)/(4!*2^1*1!) = 360
        # Xi = (3!*2!*2!*2!*1!)/(1) = 48
        # result  = 48/360
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.partitions[4]), 48 / 360)
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.hierarchical_partitions[4]), 48 / 360)

        # Omega = 4!*2!*2!*2!/2 = 96
        # Xi = (3!*2!*2!*2!*1!)/(1) = 48
        # result  = 48/360
        self.hierarchical_partitions[5].actual_level = 0
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.hierarchical_partitions[5]), 48 / 96)

        # Omega = 5!*4!*1!/(2!*2*2) = 96
        # Xi = (3!*2!*2!*2!*1!)/(1) = 48
        # result  = 48/360
        self.hierarchical_partitions[6].actual_level = 0
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.hierarchical_partitions[6]), 48 / 360)

        # Omega = 3!*2!^3*1!/(1) = 48
        # Xi = (3!*2!*2!*2!*1!)/(1) = 48
        # result  = 48/360
        self.hierarchical_partitions[7].actual_level = 0
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.hierarchical_partitions[7]), 48 / 48)

    def test_calculate_non_degree_corrected_undirected(self):
        # general Formula
        # P(A|e,b) = \frac{\prod_{r<s} e_{rs}! \prod_r e_{rr}!!}{\prod_r n_r^{e_r}}
        #             * \frac{1}{\prod_{i<j}A_{ij}! \prod_i A_{ii}!!}

        #
        # P(A|e,b) = 3!*2^1*1!/(2^5*2^3*2^1) = 3/128
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.partitions[0]), 3 / 128)
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.hierarchical_partitions[0]),
                         3 / 128)

        # P(A|e,b) = 4!/(2^4*2^4) = 3/32
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.partitions[1]), 3 / 32)
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.hierarchical_partitions[1]),
                         3 / 32)

        # P(A|e,b) = 4!*2^1*1!/(2^6*2^4*2^1) = 3/128
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.partitions[2]), 3 / 128)
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.hierarchical_partitions[2]),
                         3 / 128)

        # P(A|e,b) = 4!*2^2*2!/(2^8*2^4*2^1*2^1) = 3/256
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.partitions[3]), 3 / 256)
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.hierarchical_partitions[3]),
                         3 / 256)

        # P(A|e,b) = 4!*2^1*1!/(3^6*2^4) = 1/243
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.partitions[4]), 1 / 243)
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.hierarchical_partitions[4]),
                         1 / 243)

        # P(A|e,b) = 2^1/(2^4*(1^2)^3) = 1/8
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.hierarchical_partitions[5]),
                         1 / 8)

        # P(A|e,b) = 2!*2^1*2^1/(2^5*2^4*1^1) = 1/1
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.hierarchical_partitions[6]),
                         1 / 64)

        # P(A|e,b) = 1/1
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.hierarchical_partitions[7]),
                         1 / 1)

    def test_calculate_p_degree_sequence_uniform_undirected(self):
        # general Formula
        # P(k|e,b) = \prod_r (( n_r e_r ))^{-1}

        # P(k|e,b) = 1/( ((2 5)) * ((2 3)) ) = 1/(binom(2+5-1,5)*binom(2+3-1,3)) = 1/24
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.partitions[0]), 1 / 24)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.hierarchical_partitions[0]), 1 / 24)

        # P(k|e,b) = 1/( ((2 4)) * ((2 4)) ) = 1/(binom(2+4-1,4)*binom(2+4-1,4)) = 1/25
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.partitions[1]), 1 / 25)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.hierarchical_partitions[1]), 1 / 25)

        # P(k|e,b) = 1/( ((2 6)) * ((2 4)) ) = 1/(binom(2+6-1,6)*binom(2+4-1,4)) = 1/35
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.partitions[2]), 1 / 35)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.hierarchical_partitions[2]), 1 / 35)

        # P(k|e,b) = 1/( ((2 8)) * ((2 4)) ) = 1/(binom(2+8-1,8)*binom(2+4-1,4)) = 1/45
        self.assertAlmostEqual(self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.partitions[3]),
                               1 / 45)
        self.assertAlmostEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.hierarchical_partitions[3]),
            1 / 45)

        # P(k|e,b) = 1/( ((3 6)) * ((2 4)) ) = 1/(binom(3+6-1,6)*binom(2+4-1,4)) = 1/140
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.partitions[4]), 1 / 140)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.hierarchical_partitions[4]), 1 / 140)

        # P(k|e,b) = 1/( ((2 4)) * ((1 2))^3 ) = 1/(binom(2+4-1,4)*binom(1+2-1,2)^3) = 1/5
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.hierarchical_partitions[5]), 1 / 5)

        # P(k|e,b) = 1/( ((2 5)) * ((2 4)) ((1 1)) ) = 1/(binom(2+5-1,5)*binom(2+4-1,4)*binom(1+1-1,1)) = 1/30
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.hierarchical_partitions[6]), 1 / 30)

        # P(k|e,b) = 1/( ((1 3)) * ((1 2))^3 ((1 1)) ) = 1/(binom(1+3-1,3)*binom(1+2-1,2)^3*binom(1+1-1,1)) = 1/30
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.hierarchical_partitions[7]), 1 / 1)

    def test_calculate_p_degree_sequence_uniform_hyperprior_undirected(self):
        # general formula
        # P(k|e,b) = \prod_r 1/(n_r!*q(e_r,n_r))*\prod_k \eta_k^r!

        # P(k|e,b) = 1/(2!*2!*q(5,2)*q(3,2))*1!*1!*1!*1! = 1/(2*2*3*2) = 1/24
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.partitions[0]),
                         1 / 24)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.hierarchical_partitions[0]),
            1 / 24)

        # P(k|e,b) = 2!*2!/(2!*2!*q(4,2)*q(4,2)) = 1/(3*3) = 1/9
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.partitions[1]),
                         1 / 9)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.hierarchical_partitions[1]),
            1 / 9)

        # P(k|e,b) = 2!/(2!*2!*q(6,2)*q(4,2)) = 1/(2*4*3) =1/24
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.partitions[2]),
                         1 / 24)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.hierarchical_partitions[2]),
            1 / 24)

        # P(k|e,b) = 2!*2!/(2!*2!*q(8,2)*q(4,2))=1/(5*3) =1/15
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.partitions[3]),
                         1 / 15)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.hierarchical_partitions[3]),
            1 / 15)

        # P(k|e,b) = 2!/(3!*2!*q(6,3)*q(4,2))=2!/(3!*2!*7*3) =1/126
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.partitions[4]),
                         1 / 126)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.hierarchical_partitions[4]),
            1 / 126)

        # P(k|e,b) = 1/(2!*q(4,2)*q(2,1)^3)=1/(2!*3) =1/6
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.hierarchical_partitions[5]),
            1 / 6)

        # P(k|e,b) = 2!/(2!*2!*q(5,2)*q(4,2)*q(1,1))=2!/(2!*2!*3*3) =1/18
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.hierarchical_partitions[6]),
            1 / 18)

        # P(k|e,b) = 1
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.hierarchical_partitions[7]),
            1)

    def test_calculate_p_edge_counts_hierarchy_undirected(self):
        # formula:
        # P({e_l})*P({b_l})
        # P({e_l}) = \prod_{l=1}^L \prod_{r<s} (( n_r^ln_s^l e_rs^{l+1} ))^{-1}
        #               *\prod_r (( n_r^l(n_r^l+1)/2 e_rr^{l+1)/2 ))^{-1}
        # P({b_l}) = \prod_{l=1}^L \frac{\prod_r n_r^l!}{B_{l-1}!} (B_{l-1}-1 B_l-1)^{-1) 1/B_{l-1}
        # first test formula on simple hierarchies which are used before

        # only one real level (last term comes from l=2)
        # P({e_l}) = 1/( (( 2*2 3 ))*(( 2*(2+1)/2 2/2 ))* (( 2*(2+1)/2 0/2)) * ((2(2+1)/2 8/2 )))
        #          = 1/(binom(4+3-1,3)*binom(3+1-1,1)*binom(3+0-1,0)*binom(3+4-1,4))
        #          = 1/900
        #
        # P({b_l}) = 2!*2!/(4!*binom(4-1, 2-1)*4)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/144
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_undirected(self.hierarchical_partitions[0]),
            900 * 144)

        # P({e_l}) = 1/( (( 2*2 4 ))*(( 2*(2+1)/2 0/2 ))* (( 2*(2+1)/2 0/2)) * ((2(2+1)/2 8/2 )))
        #          = 1/(binom(4+4-1,4)*binom(3+0-1,0)*binom(3+0-1,0)*binom(3+4-1,4))
        #          = 1/525
        #
        # P({b_l}) = 2!*2!/(4!*binom(4-1, 2-1)*4)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/144
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_undirected(self.hierarchical_partitions[1]),
            525 * 144)

        # P({e_l}) = 1/( (( 2*2 4 ))*(( 2*(2+1)/2 2/2 ))* (( 2*(2+1)/2 0/2)) * ((2(2+1)/2 10/2 )))
        #          = 1/(binom(4+4-1,4)*binom(3+1-1,1)*binom(3+0-1,0)*binom(3+5-1,5))
        #          = 1/2205
        #
        # P({b_l}) = 2!*2!/(4!*binom(4-1, 2-1)*4)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/144
        self.assertAlmostEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_undirected(self.hierarchical_partitions[2]),
            2205 * 144, delta=.0000001)

        # P({e_l}) = 1/( (( 2*2 4 ))*(( 2*(2+1)/2 4/2 ))* (( 2*(2+1)/2 0/2)) * ((2(2+1)/2 12/2 )))
        #          = 1/(binom(4+4-1,4)*binom(3+2-1,2)*binom(3+0-1,0)*binom(3+6-1,6))
        #          = 1/5880
        #
        # P({b_l}) = 2!*2!/(4!*binom(4-1, 2-1)*4)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/144
        self.assertAlmostEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_undirected(self.hierarchical_partitions[3]),
            5880 * 144, delta=.0000001)

        # P({e_l}) = 1/( (( 2*3 4 ))*(( 3*(3+1)/2 2/2 ))* (( 2*(2+1)/2 0/2)) * ((2(2+1)/2 10/2 )))
        #          = 1/(binom(6+4-1,4)*binom(6+1-1,1)*binom(3+0-1,0)*binom(3+5-1,5))
        #          = 1/15876
        #
        # P({b_l}) = 3!*2!/(5!*binom(5-1, 2-1)*5)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/400
        self.assertAlmostEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_undirected(self.hierarchical_partitions[4]),
            15876 * 400, delta=.0000001)

        # P({e_l})
        #  level 1  r<s
        #            1/( (( 2*1 1 ))*(( 2*1 0 ))*(( 2*1 1 ))*(( 1*1 1 ))*(( 1*1 0 ))*(( 1*1 1 )))
        #          = 1/(binom(2+1-1,1)*binom(2+0-1,0)*binom(2+1-1,1)*binom(1+1-1,1)*binom(1-1,0)*binom(1+1-1,1))
        #          = 1/4
        #  level 1 r=s only one entry != 0
        #            1/( (( 2(2+1)/2 1 ))) =1/binom(3+1-1,1)=1/3
        #
        #  level 2
        #            1/( (( 2*2 2 ))*((2*(2+1)/2 2))*(( 2*(2+1)/2 1)))
        #          = 1/(binom(4+2-1,2)*binom(3+2-1,2)*binom(3+1-1,1))
        #          = 1/180
        #  level 3
        #            1/( (( 2*(2+1)/2 5 ))) =1/binom(3+5-1,5)=1/21
        #
        # P({b_l}) = 2!*1!*1!*1!/(5!*binom(5-1, 4-1)*5)*2!*2!/(4!*binom(4-1, 2-1)*4)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/172800
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_undirected(self.hierarchical_partitions[5]),
            12 * 180 * 21 * 172800)

        # P({e_l})
        #  level 1  r<s
        #            1/( (( 2*2 2 ))*(( 2*1 1 ))*(( 2*1 0 )))
        #          = 1/(binom(4+2-1,2)*binom(2+1-1,1)*binom(2+0-1,0))
        #          = 1/20
        #  level 1 r=s
        #            1/( (( 2(2+1)/2 1 ))*(( 2(2+1)/2 1 )) ) =1/(binom(3+1-1,1)*binom(3+1-1,1))=1/9
        #
        #  level 2
        #            1/( (( 2*1 2 ))*((2*(2+1)/2 2))*(( 1*(1+1)/2 1)))
        #          = 1/(binom(2+2-1,2)*binom(3+2-1,2)*binom(1+1-1,1))
        #          = 1/18
        #  level 3
        #            1/( (( 2*(2+1)/2 5 ))) =1/binom(3+5-1,5)=1/21
        #
        # P({b_l}) = 2!*2!*1!/(5!*binom(5-1, 3-1)*5)*2!*1!/(3!*binom(3-1, 2-1)*3)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/32400
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_undirected(self.hierarchical_partitions[6]),
            20 * 9 * 18 * 21 * 32400)

        # P({e_l})
        #  level 1
        #            1/( (( 1*1 1 ))^5)
        #          = 1/(binom(1+1-1,1)^5)
        #          = 1/1
        #
        #  level 2  r<s
        #            1/( (( 2*2 2 ))*(( 2*1 1 ))*(( 2*1 0 )))
        #          = 1/(binom(4+2-1,2)*binom(2+1-1,1)*binom(2+0-1,0))
        #          = 1/20
        #
        #  level 2 r=s
        #            1/( (( 2(2+1)/2 1 ))*(( 2(2+1)/2 1 )) ) =1/(binom(3+1-1,1)*binom(3+1-1,1))=1/9
        #
        #  level 3
        #            1/( (( 2*1 2 ))*((2*(2+1)/2 2))*(( 1*(1+1)/2 1)))
        #          = 1/(binom(2+2-1,2)*binom(3+2-1,2)*binom(1+1-1,1))
        #          = 1/18
        #  level 4
        #            1/( (( 2*(2+1)/2 5 ))) =1/binom(3+5-1,5)=1/21
        #
        #  new term + rest from above
        # P({b_l}) = 1!*1!*1!*1!*1!/(5!*binom(5-1,5-1)*5)
        #               *2!*2!*1!/(5!*binom(5-1, 3-1)*5)*2!*1!/(3!*binom(3-1, 2-1)*3)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/19440000
        self.assertAlmostEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_undirected(self.hierarchical_partitions[7]),
            20 * 9 * 18 * 21 * 19440000, delta=.001)

    def test_calculate_complete_uniform_hyperprior_undirected(self):
        self.assertEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.hierarchical_partitions[0]),
            .2 * 1 / 24 * 1 / (900 * 144))

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.hierarchical_partitions[1]),
            16 / 24 * 1 / 9 * 1 / (525 * 144))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.hierarchical_partitions[2]),
            96 / 360 * 1 / 24 * 1 / (2205 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.hierarchical_partitions[3]),
            576 / 5040 * 1 / 15 * 1 / (5880 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.hierarchical_partitions[4]),
            48 / 360 * 1 / 126 * 1 / (15876 * 400), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.hierarchical_partitions[5]),
            48 / 96 * 1 / 6 * 1 / (12 * 180 * 21 * 172800), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.hierarchical_partitions[6]),
            48 / 360 * 1 / 18 * 1 / (20 * 9 * 18 * 21 * 32400), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.hierarchical_partitions[7]),
            1 * 1 * 1 / (20 * 9 * 18 * 21 * 19440000), places=20)

    def test_calculate_complete_uniform_undirected(self):
        self.assertEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.hierarchical_partitions[0]),
            .2 * 1 / 24 * 1 / (900 * 144))

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.hierarchical_partitions[1]),
            16 / 24 * 1 / 25 * 1 / (525 * 144))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.hierarchical_partitions[2]),
            96 / 360 * 1 / 35 * 1 / (2205 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.hierarchical_partitions[3]),
            576 / 5040 * 1 / 45 * 1 / (5880 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.hierarchical_partitions[4]),
            48 / 360 * 1 / 140 * 1 / (15876 * 400), places=20)

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.hierarchical_partitions[5]),
            48 / 96 * 1 / 5 * 1 / (12 * 180 * 21 * 172800))

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.hierarchical_partitions[6]),
            48 / 360 * 1 / 30 * 1 / (20 * 9 * 18 * 21 * 32400))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.hierarchical_partitions[7]),
            1 * 1 * 1 / (20 * 9 * 18 * 21 * 19440000), places=20)

    def test_calculate_complete_non_degree_corrected_undirected(self):
        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.hierarchical_partitions[0]),
            3 / 128 * 1 / (900 * 144), places=20)

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.hierarchical_partitions[1]),
            3 / 32 * 1 / (525 * 144))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.hierarchical_partitions[2]),
            3 / 128 * 1 / (2205 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.hierarchical_partitions[3]),
            3 / 256 * 1 / (5880 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.hierarchical_partitions[4]),
            1 / 243 * 1 / (15876 * 400), places=20)

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.hierarchical_partitions[5]),
            1 / 8 * 1 / (12 * 180 * 21 * 172800))

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.hierarchical_partitions[6]),
            1 / 64 * 1 / (20 * 9 * 18 * 21 * 32400))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.hierarchical_partitions[7]),
            1 * 1 * 1 / (20 * 9 * 18 * 21 * 19440000), places=20)


class TestModelLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmDirected(TestCase):
    def setUp(self):
        # now everything directed
        self.digraphs = [nx.DiGraph() for _ in range(5)]
        nx.add_path(self.digraphs[0], [0, 0, 1, 2, 3])
        nx.add_path(self.digraphs[1], [0, 1, 2, 3, 0])
        nx.add_path(self.digraphs[2], [0, 1, 2, 3, 0, 0])
        nx.add_path(self.digraphs[4], [0, 1, 2, 3, 0, 4])
        # self.digraphs[3] = self.digraphs[2].copy()
        nx.add_path(self.digraphs[3], [0, 1, 2, 3, 0, 0])
        self.digraphs[3].add_edge(2, 2)

        self.partitions = []
        for graph in self.digraphs:
            partition = sbm.NxPartition(graph=graph, number_of_blocks=2)
            self.partitions.append(partition)
            partition.set_from_representation({node: node % partition.B for node in graph})

        self.hierarchical_partitions = []
        for graph in self.digraphs:
            partition = NxHierarchicalPartition(graph=graph, number_of_blocks=2)
            self.hierarchical_partitions.append(partition)
            partition.set_from_representation([{node: node % partition.B for node in graph}])

        # add more complex partition
        partition = NxHierarchicalPartition(graph=self.digraphs[4], number_of_blocks=5)
        self.hierarchical_partitions.append(partition)
        partition.set_from_representation([{node: node % 4 for node in self.digraphs[4]},
                                           {0: 0, 1: 0, 2: 1, 3: 1}])

        partition = NxHierarchicalPartition(graph=self.digraphs[4], number_of_blocks=5)
        self.hierarchical_partitions.append(partition)
        partition.set_from_representation([{0: 0, 1: 0, 2: 1, 3: 1, 4: 2},
                                           {0: 0, 1: 1, 2: 0}])

        partition = NxHierarchicalPartition(graph=self.digraphs[4], number_of_blocks=5)
        self.hierarchical_partitions.append(partition)
        partition.set_from_representation([{node: node for node in self.digraphs[4]},
                                           {0: 0, 1: 0, 2: 1, 3: 1, 4: 2},
                                           {0: 0, 1: 1, 2: 0}])

        # information about graphs below
        self.likelihood = ModelLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbm()

        # Hierarchical partitions

        # Information about graphs
        # --------------------------
        #  Graph 0
        # --------------------------
        # Adjacency Matrix
        # 1 1 0 0
        # 0 0 1 0
        # 0 0 0 1
        # 0 0 0 0
        #
        # out degree sequence
        # 2, 1, 1, 0
        # in degree sequence
        # 1, 1, 1, 1
        #
        # partition is b_0 = 0, 2
        #              b_1 = 1, 3
        # Edge count matrix
        # 1 2
        # 1 0
        # Summed: 3,1; 2, 2
        #
        # node counts 2, 2

        # --------------------------
        #  Graph 1
        # --------------------------
        # 0 1 0 0
        # 0 0 1 0
        # 0 0 0 1
        # 1 0 0 0
        #
        # degree sequence (both)
        # 1, 1, 1, 1
        #
        # partition is b_0 = 0, 2
        #              b_1 = 1, 3
        # Edge count matrix
        # 0 2
        # 2 0
        # Summed = 2,2;2,2

        # --------------------------
        #  Graph 2
        # --------------------------
        # 1 1 0 0
        # 0 0 1 0
        # 0 0 0 1
        # 1 0 0 0
        #
        # degree sequence (both)
        # 2,1,1,1
        #
        # partition is b_0 = 0, 2
        #              b_1 = 1, 3
        # Edge count matrix
        # 1 2
        # 2 0
        # Summed = 3,2;3;2

        # --------------------------
        #  Graph 3
        # --------------------------
        # 1 1 0 0
        # 0 0 1 0
        # 0 0 1 1
        # 1 0 0 0
        #
        # degree sequence (both)
        # 2, 1, 2, 1
        #
        # partition is b_0 = 0, 2
        #              b_1 = 1, 3
        # Edge count matrix
        # 2 2
        # 2 0
        # Summed = 4,2;4,2

        # --------------------------
        #  Graph 4
        # --------------------------
        # 0 1 0 0 1
        # 0 0 1 0 0
        # 0 0 0 1 0
        # 1 0 0 0 0
        # 0 0 0 0 0
        # out degree sequence
        # 2, 1, 1, 1, 0
        # in degree sequence
        # 1, 1, 1, 1, 1
        #
        # partition is b_0 = 0, 2, 4
        #              b_1 = 1, 3
        # Edge count matrix
        # 1 2
        # 2 0
        # Summed = 3,2;3,2

        # partition 5
        # Edge count matrix
        # 1 1 0 0
        # 0 0 1 0
        # 0 0 0 1
        # 1 0 0 0
        # Summed 2,1,1,1 (both)

        # partition 6
        # Edge count matrix
        # 1 1 1
        # 1 1 0
        # 0 0 0
        # Summed: 2,2,1; 3,2,0

    def test_calculate_p_adjacency_directed(self):
        # P(A]k,e,b) = \prod_i k_i^+! k_i^-! \prod_{rs} e_rs / (\prod_r e_r^+! e_r^-! \prod_{ij} A_{ij}!

        # P(A]k,e,b) = 2!*2!/(3!*1!*2!*2!) =1/6
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.partitions[0]), 1 / 6)
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.hierarchical_partitions[0]), 1 / 6)

        # P(A]k,e,b) = 2!*2!/(2!*2!*2!*2!) =1/4
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.partitions[1]), 1 / 4)
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.hierarchical_partitions[1]), 1 / 4)

        # P(A]k,e,b) = 2!*2!*2!*2!/(3!*2!*3!*2!) =1/9
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.partitions[2]), 1 / 9)
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.hierarchical_partitions[2]), 1 / 9)

        # P(A]k,e,b) = 2!*2!*2!*2!*2!*2!*2!/(4!*2!*4!*2!) =1/18
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.partitions[3]), 1 / 18)
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.hierarchical_partitions[3]), 1 / 18)

        # P(A]k,e,b) = 2!*2!*2!/(3!*2!*3!*2!)
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.partitions[4]), 1 / 18)
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.hierarchical_partitions[4]), 1 / 18)

        # P(A]k,e,b) = 2!/(2!*2!)
        self.hierarchical_partitions[5].actual_level = 0
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.hierarchical_partitions[5]), 1 / 2)

        # P(A]k,e,b) = 2!/(3!*2!*2!*2!)
        self.hierarchical_partitions[6].actual_level = 0
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.hierarchical_partitions[6]), 1 / 24)

        # P(A]k,e,b) =2!/2!
        self.hierarchical_partitions[7].actual_level = 0
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.hierarchical_partitions[7]), 1)

    def test_calculate_non_degree_corrected_directed(self):
        # general Formula
        # P(A|e,b) = \frac{\prod_{rs} e_{rs}!}{\prod_r n_r^{e_r^+}*n_r^{e_r^-}}
        #             * \frac{1}{\prod_{ij}A_{ij}!}

        #
        # P(A|e,b) = 2!/(2^3*2^2*2^2*2^1) = 1/128
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.partitions[0]), 1 / 128)
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.hierarchical_partitions[0]),
                         1 / 128)

        # P(A|e,b) = 2!*2!/(2^2*2^2*2^2*2^2) = 1/64
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.partitions[1]), 1 / 64)
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.hierarchical_partitions[1]),
                         1 / 64)

        # P(A|e,b) = 2!*2!/(2^3*2^2*2^3*2^2) = 1/64
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.partitions[2]), 1 / 256)
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.hierarchical_partitions[2]),
                         1 / 256)

        # P(A|e,b) = 2!*2!*2!/(2^4*2^2*2^4*2^2) = 1/512
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.partitions[3]), 1 / 512)
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.hierarchical_partitions[3]),
                         1 / 512)

        # P(A|e,b) =  2!*2!*1!/(3^3*2^2*3^3*2^2) = 1/2916
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.partitions[4]), 1 / 2916)
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.hierarchical_partitions[4]),
                         1 / 2916)

        # P(A|e,b) = 1/(2^2*2^2) = 1/16
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.hierarchical_partitions[5]),
                         1 / 16)

        # P(A|e,b) = 1/(2^2*2^2*2^3*2^2) = 1/512
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.hierarchical_partitions[6]),
                         1 / 512)

        # P(A|e,b) = 1/1
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.hierarchical_partitions[7]),
                         1 / 1)

    def test_calculate_p_degree_sequence_uniform_directed(self):
        # general Formula
        # P(k|e,b) = \prod_r (( n_r e_r^+ ))^{-1}(( n_r e_r^- ))^{-1}

        # P(k|e,b) = 1/( ((2 3)) * ((2 1))*((2 2)) * ((2 2)) )
        #          = 1/(binom(2+3-1,3)*binom(2+1-1,1)*binom(2+2-1,2)*binom(2+2-1,2)) = 1/72
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_directed(self.partitions[0]), 1 / 72)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_directed(self.hierarchical_partitions[0]), 1 / 72)

        # P(k|e,b) = 1/( ((2 2)) * ((2 2))*((2 2)) * ((2 2)) )
        #          = 1/(binom(2+2-1,2)*binom(2+2-1,2)*binom(2+2-1,2)*binom(2+2-1,2)) = 1/81
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_directed(self.partitions[1]), 1 / 81)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_directed(self.hierarchical_partitions[1]), 1 / 81)

        # P(k|e,b) = 1/( ((2 3)) * ((2 3))*((2 2)) * ((2 2)) )
        #          = 1/(binom(2+3-1,3)*binom(2+3-1,3)*binom(2+2-1,2)*binom(2+2-1,2)) = 1/144
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_directed(self.partitions[2]), 1 / 144)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_directed(self.hierarchical_partitions[2]), 1 / 144)

        # P(k|e,b) = 1/( ((2 4)) * ((2 4))*((2 2)) * ((2 2)) )
        #          = 1/(binom(2+4-1,4)*binom(2+4-1,4)*binom(2+2-1,2)*binom(2+2-1,2)) = 1/225
        self.assertAlmostEqual(self.likelihood._calculate_p_degree_sequence_uniform_directed(self.partitions[3]),
                               1 / 225)
        self.assertAlmostEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_directed(self.hierarchical_partitions[3]),
            1 / 225)

        # P(k|e,b) = 1/( ((3 3)) * ((3 3))*((2 2)) * ((2 2)) )
        #          = 1/(binom(3+3-1,3)*binom(3+3-1,3)*binom(2+2-1,2)*binom(2+2-1,2)) = 1/900
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_directed(self.partitions[4]), 1 / 900)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_directed(self.hierarchical_partitions[4]), 1 / 900)

        # P(k|e,b) = 1/( ((2 2)) * ((1 1))*((1 1)) * ((1 1))*((2 2)) * ((1 1))*((1 1)) * ((1 1)) )
        #          = 1/(binom(2+2-1,2)*binom(2+2-1,2)) = 1/9
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_directed(self.hierarchical_partitions[5]), 1 / 9)

        # P(k|e,b) = 1/( ((2 2)) * ((2 2))*((1 1)) * ((2 3))*((2 2)) * ((1 1)))
        #          = 1/(binom(2+2-1,2)*binom(2+2-1,2)*binom(2+3-1,3)*binom(2+2-1,2)) = 1/108
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_directed(self.hierarchical_partitions[6]), 1 / 108)

        # P(k|e,b) = 1
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_directed(self.hierarchical_partitions[7]), 1 / 1)

    def test_calculate_p_degree_sequence_uniform_hyperprior_directed(self):
        # general formula
        # P(k|e,b) = \prod_r 1/(n_r!*q(e_r^-,n_r)**q(e_r^-,n_r))*\prod_k \eta_{k^+,k^-}^r!

        # P(k|e,b) = 1!*1!*1!*1!/(2!*2!*q(3,2)*q(1,2)*q(2,2)*q(2,2)) = 1/(2*2*2*1*2*2) = 1/32
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.partitions[0]),
                         1 / 32)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.hierarchical_partitions[0]),
            1 / 32)

        # P(k|e,b) = 2!*2!/(2!*2!*q(2,2)*q(2,2)*q(2,2)*q(2,2)) = 4/(2*2*2*2*2*2) = 1/16
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.partitions[1]),
                         1 / 16)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.hierarchical_partitions[1]),
            1 / 16)

        # P(k|e,b) = 1!*2!/(2!*2!*q(3,2)*q(2,2)*q(3,2)*q(2,2)) = 2/(2*2*2*2*2*2) = 1/32
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.partitions[2]),
                         1 / 32)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.hierarchical_partitions[2]),
            1 / 32)

        # P(k|e,b) = 2!*2!/(2!*2!*q(4,2)*q(2,2)*q(4,2)*q(2,2)) = 4/(2*2*3*2*3*2) = 1/36
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.partitions[3]),
                         1 / 36)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.hierarchical_partitions[3]),
            1 / 36)

        # P(k|e,b) = 1!*2!/(3!*2!*q(3,3)*q(2,2)*q(3,3)*q(2,2)) = 2/(6*2*3*2*3*2) = 1/216
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.partitions[4]),
                         1 / 216)
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.hierarchical_partitions[4]),
            1 / 216)

        # P(k|e,b) = 1/(2!*q(2,2)^2*q(1,1)^6)=1/(2!*4) =1/8
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.hierarchical_partitions[5]),
            1 / 8)

        # P(k|e,b) = 2!/(2!*2!*q(2,2)*q(2,2)*q(1,1)*q(3,2)*q(2,2)*q(0,1))=2!/(2!*2!*2*2*2*2) =1/32
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.hierarchical_partitions[6]),
            1 / 32)

        # P(k|e,b) = 1
        self.assertEqual(
            self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.hierarchical_partitions[7]),
            1)

    def test_calculate_p_edge_counts_hierarchy_directed(self):
        # formula:
        # P({e_l})*P({b_l})
        # P({e_l}) = \prod_{l=1}^L \prod_{rs} (( n_r^ln_s^l e_rs^{l+1} ))^{-1}
        # P({b_l}) = \prod_{l=1}^L \frac{\prod_r n_r^l!}{B_{l-1}!} (B_{l-1}-1 B_l-1)^{-1) 1/B_{l-1}
        # first test formula on simple hierarchies which are used before

        # only one real level (last term comes from l=2)
        # P({e_l}) = 1/( (( 2*2 1 ))*(( 2*2 2 ))* (( 2*2 1))*((2*2 4)))
        #          = 1/(binom(4+1-1,1)*binom(4+2-1,2)*binom(4+1-1,1)*binom(4+4-1,4))
        #          = 1/5600
        #
        # P({b_l}) = 2!*2!/(4!*binom(4-1, 2-1)*4)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/144
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_directed(self.hierarchical_partitions[0]),
            5600 * 144)

        # P({e_l}) = 1/( (( 2*2 2 ))* (( 2*2 2)) * ((2*2 4 )))
        #          = 1/(binom(4+2-1,2)*binom(4+2-1,2)*binom(4+4-1,4))
        #          = 1/3500
        #
        # P({b_l}) = 2!*2!/(4!*binom(4-1, 2-1)*4)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/144
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_directed(self.hierarchical_partitions[1]),
            3500 * 144)

        # P({e_l}) = 1/( (( 2*2 1 ))*(( 2*2 2 ))* (( 2*2 2)) * ((2*2 5 )))
        #          = 1/(binom(4+1-1,1)*binom(4+2-1,2)*binom(4+2-1,2)*binom(4+5-1,5))
        #          = 1/22400
        #
        # P({b_l}) = 2!*2!/(4!*binom(4-1, 2-1)*4)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/144
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_directed(self.hierarchical_partitions[2]),
            22400 * 144)

        # P({e_l}) = 1/( (( 2*2 2 ))*(( 2*2 2))* (( 2*2 2)) * ((2*2 6 )))
        #          = 1/(binom(4+2-1,2)*binom(4+2-1,2)*binom(4+2-1,2)*binom(4+6-1,6))
        #          = 1/84000
        #
        # P({b_l}) = 2!*2!/(4!*binom(4-1, 2-1)*4)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/144
        self.assertAlmostEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_directed(self.hierarchical_partitions[3]),
            84000 * 144, delta=.0000001)

        # P({e_l}) = 1/( (( 3*3 1 ))*(( 2*3 2 ))* (( 2*3 2)) * ((2*2 5 )))
        #          = 1/(binom(9+1-1,1)*binom(6+2-1,2)*binom(6+2-1,2)*binom(4+5-1,5))
        #          = 1/222264
        #
        # P({b_l}) = 3!*2!/(5!*binom(5-1, 2-1)*5)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/400
        self.assertAlmostEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_directed(self.hierarchical_partitions[4]),
            222264 * 400, delta=.0000001)

        # P({e_l})
        #  level 1
        #            1/( (( 2*1 1 ))*(( 2*1 1 ))*(( 2*2 1 )))
        #          = 1/(binom(2+1-1,1)*binom(2+1-1,1)*binom(4+1-1,1))
        #          = 1/16
        #
        #  level 2
        #            1/( (( 2*2 1 ))*((2*2 2))*((2*2 1))*((2*2 1)))
        #          = 1/(binom(4+1-1,1)*binom(4+2-1,2)*binom(4+1-1,1)*binom(4+1-1,1))
        #          = 1/640
        #  level 3
        #            1/( (( 2*2 5 ))) =1/binom(4+5-1,5)=1/56
        #
        # P({b_l}) = 2!*1!*1!*1!/(5!*binom(5-1, 4-1)*5)*2!*2!/(4!*binom(4-1, 2-1)*4)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/172800
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_directed(self.hierarchical_partitions[5]),
            16 * 640 * 56 * 172800)

        # P({e_l})
        #  level 1
        #            1/( (( 2*2 1 ))*(( 2*2 1 ))*(( 2*1 1 ))*(( 2*2 1 ))*(( 2*2 1 )))
        #          = 1/(binom(4+1-1,1)^4*binom(2+1-1,1))
        #          = 1/512
        #
        #  level 2
        #            1/( (( 2*2 2 ))*(( 2*1 1 ))*(( 1*1 1 ))*(( 1*2 1 )))
        #          = 1/(binom(4+2-1,2)*binom(2+1-1,1)*binom(1+1-1,1)*binom(2+1-1,1))
        #          = 1/40
        #  level 3
        #            1/( (( 2*2 5 ))) =1/binom(4+5-1,5)=1/56
        #
        # P({b_l}) = 2!*2!*1!/(5!*binom(5-1, 3-1)*5)*2!*1!/(3!*binom(3-1, 2-1)*3)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/32400
        self.assertAlmostEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_directed(self.hierarchical_partitions[6]),
            512 * 40 * 56 * 32400, delta=.00001)

        # P({e_l})
        #  level 1
        #            1/( (( 1*1 1 ))^5)
        #          = 1/(binom(1+1-1,1)^5)
        #          = 1/1
        #
        #  level 2
        #            1/( (( 2*2 1 ))*(( 2*2 1 ))*(( 2*1 1 ))*(( 2*2 1 ))*(( 2*2 1 )))
        #          = 1/(binom(4+1-1,1)^4*binom(2+1-1,1))
        #          = 1/512
        #
        #  level 3
        #            1/( (( 2*2 2 ))*(( 2*1 1 ))*(( 1*1 1 ))*(( 1*2 1 )))
        #          = 1/(binom(4+2-1,2)*binom(2+1-1,1)*binom(1+1-1,1)*binom(2+1-1,1))
        #          = 1/40
        #  level 4
        #            1/( (( 2*2 5 ))) =1/binom(4+5-1,5)=1/56
        #
        #  new term + rest from above
        # P({b_l}) = 1!*1!*1!*1!*1!/(5!*binom(5-1,5-1)*5)
        #               *2!*2!*1!/(5!*binom(5-1, 3-1)*5)*2!*1!/(3!*binom(3-1, 2-1)*3)*2!/(2!*binom(2-1,1-1)*2)
        #          = 1/19440000
        self.assertAlmostEqual(
            1 / self.likelihood._calculate_p_edge_counts_hierarchy_directed(self.hierarchical_partitions[7]),
            512 * 40 * 56 * 19440000, delta=.01)

    def test_calculate_complete_uniform_hyperprior_directed(self):
        self.assertEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.hierarchical_partitions[0]),
            1 / 6 * 1 / 32 * 1 / (5600 * 144))

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.hierarchical_partitions[1]),
            1 / 4 * 1 / 16 * 1 / (3500 * 144))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.hierarchical_partitions[2]),
            1 / 9 * 1 / 32 * 1 / (22400 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.hierarchical_partitions[3]),
            1 / 18 * 1 / 36 * 1 / (84000 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.hierarchical_partitions[4]),
            1 / 18 * 1 / 216 * 1 / (222264 * 400), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.hierarchical_partitions[5]),
            1 / 2 * 1 / 8 * 1 / (16 * 640 * 56 * 172800), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.hierarchical_partitions[6]),
            1 / 24 * 1 / 32 * 1 / (512 * 40 * 56 * 32400), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.hierarchical_partitions[7]),
            1 * 1 * 1 / (512 * 40 * 56 * 19440000), places=20)

    def test_calculate_complete_uniform_directed(self):
        self.assertEqual(
            self.likelihood.calculate_complete_uniform_directed(self.hierarchical_partitions[0]),
            1 / 6 * 1 / 72 * 1 / (5600 * 144))

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_directed(self.hierarchical_partitions[1]),
            1 / 4 * 1 / 81 * 1 / (3500 * 144))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_directed(self.hierarchical_partitions[2]),
            1 / 9 * 1 / 144 * 1 / (22400 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_directed(self.hierarchical_partitions[3]),
            1 / 18 * 1 / 225 * 1 / (84000 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_directed(self.hierarchical_partitions[4]),
            1 / 18 * 1 / 900 * 1 / (222264 * 400), places=20)

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_directed(self.hierarchical_partitions[5]),
            1 / 2 * 1 / 9 * 1 / (16 * 640 * 56 * 172800))

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_directed(self.hierarchical_partitions[6]),
            1 / 24 * 1 / 108 * 1 / (512 * 40 * 56 * 32400))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_directed(self.hierarchical_partitions[7]),
            1 * 1 * 1 / (512 * 40 * 56 * 19440000), places=20)

    def test_calculate_complete_non_degree_corrected_directed(self):
        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.hierarchical_partitions[0]),
            1 / 128 * 1 / (5600 * 144), places=20)

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.hierarchical_partitions[1]),
            1 / 64 * 1 / (3500 * 144))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.hierarchical_partitions[2]),
            1 / 256 * 1 / (22400 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.hierarchical_partitions[3]),
            1 / 512 * 1 / (84000 * 144), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.hierarchical_partitions[4]),
            1 / 2916 * 1 / (222264 * 400), places=20)

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.hierarchical_partitions[5]),
            1 / 16 * 1 / (16 * 640 * 56 * 172800))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.hierarchical_partitions[6]),
            1 / 512 * 1 / (512 * 40 * 56 * 32400), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.hierarchical_partitions[7]),
            1 * 1 * 1 / (512 * 40 * 56 * 19440000), places=20)
