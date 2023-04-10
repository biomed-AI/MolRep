from unittest import TestCase

import networkx as nx

from pysbm import sbm
from pysbm.sbm.peixotos_flat_sbm_full import ModelLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm


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

        # information about graphs below
        self.likelihood = ModelLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm()

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

        # Omega = (4!*4!)/(4!) = 24
        # Xi = (2!*2!*2!*2!)/(1) = 16
        # result  = 16/24=.8
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.partitions[1]), 16 / 24)

        # Omega = (6!*4!)/(4!*2^1*1!) = 360
        # Xi = (4!*2!*2!*2!)/(2) = 96
        # result  = 96/360
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.partitions[2]), 96 / 360)

        # Omega = (8!*4!)/(4!*2^2*2!) = 5040
        # Xi = (4!*2!*4!*2!)/(2*2) = 576
        # result  = 576/5040
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.partitions[3]), 576 / 5040)

        # Omega = (6!*4!)/(4!*2^1*1!) = 360
        # Xi = (3!*2!*2!*2!*1!)/(1) = 48
        # result  = 48/360
        self.assertEqual(self.likelihood._calculate_p_adjacency_undirected(self.partitions[4]), 48 / 360)

    def test_calculate_non_degree_corrected_undirected(self):
        # general Formula
        # P(A|e,b) = \frac{\prod_{r<s} e_{rs}! \prod_r e_{rr}!!}{\prod_r n_r^{e_r}}
        #             * \frac{1}{\prod_{i<j}A_{ij}! \prod_i A_{ii}!!}

        #
        # P(A|e,b) = 3!*2^1*1!/(2^5*2^3*2^1) = 3/128
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.partitions[0]), 3 / 128)

        # P(A|e,b) = 4!/(2^4*2^4) = 3/32
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.partitions[1]), 3 / 32)

        # P(A|e,b) = 4!*2^1*1!/(2^6*2^4*2^1) = 3/128
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.partitions[2]), 3 / 128)

        # P(A|e,b) = 4!*2^2*2!/(2^8*2^4*2^1*2^1) = 3/256
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.partitions[3]), 3 / 256)

        # P(A|e,b) = 4!*2^1*1!/(3^6*2^4) = 1/243
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_undirected(self.partitions[4]), 1 / 243)

    def test_calculate_p_degree_sequence_uniform_undirected(self):
        # general Formula
        # P(k|e,b) = \prod_r (( n_r e_r ))^{-1}

        # P(k|e,b) = 1/( ((2 5)) * ((2 3)) ) = 1/(binom(2+5-1,5)*binom(2+3-1,3)) = 1/24
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.partitions[0]), 1 / 24)

        # P(k|e,b) = 1/( ((2 4)) * ((2 4)) ) = 1/(binom(2+4-1,4)*binom(2+4-1,4)) = 1/25
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.partitions[1]), 1 / 25)

        # P(k|e,b) = 1/( ((2 6)) * ((2 4)) ) = 1/(binom(2+6-1,6)*binom(2+4-1,4)) = 1/35
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.partitions[2]), 1 / 35)

        # P(k|e,b) = 1/( ((2 8)) * ((2 4)) ) = 1/(binom(2+8-1,8)*binom(2+4-1,4)) = 1/45
        self.assertAlmostEqual(self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.partitions[3]),
                               1 / 45)

        # P(k|e,b) = 1/( ((3 6)) * ((2 4)) ) = 1/(binom(3+6-1,6)*binom(2+4-1,4)) = 1/140
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_undirected(self.partitions[4]), 1 / 140)

    def test_calculate_p_degree_sequence_uniform_hyperprior_undirected(self):
        # general formula
        # P(k|e,b) = \prod_r 1/(n_r!*q(e_r,n_r))*\prod_k \eta_k^r!

        # P(k|e,b) = 1/(2!*2!*q(5,2)*q(3,2))*1!*1!*1!*1! = 1/(2*2*3*2) = 1/24
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.partitions[0]),
                         1 / 24)

        # P(k|e,b) = 2!*2!/(2!*2!*q(4,2)*q(4,2)) = 1/(3*3) = 1/9
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.partitions[1]),
                         1 / 9)

        # P(k|e,b) = 2!/(2!*2!*q(6,2)*q(4,2)) = 1/(2*4*3) =1/24
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.partitions[2]),
                         1 / 24)

        # P(k|e,b) = 2!*2!/(2!*2!*q(8,2)*q(4,2))=1/(5*3) =1/15
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.partitions[3]),
                         1 / 15)

        # P(k|e,b) = 2!/(3!*2!*q(6,3)*q(4,2))=2!/(3!*2!*7*3) =1/126
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(self.partitions[4]),
                         1 / 126)

    def test_calculate_p_edge_counts_undirected(self):
        # formula:
        # P(e)*P(b)
        # P(e) = (( B(B+1)/2 E ))^{-1}
        # P(b) = \frac{\prod_r n_r!}{N!} (N-1 B-1)^{-1} 1/N

        # P(e) = 1/(( 2*(2+1)/2 4 )) = 1/binom(3+4-1,4) = 1/15
        # P(b) = 2!*2!/(4!*binom(4-1,2-1)*4) = 1/72
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_undirected(self.partitions[0]),
            15 * 72)

        # P(e) = 1/(( 2*(2+1)/2 4 )) = 1/binom(3+4-1,4) = 1/15
        # P(b) = 2!*2!/(4!*binom(4-1,2-1)*4) = 1/72
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_undirected(self.partitions[1]),
            15 * 72)

        # P(e) = 1/(( 2*(2+1)/2 5 )) = 1/binom(3+5-1,5) = 1/21
        # P(b) = 2!*2!/(4!*binom(4-1,2-1)*4) = 1/72
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_undirected(self.partitions[2]),
            21 * 72)

        # P(e) = 1/(( 2*(2+1)/2 6 )) = 1/binom(3+6-1,6) = 1/28
        # P(b) = 2!*2!/(4!*binom(4-1,2-1)*4) = 1/72
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_undirected(self.partitions[3]),
            28 * 72)

        # P(e) = 1/(( 2*(2+1)/2 5 )) = 1/binom(3+5-1,5) = 1/21
        # P(b) = 3!*2!/(5!*binom(5-1,2-1)*5) = 1/200
        self.assertAlmostEqual(
            1 / self.likelihood._calculate_p_edge_counts_undirected(self.partitions[4]),
            21 * 200, places=10)

    def test_calculate_complete_uniform_hyperprior_undirected(self):
        self.assertEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.partitions[0]),
            .2 * 1 / 24 * 1 / (15 * 72))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.partitions[1]),
            16 / 24 * 1 / 9 * 1 / (15 * 72), places=15)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.partitions[2]),
            96 / 360 * 1 / 24 * 1 / (21 * 72), places=15)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.partitions[3]),
            576 / 5040 * 1 / 15 * 1 / (28 * 72), places=15)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_undirected(self.partitions[4]),
            48 / 360 * 1 / 126 * 1 / (21 * 200), places=15)

    def test_calculate_complete_uniform_undirected(self):
        self.assertEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.partitions[0]),
            .2 * 1 / 24 * 1 / (15 * 72))

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.partitions[1]),
            16 / 24 * 1 / 25 * 1 / (15 * 72))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.partitions[2]),
            96 / 360 * 1 / 35 * 1 / (21 * 72), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.partitions[3]),
            576 / 5040 * 1 / 45 * 1 / (28 * 72), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_undirected(self.partitions[4]),
            48 / 360 * 1 / 140 * 1 / (21 * 200), places=20)

    def test_calculate_complete_non_degree_corrected_undirected(self):
        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.partitions[0]),
            3 / 128 * 1 / (15 * 72))

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.partitions[1]),
            3 / 32 * 1 / (15 * 72))

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.partitions[2]),
            3 / 128 * 1 / (21 * 72))

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.partitions[3]),
            3 / 256 * 1 / (28 * 72))

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_undirected(self.partitions[4]),
            1 / 243 * 1 / (21 * 200))


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

        # information about graphs below
        self.likelihood = ModelLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm()

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

        # P(A]k,e,b) = 2!*2!/(2!*2!*2!*2!) =1/4
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.partitions[1]), 1 / 4)

        # P(A]k,e,b) = 2!*2!*2!*2!/(3!*2!*3!*2!) =1/9
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.partitions[2]), 1 / 9)

        # P(A]k,e,b) = 2!*2!*2!*2!*2!*2!*2!/(4!*2!*4!*2!) =1/18
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.partitions[3]), 1 / 18)

        # P(A]k,e,b) = 2!*2!*2!/(3!*2!*3!*2!)
        self.assertEqual(self.likelihood._calculate_p_adjacency_directed(self.partitions[4]), 1 / 18)

    def test_calculate_non_degree_corrected_directed(self):
        # general Formula
        # P(A|e,b) = \frac{\prod_{rs} e_{rs}!}{\prod_r n_r^{e_r^+}*n_r^{e_r^-}}
        #             * \frac{1}{\prod_{ij}A_{ij}!}

        #
        # P(A|e,b) = 2!/(2^3*2^2*2^2*2^1) = 1/128
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.partitions[0]), 1 / 128)

        # P(A|e,b) = 2!*2!/(2^2*2^2*2^2*2^2) = 1/64
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.partitions[1]), 1 / 64)

        # P(A|e,b) = 2!*2!/(2^3*2^2*2^3*2^2) = 1/64
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.partitions[2]), 1 / 256)

        # P(A|e,b) = 2!*2!*2!/(2^4*2^2*2^4*2^2) = 1/512
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.partitions[3]), 1 / 512)

        # P(A|e,b) =  2!*2!*1!/(3^3*2^2*3^3*2^2) = 1/2916
        self.assertEqual(self.likelihood._calculate_non_degree_corrected_directed(self.partitions[4]), 1 / 2916)

    def test_calculate_p_degree_sequence_uniform_directed(self):
        # general Formula
        # P(k|e,b) = \prod_r (( n_r e_r^+ ))^{-1}(( n_r e_r^- ))^{-1}

        # P(k|e,b) = 1/( ((2 3)) * ((2 1))*((2 2)) * ((2 2)) )
        #          = 1/(binom(2+3-1,3)*binom(2+1-1,1)*binom(2+2-1,2)*binom(2+2-1,2)) = 1/72
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_directed(self.partitions[0]), 1 / 72)

        # P(k|e,b) = 1/( ((2 2)) * ((2 2))*((2 2)) * ((2 2)) )
        #          = 1/(binom(2+2-1,2)*binom(2+2-1,2)*binom(2+2-1,2)*binom(2+2-1,2)) = 1/81
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_directed(self.partitions[1]), 1 / 81)

        # P(k|e,b) = 1/( ((2 3)) * ((2 3))*((2 2)) * ((2 2)) )
        #          = 1/(binom(2+3-1,3)*binom(2+3-1,3)*binom(2+2-1,2)*binom(2+2-1,2)) = 1/144
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_directed(self.partitions[2]), 1 / 144)

        # P(k|e,b) = 1/( ((2 4)) * ((2 4))*((2 2)) * ((2 2)) )
        #          = 1/(binom(2+4-1,4)*binom(2+4-1,4)*binom(2+2-1,2)*binom(2+2-1,2)) = 1/225
        self.assertAlmostEqual(self.likelihood._calculate_p_degree_sequence_uniform_directed(self.partitions[3]),
                               1 / 225)

        # P(k|e,b) = 1/( ((3 3)) * ((3 3))*((2 2)) * ((2 2)) )
        #          = 1/(binom(3+3-1,3)*binom(3+3-1,3)*binom(2+2-1,2)*binom(2+2-1,2)) = 1/900
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_directed(self.partitions[4]), 1 / 900)

    def test_calculate_p_degree_sequence_uniform_hyperprior_directed(self):
        # general formula
        # P(k|e,b) = \prod_r 1/(n_r!*q(e_r^-,n_r)**q(e_r^-,n_r))*\prod_k \eta_{k^+,k^-}^r!

        # P(k|e,b) = 1!*1!*1!*1!/(2!*2!*q(3,2)*q(1,2)*q(2,2)*q(2,2)) = 1/(2*2*2*1*2*2) = 1/32
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.partitions[0]),
                         1 / 32)

        # P(k|e,b) = 2!*2!/(2!*2!*q(2,2)*q(2,2)*q(2,2)*q(2,2)) = 4/(2*2*2*2*2*2) = 1/16
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.partitions[1]),
                         1 / 16)

        # P(k|e,b) = 1!*2!/(2!*2!*q(3,2)*q(2,2)*q(3,2)*q(2,2)) = 2/(2*2*2*2*2*2) = 1/32
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.partitions[2]),
                         1 / 32)

        # P(k|e,b) = 2!*2!/(2!*2!*q(4,2)*q(2,2)*q(4,2)*q(2,2)) = 4/(2*2*3*2*3*2) = 1/36
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.partitions[3]),
                         1 / 36)

        # P(k|e,b) = 1!*2!/(3!*2!*q(3,3)*q(2,2)*q(3,3)*q(2,2)) = 2/(6*2*3*2*3*2) = 1/216
        self.assertEqual(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(self.partitions[4]),
                         1 / 216)

    def test_calculate_p_edge_counts_directed(self):
        # formula:
        # P(e)*P(b)
        # P(e) = (( B*B E ))^{-1}
        # P(b) = \frac{\prod_r n_r!}{N!} (N-1 B-1)^{-1} 1/N

        # P(e) = 1/(( 2*2 4 )) = 1/binom(4+4-1,4) = 1/35
        # P(b) = 2!*2!/(4!*binom(4-1,2-1)*4) = 1/72
        self.assertAlmostEqual(
            1 / self.likelihood._calculate_p_edge_counts_directed(self.partitions[0]),
            35 * 72, delta=0.00001)

        # P(e) = 1/(( 2*2 4 )) = 1/binom(4+4-1,4) = 1/35
        # P(b) = 2!*2!/(4!*binom(4-1,2-1)*4) = 1/72
        self.assertAlmostEqual(
            1 / self.likelihood._calculate_p_edge_counts_directed(self.partitions[1]),
            35 * 72, delta=0.00001)

        # P(e) = 1/(( 2*2 5 )) = 1/binom(4+5-1,5) = 1/56
        # P(b) = 2!*2!/(4!*binom(4-1,2-1)*4) = 1/72
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_directed(self.partitions[2]),
            56 * 72)

        # P(e) = 1/(( 2*2 6 )) = 1/binom(4+6-1,6) = 1/84
        # P(b) = 2!*2!/(4!*binom(4-1,2-1)*4) = 1/72
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_directed(self.partitions[3]),
            84 * 72)

        # P(e) = 1/(( 2*2 5 )) = 1/binom(4+5-1,5) = 1/56
        # P(b) = 3!*2!/(5!*binom(5-1,2-1)*5) = 1/200
        self.assertEqual(
            1 / self.likelihood._calculate_p_edge_counts_directed(self.partitions[4]),
            56 * 200)

    def test_calculate_complete_uniform_hyperprior_directed(self):
        self.assertEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.partitions[0]),
            1 / 6 * 1 / 32 * 1 / (35 * 72))

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.partitions[1]),
            1 / 4 * 1 / 16 * 1 / (35 * 72), places=20)

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.partitions[2]),
            1 / 9 * 1 / 32 * 1 / (56 * 72))

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.partitions[3]),
            1 / 18 * 1 / 36 * 1 / (84 * 72))

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_hyperprior_directed(self.partitions[4]),
            1 / 18 * 1 / 216 * 1 / (56 * 200))

    def test_calculate_complete_uniform_directed(self):
        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_directed(self.partitions[0]),
            1 / 6 * 1 / 72 * 1 / (35 * 72), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_directed(self.partitions[1]),
            1 / 4 * 1 / 81 * 1 / (35 * 72), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_uniform_directed(self.partitions[2]),
            1 / 9 * 1 / 144 * 1 / (56 * 72), places=20)

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_directed(self.partitions[3]),
            1 / 18 * 1 / 225 * 1 / (84 * 72))

        self.assertEqual(
            self.likelihood.calculate_complete_uniform_directed(self.partitions[4]),
            1 / 18 * 1 / 900 * 1 / (56 * 200))

    def test_calculate_complete_non_degree_corrected_directed(self):
        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.partitions[0]),
            1 / 128 * 1 / (35 * 72), places=20)

        self.assertAlmostEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.partitions[1]),
            1 / 64 * 1 / (35 * 72), places=20)

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.partitions[2]),
            1 / 256 * 1 / (56 * 72))

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.partitions[3]),
            1 / 512 * 1 / (84 * 72))

        self.assertEqual(
            self.likelihood.calculate_complete_non_degree_corrected_directed(self.partitions[4]),
            1 / 2916 * 1 / (56 * 200))
