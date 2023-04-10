import unittest as ut

import networkx as nx

from pysbm.sbm import DegreeCorrectedMicrocanonicalEntropy
from pysbm.sbm.nxpartitiongraphbased import NxPartitionGraphBased


class TestNxPartitionGraphBasedCovariates(ut.TestCase):
    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionGraphBasedCovariates, self).__init__(methodName)
        self.is_directed = False
        self.test_class = NxPartitionGraphBased
        self.comparision_class = NxPartitionGraphBased  # NxPartition
        self.objective_function = DegreeCorrectedMicrocanonicalEntropy(self.is_directed)

    def setUp(self):
        if self.is_directed:
            self.weighted_graph = nx.DiGraph()
        else:
            self.weighted_graph = nx.Graph()

        nx.add_star(self.weighted_graph, range(5))
        nx.add_path(self.weighted_graph, range(4, 10))
        # add a selfloop
        self.weighted_graph.add_edge(4, 4)

        self.covariate_graph = self.weighted_graph.copy()

        for from_node in self.weighted_graph:
            for to_node in self.weighted_graph:
                if self.weighted_graph.has_edge(from_node, to_node):
                    self.weighted_graph[from_node][to_node]["weight"] = from_node + to_node
                    self.covariate_graph[from_node][to_node]["covariate"] = from_node + to_node

        representation = {node: node % 3 for node in self.weighted_graph}
        self.weighted_partition = self.comparision_class(self.weighted_graph, representation=representation,
                                                         weighted_graph=True)
        self.covariate_partition = self.test_class(self.covariate_graph, weighted_graph=False, with_covariate=True,
                                                   representation=representation)

        self.unweighted_partition = self.comparision_class(self.weighted_graph, representation=representation,
                                                           weighted_graph=False)

    def compare_weighted_and_covariate_partition(self, weighted_partition, covariate_partition):
        """Simple method for the comparision of the saved weighted edge counts and the summed covariates"""
        # first some basic checks
        self.assertEqual(weighted_partition.get_representation(), covariate_partition.get_representation())
        self.assertEqual(weighted_partition.B, covariate_partition.B)
        self.assertEqual(weighted_partition.is_graph_directed(), covariate_partition.is_graph_directed())
        self.assertEqual(covariate_partition.is_graph_directed(), self.is_directed)

        for from_block in range(weighted_partition.B):
            for to_block in range(weighted_partition.B):
                if weighted_partition.get_edge_count(from_block, to_block) > 0:
                    if self.is_directed or from_block != to_block:
                        self.assertEqual(weighted_partition.get_edge_count(from_block, to_block),
                                         covariate_partition.get_sum_of_covariates(from_block, to_block))
                    else:
                        self.assertEqual(weighted_partition.get_edge_count(from_block, to_block) / 2,
                                         covariate_partition.get_sum_of_covariates(from_block, to_block))

    def test_get_sum_of_covariates(self):
        """Test calculation and updating of summed covariates"""
        # compare initial
        self.compare_weighted_and_covariate_partition(self.weighted_partition, self.covariate_partition)

        for _ in range(10):
            node, _, to_block = self.weighted_partition.get_random_move()
            self.weighted_partition.move_node(node, to_block)
            self.covariate_partition.move_node(node, to_block)

            self.compare_weighted_and_covariate_partition(self.weighted_partition, self.covariate_partition)

    def test_precalc_move(self):
        """Test pre calculation of move with the information about the covariates"""
        for _ in range(10):
            node, _, to_block = self.weighted_partition.get_random_move()
            self.weighted_partition.move_node(node, to_block)
            self.covariate_partition.move_node(node, to_block)
            self.unweighted_partition.move_node(node, to_block)

            for _ in range(10):
                move = self.weighted_partition.get_random_move()

                self.assertEqual(self.covariate_partition.precalc_move(move, self.objective_function),
                                 self.unweighted_partition.precalc_move(move, self.objective_function) \
                                 + self.weighted_partition.precalc_move(move, self.objective_function))


class TestNxPartitionGraphBasedCovariatesDirected(TestNxPartitionGraphBasedCovariates):
    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionGraphBasedCovariatesDirected, self).__init__(methodName)
        self.is_directed = True
