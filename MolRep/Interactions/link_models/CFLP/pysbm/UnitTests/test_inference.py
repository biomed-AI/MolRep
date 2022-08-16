import random as rd
import unittest as ut

import networkx as nx
import numpy as np

from pysbm import sbm


class TestSBMInference(ut.TestCase):
    """ Test Class for SBMInference and subclasses """

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestSBMInference, self).__init__(methodName)
        self.is_directed = False

    def setUp(self):
        self.graph = nx.karate_club_graph()

        self.partition = sbm.NxPartition(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True)

        self.objective_function = sbm.TraditionalUnnormalizedLogLikelyhood(is_directed=False)

        # set up -> worse enough random partition
        #  optimum by around -240 start at least below -260
        while self.objective_function.calculate(self.partition) > -260:
            self.partition.random_partition(number_of_blocks=2)

    def test_base_class(self):
        """Test Class SBMInference"""
        inference = sbm.Inference(self.graph, self.objective_function, self.partition)

        # test reminders
        with self.assertRaises(NotImplementedError):
            inference.infer_stepwise()
        with self.assertRaises(NotImplementedError):
            inference.infer_stochastic_block_model()

        # test getter and setter of objective_function
        self.assertEqual(inference.objective_function, self.objective_function)

        other_objective_function = sbm.DegreeCorrectedMicrocanonicalEntropy(is_directed=False)
        self.assertNotEqual(inference.objective_function, other_objective_function)

        inference.objective_function = other_objective_function
        self.assertEqual(inference.objective_function, other_objective_function)

        # start value of get moves..
        self.assertEqual(inference.get_moves(), 0)

    def test_base_behaviour(self):
        """Test that all inference method improves objective function"""

        inference_classes = [sbm.MetropolisInference,
                             sbm.MetropolisHastingInference,
                             sbm.EMInference,
                             sbm.KarrerInference,
                             sbm.PeixotoInference,
                             sbm.KerninghanLinInference]
        may_not_decrease_every_time = {sbm.MetropolisInference,
                                       sbm.MetropolisHastingInference, }
        # try n times for all inference methods to improve the starting partition

        for inference_class in inference_classes:
            random_seed = 42
            np.random.seed(random_seed)
            rd.seed(random_seed)

            optimized = False

            for _ in range(3):
                # every loop with different starting partition
                self.partition.random_partition(number_of_blocks=2)
                start_value = self.objective_function.calculate(self.partition)

                new_partition = self.partition.copy()
                inference = inference_class(self.graph, self.objective_function,
                                            new_partition)

                inference.infer_stochastic_block_model()
                if inference_class in may_not_decrease_every_time:
                    if self.objective_function.calculate(new_partition) > start_value:
                        optimized = True
                else:
                    self.assertGreater(self.objective_function.calculate(new_partition),
                                       start_value)

            # for MCMC method at least one try should have an improve
            if inference_class in may_not_decrease_every_time:
                self.assertTrue(optimized)

    def test_metropolis_inference(self):
        """Test extra functions of Metropolis inference"""

        inference = sbm.MetropolisInference(self.graph, self.objective_function, self.partition)

        # too many parameters...
        with self.assertRaises(ValueError):
            inference.infer_stochastic_block_model("steps", "save_min", "too_much")

        # run one step
        inference.infer_stepwise()
        self.assertEqual(inference.steps, 1)
        self.assertLessEqual(inference.get_moves(), 1)

        # run for only 10 steps
        inference.infer_stochastic_block_model(10)
        self.assertEqual(inference.steps, 11)
        self.assertLessEqual(inference.get_moves(), 11)

        # standard no saving of minimal partition
        self.assertEqual(inference.min_partition, None)

        # run for 300 steps with saving min
        inference.infer_stochastic_block_model(600, True)
        self.assertEqual(inference.steps, 611)
        self.assertLessEqual(inference.get_moves(), 611)
        # at least some moves performed
        self.assertGreater(inference.get_moves(), 10)
        self.assertEqual(len(inference.min_partition), self.partition.get_number_of_nodes())

        # try some more minimum saves
        for _ in range(10):
            self.partition.random_partition(number_of_blocks=2)
            inference.partition = self.partition
            inference.infer_stochastic_block_model(500, True)

        if not self.is_directed:
            # run even without delta
            objective_without_delta = sbm.TraditionalMicrocanonicalEntropy(
                is_directed=self.partition.is_graph_directed())

            # ensure no delta function
            def stub(*args):
                raise NotImplementedError()

            objective_without_delta.function_calculate_delta_directed = stub
            objective_without_delta.function_calculate_delta_undirected = stub
            objective_without_delta.is_directed = True
            objective_without_delta.is_directed = False
            objective_without_delta.is_directed = self.partition.is_graph_directed()

            inference.objective_function = objective_without_delta

            inference.use_delta = False
            inference.infer_stepwise()
            inference.infer_stochastic_block_model(1000)

    def test_karrer_with_no_negative_move(self):
        """ Test Karrer without negative moves """
        start_value = self.objective_function.calculate(self.partition)

        partition = self.partition.copy()
        inference = sbm.KarrerInference(self.graph, self.objective_function,
                                        partition, no_negative_move=True)

        inference.infer_stochastic_block_model()

        self.assertGreaterEqual(self.objective_function.calculate(partition),
                                start_value)


# noinspection PyClassHasNoInit,PyAbstractClass
class ObjectiveFunctionDummy(sbm.TraditionalUnnormalizedLogLikelyhood):
    """ Dummy for easier test of acceptance probability """

    def calculate_delta(self, partition, from_block, to_block, *args):
        return 0


class TestMetropolisHastingInference(ut.TestCase):
    """Test Class for Metropolis Hasting Inference"""

    def setUp(self):
        # unconnected example
        graph = nx.Graph()
        # create graph(s)
        #  0 - 1
        # 2 -  3 - 4
        # 6 - 5 -7
        # 8 - 9
        # |   |
        # 10 - 11
        # 12 - 13 (selfloop to 13)
        # 14 - 15 (selfloop) - 16 - 17 (selfloop) - 18
        # ensure that each node can be moved, i.e. in every block are at least 2 nodes
        graph.add_edges_from([(0, 1),
                              (2, 3), (3, 4),
                              (5, 6), (5, 7),
                              (8, 9), (9, 10), (10, 11), (11, 8),
                              (12, 13), (13, 13),
                              (14, 15), (15, 15), (15, 16), (16, 17), (17, 17), (17, 18)
                              ])
        self.partition = sbm.NxPartition(graph=graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=False)

        # set defined block state
        for node in range(self.partition.get_number_of_nodes()):
            self.partition.move_node(node, node % 2)

        self.objective_function = ObjectiveFunctionDummy(is_directed=False)

        self.inference = sbm.MetropolisHastingInference(graph,
                                                        self.objective_function,
                                                        self.partition)

        # same with 3 blocks
        partition_3 = sbm.NxPartition(graph=graph,
                                      number_of_blocks=3,
                                      calculate_degree_of_blocks=True,
                                      save_neighbor_edges=True,
                                      save_neighbor_of_blocks=True,
                                      weighted_graph=False)
        # set defined block state
        for node in range(self.partition.get_number_of_nodes()):
            partition_3.move_node(node, node % 3)

        inference_3 = sbm.MetropolisHastingInference(graph,
                                                     self.objective_function,
                                                     partition_3)

        # create directed graph with the same examples like before,
        #  i.e. always insert both possible edges
        digraph = nx.DiGraph(graph)
        # add directed cycle
        # 19  -> 20 -> 21
        #  | <- - - - |
        digraph.add_edges_from([(19, 20), (20, 21), (21, 19)])

        self.dipartition = sbm.NxPartition(graph=digraph,
                                           number_of_blocks=2,
                                           calculate_degree_of_blocks=True,
                                           save_neighbor_edges=True,
                                           save_neighbor_of_blocks=True,
                                           weighted_graph=False)

        # set defined block state
        for node in range(self.dipartition.get_number_of_nodes()):
            self.dipartition.move_node(node, node % 2)

        self.di_inference = sbm.MetropolisHastingInference(digraph,
                                                           self.objective_function,
                                                           self.dipartition)

        dipartition_3 = sbm.NxPartition(graph=digraph,
                                        number_of_blocks=3,
                                        calculate_degree_of_blocks=True,
                                        save_neighbor_edges=True,
                                        save_neighbor_of_blocks=True,
                                        weighted_graph=False)
        # set defined block state
        for node in range(self.dipartition.get_number_of_nodes()):
            dipartition_3.move_node(node, node % 3)

        di_inference_3 = sbm.MetropolisHastingInference(digraph,
                                                        self.objective_function,
                                                        dipartition_3)

        self.test_cases = [
            (self.partition, self.inference),
            (partition_3, inference_3),
            (self.dipartition, self.di_inference),
            (dipartition_3, di_inference_3),
        ]

    def _calculate_block_moving_probability_full(self, partition=None, inference=None):
        """
Calculate the moving probability for all nodes
        """
        if partition is None:
            partition = self.partition
        if inference is None:
            inference = self.inference
        reference = [{} for _ in range(partition.get_number_of_nodes())]
        for node in range(partition.get_number_of_nodes()):
            # calculate the fraction of neighbors belonging to certain blocks
            reference[node] = self._calculate_block_moving_probability(node, partition, inference)

        return reference

    def _calculate_block_moving_probability(self, node, partition=None, inference=None):
        """
        calculate the formula sum_(neighbor blocks) (fraction of neighbors belonging to that block)\
          p(r->s|t) (where r old block, t = neighbor block and s = block = new block)
        """

        if partition is None:
            partition = self.partition
        if inference is None:
            inference = self.inference

        probability = {i: 0 for i in range(partition.B)}
        neighbor_block_fraction = {}
        for neighbor in partition.get_neighbors_of_node(node):
            neighbor_block = partition.get_block_of_node(neighbor)
            neighbor_block_fraction[neighbor_block] = \
                neighbor_block_fraction.get(neighbor_block, 0) + 1.0

        # print neighbor_block_fraction

        for block in range(partition.B):
            for neighbor_block in neighbor_block_fraction:
                if partition.is_graph_directed():
                    probability[block] = probability.get(block, 0) \
                                         + neighbor_block_fraction[neighbor_block] \
                                         / len(partition.get_neighbors_of_node(node)) \
                                         * inference._probability_directed(block, neighbor_block)
                else:
                    probability[block] = probability.get(block, 0) \
                                         + neighbor_block_fraction[neighbor_block] \
                                         / len(partition.get_neighbors_of_node(node)) \
                                         * inference._probability_undirected(block, neighbor_block)
                #   print probability
                #   print inference._probability(block, neighbor_block)
        return probability

    def test_block_selecting(self):
        """ Test Selecting of new blocks"""

        # first test with too many arguments
        with self.assertRaises(ValueError):
            self.inference.infer_stochastic_block_model("too", "many", "arguments")

        number_of_tries = 1000

        for partition, inference in self.test_cases:
            inference.epsilon = 0.0

            hits = [{} for _ in range(partition.get_number_of_nodes())]

            # sample select new block and calculate right probability
            for node in range(partition.get_number_of_nodes()):
                for _ in range(number_of_tries):
                    if partition.is_graph_directed():
                        selected_block = inference._select_new_block_directed(node)
                    else:
                        selected_block = inference._select_new_block_undirected(node)
                    hits[node][selected_block] = hits[node].get(selected_block, 0) + 1

            reference = self._calculate_block_moving_probability_full(partition, inference)

            # check results
            for node in range(partition.get_number_of_nodes()):
                for selected_block in hits[node]:
                    self.assertAlmostEqual((hits[node][selected_block] * 1.0) / number_of_tries,
                                           reference[node][selected_block],
                                           delta=0.05)

    def test_acceptance_probability(self):
        """Test if the move probabilities are calculated right"""

        # test on directed and undirected graphs
        for partition, inference in self.test_cases:
            inference.epsilon = 0.0
            # print inference
            for node in range(partition.get_number_of_nodes()):
                #   print node
                old_block = partition.get_block_of_node(node)
                bottom = self._calculate_block_moving_probability(node,
                                                                  partition,
                                                                  inference)
                for i in range(partition.B):
                    new_block = (old_block + i) % partition.B
                    #  print str(node)+":"+str(old_block)+"->"+str(new_block)
                    if partition.is_graph_directed():
                        calculated_probability = inference._calculate_acceptance_probability_directed(
                            node, old_block, new_block, debug=True)
                    else:
                        calculated_probability = inference._calculate_acceptance_probability_undirected(
                            node, old_block, new_block, debug=True)

                    partition.move_node(node, new_block)

                    top = self._calculate_block_moving_probability(node,
                                                                   partition,
                                                                   inference)
                    partition.move_node(node, old_block)

                    self.assertAlmostEqual(calculated_probability[0], top[old_block])
                    self.assertAlmostEqual(calculated_probability[1], bottom[new_block])


class TestSBMInferenceTextNodes(TestSBMInference):
    """ Test Class for SBMInference and subclasses on Text nodes"""

    def setUp(self):
        self.graph = nx.Graph()

        for from_node, to_node in nx.karate_club_graph().edges():
            self.graph.add_edge("P" + str(from_node), "P" + str(to_node))

        self.partition = sbm.NxPartition(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True)

        self.objective_function = sbm.TraditionalUnnormalizedLogLikelyhood(is_directed=False)

        # set up -> worse enough random partition
        #  optimum by around -240 start at least below -260
        while self.objective_function.calculate(self.partition) > -260:
            self.partition.random_partition(number_of_blocks=2)


class TestSBMInferenceDirected(TestSBMInference):
    """ Test Class for SBMInference and subclasses on directed Graphs"""

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestSBMInferenceDirected, self).__init__(methodName)
        self.is_directed = True

    def setUp(self):
        self.graph = nx.DiGraph(nx.karate_club_graph())

        self.partition = sbm.NxPartition(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True)

        self.objective_function = sbm.TraditionalUnnormalizedLogLikelyhood(is_directed=True)

        # set up -> worse enough random partition
        #  optimum by around -240 start at least below -260
        while self.objective_function.calculate(self.partition) > -260:
            self.partition.random_partition(number_of_blocks=2)


class TestSBMInferenceDirectedTextNodes(TestSBMInferenceDirected):
    """ Test Class for SBMInference and subclasses on directed graphs with text nodes"""

    def setUp(self):
        self.graph = nx.DiGraph()

        for from_node, to_node in nx.karate_club_graph().edges():
            self.graph.add_edge("P" + str(from_node), "P" + str(to_node))
            self.graph.add_edge("P" + str(to_node), "P" + str(from_node))

        self.partition = sbm.NxPartition(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True)

        self.objective_function = sbm.TraditionalUnnormalizedLogLikelyhood(is_directed=True)

        # set up -> worse enough random partition
        #  optimum by around -240 start at least below -260
        while self.objective_function.calculate(self.partition) > -260:
            self.partition.random_partition(number_of_blocks=2)


class TestPeixotoInference(ut.TestCase):
    """ Test Class for SBMInference and subclasses """

    def setUp(self):
        # first check probabilities of new block
        self.graph = nx.Graph()
        # create graph
        #  0 - 1
        # |  \ | /\ 4
        # 3 -  2
        self.graph.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 2), (1, 4), (2, 4)])
        self.partition = sbm.NxPartition(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True)

        self.objective_function = sbm.TraditionalUnnormalizedLogLikelyhood(is_directed=False)
        self.inference = sbm.PeixotoInference(self.graph, self.objective_function, self.partition)
        self.inference.epsilon = 0.0

    def test_select_new_block(self):
        """Test probabilities of selecting block merges"""
        # set defined block state
        for node in range(self.partition.get_number_of_nodes()):
            self.partition.move_node(node, node % 2)
        self.partition.B = 2

        # calculate fraction of neighbors of all blocks
        fraction_of_neighbors = {0: [0.0, 0.0], 1: [0.0, 0.0]}
        for node in self.partition.get_nodes_iter():
            own_block = self.partition.get_block_of_node(node)
            for neighbor in self.partition.get_neighbors_of_node(node):
                neighbor_block = self.partition.get_block_of_node(neighbor)
                fraction_of_neighbors[own_block][neighbor_block] += 1
                # calculate the right probabilities
        probability = {0: [0.0, 0.0], 1: [0.0, 0.0]}
        for r in fraction_of_neighbors:
            total_neighbors = sum(fraction_of_neighbors[r])
            for s in range(2):
                for t in range(2):
                    probability[r][s] += fraction_of_neighbors[r][t] / total_neighbors \
                                         * (self.partition.get_edge_count(t, s) + self.inference.epsilon) \
                                         / (self.partition.get_degree_of_block(t)
                                            + self.inference.epsilon * self.partition.B)

                    # get results from implementation
        number_of_tries = 10000
        next_block_hits = {0: [0.0, 0.0], 1: [0.0, 0.0]}
        for block in range(2):
            for _ in range(number_of_tries):
                next_block = self.inference._select_new_block_for_block(block)
                next_block_hits[block][next_block] += 1

                # see if its matches with the probabilities
        for block in probability:
            for next_block in range(2):
                self.assertAlmostEqual(next_block_hits[block][next_block] / number_of_tries,
                                       probability[block][next_block],
                                       delta=0.1)

    def test_cleanup_block_merges(self):
        """Test cleaning merges"""
        # prepare a bigger graph (10 nodes)
        self.graph.add_edges_from([(5, 6), (7, 8), (8, 9)])
        self.partition = sbm.NxPartition(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True)
        self.inference = sbm.PeixotoInference(self.graph, self.objective_function, self.partition)

        # minimum move on (delta, block_1, block_2)
        merges_with_delta = [(0, 8, 4), (0, 1, 2), (0, 1, 0), (0, 5, 6), (0, 2, 4)]
        result, new_block_number = self.inference._cleanup_merges(merges_with_delta)

        # no loop therefore decrease by length
        self.assertEqual(new_block_number, 5)
        # test if all higher blocks are shifted
        for i in range(new_block_number, 10):
            self.assertTrue(i in result)
            self.assertLess(result[i], new_block_number)
        # Check merges
        for block in [1, 2, 4, 8]:
            self.assertEqual(result[block], 0)

        self.assertEqual(result[5], result[6])

        # next try with some loops
        merges_with_delta = [(0, 8, 4), (0, 1, 2), (0, 1, 8), (0, 4, 2), (0, 3, 6), (0, 6, 3)]
        result, new_block_number = self.inference._cleanup_merges(merges_with_delta)

        self.assertEqual(new_block_number, 6)
        # test if all higher blocks are shifted
        for i in range(new_block_number, 10):
            self.assertTrue(i in result)
            self.assertLess(result[i], new_block_number)
        # Check merges
        for block in [2, 4, 8]:
            self.assertEqual(result[block], 1)
        self.assertEqual(result[6], 3)

        # test right updating of observed
        merges_with_delta = [(0, 2, 1), (0, 3, 0), (0, 2, 3), (0, 5, 4), (0, 6, 7), (0, 5, 6)]
        result, new_block_number = self.inference._cleanup_merges(merges_with_delta)
        self.assertEqual(new_block_number, 4)
        # test if all higher blocks are shifted
        for i in range(new_block_number, 10):
            self.assertTrue(i in result)
            self.assertLess(result[i], new_block_number)

    def test_objective_function(self):
        """Test passing of the new objective function"""
        #
        objective_function = sbm.DegreeCorrectedMicrocanonicalEntropy(is_directed=False)
        self.inference.objective_function = objective_function
        self.assertEqual(self.inference.objective_function, objective_function)
        self.assertEqual(self.inference.mcmc.objective_function, objective_function)

        # check if no error occurs
        self.inference.infer_stepwise()

    def test_infer_stepwise(self):
        """Test stopping of infer stepwise"""
        with self.assertRaises(StopIteration):
            while True:
                self.inference.infer_stepwise()


class TestKerninghanLinInference(ut.TestCase):

    def test_inference(self):
        graph = nx.Graph()
        graph.add_edges_from(
            [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13),
             (0, 17), (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
             (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12), (3, 13), (4, 6),
             (4, 10), (5, 6), (5, 10), (5, 16), (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32),
             (14, 33), (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32), (20, 33), (22, 32), (22, 33),
             (23, 25), (23, 27), (23, 29), (23, 32), (23, 33), (24, 25), (24, 27), (24, 31), (25, 31), (26, 29),
             (26, 33), (27, 33), (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33), (31, 32), (31, 33),
             (32, 33)])

        partition = sbm.NxPartition(graph,
                                    representation={0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0, 9: 0, 10: 1,
                                                     11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 1, 17: 0, 18: 0, 19: 0,
                                                     20: 0, 21: 0, 22: 0, 23: 1, 24: 1, 25: 0, 26: 1, 27: 0, 28: 0,
                                                     29: 0, 30: 1, 31: 1, 32: 1, 33: 1})

        objective_function = sbm.DegreeCorrectedUnnormalizedLogLikelyhood(is_directed=False)

        inference = sbm.KerninghanLinInference(graph, objective_function, partition)

        inference.infer_stochastic_block_model()


class TestSpectralInference(ut.TestCase):

    def test_infer_stochastic_block_model(self):
        # dummy objective function
        objective_function = sbm.DegreeCorrectedUnnormalizedLogLikelyhood(is_directed=False)

        graph = nx.DiGraph()
        graph.add_edges_from(
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7), (0, 4)])
        graph[0][1]['weight'] = 20
        representation = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
        partition = sbm.NxPartition(graph, 2, True, True, False, False, False, representation)
        x = sbm.SpectralInference(graph, objective_function, partition)
        a = x.infer_stochastic_block_model()
