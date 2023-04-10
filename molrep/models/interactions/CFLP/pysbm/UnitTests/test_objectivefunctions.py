import copy
import random as rd
import unittest as ut

import networkx as nx

from pysbm import test_ground, sbm


class TestObjectiveFunctions(ut.TestCase):
    """ Test Class for objective functions """

    def setUp(self):
        self.graph = nx.Graph()
        self.number_of_nodes = 10
        for i in range(self.number_of_nodes):
            self.graph.add_edge(i, (i + 1) % self.number_of_nodes)
        # beside complete circle add one edge to connect all blocks
        self.graph.add_edge(0, 2)
        # here first without selfloops and undirected
        self.partition = sbm.NxPartition(graph=self.graph,
                                         number_of_blocks=3,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True)
        self._fixed_starting_point()

        self.objective_functions = []

        self.objective_functions.append(sbm.TraditionalUnnormalizedLogLikelyhood(is_directed=False))
        self.objective_functions.append(sbm.DegreeCorrectedUnnormalizedLogLikelyhood(is_directed=False))
        self.objective_functions.append(sbm.TraditionalMicrocanonicalEntropy(is_directed=False))
        self.objective_functions.append(sbm.TraditionalMicrocanonicalEntropyDense(is_directed=False))
        self.objective_functions.append(sbm.DegreeCorrectedMicrocanonicalEntropy(is_directed=False))
        self.objective_functions.append(sbm.IntegratedCompleteLikelihoodExactJeffrey(is_directed=False))
        self.objective_functions.append(sbm.IntegratedCompleteLikelihoodExactUniform(is_directed=False))
        self.objective_functions.append(sbm.NewmanReinertDegreeCorrected(is_directed=False))
        self.objective_functions.append(sbm.NewmanReinertDegreeCorrected(is_directed=False))
        self.objective_functions.append(sbm.NewmanReinertNonDegreeCorrected(is_directed=False))
        self.objective_functions.append(sbm.LogLikelihoodOfFlatMicrocanonicalNonDegreeCorrected(is_directed=False))
        self.objective_functions.append(sbm.LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedUniform(is_directed=False))
        self.objective_functions.append(
            sbm.LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedUniformHyperprior(is_directed=False))

    def _fixed_starting_point(self):
        #        start from a fixed partition as above
        for i in range(self.number_of_nodes):
            self.partition.move_node(i, i % self.partition.B)

    def test_delta(self):
        """Test calculation of delta"""

        #        try some moves,
        #         first move node with selfloop and after that random on random partition
        node = 0
        while self.partition.get_number_of_nodes_in_block(
                self.partition.get_block_of_node(node)) == 1:
            node = node + 1
        move = (node,
                self.partition.get_block_of_node(node),
                (self.partition.get_block_of_node(node) + 1) % self.partition.B)
        node, from_block, to_block = move
        for _ in range(20):
            for objective_function in self.objective_functions:
                start_value = objective_function.calculate(self.partition)
                parameters = self.partition.precalc_move(move, objective_function)
                calculated_delta = objective_function.calculate_delta(self.partition,
                                                                      from_block,
                                                                      to_block,
                                                                      *parameters)
                self.partition.move_node(node, to_block)
                real_delta = objective_function.calculate(self.partition) - start_value
                self.assertAlmostEqual(calculated_delta, real_delta, delta=0.0001,
                                       msg="\nError in " + str(objective_function)
                                           + "\nPartition: " + str(self.partition)
                                           + "\nmove: " + str(move)
                                           + "\nparameters: " + str(parameters)
                                       )
                self.partition.move_node(node, from_block)

            self.partition.random_partition(number_of_blocks=self.partition.B)
            move = self.partition.get_random_move()
            node, from_block, to_block = move

    def test_block_merge(self):
        """Test delta calculation of block merges"""

        #        for simplicity merge the last two block
        block = self.partition.B - 1
        merge_with_block = self.partition.B - 2
        merge = {block: merge_with_block}

        #        create a new partition where one can perform the merge
        merged_partition = self.partition.copy()
        merged_partition.merge_blocks(merge, self.partition.B - 1)

        for objective_function in self.objective_functions:
            #            use protected methods from Peixoto
            inference = sbm.PeixotoInference(self.partition._graph,
                                             objective_function,
                                             self.partition,
                                             no_partition_reset=True)

            parameter, reference = inference._precalc_block_merge(block)
            #            save right parameters for later comparision
            saved_parameter = copy.deepcopy(parameter)
            #            modify general parameters for actual block merge
            saved_data = inference._adjustment_delta_pre(reference, block, merge_with_block)

            #                calculate real delta using the merged partition
            real_delta = objective_function.calculate(merged_partition) - objective_function.calculate(self.partition)

            #             calculate via parameter
            calculated_delta = objective_function.calculate_delta(
                self.partition, block, merge_with_block, *parameter)

            #            check if both are almost the same
            self.assertAlmostEqual(calculated_delta, real_delta, delta=0.0001,
                                   msg="Objective function:" + str(objective_function)
                                       + "\nParameters: " + str(parameter))

            #            reset neighbor_info kit
            inference._adjustment_delta_post(reference, merge_with_block, saved_data)
            #            test if reset works correctly
            self.assertEqual(parameter, saved_parameter)

    def test_extra_functions(self):
        """Test counting deltas, calculate from inference, ... """

        max_counter = 10
        from_block = 0
        to_block = 0
        parameters = {}
        for objective_function in self.objective_functions:
            #            test counting of delta calculations
            for i in range(1, max_counter):
                move = self.partition.get_random_move()
                _, from_block, to_block = move

                parameters = self.partition.precalc_move(move, objective_function)
                objective_function.calculate_delta(self.partition,
                                                   from_block,
                                                   to_block,
                                                   *parameters)
                self.assertEqual(objective_function.number_of_calculated_deltas, i)
            # test wrong parameters length
            with self.assertRaises(ValueError):
                objective_function.calculate_delta(self.partition, from_block, to_block)

            # test that delta for null move is null
            self.assertEqual(objective_function.calculate_delta(self.partition, 0, 0, *parameters),
                             0)

    def test_denseTraditionalMicrocanonicalEntropy(self):
        """Test dense"""
        #        only do this test in basic version
        if self.__class__ != TestObjectiveFunctions:
            return
        dense_objective_function = sbm.TraditionalMicrocanonicalEntropyDense(is_directed=False)
        sparse_objective_function = sbm.TraditionalMicrocanonicalEntropy(is_directed=False)

        #        roughly the same value (really rough)
        self.assertAlmostEqual(dense_objective_function.calculate(self.partition),
                               sparse_objective_function.calculate(self.partition),
                               delta=10)

    def test_base_class(self):
        """Test correct mapping of functions and switch on is_directed"""
        objective_function = sbm.ObjectiveFunction(self.partition.is_graph_directed(),
                                                   lambda x: "complete undirected",
                                                   lambda x: "complete directed",
                                                   lambda *args: "delta undirected",
                                                   lambda *args: "delta directed")
        if self.partition.is_graph_directed():
            self.assertEqual(objective_function.calculate(self.partition), "complete directed")
            self.assertEqual(objective_function.calculate_delta(self.partition, 0, 1), "delta directed")
        else:
            self.assertEqual(objective_function.calculate(self.partition), "complete undirected")
            self.assertEqual(objective_function.calculate_delta(self.partition, 0, 1), "delta undirected")

        objective_function.is_directed = True
        digraph = nx.DiGraph()
        digraph.add_edge(0, 1)
        directed_partition = sbm.NxPartition(digraph, number_of_blocks=1)
        self.assertEqual(objective_function.calculate(directed_partition), "complete directed")
        self.assertEqual(objective_function.calculate_delta(directed_partition, 0, 1), "delta directed")

        objective_function.is_directed = False
        graph = nx.Graph()
        graph.add_edge(0, 1)
        undirected_partition = sbm.NxPartition(graph, number_of_blocks=1)
        self.assertEqual(objective_function.calculate(undirected_partition), "complete undirected")
        self.assertEqual(objective_function.calculate_delta(undirected_partition, 0, 1), "delta undirected")

    def test_empty_to_group(self):
        objective_function = self.objective_functions[0]

        graph = nx.complete_graph(5)
        if objective_function.is_directed:
            graph = nx.DiGraph(graph)
        partition = sbm.NxPartition(graph, representation={0: 2, 1: 1, 2: 0, 3: 0, 4: 0})
        partition.move_node(1, 0)

        precalc_info = partition.precalc_move((0, 2, 1), objective_function)
        for objective_function in self.objective_functions:
            objective_function.calculate_delta(partition, 2, 1, *precalc_info)

        # first find
        generator = test_ground.PlantedPartitionGenerator(4, 32, 0.9696969696969697, 0.0)
        graph, _, _ = generator.generate(directed=False, seed=0)
        if objective_function.is_directed:
            graph = nx.DiGraph(graph)

        partition = sbm.NxPartition(graph,
                                    representation={0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2,
                                                    11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2,
                                                    20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2,
                                                    29: 2, 30: 2, 31: 2, 32: 3, 33: 3, 34: 3, 35: 3, 36: 3, 37: 3,
                                                    38: 3, 39: 3, 40: 3, 41: 3, 42: 3, 43: 3, 44: 3, 45: 3, 46: 3,
                                                    47: 3, 48: 3, 49: 3, 50: 3, 51: 3, 52: 3, 53: 3, 54: 3, 55: 3,
                                                    56: 3, 57: 3, 58: 3, 59: 3, 60: 3, 61: 3, 62: 3, 63: 3, 64: 2,
                                                    65: 2, 66: 2, 67: 2, 68: 2, 69: 2, 70: 2, 71: 2, 72: 2, 73: 2,
                                                    74: 2, 75: 2, 76: 2, 77: 2, 78: 2, 79: 2, 80: 2, 81: 2, 82: 2,
                                                    83: 2, 84: 2, 85: 2, 86: 2, 87: 2, 88: 2, 89: 2, 90: 2, 91: 2,
                                                    92: 2, 93: 2, 94: 2, 95: 2, 96: 0, 97: 0, 98: 0, 99: 0, 100: 0,
                                                    101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106: 0, 107: 0, 108: 0,
                                                    109: 0, 110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0,
                                                    117: 0, 118: 0, 119: 0, 120: 0, 121: 0, 122: 0, 123: 0, 124: 0,
                                                    125: 0, 126: 0, 127: 0})

        precalc_info = partition.precalc_move((0, 2, 1), objective_function)
        for objective_function in self.objective_functions:
            objective_function.calculate_delta(partition, 2, 1, *precalc_info)


class TestObjectiveFunctionsExtended(TestObjectiveFunctions):
    """ Test Class for objective functions """

    def setUp(self):
        super(TestObjectiveFunctionsExtended, self).setUp()
        self.partitions = []

        #        alternate with selfloop
        self.graph = nx.Graph()
        self.number_of_nodes = 10
        for i in range(self.number_of_nodes):
            self.graph.add_edge(i, (i + 1) % self.number_of_nodes)
        # beside complete circle add one edge to connect all blocks
        self.graph.add_edge(0, 2)
        #   include one self loop
        self.graph.add_edge(0, 0)
        self.partition = sbm.NxPartition(graph=self.graph,
                                         number_of_blocks=3,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True)
        self._fixed_starting_point()

        self.partitions.append(self.partition)
        # generate some random graphs with increasing chance
        for i in range(8):
            graph = nx.gnp_random_graph(10, i * .1 + .1)
            partition = sbm.NxPartition(graph=graph,
                                        number_of_blocks=3,
                                        calculate_degree_of_blocks=True,
                                        save_neighbor_edges=True,
                                        save_neighbor_of_blocks=True)
            self.partitions.append(partition)

    def _partitions_caller(self, call_function):
        #        call call_function with different
        for partition in self.partitions:
            self.partition = partition
            call_function()

    def test_delta(self):
        self._partitions_caller(super(TestObjectiveFunctionsExtended, self).test_delta)

    def test_block_merge(self):
        self._partitions_caller(super(TestObjectiveFunctionsExtended, self).test_block_merge)


class TestObjectiveFunctionsDirected(TestObjectiveFunctionsExtended):
    """Test class for testing directed versions of objective function"""

    def setUp(self):
        #        basically we only set up the test with different graphs and
        #         after that the same tests as before apply
        super(TestObjectiveFunctionsDirected, self).setUp()
        self.partitions = []
        self.graph = nx.DiGraph()
        self.number_of_nodes = 10
        for i in range(self.number_of_nodes):
            self.graph.add_edge(i, (i + 1) % self.number_of_nodes)
        # beside complete circle add one edge to connect all blocks
        self.graph.add_edge(0, 2)
        # here first without selfloops and undirected
        self.partition = sbm.NxPartition(graph=self.graph,
                                         number_of_blocks=4,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True)
        self._fixed_starting_point()
        self.partitions.append(self.partition)

        #        with selfloop
        self.graph = self.graph.copy()
        self.graph.add_edge(0, 0)
        self.partition = sbm.NxPartition(graph=self.graph,
                                         number_of_blocks=3,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True)
        self._fixed_starting_point()
        self.partitions.append(self.partition)

        for i in range(1, 10):
            #            attention creates MultiDigraph
            graph = nx.scale_free_graph(i * 10 + 1)
            # at the moment algorithms can only handle DiGraphs therefore cast
            partition = sbm.NxPartition(graph=nx.DiGraph(graph),
                                        number_of_blocks=rd.randint(2, 9),
                                        calculate_degree_of_blocks=True,
                                        save_neighbor_edges=True,
                                        save_neighbor_of_blocks=True)
            self.partitions.append(partition)

        for objective_function in self.objective_functions:
            objective_function.is_directed = True


if __name__ == '__main__':
    Loader = ut.TestLoader()
    suite = Loader.loadTestsFromTestCase(TestObjectiveFunctions)
    suite.addTests(Loader.loadTestsFromTestCase(TestObjectiveFunctionsDirected))
    suite.addTests(Loader.loadTestsFromTestCase(TestObjectiveFunctionsExtended))
    ut.TextTestRunner(verbosity=2).run(suite)
