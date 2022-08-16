import inspect
import random as rd
import unittest as ut

import networkx as nx
import numpy as np
import six

from pysbm import sbm


class TestPartition(ut.TestCase):
    """ Test Class for Partition """

    def test_reminders(self):
        """Test NotImplementedError reminders"""
        partition = sbm.Partition(None, fill_random=False)

        already_implemented = {'set_from_representation'}

        # this code call all methods of partition
        #  the arguments passed are the names of the arguments
        #  the first one is skipped because it is self and already given
        for method_name, method in inspect.getmembers(partition, inspect.ismethod):
            if str(method_name) in already_implemented:
                continue
            arg_spec = inspect.getfullargspec(method)
            with self.assertRaises(NotImplementedError):
                method(*arg_spec[0][1:])


class TestNxPartition(ut.TestCase):
    """ Test Class for NxPartition """

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartition, self).__init__(methodName)
        self.is_weighted = False
        self.test_class = sbm.NxPartition

    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        # # test if weights get ignored
        # self.graph.add_edge(1,2, weight=100)
        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=False,
                                         save_neighbor_edges=False,
                                         save_neighbor_of_blocks=False,
                                         weighted_graph=self.is_weighted)
        self._fixed_starting_point()

    def _fixed_starting_point(self):
        # start from a fixed partition
        self.partition.set_from_representation({1: 0, 2: 1, 3: 0, 4: 1})

    def test_move_node(self):
        """Test moving blocks and representation"""
        representation = {}
        for i in range(1, 5):
            self.assertEqual(self.partition.get_block_of_node(i), (i + 1) % 2)
            representation[i] = (i + 1) % 2

        self.assertEqual(str(self.partition), str(representation))

        self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 2)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 2)

        self.partition.move_node(1, 1)
        self.assertEqual(self.partition.get_block_of_node(1), 1)

        self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 1)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 3)

    def test_random_partition(self):
        """Test random generating of partition"""

        partition_count = {}
        number_of_tries = 1000
        for _ in range(number_of_tries):
            self.partition.random_partition(number_of_blocks=2)
            key = str(self.partition)
            partition_count.get(key, 0)
            partition_count[key] = partition_count.get(key, 0) + 1

        # roughly not generates all times the same but uniformly sample partitions
        number_of_partitions = len(partition_count)
        for key in partition_count:
            self.assertAlmostEqual(float(partition_count[key]) / number_of_tries,
                                   1.0 / number_of_partitions,
                                   delta=0.1)
        # test right random B
        for _ in range(10):
            self.partition.random_partition()
            self.assertTrue(1 <= self.partition.B <= self.partition.get_number_of_nodes())

        # test that in every block is at least one node
        # therefore work on a bigger graph
        self.graph = nx.Graph()
        for i in range(30):
            if self.is_weighted:
                self.graph.add_edge(i, (i + 1) % 30, weight=1)
            else:
                self.graph.add_edge(i, (i + 1) % 30)

        self.partition = self.test_class(graph=self.graph,
                                         calculate_degree_of_blocks=False,
                                         save_neighbor_edges=False,
                                         save_neighbor_of_blocks=False,
                                         weighted_graph=self.is_weighted)

        for _ in range(1000):
            self.partition.random_partition()
            for block in range(self.partition.B):
                self.assertGreaterEqual(self.partition.get_number_of_nodes_in_block(block), 1)

    def test_random_move(self):
        """ Test random moves"""
        # test that it only return "accepted" moves, i.e. the resulting partition
        #  after the proposed move does not contain an empty block
        for _ in range(100):
            self.partition.random_partition()
            # ensure more than one block and at least one node movable
            while (self.partition.B == 1
                   or self.partition.B == self.partition.get_number_of_nodes()):
                self.partition.random_partition()
            for __ in range(100):
                _, block, _ = self.partition.get_random_move()
                self.assertGreaterEqual(self.partition.get_number_of_nodes_in_block(block), 1)

    def test_get_basics(self):
        """Test all basic getters"""
        # get number of nodes
        self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 2)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 2)

        # Test Edge Counts
        self.assertEqual(self.partition.get_edge_count(0, 0), 0)
        self.assertEqual(self.partition.get_edge_count(1, 0), 4)
        self.assertEqual(self.partition.get_edge_count(0, 1), 4)
        self.assertEqual(self.partition.get_edge_count(1, 1), 0)

        # test read neighbors
        six.assertCountEqual(self, self.partition.get_neighbors_of_node(1), [2, 4])
        six.assertCountEqual(self, self.partition.get_neighbors_of_node(2), [3, 1])

        # test representation
        self.assertEqual(self.partition.get_representation(), {1: 0, 2: 1, 3: 0, 4: 1})
        # test readable representation
        self.assertEqual(str(self.partition), str({1: 0, 2: 1, 3: 0, 4: 1}))
        # test only block membership as array
        self.assertEqual(self.partition.get_block_memberships(), [0, 1, 0, 1])
        # test nodes count
        self.assertEqual(self.partition.get_number_of_nodes(), 4)

        # test random node
        #  first al nodes allowed
        hit_nodes = {}
        number_of_tries = 1000
        for _ in range(number_of_tries):
            node = self.partition.get_random_node()
            hit_nodes[node] = 1 + hit_nodes.get(node, 0)

        for node in range(1, 5):
            self.assertAlmostEqual(float(hit_nodes[node]) / number_of_tries, 0.25, delta=0.1)

        # now block one node by making it the only one in the block
        # node 1 now alone in block 0 -> zero hits
        self.partition.move_node(3, 1)
        hit_nodes = {}
        for _ in range(number_of_tries):
            node = self.partition.get_random_node()
            hit_nodes[node] = 1 + hit_nodes.get(node, 0)

        self.assertEqual(hit_nodes.get(1, 0), 0)

        for node in range(2, 5):
            self.assertAlmostEqual(float(hit_nodes[node]) / number_of_tries, 1.0 / 3, delta=0.1)

        # move node back and test random move
        self.partition.move_node(3, 0)
        hit_moves = {}
        for _ in range(number_of_tries):
            move = self.partition.get_random_move()
            hit_moves[move] = 1 + hit_moves.get(move, 0)

        # for every node a move is possible
        self.assertEqual(len(hit_moves), 4)
        for move in hit_moves:
            self.assertAlmostEqual(float(hit_moves[move]) / number_of_tries, .25, delta=0.1)

        # simple test of block_merge
        self.partition.merge_blocks({1: 0}, 1)
        self.assertEqual(self.partition.get_block_memberships(), [0, 0, 0, 0])
        self.assertEqual(self.partition.get_edge_count(0, 0), 8)
        self.assertEqual(self.partition.B, 1)

        # Test Error on all in own block
        self.partition = self.test_class(self.graph, number_of_blocks=4)
        with self.assertRaises(sbm.NoFreeNodeException):
            self.partition.get_random_node()

        # Test no_single flag -> no error
        self.partition.get_random_node(no_single=False)

        # Test node iter
        six.assertCountEqual(self, self.graph.nodes(), list(self.partition.get_nodes_iter()))

    def test_block_merges(self):
        """Test block merge and the effect on the partition"""
        # every node in a single block and all extras
        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=4,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        self.partition.set_from_representation({node + 1: node for node in range(4)})
        # merge two blocks
        self.partition.merge_blocks({3: 1, 2: 0}, 2)

        self.assertEqual(self.partition.B, 2)
        # get number of nodes
        self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 2)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 2)

        # Test Edge Counts
        self.assertEqual(self.partition.get_edge_count(0, 0), 0)
        self.assertEqual(self.partition.get_edge_count(1, 0), 4)
        self.assertEqual(self.partition.get_edge_count(0, 1), 4)
        self.assertEqual(self.partition.get_edge_count(1, 1), 0)

        # test read neighbors
        six.assertCountEqual(self, self.partition.get_neighbors_of_block(0), {1})
        six.assertCountEqual(self, self.partition.get_neighbors_of_block(1), {0})

        # test degree of blocks
        self.assertEqual(self.partition.get_degree_of_block(0), 4)
        self.assertEqual(self.partition.get_degree_of_block(1), 4)

    def test_neighbor_block(self):
        """ Test correct detecting of neighboring blocks """
        for i in range(50):
            # each 5th try create a new random graph
            #  else only move some nodes randomly
            if i % 5 == 0:
                self.graph = nx.gnp_random_graph(100, rd.random())
                if self.is_weighted:
                    for from_node, to_node in self.graph.edges():
                        self.graph[from_node][to_node]['weight'] = 1
                self.partition = self.test_class(graph=self.graph,
                                                 number_of_blocks=rd.randint(2, 99),
                                                 calculate_degree_of_blocks=False,
                                                 save_neighbor_edges=False,
                                                 save_neighbor_of_blocks=True,
                                                 weighted_graph=self.is_weighted)
            else:
                for _ in range(20):
                    node, _, to_block = self.partition.get_random_move()
                    self.partition.move_node(node, to_block)
                    # create correct sets based on edge counts
            list_of_neighbors = [set() for _ in range(self.partition.B)]
            for block in range(self.partition.B):
                #   check all possible neighbors
                for possible_neighbor in range(self.partition.B):
                    if self.partition.get_edge_count(block, possible_neighbor) != 0 \
                            or self.partition.get_edge_count(possible_neighbor, block) != 0:
                        list_of_neighbors[block].add(possible_neighbor)
                        # check entries
            for block in range(self.partition.B):
                self.assertEqual(self.partition.get_neighbors_of_block(block),
                                 list_of_neighbors[block])

    def test_extra_calculations(self):
        """Test degree saving, neighbor edge saving, neighbor blocks saving"""

        # test degree count
        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=False,
                                         save_neighbor_of_blocks=False,
                                         weighted_graph=self.is_weighted)

        self._fixed_starting_point()
        self.assertEqual(self.partition.get_degree_of_block(0), 4)
        self.assertEqual(self.partition.get_degree_of_block(1), 4)

        # Test saving neighbor edges
        self.partition.set_save_neighbor_edges(True)
        for block in range(2):
            hits = {}
            number_of_tries = 1000
            for _ in range(number_of_tries):
                edge = self.partition.get_random_neighboring_edge_of_block(block)
                hits[edge] = 1 + hits.get(edge, 0)

            self.assertEqual(len(hits), 4)
            # saved edges are in the from (node_of_own_block, other node)
            if block == 0:
                six.assertCountEqual(self, hits.keys(), [(1, 2), (1, 4), (3, 2), (3, 4)])
            elif block == 1:
                six.assertCountEqual(self, hits.keys(), [(2, 1), (2, 3), (4, 1), (4, 3)])
            for edge in hits:
                self.assertAlmostEqual(float(hits[edge]) / number_of_tries, 0.25, delta=0.1)

        # delete information
        self.partition.set_save_neighbor_edges(False)
        with self.assertRaises(Exception):
            self.partition.get_random_neighboring_edge_of_block(0)

        # test saving neighbors
        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=False,
                                         save_neighbor_edges=False,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        self._fixed_starting_point()

        self.assertEqual(self.partition.get_neighbors_of_block(0), {1})
        self.assertEqual(self.partition.get_neighbors_of_block(1), {0})

        # # test failing on weird objective function
        # with self.assertRaises(NotImplementedError):
        #     self.partition.precalc_move((0, 1, 0), None)

    def test_copy(self):
        """Test copying partitions"""
        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        another_partition = self.partition.copy()

        self.assertFalse(another_partition is None)

        self.assertEqual(self.partition.get_block_memberships(),
                         another_partition.get_block_memberships())

        # test if copy contains the same data
        for i in range(2):
            self.assertEqual(self.partition.get_degree_of_block(i),
                             another_partition.get_degree_of_block(i))
            self.assertEqual(self.partition.get_neighbors_of_block(i),
                             another_partition.get_neighbors_of_block(i))
            for j in range(2):
                self.assertEqual(self.partition.get_edge_count(i, j),
                                 another_partition.get_edge_count(i, j))

        # now check that not both are modified
        node, _, to_block = another_partition.get_random_move()
        another_partition.move_node(node, to_block)

        difference_found = False

        for i in range(2):
            self.assertNotEqual(self.partition.get_degree_of_block(i),
                                another_partition.get_degree_of_block(i))
            for j in range(2):
                if self.partition.get_edge_count(i, j) != another_partition.get_edge_count(i, j):
                    difference_found = True
                    # at least one edge count must change due to node move
        self.assertTrue(difference_found)

        # check change of neighbors
        another_partition.merge_blocks({1: 0}, 1)
        self.assertTrue(another_partition.get_neighbors_of_block(0) == {0})
        self.assertFalse(another_partition.get_neighbors_of_block(0) == self.partition.get_neighbors_of_block(0))

    def test_copy_neighbor_edges(self):
        """Specific test of neighboring edges because test is implementation specific"""
        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        another_partition = self.partition.copy()
        # test if copy contains the same data
        for i in range(2):
            # next line depends on implementation of NxPartition AND ListDict!
            self.assertTrue(set(self.partition.neighboring_edges[i].items)
                            ^ set(another_partition.neighboring_edges[i].items) == set([]))

        node, _, to_block = another_partition.get_random_move()
        another_partition.move_node(node, to_block)

        for i in range(2):
            self.assertFalse(set(self.partition.neighboring_edges[i].items)
                             ^ set(another_partition.neighboring_edges[i].items) == set([]))

    def test_selfloops(self):
        """Test Graph with a selfloop"""
        if self.is_weighted:
            self.graph.add_edge(1, 1, weight=1)
        else:
            self.graph.add_edge(1, 1)

        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        self._fixed_starting_point()

        # first test if the starting point is correct
        #  after this check move the node with the selfloop in another block and
        #  check again

        # Test Edge Counts
        self.assertEqual(self.partition.get_edge_count(0, 0), 2)
        self.assertEqual(self.partition.get_edge_count(1, 0), 4)
        self.assertEqual(self.partition.get_edge_count(0, 1), 4)
        self.assertEqual(self.partition.get_edge_count(1, 1), 0)

        # node count
        self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 2)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 2)

        # degree
        self.assertEqual(self.partition.get_degree_of_block(0), 6)
        self.assertEqual(self.partition.get_degree_of_block(1), 4)

        # move node with selfloop and check if the counters are right
        self.partition.move_node(1, 1)
        self.assertEqual(self.partition.get_edge_count(0, 0), 0)
        self.assertEqual(self.partition.get_edge_count(1, 0), 2)
        self.assertEqual(self.partition.get_edge_count(0, 1), 2)
        self.assertEqual(self.partition.get_edge_count(1, 1), 6)

        # node count
        self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 1)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 3)

        # degree
        self.assertEqual(self.partition.get_degree_of_block(0), 2)
        self.assertEqual(self.partition.get_degree_of_block(1), 8)

    def test_precalc_move(self):
        """Check precalculation of information for delta calculation"""

        objective_function = sbm.TraditionalUnnormalizedLogLikelyhood(self.partition.is_graph_directed())

        # get data and check
        kit, selfloops, degree = self.partition.precalc_move((1, 0, 1), objective_function)
        self.assertEqual(kit, {0: 0, 1: 2})
        self.assertEqual(selfloops, 0)
        self.assertEqual(degree, 2)

        # add more nodes
        if self.is_weighted:
            self.graph.add_edges_from([(0, 1), (1, 3), (1, 1)], weight=1)
        else:
            self.graph.add_edges_from([(0, 1), (1, 3), (1, 1)])
        self.partition = self.test_class(self.graph, self.partition.B, weighted_graph=self.is_weighted)

        # fix partition
        self.partition.set_from_representation({0: 0, 1: 0, 2: 1, 3: 0, 4: 1})

        kit, selfloops, degree = self.partition.precalc_move((1, 0, 1), objective_function)
        self.assertEqual(kit, {0: 2, 1: 2})
        self.assertEqual(selfloops, 1)
        self.assertEqual(degree, 6)

        # try error with wrong weights
        self.partition.set_from_representation({0: 1, 1: 1, 2: 0, 3: 0, 4: 0})
        self.partition.set_from_representation({0: 0, 1: 0, 2: 1, 3: 0, 4: 1})

        kit, selfloops, degree = self.partition.precalc_move((1, 0, 1), objective_function)
        self.assertEqual(kit, {0: 2, 1: 2})
        self.assertEqual(selfloops, 1)
        self.assertEqual(degree, 6)

    def test_degree_distributions(self):
        # distribution per block
        self.assertEqual(self.partition.get_degree_distribution_of_blocks(probability=False), [{2: 2}, {2: 2}])
        self.assertEqual(self.partition.get_degree_distribution_of_blocks(probability=True), [{2: 1.0}, {2: 1.0}])

        # add more nodes
        if self.is_weighted:
            self.graph.add_edges_from([(0, 1), (1, 3), (1, 1)], weight=1)
        else:
            self.graph.add_edges_from([(0, 1), (1, 3), (1, 1)])
        self.partition = self.test_class(self.graph, self.partition.B, weighted_graph=self.is_weighted)
        self.partition.set_from_representation({0: 0, 1: 0, 2: 1, 3: 0, 4: 1})

        self.assertEqual(self.partition.get_degree_distribution_of_blocks(probability=False),
                         [{1: 1, 3: 1, 6: 1}, {2: 2}])
        self.assertEqual(self.partition.get_degree_distribution_of_blocks(probability=True),
                         [{1: 1 / 3, 3: 1 / 3, 6: 1 / 3}, {2: 1.0}])

    def test_degree_iter(self):
        self.assertEqual(list(self.partition.get_degree_iter()), [2, 2, 2, 2])
        # check if iteration is possible
        for degree in self.partition.get_degree_iter():
            self.assertEqual(degree, 2)

        # add more nodes
        if self.is_weighted:
            self.graph.add_edges_from([(0, 1), (1, 3), (1, 1)], weight=1)
        else:
            self.graph.add_edges_from([(0, 1), (1, 3), (1, 1)])
        self.partition = self.test_class(self.graph, self.partition.B, weighted_graph=self.is_weighted)
        self.partition.set_from_representation({0: 0, 1: 0, 2: 1, 3: 0, 4: 1})

        # correct list is that because with node names 1:6, 2:2, 3:2, 4:2, 0:1
        self.assertEqual(list(self.partition.get_degree_iter()), [6, 2, 3, 2, 1])

    def test_matrix_representation(self):
        np.testing.assert_array_equal([[0, 1, 0, 1],
                                       [1, 0, 1, 0],
                                       [0, 1, 0, 1],
                                       [1, 0, 1, 0]],
                                      self.partition.get_graph_matrix_representation(with_weights=self.is_weighted))

    def test_creation_from_representation(self):
        new_partition = self.test_class(self.graph, fill_random=False, representation={1: 0, 2: 0, 3: 1, 4: 1})
        self.assertEqual(new_partition.get_block_of_node(1), 0)
        self.assertEqual(new_partition.get_block_of_node(2), 0)
        self.assertEqual(new_partition.get_block_of_node(3), 1)
        self.assertEqual(new_partition.get_block_of_node(4), 1)

    def test_empty_create_assign_B(self):
        """Check that B is not None for empty init"""
        new_partition = self.test_class(self.graph)
        self.assertNotEqual(new_partition.B, None)
        self.assertGreater(new_partition.B, 0)
        self.assertLessEqual(new_partition.B, len(self.graph))


class TestNxPartitionDirected(TestNxPartition):
    """ Test Class for NxPartition """

    def setUp(self):
        self.graph = nx.DiGraph()
        # circle with 10 nodes
        for i in range(10):
            self.graph.add_edge(i, (i + 1) % 10)
        self.graph.add_edge(0, 5)
        self.graph.add_edge(1, 4)
        # # test if weights get ignored
        # self.graph.add_edge(1, 2, weight=100)

        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=3,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=False,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        self._fixed_starting_point()

    def _fixed_starting_point(self):
        # fixed partition
        self.partition.set_from_representation(
            {node: node % self.partition.B for node in self.partition.get_nodes_iter()})

    def test_get_basics(self):
        """Test Partition behaving on directed graph"""

        # test edge count
        edge_count = [[1, 3, 1],
                      [0, 1, 3],
                      [3, 0, 0]]
        blocksize = [4, 3, 3]
        self.assertEqual(self.partition.B, 3)
        for i in range(self.partition.B):
            self.assertEqual(self.partition.get_number_of_nodes_in_block(i), blocksize[i])
            for j in range(self.partition.B):
                self.assertEqual(self.partition.get_edge_count(i, j), edge_count[i][j])

                #  block_degrees
        block_out_degrees = [5, 4, 3]
        block_in_degrees = [4, 4, 4]
        for block in range(self.partition.B):
            self.assertEqual(self.partition.get_out_degree_of_block(block),
                             block_out_degrees[block])
            self.assertEqual(self.partition.get_in_degree_of_block(block),
                             block_in_degrees[block])

        # neighboring edges
        self.partition.set_save_neighbor_edges(True)
        for block in range(3):
            hits = {}
            number_of_tries = 1000
            for _ in range(number_of_tries):
                edge = self.partition.get_random_neighboring_edge_of_block(block)
                hits[edge] = 1 + hits.get(edge, 0)

            # saved edges are in the from (node_of_own_block, other node)
            #  first row outgoing edges, second incoming
            if block == 0:
                six.assertCountEqual(self, hits.keys(), [(0, 1), (0, 5), (3, 4), (6, 7), (9, 0),
                                                         (3, 2), (6, 5), (9, 8), (0, 9)])
            elif block == 1:
                six.assertCountEqual(self, hits.keys(), [(1, 2), (1, 4), (4, 5), (7, 8),
                                                         (1, 0), (4, 3), (7, 6), (4, 1)])
            elif block == 2:
                six.assertCountEqual(self, hits.keys(), [(2, 3), (5, 6), (8, 9),
                                                         (2, 1), (5, 4), (8, 7), (5, 0)])
            for edge in hits:
                self.assertAlmostEqual(float(hits[edge]) / number_of_tries, 1.0 / len(hits), delta=0.1)

                # neighbor blocks
        self.assertEqual(self.partition.get_neighbors_of_block(0), set(range(3)))
        self.assertEqual(self.partition.get_neighbors_of_block(1), {0, 1, 2})
        self.assertEqual(self.partition.get_neighbors_of_block(2), {0, 1})

    def test_get_neighbors_of_node(self):
        """
        Test if in directed case all neighbors are returned, i.e.
        successors and predecessors have to be included.
        """
        # test all nodes
        number_of_nodes = self.partition.get_number_of_nodes()
        for node in range(number_of_nodes):
            # neighbors from edges forming the circle
            neighbors = [(node - 1 + number_of_nodes) % number_of_nodes,
                         (node + 1) % number_of_nodes]
            # neighbors from other edges
            if node == 0:
                neighbors.append(5)
            elif node == 5:
                neighbors.append(0)
            elif node == 1:
                neighbors.append(4)
            elif node == 4:
                neighbors.append(1)
            six.assertCountEqual(self, self.partition.get_neighbors_of_node(node), neighbors)

    def test_edge_in_both_directions(self):
        """ Test with edge in both directions """

        if self.is_weighted:
            self.graph.add_edge(1, 0, weight=1)
        else:
            self.graph.add_edge(1, 0)

        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=3,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=False,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        # fixed partition (this time ensure that every node is moved)
        self.partition.set_from_representation(
            {node: (node + 1) % self.partition.B for node in self.partition.get_nodes_iter()})

        self._fixed_starting_point()

        # test edge count
        edge_count = [[1, 3, 1],
                      [1, 1, 3],
                      [3, 0, 0]]
        blocksize = [4, 3, 3]
        self.assertEqual(self.partition.B, 3)
        for i in range(self.partition.B):
            self.assertEqual(self.partition.get_number_of_nodes_in_block(i), blocksize[i])
            for j in range(self.partition.B):
                self.assertEqual(self.partition.get_edge_count(i, j), edge_count[i][j])

                #  block_degrees
        block_out_degrees = [5, 5, 3]
        block_in_degrees = [5, 4, 4]
        for block in range(self.partition.B):
            self.assertEqual(self.partition.get_out_degree_of_block(block),
                             block_out_degrees[block])
            self.assertEqual(self.partition.get_in_degree_of_block(block),
                             block_in_degrees[block])

    def test_selfloops(self):
        """Test directed Graph with a selfloop"""
        if self.is_weighted:
            self.graph.add_edge(1, 0, weight=1)
            self.graph.add_edge(2, 2, weight=1)
        else:
            self.graph.add_edge(1, 0)
            self.graph.add_edge(2, 2)
        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=3,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=False,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        # fixed partition (this time ensure that every node is moved)
        for node in self.partition.get_nodes_iter():
            self.partition.move_node(node, (node + 1) % 3)
        for node in self.partition.get_nodes_iter():
            self.partition.move_node(node, node % 3)

        # test edge count
        edge_count = [[1, 3, 1],
                      [1, 1, 3],
                      [3, 0, 1]]
        blocksize = [4, 3, 3]
        self.assertEqual(self.partition.B, 3)
        for i in range(self.partition.B):
            self.assertEqual(self.partition.get_number_of_nodes_in_block(i), blocksize[i])
            for j in range(self.partition.B):
                self.assertEqual(self.partition.get_edge_count(i, j), edge_count[i][j])

                #  block_degrees
        block_out_degrees = [5, 5, 4]
        block_in_degrees = [5, 4, 5]
        for block in range(self.partition.B):
            self.assertEqual(self.partition.get_out_degree_of_block(block),
                             block_out_degrees[block])
            self.assertEqual(self.partition.get_in_degree_of_block(block),
                             block_in_degrees[block])

    def test_block_merges(self):
        """Test block merge and the effect on the partition (directed)"""
        # every node in a single block and all extras
        if self.is_weighted:
            self.graph.add_edge(2, 5, weight=1)
            self.graph.add_edge(2, 4, weight=1)
        else:
            self.graph.add_edge(2, 5)
            self.graph.add_edge(2, 4)

        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=4,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        self.partition.set_from_representation(
            {node: node % self.partition.B for node in range(self.partition.get_number_of_nodes())})

        self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 3)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 3)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(2), 2)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(3), 2)
        # merge two blocks
        self.partition.merge_blocks({3: 1, 2: 0}, 2)

        self.assertEqual(self.partition.B, 2)
        # get number of nodes
        self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 5)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 5)

        # Test Edge Counts
        self.assertEqual(self.partition.get_edge_count(0, 0), 1)
        self.assertEqual(self.partition.get_edge_count(1, 0), 6)
        self.assertEqual(self.partition.get_edge_count(0, 1), 7)
        self.assertEqual(self.partition.get_edge_count(1, 1), 0)

        # test read neighbors
        six.assertCountEqual(self, self.partition.get_neighbors_of_block(0), {0, 1})
        six.assertCountEqual(self, self.partition.get_neighbors_of_block(1), {0})

        # test degree of blocks
        self.assertEqual(self.partition.get_in_degree_of_block(0), 7)
        self.assertEqual(self.partition.get_in_degree_of_block(1), 7)
        self.assertEqual(self.partition.get_out_degree_of_block(0), 8)
        self.assertEqual(self.partition.get_out_degree_of_block(1), 6)

    def test_extra_calculations(self):
        """Test in/out degree saving, neighbor edge saving, neighbor blocks saving"""
        if self.is_weighted:
            self.graph.add_edge(0, 7, weight=1)
        else:
            self.graph.add_edge(0, 7)
        # test degree count
        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=False,
                                         save_neighbor_of_blocks=False,
                                         weighted_graph=self.is_weighted)

        self._fixed_starting_point()
        self.assertEqual(self.partition.get_degree_of_block(0), 7)
        self.assertEqual(self.partition.get_degree_of_block(1), 6)
        self.assertEqual(self.partition.get_in_degree_of_block(0), 6)
        self.assertEqual(self.partition.get_in_degree_of_block(1), 7)
        self.assertEqual(self.partition.get_out_degree_of_block(0), 7)
        self.assertEqual(self.partition.get_out_degree_of_block(1), 6)

        # Test saving neighbor edges
        self.partition.set_save_neighbor_edges(True)
        for block in range(2):
            hits = {}
            number_of_tries = 1000
            for _ in range(number_of_tries):
                edge = self.partition.get_random_neighboring_edge_of_block(block)
                hits[edge] = 1 + hits.get(edge, 0)

            # saved edges are in the from (node_of_own_block, other node)
            #  first two rows outgoing edges the other two incoming edges
            if block == 0:
                self.assertEqual(len(hits), 13)
                six.assertCountEqual(self, hits.keys(), [(0, 1), (2, 3), (4, 5), (6, 7),
                                                         (8, 9), (0, 5), (0, 7),
                                                         (2, 1), (4, 3), (6, 5), (8, 7),
                                                         (0, 9), (4, 1)])
            elif block == 1:
                self.assertEqual(len(hits), 13)
                six.assertCountEqual(self, hits.keys(), [(1, 2), (3, 4), (5, 6), (7, 8),
                                                         (9, 0), (1, 4),
                                                         (1, 0), (3, 2), (5, 4), (7, 6),
                                                         (9, 8), (5, 0), (7, 0)])
            for edge in hits:
                self.assertAlmostEqual(float(hits[edge]) / number_of_tries, 1.0 / len(hits), delta=0.1)
                # deletion of information
        self.partition.set_save_neighbor_edges(False)
        with self.assertRaises(Exception):
            self.partition.get_random_neighboring_edge_of_block(0)

        # test saving neighbors
        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=False,
                                         save_neighbor_edges=False,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        self._fixed_starting_point()

        self.assertEqual(self.partition.get_neighbors_of_block(0), {1})
        self.assertEqual(self.partition.get_neighbors_of_block(1), {0})

        # deprecated # test failing on weird objective function
        # with self.assertRaises(NotImplementedError):
        #     self.partition.precalc_move((0, 1, 0), None)

    def test_move_node(self):
        """Test moving blocks and representation of directed graph"""
        representation = {}
        for node in range(self.partition.get_number_of_nodes()):
            self.assertEqual(self.partition.get_block_of_node(node), node % self.partition.B)
            representation[node] = node % self.partition.B

        self.assertEqual(str(self.partition), str(representation))

        self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 4)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 3)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(2), 3)

        self.partition.move_node(0, 1)
        self.assertEqual(self.partition.get_block_of_node(1), 1)

        self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 3)
        self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 4)

    def test_precalc_move(self):
        """Precalculation on directed graph"""

        objective_function = sbm.TraditionalUnnormalizedLogLikelyhood(self.partition.is_graph_directed())

        # get data and check
        kit, kti, selfloops, in_degree, out_degree = self.partition.precalc_move((1, 1, 0), objective_function)
        self.assertEqual(kit, {0: 0, 1: 1, 2: 1})
        self.assertEqual(kti, {0: 1, 1: 0})
        self.assertEqual(selfloops, 0)
        self.assertEqual(in_degree, 1)
        self.assertEqual(out_degree, 2)

        # add more nodes
        if self.is_weighted:
            self.graph.add_edges_from([(3, 1), (1, 3), (1, 1)], weight=1)
        else:
            self.graph.add_edges_from([(3, 1), (1, 3), (1, 1)])
        self.partition = self.test_class(self.graph,
                                         self.partition.B,
                                         weighted_graph=self.is_weighted)
        self._fixed_starting_point()

        kit, kti, selfloops, in_degree, out_degree = self.partition.precalc_move((1, 1, 0), objective_function)
        self.assertEqual(kit, {0: 1, 1: 1, 2: 1})
        self.assertEqual(kti, {0: 2, 1: 0})
        self.assertEqual(selfloops, 1)
        self.assertEqual(in_degree, 3)
        self.assertEqual(out_degree, 4)

    def test_degree_distributions(self):
        # distribution per block
        # here graph consists of circle with 10 nodes and 2 shortcuts and 3 blocks!
        self.assertEqual(self.partition.get_degree_distribution_of_blocks(probability=False),
                         ([{1: 4}, {1: 2, 2: 1}, {1: 2, 2: 1}], [{1: 3, 2: 1}, {2: 1, 1: 2}, {1: 3}]))
        self.assertEqual(self.partition.get_degree_distribution_of_blocks(probability=True), (
            [{1: 1.0}, {1: 2 / 3, 2: 1 / 3}, {1: 2 / 3, 2: 1 / 3}],
            [{1: 3 / 4, 2: 1 / 4}, {2: 1 / 3, 1: 2 / 3}, {1: 1.0}]))

        # add more nodes
        if self.is_weighted:
            self.graph.add_edges_from([(10, 1), (1, 3), (1, 1)], weight=1)
        else:
            self.graph.add_edges_from([(10, 1), (1, 3), (1, 1)])
        self.partition = self.test_class(self.graph, self.partition.B, weighted_graph=self.is_weighted)
        self._fixed_starting_point()

        self.assertEqual(self.partition.get_degree_distribution_of_blocks(probability=False),
                         ([{1: 3, 2: 1}, {0: 1, 1: 1, 2: 1, 3: 1}, {1: 2, 2: 1}], [{1: 3, 2: 1}, {1: 3, 4: 1}, {1: 3}]))
        self.assertEqual(self.partition.get_degree_distribution_of_blocks(probability=True), (
            [{1: 3 / 4, 2: 1 / 4}, {0: 1 / 4, 1: 1 / 4, 2: 1 / 4, 3: 1 / 4}, {1: 2 / 3, 2: 1 / 3}],
            [{1: 3 / 4, 2: 1 / 4}, {1: 3 / 4, 4: 1 / 4}, {1: 1.0}]))

    def test_degree_iter(self):
        correct_degree = [3, 3, 2, 2, 3, 3, 2, 2, 2, 2]
        correct_in_degree = [1, 1, 1, 1, 2, 2, 1, 1, 1, 1]
        correct_out_degree = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        self.assertEqual(list(self.partition.get_degree_iter()), correct_degree)
        self.assertEqual(list(self.partition.get_in_degree_iter()), correct_in_degree)
        self.assertEqual(list(self.partition.get_out_degree_iter()), correct_out_degree)
        # check if iteration is possible
        for i, degree in enumerate(self.partition.get_degree_iter()):
            self.assertEqual(degree, correct_degree[i])
        for i, degree in enumerate(self.partition.get_in_degree_iter()):
            self.assertEqual(degree, correct_in_degree[i])
        for i, degree in enumerate(self.partition.get_out_degree_iter()):
            self.assertEqual(degree, correct_out_degree[i])

        # add more nodes
        if self.is_weighted:
            self.graph.add_edges_from([(10, 1), (1, 3), (1, 1)], weight=1)
        else:
            self.graph.add_edges_from([(10, 1), (1, 3), (1, 1)])
        self.partition = self.test_class(self.graph, self.partition.B, weighted_graph=self.is_weighted)
        self._fixed_starting_point()

        # last entry belongs to node 10!
        self.assertEqual(list(self.partition.get_degree_iter()), [3, 7, 2, 3, 3, 3, 2, 2, 2, 2, 1])
        self.assertEqual(list(self.partition.get_in_degree_iter()), [1, 3, 1, 2, 2, 2, 1, 1, 1, 1, 0])
        self.assertEqual(list(self.partition.get_out_degree_iter()), [2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_joint_degree_distribution_of_blocks(self):
        self.assertEqual(self.partition.get_joint_in_out_degree_distribution_of_blocks(),
                         [{(1, 1): 3, (1, 2): 1}, {(1, 1): 1, (1, 2): 1, (2, 1): 1}, {(1, 1): 2, (2, 1): 1}])
        # total {(1, 2): 2, (1, 1): 6, (2, 1): 2}
        # add more nodes
        if self.is_weighted:
            self.graph.add_edges_from([(10, 1), (1, 3), (1, 1)], weight=1)
        else:
            self.graph.add_edges_from([(10, 1), (1, 3), (1, 1)])
        self.partition = self.test_class(self.graph, self.partition.B, weighted_graph=self.is_weighted)
        self._fixed_starting_point()

        self.assertEqual(self.partition.get_joint_in_out_degree_distribution_of_blocks(),
                         [{(1, 1): 2, (1, 2): 1, (2, 1): 1}, {(0, 1): 1, (1, 1): 1, (2, 1): 1, (3, 4): 1},
                          {(1, 1): 2, (2, 1): 1}])
        # total {(1, 2): 1, (1, 1): 5, (2, 1): 3, (3, 4): 1}

    def test_matrix_representation(self):
        np.testing.assert_array_equal([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                      self.partition.get_graph_matrix_representation(with_weights=self.is_weighted))

    def test_creation_from_representation(self):
        new_partition = self.test_class(self.graph, fill_random=False,
                                        representation={node: node % 2 for node in self.graph})
        for node in self.graph:
            self.assertEqual(new_partition.get_block_of_node(node), node % 2)

    def test_get_possible_blocks(self):
        """Check possible blocks - default all"""
        # in a flat partition (no hierarchy) all blocks are possible
        for block in range(self.partition.B):
            self.assertEqual(self.partition.get_possible_blocks(block), range(self.partition.B))


class TestNxPartitionMultiWeighted(TestNxPartition):
    """ Test Class for NxPartition on weighted undirected multigraphs """

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionMultiWeighted, self).__init__(methodName)
        self.is_weighted = True

    def setUp(self):
        self.graph = nx.MultiGraph()
        # lazy version with multiple edges and weights
        self.graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)], weight=.5)
        self.graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)], weight=.5)
        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=False,
                                         save_neighbor_edges=False,
                                         save_neighbor_of_blocks=False,
                                         weighted_graph=self.is_weighted)
        self._fixed_starting_point()

    def test_weighted_edge_sampling(self):
        """Check if weighted edge are correctly sampled according to their weight"""
        self.graph.add_edge(1, 2, weight=5)

        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=2,
                                         calculate_degree_of_blocks=False,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=False,
                                         weighted_graph=self.is_weighted)
        self._fixed_starting_point()

        hits = {}
        number_of_tries = 1000
        for _ in range(number_of_tries):
            edge = self.partition.get_random_neighboring_edge_of_block(0)
            hits[edge] = hits.get(edge, 0) + 1

        # edge (1, 2) has a total weight of 6 the other 3 only each a weight of 1
        # the probability should be around weight/(sum of all weights = 9)
        self.assertAlmostEqual((hits[(1, 2)] * 1.0) / number_of_tries, 6.0 / 9, delta=.1)
        self.assertAlmostEqual((hits[(3, 2)] * 1.0) / number_of_tries, 1.0 / 9, delta=.1)
        self.assertAlmostEqual((hits[(1, 4)] * 1.0) / number_of_tries, 1.0 / 9, delta=.1)
        self.assertAlmostEqual((hits[(3, 4)] * 1.0) / number_of_tries, 1.0 / 9, delta=.1)


class TestNxPartitionDirectedMultiWeighted(TestNxPartitionDirected):
    """ Test Class for NxPartition """

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionDirectedMultiWeighted, self).__init__(methodName)
        self.is_weighted = True

    def setUp(self):
        self.graph = nx.MultiDiGraph()
        # circle with 10 nodes
        for i in range(10):
            self.graph.add_edge(i, (i + 1) % 10, weight=.5)
            self.graph.add_edge(i, (i + 1) % 10, weight=.5)
        self.graph.add_edge(0, 5, weight=.5)
        self.graph.add_edge(1, 4, weight=.5)
        self.graph.add_edge(0, 5, weight=.5)
        self.graph.add_edge(1, 4, weight=.5)

        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=3,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=False,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        self._fixed_starting_point()

    def test_weighted_edge_sampling(self):
        """Check if weighted edge are correctly sampled according to their weight"""

        self.graph.add_edge(0, 7, weight=5)

        self.partition = self.test_class(graph=self.graph,
                                         number_of_blocks=3,
                                         calculate_degree_of_blocks=True,
                                         save_neighbor_edges=True,
                                         save_neighbor_of_blocks=True,
                                         weighted_graph=self.is_weighted)

        self._fixed_starting_point()

        hits = {}
        number_of_tries = 1000
        for _ in range(number_of_tries):
            edge = self.partition.get_random_neighboring_edge_of_block(0)
            hits[edge] = hits.get(edge, 0) + 1

        # check if edge are sampled with the probability given by their weight
        #  (edge 0->7 is the only edge with weight 5 others weight 1)
        self.assertEqual(len(hits), 10)
        for edge in hits:
            if edge != (0, 7):
                self.assertAlmostEqual((hits[edge] * 1.0) / number_of_tries, 1.0 / (len(hits) + 4),
                                       delta=.1)
            else:
                self.assertAlmostEqual((hits[edge] * 1.0) / number_of_tries, 5.0 / (len(hits) + 4),
                                       delta=.1)


class TestNxPartitionWithMoveCounter(ut.TestCase):

    def test_basics(self):
        # only creating and a little moving
        graph = nx.karate_club_graph()

        representation = {node: node % 2 for node in graph}

        partition = sbm.NxPartitionWithMoveCounter(graph=graph, representation=representation)

        self.assertEqual(representation, partition.get_representation())
        self.assertEqual(0, partition.node_moves)

        # test counting of moves
        partition.move_node(0, 1)
        self.assertEqual(1, partition.node_moves)
        partition.move_node(0, 0)
        self.assertEqual(2, partition.node_moves)
        # null moves should not increase counter
        partition.move_node(0, 0)
        self.assertEqual(2, partition.node_moves)

        self.assertEqual(representation, partition.get_representation())

        with self.assertRaises(NotImplementedError):
            partition.get_sum_of_covariates(0, 1)
