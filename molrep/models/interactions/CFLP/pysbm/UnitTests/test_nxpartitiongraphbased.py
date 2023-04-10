import unittest as ut

import networkx as nx
import numpy as np
import six

import pysbm.UnitTests.test_partition as test_sbmpartition
from pysbm.sbm.nxpartitiongraphbased import NxHierarchicalPartition
from pysbm.sbm.nxpartitiongraphbased import NxPartitionGraphBased
from pysbm.sbm.nxpartitiongraphbased import NxPartitionGraphBasedHierarchy
from pysbm.sbm.objective_functions import TraditionalUnnormalizedLogLikelyhood
from pysbm.sbm.exceptions import NoFreeNodeException


# function which are later methods of the classes
def graph_based_test_copy_neighbor_edges(self):
    """Specific test of neighboring edges because test is implementation specific"""
    self.partition = self.test_class(graph=self.graph,
                                     number_of_blocks=2,
                                     calculate_degree_of_blocks=True,
                                     save_neighbor_edges=True,
                                     save_neighbor_of_blocks=True,
                                     weighted_graph=self.is_weighted)

    another_partition = self.partition.copy()
    self.assertTrue(general_test_saved_edges(self, self.partition, another_partition))

    node, _, to_block = another_partition.get_random_move()
    another_partition.move_node(node, to_block)

    self.assertFalse(general_test_saved_edges(self, self.partition, another_partition))


def general_test_saved_edges(self, first_partition, second_partition):
    # test if copy contains the same data
    for i in range(2):
        # next line depends on implementation of NxPartitionGraphBased AND ListDict!
        if set(first_partition.get_partition_as_a_graph().nodes[i][self.partition._NEIGHBORING_EDGES_A].items) \
                ^ set(second_partition.get_partition_as_a_graph().nodes[i][
                          second_partition._NEIGHBORING_EDGES_A].items) == set([]):
            return True
    else:
        return False


def undirected_test_move_node_with_return_values(self):
    """Test correct returns of move node (undirected)"""
    node = 1
    new_block = 2

    # check no return on normal move
    return_values = self.partition.move_node(node, self.partition.get_block_of_node(node))
    self.assertTrue(return_values is None)

    return_values = self.partition.move_node(node, 1)
    self.assertTrue(return_values is None)

    # move node back to start point
    self.partition.move_node(node, 0)

    # check return values for new block
    return_values = self.partition.move_node(node, new_block)
    self.assertTrue(return_values is not None)
    self.assertEqual(len(return_values), 5)
    # is new block? = True
    self.assertTrue(return_values[0])
    # label of old block
    self.assertEqual(return_values[1], 0)
    # label of newly created block
    self.assertEqual(return_values[2], 2)
    if self.is_weighted:
        # old edges on block basis (for edge (1,2) and (4,1) double with weight .5)
        self.assertEqual(return_values[3], [(0, 1, .5), (0, 1, .5), (0, 1, .5), (0, 1, .5)])
        self.assertEqual(return_values[4], [(2, 1, .5), (2, 1, .5), (2, 1, .5), (2, 1, .5)])
    else:
        # old edges on block basis (for edge (1,2) and (4,1) no weight -> all weight 1)
        self.assertEqual(return_values[3], [(0, 1, 1), (0, 1, 1)])
        self.assertEqual(return_values[4], [(2, 1, 1), (2, 1, 1)])

    # delete block 0 (not last block, have to relabel things)
    return_values = self.partition.move_node(3, 1)
    # now length 5 (include relabeled edges)
    self.assertEqual(len(return_values), 6)
    # is new block?
    self.assertFalse(return_values[0])
    # label of old block
    self.assertEqual(return_values[1], 0)
    if self.is_weighted:
        # old edges on block basis (for edge (2,3) and (3,4) double with weight .5)
        self.assertEqual(return_values[2], [(0, 1, .5), (0, 1, .5), (0, 1, .5), (0, 1, .5)])
        self.assertEqual(return_values[3], [(1, 1, .5), (1, 1, .5), (1, 1, .5), (1, 1, .5)])
    else:
        # old edges on block basis (for edge (2,3) and (3,4) no weight -> all weight 1)
        self.assertEqual(return_values[2], [(0, 1, 1), (0, 1, 1)])
        self.assertEqual(return_values[3], [(1, 1, 1), (1, 1, 1)])
    # old relabeled edges (for edge (1,2) and (4,1) no weight -> total weight 2)
    self.assertEqual(return_values[4], [(2, 1, 2)])
    # new relabeled edges (for edge (1,2) and (4,1) no weight -> total weight 2)
    self.assertEqual(return_values[5], [(0, 1, 2)])

    # now delete block 1 (now the last block)
    return_values = self.partition.move_node(3, 0)
    self.assertTrue(return_values is None)
    return_values = self.partition.move_node(2, 0)
    self.assertTrue(return_values is None)

    return_values = self.partition.move_node(4, 0)
    # now length 5 (include relabeled edges)
    self.assertEqual(len(return_values), 6)
    # is new block?
    self.assertFalse(return_values[0])
    # label of old block
    self.assertEqual(return_values[1], 1)
    if self.is_weighted:
        # old edges on block basis (for edge (4,1) and (3,4) double with weight .5)
        self.assertEqual(return_values[2], [(1, 0, .5), (1, 0, .5), (1, 0, .5), (1, 0, .5)])
        self.assertEqual(return_values[3], [(0, 0, .5), (0, 0, .5), (0, 0, .5), (0, 0, .5)])
    else:
        # old edges on block basis (for edge (4,1) and (3,4) no weight -> all weight 1)
        self.assertEqual(return_values[2], [(1, 0, 1), (1, 0, 1)])
        self.assertEqual(return_values[3], [(0, 0, 1), (0, 0, 1)])
    # no relabeled edges because was last block
    self.assertEqual(return_values[4], None)
    self.assertEqual(return_values[5], None)


def undirected_test_split_and_merge(self):
    """Test spliting and merging of nodes/blocks (undirected)"""
    self.partition.set_from_representation({1: 0, 2: 0, 3: 1, 4: 2})
    # create partition of blocks
    partition = self.test_class(self.partition.get_partition_as_a_graph(),
                                number_of_blocks=2,
                                save_neighbor_edges=True,
                                weighted_graph=True)
    partition.set_from_representation({0: 0, 1: 0, 2: 1})

    # check error raising of split node if new node is other
    with self.assertRaises(ValueError):
        partition.split_node(0, 100, [], [])

    # check starting edge count
    self.assertEqual(partition.get_edge_count(0, 0), 4)
    self.assertEqual(partition.get_edge_count(0, 1), 2)
    self.assertEqual(partition.get_edge_count(1, 1), 0)

    # check neighboring edge before
    edges = {}
    number_of_tries = 200
    correct_entries = {(0, 0), (0, 1), (1, 2), (1, 0), (0, 2)}
    for _ in range(number_of_tries):
        edge = partition.get_random_neighboring_edge_of_block(0)
        edges[edge] = edges.get(edge, 0) + 1
    self.assertEqual(set(edges.keys()), correct_entries)
    # check number of hits
    # one edge count twice all other simple
    self.assertAlmostEqual(edges[(0, 0)] / number_of_tries, 2 / (len(correct_entries) + 1), delta=.1)
    del edges[(0, 0)]
    for edge in edges:
        self.assertAlmostEqual(edges[edge] / number_of_tries, 1 / (len(correct_entries) + 1), delta=.1)

    # -----------------------
    # perform move
    return_values = self.partition.move_node(1, 3)
    partition.split_node(*return_values[1:])

    # test block membership of new node
    self.assertEqual(partition.get_block_of_node(3), 0)
    self.assertEqual(partition.get_number_of_nodes(), 4)

    # check edge count afterwards
    self.assertEqual(partition.get_edge_count(0, 0), 4)
    self.assertEqual(partition.get_edge_count(0, 1), 2)
    self.assertEqual(partition.get_edge_count(1, 1), 0)

    # check neighboring edge
    edges = {}
    number_of_tries = 200
    correct_entries = {(0, 3), (0, 1), (1, 2), (1, 0), (3, 2), (3, 0)}
    for _ in range(number_of_tries):
        edge = partition.get_random_neighboring_edge_of_block(0)
        edges[edge] = edges.get(edge, 0) + 1
    self.assertEqual(set(edges.keys()), correct_entries)
    # check number of hits
    for edge in edges:
        self.assertAlmostEqual(edges[edge] / number_of_tries, 1 / len(correct_entries), delta=.1)

    # 2 ------
    # perform move
    return_values = self.partition.move_node(1, 0)
    partition.merge_node(*return_values[1:])

    # check if node was deleted
    self.assertEqual(partition.get_number_of_nodes(), 3)

    # check edge count afterwards
    self.assertEqual(partition.get_edge_count(0, 0), 4)
    self.assertEqual(partition.get_edge_count(0, 1), 2)
    self.assertEqual(partition.get_edge_count(1, 1), 0)

    # check neighboring edge after
    edges = {}
    number_of_tries = 200
    correct_entries = {(0, 0), (0, 1), (1, 2), (1, 0), (0, 2)}
    for _ in range(number_of_tries):
        edge = partition.get_random_neighboring_edge_of_block(0)
        edges[edge] = edges.get(edge, 0) + 1
    self.assertEqual(set(edges.keys()), correct_entries)
    # check number of hits
    # one edge count twice all other simple
    self.assertAlmostEqual(edges[(0, 0)] / number_of_tries, 2 / (len(correct_entries) + 1), delta=.1)
    del edges[(0, 0)]
    for edge in edges:
        self.assertAlmostEqual(edges[edge] / number_of_tries, 1 / (len(correct_entries) + 1), delta=.1)

    # --- test remove of not last block
    # simple perform a null move by in fact relabel block 0
    return_values = self.partition.move_node(1, 3)
    partition.split_node(*return_values[1:])
    return_values = self.partition.move_node(2, 3)
    partition.merge_node(*return_values[1:])

    # check if node was deleted
    self.assertEqual(partition.get_number_of_nodes(), 3)

    # check edge count afterwards
    self.assertEqual(partition.get_edge_count(0, 0), 4)
    self.assertEqual(partition.get_edge_count(0, 1), 2)
    self.assertEqual(partition.get_edge_count(1, 1), 0)

    # check neighboring edge after
    edges = {}
    number_of_tries = 200
    correct_entries = {(0, 0), (0, 1), (1, 2), (1, 0), (0, 2)}
    for _ in range(number_of_tries):
        edge = partition.get_random_neighboring_edge_of_block(0)
        edges[edge] = edges.get(edge, 0) + 1
    self.assertEqual(set(edges.keys()), correct_entries)
    # check number of hits
    # one edge count twice all other simple
    self.assertAlmostEqual(edges[(0, 0)] / number_of_tries, 2 / (len(correct_entries) + 1), delta=.1)
    del edges[(0, 0)]
    for edge in edges:
        self.assertAlmostEqual(edges[edge] / number_of_tries, 1 / (len(correct_entries) + 1), delta=.1)


def undirected_test_move_node_with_block_number_change(self):
    # switch everything on
    partition = self.test_class(graph=self.graph,
                                number_of_blocks=2,
                                calculate_degree_of_blocks=True,
                                save_neighbor_edges=True,
                                save_neighbor_of_blocks=True,
                                weighted_graph=self.is_weighted)
    partition.set_from_representation({1: 0, 2: 1, 3: 0, 4: 1})

    node = 1
    old_number_of_blocks = self.partition.B
    self.partition.move_node(node, old_number_of_blocks)
    self.assertEqual(self.partition.B, old_number_of_blocks + 1)
    self.assertEqual(self.partition.get_block_of_node(node), old_number_of_blocks)

    # test edge count
    self.assertEqual(self.partition.get_edge_count(old_number_of_blocks, 1), 2)
    self.assertEqual(self.partition.get_edge_count(1, old_number_of_blocks), 2)
    self.assertEqual(self.partition.get_edge_count(old_number_of_blocks, 0), 0)
    self.assertEqual(self.partition.get_edge_count(0, old_number_of_blocks), 0)
    self.assertEqual(self.partition.get_edge_count(old_number_of_blocks, old_number_of_blocks), 0)

    # test node count
    self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 1)
    self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 2)
    self.assertEqual(self.partition.get_number_of_nodes_in_block(2), 1)

    # check neighbors
    partition.move_node(node, old_number_of_blocks)
    self.assertEqual(partition.get_neighbors_of_block(old_number_of_blocks), {1})
    self.assertEqual(partition.get_neighbors_of_block(1), {0, 2})
    self.assertEqual(partition.get_neighbors_of_block(0), {1})

    # check neighboring edge
    edges = set()
    for _ in range(20):
        edges.add(partition.get_random_neighboring_edge_of_block(2))
    self.assertEqual(edges, {(1, 4), (1, 2)})

    # test null move
    self.partition.move_node(node, old_number_of_blocks + 1)
    self.assertEqual(self.partition.B, old_number_of_blocks + 1)
    self.assertEqual(self.partition.get_block_of_node(node), old_number_of_blocks)

    # delete new block (last)
    self.partition.move_node(node, old_number_of_blocks - 1)
    self.assertEqual(self.partition.B, old_number_of_blocks)
    self.assertEqual(self.partition.get_block_of_node(node), old_number_of_blocks - 1)
    # node counts
    self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 1)
    self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 3)

    # delete block (not last) (now all in one block)
    self.partition.move_node(3, old_number_of_blocks - 1)
    self.assertEqual(self.partition.B, 1)


def undirected_test_move_node_edge_changes(self):
    """Test moving nodes without creation or remove of blocks"""
    # create bigger graph with selfloops
    if self.is_weighted:
        graph = nx.MultiGraph()
        graph.add_edges_from([(node, (node + 1) % 10) for node in range(10)], weight=.5)
        graph.add_edge(0, 0, weight=.5)
        graph.add_edge(1, 1, weight=.5)
        graph.add_edges_from([(node, (node + 1) % 10) for node in range(10)], weight=.5)
        graph.add_edge(0, 0, weight=.5)
        graph.add_edge(1, 1, weight=.5)
    else:
        graph = nx.Graph()
        graph.add_edges_from([(node, (node + 1) % 10) for node in range(10)])
        graph.add_edge(0, 0)
        graph.add_edge(1, 1)
    # # test if weights get ignored
    # self.graph.add_edge(1,2, weight=100)
    partition = self.test_class(graph=graph,
                                number_of_blocks=2,
                                calculate_degree_of_blocks=True,
                                save_neighbor_edges=True,
                                save_neighbor_of_blocks=True,
                                weighted_graph=self.is_weighted)
    partition.set_from_representation({node: node % 2 for node in graph})

    return_values = partition.move_node(2, 1, return_edge_changes=True)
    self.assertEqual(len(return_values), 2)
    if self.is_weighted:
        six.assertCountEqual(self, return_values[0], [(0, 1, .5), (0, 1, .5), (0, 1, .5), (0, 1, .5)])
        six.assertCountEqual(self, return_values[1], [(1, 1, .5), (1, 1, .5), (1, 1, .5), (1, 1, .5)])
    else:
        six.assertCountEqual(self, return_values[0], [(0, 1, 1), (0, 1, 1)])
        six.assertCountEqual(self, return_values[1], [(1, 1, 1), (1, 1, 1)])

    return_values = partition.move_node(1, 1, return_edge_changes=True)
    self.assertEqual(return_values, None)
    # check correct handling
    return_values = partition.move_node(1, 0, return_edge_changes=True)
    self.assertEqual(len(return_values), 2)
    if self.is_weighted:
        six.assertCountEqual(self, return_values[0],
                             [(1, 0, .5), (1, 1, .5), (1, 1, .5), (1, 0, .5), (1, 1, .5), (1, 1, .5)])
        six.assertCountEqual(self, return_values[1],
                             [(0, 0, .5), (0, 1, .5), (0, 0, .5), (0, 0, .5), (0, 1, .5), (0, 0, .5)])
    else:
        six.assertCountEqual(self, return_values[0], [(1, 0, 1), (1, 1, 1), (1, 1, 1)])
        six.assertCountEqual(self, return_values[1], [(0, 0, 1), (0, 1, 1), (0, 0, 1)])


def directed_test_move_node_with_return_values(self):
    """Test correct returns of move node (directed)"""
    node = 0
    new_block = 3

    # check no return on normal move
    return_values = self.partition.move_node(node, self.partition.get_block_of_node(node))
    self.assertTrue(return_values is None)

    return_values = self.partition.move_node(node, 1)
    self.assertTrue(return_values is None)
    return_values = self.partition.move_node(node, 2)
    self.assertTrue(return_values is None)

    # move node back to start point
    self.partition.move_node(node, 0)

    # check return values for new block
    return_values = self.partition.move_node(node, new_block)
    self.assertTrue(return_values is not None)
    self.assertEqual(len(return_values), 5)
    # is new block? = True
    self.assertTrue(return_values[0])
    # label of old block
    self.assertEqual(return_values[1], 0)
    # label of newly created block
    self.assertEqual(return_values[2], 3)
    if self.is_weighted:
        # old edges on block basis (for edge (0,1), (0, 5) and (9, 0) double with weight .5)
        six.assertCountEqual(self, return_values[3],
                             [(0, 1, .5), (0, 2, .5), (0, 0, .5), (0, 1, .5), (0, 2, .5), (0, 0, .5)])
        six.assertCountEqual(self, return_values[4],
                             [(3, 1, .5), (3, 2, .5), (0, 3, .5), (3, 1, .5), (3, 2, .5), (0, 3, .5)])
    else:
        # old edges on block basis (for edge (0,1), (0, 5) and (9, 0) no weight -> all weight 1)
        self.assertEqual(return_values[3], [(0, 1, 1), (0, 2, 1), (0, 0, 1)])
        self.assertEqual(return_values[4], [(3, 1, 1), (3, 2, 1), (0, 3, 1)])

    # delete block 0 (not last block, have to relabel things)
    return_values = self.partition.move_node(3, 1)
    self.assertTrue(return_values is None)
    return_values = self.partition.move_node(6, 1)
    self.assertTrue(return_values is None)
    return_values = self.partition.move_node(9, 1)
    # now length 5 (include relabeled edges)
    self.assertEqual(len(return_values), 6)
    # is new block?
    self.assertFalse(return_values[0])
    # label of old block
    self.assertEqual(return_values[1], 0)
    if self.is_weighted:
        # old edges on block basis (for edge (9,0) and (8,9) double with weight .5)
        six.assertCountEqual(self, return_values[2], [(0, 3, .5), (2, 0, .5), (0, 3, .5), (2, 0, .5)])
        six.assertCountEqual(self, return_values[3], [(1, 3, .5), (2, 1, .5), (1, 3, .5), (2, 1, .5)])
    else:
        # old edges on block basis (for edge (9,0) and (8,9) no weight -> all weight 1)
        self.assertEqual(return_values[2], [(0, 3, 1), (2, 0, 1)])
        self.assertEqual(return_values[3], [(1, 3, 1), (2, 1, 1)])
    # old relabeled edges (for edge (0,1) (1 block 1), (0, 5) (5 block 2) and (9, 0) (9 block 1) no weight
    # (or sumed weight 1)
    self.assertEqual(return_values[4], [(3, 1, 1), (3, 2, 1), (1, 3, 1)])
    # new relabeled edges
    self.assertEqual(return_values[5], [(0, 1, 1), (0, 2, 1), (1, 0, 1)])

    # now delete block 2 (now the last block)
    return_values = self.partition.move_node(2, 0)
    self.assertTrue(return_values is None)
    return_values = self.partition.move_node(5, 0)
    self.assertTrue(return_values is None)

    return_values = self.partition.move_node(8, 0)
    # now length 5 (include relabeled edges)
    self.assertEqual(len(return_values), 6)
    # is new block?
    self.assertFalse(return_values[0])
    # label of old block
    self.assertEqual(return_values[1], 2)
    if self.is_weighted:
        # old edges on block basis (for edge (4,1) and (3,4) double with weight .5)
        six.assertCountEqual(self, return_values[2], [(2, 1, .5), (1, 2, .5), (2, 1, .5), (1, 2, .5)])
        six.assertCountEqual(self, return_values[3], [(0, 1, .5), (1, 0, .5), (0, 1, .5), (1, 0, .5)])
    else:
        # old edges on block basis (for edge (8,9) and (7,8) no weight -> all weight 1)
        self.assertEqual(return_values[2], [(2, 1, 1), (1, 2, 1)])
        self.assertEqual(return_values[3], [(0, 1, 1), (1, 0, 1)])
    # no relabeled edges because was last block
    self.assertEqual(return_values[4], None)
    self.assertEqual(return_values[5], None)


def directed_test_split_and_merge(self):
    """Test spliting and merging of nodes/blocks (directed)"""
    self.partition.set_from_representation({0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 3, 7: 4, 8: 5, 9: 5})
    # create partition of blocks
    partition = self.test_class(self.partition.get_partition_as_a_graph(),
                                number_of_blocks=4,
                                save_neighbor_edges=True,
                                weighted_graph=True)
    partition.set_from_representation({0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3})

    # check error raising of split node if new node is other
    with self.assertRaises(ValueError):
        partition.split_node(0, 100, [], [])

    # check starting edge count
    correct_edges = [[3, 2, 1, 0],
                     [0, 0, 1, 0],
                     [0, 0, 1, 1],
                     [1, 0, 0, 2]]
    for i in range(4):
        for j in range(4):
            self.assertEqual(partition.get_edge_count(i, j), correct_edges[i][j],
                             "Wrong edge count between group" + str((i, j)))

    # check neighboring edge before
    edges = {}
    number_of_tries = 200
    correct_entries = {(0, 0), (0, 1), (1, 2), (1, 0), (0, 2), (0, 3), (0, 5)}
    for _ in range(number_of_tries):
        edge = partition.get_random_neighboring_edge_of_block(0)
        edges[edge] = edges.get(edge, 0) + 1
    self.assertEqual(set(edges.keys()), correct_entries)
    # check number of hits
    # one edge count twice all other simple
    self.assertAlmostEqual(edges[(0, 0)] / number_of_tries, 4 / (len(correct_entries) + 3), delta=.1)
    del edges[(0, 0)]
    for edge in edges:
        self.assertAlmostEqual(edges[edge] / number_of_tries, 1 / (len(correct_entries) + 3), delta=.1)

    # -----------------------
    # perform move
    return_values = self.partition.move_node(1, 6)
    partition.split_node(*return_values[1:])

    # test block membership of new node
    self.assertEqual(partition.get_block_of_node(6), 0)
    self.assertEqual(partition.get_number_of_nodes(), 7)

    # check edge count afterwards
    for i in range(4):
        for j in range(4):
            self.assertEqual(partition.get_edge_count(i, j), correct_edges[i][j],
                             "Wrong edge count between group" + str((i, j)))

    # check neighboring edge
    edges = {}
    number_of_tries = 200
    correct_entries = {(0, 3), (0, 1), (1, 2), (1, 0), (6, 2), (6, 0), (0, 6), (0, 5)}
    for _ in range(number_of_tries):
        edge = partition.get_random_neighboring_edge_of_block(0)
        edges[edge] = edges.get(edge, 0) + 1
    self.assertEqual(set(edges.keys()), correct_entries)
    # those edges are twice their (1,2) + (1,4)
    self.assertAlmostEqual(edges[(6, 0)] / number_of_tries, 2 / (len(correct_entries) + 2), delta=.1)
    self.assertAlmostEqual(edges[(0, 6)] / number_of_tries, 2 / (len(correct_entries) + 2), delta=.1)
    del edges[(6, 0)]
    del edges[(0, 6)]
    # check number of hits
    for edge in edges:
        self.assertAlmostEqual(edges[edge] / number_of_tries, 1 / (len(correct_entries) + 2), delta=.1)

    # 2 ------
    # perform move
    return_values = self.partition.move_node(1, 0)
    partition.merge_node(*return_values[1:])

    # check if node was deleted
    self.assertEqual(partition.get_number_of_nodes(), 6)

    # check edge count afterwards
    # check edge count afterwards
    for i in range(4):
        for j in range(4):
            self.assertEqual(partition.get_edge_count(i, j), correct_edges[i][j],
                             "Wrong edge count between group" + str((i, j)))

    # check neighboring edge after
    edges = {}
    number_of_tries = 200
    correct_entries = {(0, 0), (0, 1), (1, 2), (1, 0), (0, 2), (0, 3), (0, 5)}
    for _ in range(number_of_tries):
        edge = partition.get_random_neighboring_edge_of_block(0)
        edges[edge] = edges.get(edge, 0) + 1
    self.assertEqual(set(edges.keys()), correct_entries)
    # check number of hits
    # one edge count twice all other simple
    self.assertAlmostEqual(edges[(0, 0)] / number_of_tries, 4 / (len(correct_entries) + 3), delta=.1)
    del edges[(0, 0)]
    for edge in edges:
        self.assertAlmostEqual(edges[edge] / number_of_tries, 1 / (len(correct_entries) + 3), delta=.1)

    # --- test remove of not last block
    # simple perform a null move by in fact relabel block 0
    # perform moves
    return_values = self.partition.move_node(1, 6)
    partition.split_node(*return_values[1:])
    self.assertEqual(partition.get_block_of_node(6), 0)
    return_values = self.partition.move_node(0, 6, return_edge_changes=True)
    partition.change_saved_edges(*return_values)
    return_values = self.partition.move_node(2, 6)
    partition.merge_node(*return_values[1:])

    # same values as before

    # check if node was deleted
    self.assertEqual(partition.get_number_of_nodes(), 6)

    # check edge count afterwards
    # check edge count afterwards
    for i in range(4):
        for j in range(4):
            self.assertEqual(partition.get_edge_count(i, j), correct_edges[i][j],
                             "Wrong edge count between group" + str((i, j)))

    # check neighboring edge after
    edges = {}
    number_of_tries = 200
    correct_entries = {(0, 0), (0, 1), (1, 2), (1, 0), (0, 2), (0, 3), (0, 5)}
    for _ in range(number_of_tries):
        edge = partition.get_random_neighboring_edge_of_block(0)
        edges[edge] = edges.get(edge, 0) + 1
    self.assertEqual(set(edges.keys()), correct_entries)
    # check number of hits
    # one edge count twice all other simple
    self.assertAlmostEqual(edges[(0, 0)] / number_of_tries, 4 / (len(correct_entries) + 3), delta=.1)
    del edges[(0, 0)]
    for edge in edges:
        self.assertAlmostEqual(edges[edge] / number_of_tries, 1 / (len(correct_entries) + 3), delta=.1)


def directed_test_move_node_with_block_number_change(self):
    # switch everything on
    partition = self.test_class(graph=self.graph,
                                number_of_blocks=3,
                                calculate_degree_of_blocks=True,
                                save_neighbor_edges=True,
                                save_neighbor_of_blocks=True,
                                weighted_graph=self.is_weighted)
    partition.set_from_representation({node: node % partition.B for node in self.partition.get_nodes_iter()})

    node = 2
    old_number_of_blocks = self.partition.B
    self.partition.move_node(node, old_number_of_blocks)
    self.assertEqual(self.partition.B, old_number_of_blocks + 1)
    self.assertEqual(self.partition.get_block_of_node(node), old_number_of_blocks)

    # test edge count
    self.assertEqual(self.partition.get_edge_count(old_number_of_blocks, 2), 0)
    self.assertEqual(self.partition.get_edge_count(2, old_number_of_blocks), 0)
    self.assertEqual(self.partition.get_edge_count(old_number_of_blocks, 1), 0)
    self.assertEqual(self.partition.get_edge_count(1, old_number_of_blocks), 1)
    self.assertEqual(self.partition.get_edge_count(old_number_of_blocks, 0), 1)
    self.assertEqual(self.partition.get_edge_count(0, old_number_of_blocks), 0)
    self.assertEqual(self.partition.get_edge_count(old_number_of_blocks, old_number_of_blocks), 0)

    # test node count
    self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 4)
    self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 3)
    self.assertEqual(self.partition.get_number_of_nodes_in_block(2), 2)
    self.assertEqual(self.partition.get_number_of_nodes_in_block(3), 1)

    # check neighbors
    partition.move_node(node, old_number_of_blocks)
    self.assertEqual(partition.get_neighbors_of_block(old_number_of_blocks), {0, 1})
    self.assertEqual(partition.get_neighbors_of_block(2), {0, 1})
    self.assertEqual(partition.get_neighbors_of_block(1), {0, 1, 2, 3})
    self.assertEqual(partition.get_neighbors_of_block(0), {0, 1, 2, 3})

    # check neighboring edge
    edges = set()
    for _ in range(20):
        edges.add(partition.get_random_neighboring_edge_of_block(3))
    self.assertEqual(edges, {(2, 1), (2, 3)})

    # test null move
    self.partition.move_node(node, old_number_of_blocks + 1)
    self.assertEqual(self.partition.B, old_number_of_blocks + 1)
    self.assertEqual(self.partition.get_block_of_node(node), old_number_of_blocks)

    # delete new block (last)
    self.partition.move_node(node, old_number_of_blocks - 1)
    self.assertEqual(self.partition.B, old_number_of_blocks)
    self.assertEqual(self.partition.get_block_of_node(node), old_number_of_blocks - 1)
    # node counts
    self.assertEqual(self.partition.get_number_of_nodes_in_block(0), 4)
    self.assertEqual(self.partition.get_number_of_nodes_in_block(1), 3)
    self.assertEqual(self.partition.get_number_of_nodes_in_block(2), 3)

    # delete block (not last)
    self.partition.move_node(1, 0)
    self.partition.move_node(4, 0)
    self.partition.move_node(7, 0)
    self.assertEqual(self.partition.B, 2)


def directed_test_move_node_edge_changes(self):
    """Test moving nodes without creation or remove of blocks"""
    # create bigger graph with selfloops
    if self.is_weighted:
        graph = nx.MultiDiGraph()
        graph.add_edges_from([(node, (node + 1) % 10) for node in range(10)], weight=.5)
        graph.add_edge(0, 0, weight=.5)
        graph.add_edge(1, 1, weight=.5)
        graph.add_edges_from([(node, (node + 1) % 10) for node in range(10)], weight=.5)
        graph.add_edge(0, 0, weight=.5)
        graph.add_edge(1, 1, weight=.5)
    else:
        graph = nx.DiGraph()
        graph.add_edges_from([(node, (node + 1) % 10) for node in range(10)])
        graph.add_edge(0, 0)
        graph.add_edge(1, 1)
    # # test if weights get ignored
    # self.graph.add_edge(1,2, weight=100)
    partition = self.test_class(graph=graph,
                                number_of_blocks=2,
                                calculate_degree_of_blocks=True,
                                save_neighbor_edges=True,
                                save_neighbor_of_blocks=True,
                                weighted_graph=self.is_weighted)
    partition.set_from_representation({node: node % 2 for node in graph})

    return_values = partition.move_node(2, 1, return_edge_changes=True)
    self.assertEqual(len(return_values), 2)
    if self.is_weighted:
        six.assertCountEqual(self, return_values[0], [(0, 1, .5), (0, 1, .5), (1, 0, .5), (1, 0, .5)])
        six.assertCountEqual(self, return_values[1], [(1, 1, .5), (1, 1, .5), (1, 1, .5), (1, 1, .5)])
    else:
        six.assertCountEqual(self, return_values[0], [(0, 1, 1), (1, 0, 1)])
        six.assertCountEqual(self, return_values[1], [(1, 1, 1), (1, 1, 1)])

    return_values = partition.move_node(1, 1, return_edge_changes=True)
    self.assertEqual(return_values, None)
    # check correct handling
    return_values = partition.move_node(1, 0, return_edge_changes=True)
    self.assertEqual(len(return_values), 2)
    if self.is_weighted:
        six.assertCountEqual(self, return_values[0],
                             [(0, 1, .5), (1, 1, .5), (1, 1, .5), (0, 1, .5), (1, 1, .5), (1, 1, .5)])
        six.assertCountEqual(self, return_values[1],
                             [(0, 0, .5), (0, 1, .5), (0, 0, .5), (0, 0, .5), (0, 1, .5), (0, 0, .5)])
    else:
        six.assertCountEqual(self, return_values[0], [(0, 1, 1), (1, 1, 1), (1, 1, 1)])
        six.assertCountEqual(self, return_values[1], [(0, 0, 1), (0, 1, 1), (0, 0, 1)])


def general_test_set_from_representation_different_number_of_block(self):
    """Test setting partition from representation with different block sizes"""
    old_number_of_blocks = self.partition.B

    new_representation = {}
    node = 0
    for block, node in enumerate(self.partition.get_nodes_iter()):
        new_representation[node] = block

    self.assertGreater(len(new_representation), old_number_of_blocks)
    self.partition.set_from_representation(new_representation)
    self.assertEqual(self.partition.B, len(new_representation))
    # check if it works
    for block, node in enumerate(self.partition.get_nodes_iter()):
        self.assertEqual(self.partition.get_block_of_node(node), block)

    # move again but shifted
    for node in self.partition.get_nodes_iter():
        new_representation[node] = (new_representation[node] + 1) % len(new_representation)
    self.partition.set_from_representation(new_representation)
    self.assertEqual(self.partition.B, len(new_representation))
    # check if it works
    for block, node in enumerate(self.partition.get_nodes_iter()):
        self.assertEqual(self.partition.get_block_of_node(node), (block + 1) % len(new_representation))

    # move all in one block
    for node in self.partition.get_nodes_iter():
        new_representation[node] = 0

    # test if check is done
    partition = self.partition.copy()
    wrong_representation = new_representation.copy()
    del wrong_representation[node]
    with self.assertRaises(ValueError):
        partition.set_from_representation(wrong_representation)

    # now correct
    self.partition.set_from_representation(new_representation)
    self.assertEqual(self.partition.B, 1)
    # check if it works
    for block, node in enumerate(self.partition.get_nodes_iter()):
        self.assertEqual(self.partition.get_block_of_node(node), 0)


def general_test_move_node_with_remove_block_and_save_neighbor(self):
    self.partition = self.test_class(graph=self.graph,
                                     calculate_degree_of_blocks=True,
                                     save_neighbor_edges=True,
                                     save_neighbor_of_blocks=True,
                                     weighted_graph=self.is_weighted)

    self.partition.set_from_representation({node: node % len(self.graph) for node in self.graph})

    # old_block = self.partition.get_block_of_node(1)
    self.partition.move_node(1, 0)
    # create new for comparision
    comparision_partition = self.test_class(graph=self.graph,
                                            calculate_degree_of_blocks=True,
                                            save_neighbor_edges=True,
                                            save_neighbor_of_blocks=True,
                                            weighted_graph=self.is_weighted)
    comparision_partition.set_from_representation(self.partition.get_representation())
    # be sure that with the set from representation everything was fine
    self.assertEqual(comparision_partition.get_representation(), self.partition.get_representation())
    # check old_block is removed from neighbors list
    for block in range(self.partition.B):
        self.assertEqual(comparision_partition.get_neighbors_of_block(block),
                         self.partition.get_neighbors_of_block(block))
        # check if no greater block is included
        self.assertGreater(self.partition.B, max(self.partition.get_neighbors_of_block(block)))

    self.assertEqual(self.partition.get_number_of_nodes_in_block(self.partition.B - 1), 1)

    # find node of last block
    old_block = self.partition.B - 1
    node = 0
    for node in self.graph:
        if self.partition.get_block_of_node(node) == old_block:
            break
    # move it away
    self.partition.move_node(node, 0)
    # create new for comparision
    comparision_partition = self.test_class(graph=self.graph,
                                            calculate_degree_of_blocks=True,
                                            save_neighbor_edges=True,
                                            save_neighbor_of_blocks=True,
                                            weighted_graph=self.is_weighted)
    comparision_partition.set_from_representation(self.partition.get_representation())
    # be sure that with the set from representation everything was fine
    self.assertEqual(comparision_partition.get_representation(), self.partition.get_representation())
    # check old_block is removed from neighbors list
    for block in range(self.partition.B):
        self.assertEqual(comparision_partition.get_neighbors_of_block(block),
                         self.partition.get_neighbors_of_block(block))
        # check if no greater block is included
        self.assertGreater(self.partition.B, max(self.partition.get_neighbors_of_block(block)))


def directed_test_set_from_representation(self):
    """Test error with set from representation"""
    self.partition.set_from_representation({0: 0, 4: 1, 5: 2, 8: 3, 6: 4, 2: 5, 9: 6, 3: 1, 7: 6, 1: 6})
    true_partition = {7: 7, 1: 0, 6: 6, 9: 1, 8: 8, 4: 4, 3: 3, 0: 0, 2: 2, 5: 5}
    self.partition.set_from_representation(true_partition)
    self.assertEqual(true_partition, self.partition.get_representation())

    # test second error with node key error of saved neighbors
    self.partition.set_from_representation({7: 0, 9: 1, 1: 2, 4: 3, 6: 4, 3: 5, 8: 6, 2: 2, 5: 5, 0: 2})
    true_partition = {7: 7, 9: 1, 4: 4, 6: 6, 5: 5, 2: 2, 8: 8, 3: 3, 0: 0, 1: 0}
    self.partition.set_from_representation(true_partition)
    self.assertEqual(true_partition, self.partition.get_representation())


def general_test_save_degree_distribution(self):
    new_partition = self.test_class(graph=self.graph,
                                    number_of_blocks=2,
                                    calculate_degree_of_blocks=False,
                                    save_neighbor_edges=False,
                                    save_neighbor_of_blocks=False,
                                    save_degree_distributions=True,
                                    representation={node: node % 2 for node in self.graph},
                                    weighted_graph=self.is_weighted)
    self.partition.set_from_representation({node: node % 2 for node in self.graph})

    for _ in range(10):
        self.assertEqual(self.partition.get_degree_distribution_of_blocks(True),
                         new_partition.get_degree_distribution_of_blocks(True))

        self.assertEqual(self.partition.get_degree_distribution_of_blocks(False),
                         new_partition.get_degree_distribution_of_blocks(False))

        if self.partition.is_graph_directed():
            self.assertEqual(new_partition.get_joint_in_out_degree_distribution_of_blocks(),
                             self.partition.get_joint_in_out_degree_distribution_of_blocks())

            degree_distributions = new_partition.get_joint_in_out_degree_distribution_of_blocks()
        else:
            degree_distributions = new_partition.get_degree_distribution_of_blocks(False)

        self.assertEqual(len(degree_distributions), self.partition.B)
        for block in range(len(degree_distributions)):
            for degree in degree_distributions[block]:
                self.assertEqual(new_partition.get_number_of_nodes_with_same_degree_in_block(block, degree),
                                 degree_distributions[block][degree])

        node, _, to_block = self.partition.get_random_move()
        self.partition.move_node(node, to_block)
        new_partition.move_node(node, to_block)


def general_test_handle_change_information(self):
    # first try error
    with self.assertRaises(ValueError):
        self.partition.handle_change_information(0)

    above_partition = NxPartitionGraphBasedHierarchy(
        save_neighbor_edges=True,
        graph=self.partition.get_partition_as_a_graph(),
        representation={block: 0 for block in range(self.partition.B)},
        weighted_graph=True)

    # normal move
    node, _, to_block = self.partition.get_random_move()
    moves = [(node, to_block), (node, self.partition.B), (node, 0)]
    return_values = []
    for node, to_block in moves:
        return_values = self.partition.move_node(node, to_block, return_edge_changes=True)
        above_partition.handle_change_information(*return_values)

        comparision_partition = NxPartitionGraphBasedHierarchy(
            save_neighbor_edges=True,
            graph=self.partition.get_partition_as_a_graph(),
            representation={block: 0 for block in range(self.partition.B)},
            weighted_graph=True)

        self.assertTrue(general_test_saved_edges(self, above_partition, comparision_partition))

    above_unweighted_partition = NxPartitionGraphBasedHierarchy(
        save_neighbor_edges=True,
        graph=self.partition.get_partition_as_a_graph(),
        representation={block: 0 for block in range(self.partition.B)},
        weighted_graph=False)

    with self.assertRaises(NotImplementedError):
        above_unweighted_partition.handle_change_information(*return_values)


class TestNxPartitionGraphBased(test_sbmpartition.TestNxPartition):
    """ Test Class for NxPartitionGraphBased """

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionGraphBased, self).__init__(methodName)
        self.is_weighted = False
        self.test_class = NxPartitionGraphBased

    # rewrite methods
    test_set_from_representation_different_number_of_block = \
        general_test_set_from_representation_different_number_of_block
    test_copy_neighbor_edges = graph_based_test_copy_neighbor_edges


    test_move_node_with_remove_block_and_save_neighbor = general_test_move_node_with_remove_block_and_save_neighbor
    test_save_degree_distribution = general_test_save_degree_distribution

    test_move_node_with_block_number_change = undirected_test_move_node_with_block_number_change


class TestNxPartitionGraphBasedDirected(test_sbmpartition.TestNxPartitionDirected):
    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionGraphBasedDirected, self).__init__(methodName)
        self.is_weighted = False
        self.test_class = NxPartitionGraphBased

    # rewrite methods
    test_set_from_representation_different_number_of_block = \
        general_test_set_from_representation_different_number_of_block
    test_copy_neighbor_edges = graph_based_test_copy_neighbor_edges

    test_move_node_with_remove_block_and_save_neighbor = general_test_move_node_with_remove_block_and_save_neighbor
    test_save_degree_distribution = general_test_save_degree_distribution

    test_move_node_with_block_number_change = directed_test_move_node_with_block_number_change
    test_set_from_representation = directed_test_set_from_representation


class TestNxPartitionGraphBasedMultiWeighted(test_sbmpartition.TestNxPartitionMultiWeighted):
    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionGraphBasedMultiWeighted, self).__init__(methodName)
        self.is_weighted = True
        self.test_class = NxPartitionGraphBased

    # rewrite neighbor edge test of copy
    test_set_from_representation_different_number_of_block = \
        general_test_set_from_representation_different_number_of_block
    test_copy_neighbor_edges = graph_based_test_copy_neighbor_edges

    test_save_degree_distribution = general_test_save_degree_distribution

    test_move_node_with_block_number_change = undirected_test_move_node_with_block_number_change


class TestNxPartitionGraphBasedDirectedMultiWeighted(test_sbmpartition.TestNxPartitionDirectedMultiWeighted):
    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionGraphBasedDirectedMultiWeighted, self).__init__(methodName)
        self.is_weighted = True
        self.test_class = NxPartitionGraphBased

    # rewrite neighbor edge test of copy
    test_set_from_representation_different_number_of_block = \
        general_test_set_from_representation_different_number_of_block
    test_copy_neighbor_edges = graph_based_test_copy_neighbor_edges

    test_save_degree_distribution = general_test_save_degree_distribution

    test_move_node_with_block_number_change = directed_test_move_node_with_block_number_change


class TestNxPartitionGraphBasedHierarchy(TestNxPartitionGraphBased):
    """ Test Class for NxPartitionGraphBasedHierarchy """

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionGraphBasedHierarchy, self).__init__(methodName)
        self.is_weighted = False
        self.test_class = NxPartitionGraphBasedHierarchy

    test_handle_change_information = general_test_handle_change_information

    # add methods
    test_move_node_with_return_values = undirected_test_move_node_with_return_values
    test_split_and_merge = undirected_test_split_and_merge
    test_move_node_with_edge_changes = undirected_test_move_node_edge_changes


class TestNxPartitionGraphBasedHierarchyDirected(TestNxPartitionGraphBasedDirected):

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionGraphBasedHierarchyDirected, self).__init__(methodName)
        self.is_weighted = False
        self.test_class = NxPartitionGraphBasedHierarchy

    test_handle_change_information = general_test_handle_change_information

    # add methods
    test_move_node_with_return_values = directed_test_move_node_with_return_values
    test_split_and_merge = directed_test_split_and_merge
    test_move_node_with_edge_changes = directed_test_move_node_edge_changes


class TestNxPartitionGraphBasedHierarchyMultiWeighted(TestNxPartitionGraphBasedMultiWeighted):

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionGraphBasedHierarchyMultiWeighted, self).__init__(methodName)
        self.is_weighted = True
        self.test_class = NxPartitionGraphBasedHierarchy

    test_handle_change_information = general_test_handle_change_information

    # add methods
    test_move_node_with_return_values = undirected_test_move_node_with_return_values
    test_split_and_merge = undirected_test_split_and_merge
    test_move_node_with_edge_changes = undirected_test_move_node_edge_changes


class TestNxPartitionGraphBasedHierarchyDirectedMultiWeighted(TestNxPartitionGraphBasedDirectedMultiWeighted):

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxPartitionGraphBasedHierarchyDirectedMultiWeighted, self).__init__(methodName)
        self.is_weighted = True
        self.test_class = NxPartitionGraphBasedHierarchy

    test_handle_change_information = general_test_handle_change_information

    # add methods
    test_move_node_with_return_values = directed_test_move_node_with_return_values
    test_split_and_merge = directed_test_split_and_merge
    test_move_node_with_edge_changes = directed_test_move_node_edge_changes


class TestNxHierarchicalPartition(ut.TestCase):

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxHierarchicalPartition, self).__init__(methodName)
        self.is_weighted = False

    def setUp(self):
        # create simple graph with 4 full connected nodes
        if self.is_weighted:
            self.graph = nx.DiGraph()
            for from_node in range(4):
                for to_node in range(4):
                    if from_node != to_node:
                        self.graph.add_edge(from_node, to_node, weight=.5)
                        self.graph.add_edge(from_node, to_node, weight=.5)
        else:
            self.graph = nx.DiGraph(nx.complete_graph(4))
            # # Problem of the underlying class test there first and then include here too
            # # check if edge weights are ignored
            # self.graph.add_edge(0, 1, weight=100)

        self.partition = NxHierarchicalPartition(self.graph, number_of_blocks=len(self.graph),
                                                 weighted_graph=self.is_weighted)

        self.representations = [
            [{node: node for node in self.graph}],
            [{node: node for node in self.graph},
             {0: 0, 1: 0, 2: 1, 3: 1}],
            [{0: 0, 1: 0, 2: 1, 3: 2},
             {0: 0, 1: 0, 2: 1}],
            [{0: 0, 1: 0, 2: 1, 3: 2},
             {0: 0, 1: 0, 2: 1},
             {0: 0, 1: 0}]
        ]

    def test_get_number_of_blocks_in_level(self):
        # check length detection for prepared representation
        self.partition.actual_level = 0
        for representation in self.representations:
            self.partition.set_from_representation(representation)
            # for reach level -1 is one way to acces the number of nodes in the underlying graph
            self.assertEqual(self.partition.get_number_of_blocks_in_level(-1), len(self.graph))
            # for all other levels the number of blocks is given by the maximal block number (+1)
            #  caused by numbering from 0... B-1
            for level, level_representation in enumerate(representation):
                self.assertEqual(self.partition.get_number_of_blocks_in_level(level),
                                 max(level_representation.values()) + 1)

    def test_iter_levels(self):
        # returned level iter should match with the number of levels given
        for representation in self.representations:
            self.partition.set_from_representation(representation)
            self.assertEqual(list(self.partition.iter_levels()), list(range(len(representation))))

    def test_set_from_representations_basic(self):
        for representation in self.representations:
            self.partition.set_from_representation(representation)
            for level, level_representation in enumerate(representation):
                self.assertEqual(self.partition.partitions[level].partition, level_representation)

    def _comparator_of_two_partitions(self, first_partition, second_partition):
        self.assertEqual(list(first_partition.get_degree_iter()),
                         list(second_partition.get_degree_iter()))

        # test get directed
        self.assertEqual(first_partition.is_graph_directed(), second_partition.is_graph_directed())

        if first_partition.is_graph_directed():
            # these tests only for directed partitions
            self.assertEqual(first_partition.get_joint_in_out_degree_distribution_of_blocks(),
                             second_partition.get_joint_in_out_degree_distribution_of_blocks())

            self.assertEqual(list(first_partition.get_in_degree_iter()),
                             list(second_partition.get_in_degree_iter()))

            self.assertEqual(list(first_partition.get_out_degree_iter()),
                             list(second_partition.get_out_degree_iter()))

        np.testing.assert_array_equal(first_partition.get_graph_matrix_representation(),
                                      second_partition.get_graph_matrix_representation())

        np.testing.assert_array_equal(first_partition.get_graph_matrix_representation(with_weights=True),
                                      second_partition.get_graph_matrix_representation(with_weights=True))

        np.testing.assert_array_equal(first_partition.get_graph_matrix_representation(with_weights=False),
                                      second_partition.get_graph_matrix_representation(with_weights=False))

        self.assertEqual(first_partition.B, second_partition.B)

        for block in range(second_partition.B):
            self.assertEqual(first_partition.get_number_of_nodes_in_block(block),
                             second_partition.get_number_of_nodes_in_block(block))

            self.assertEqual(first_partition.get_neighbors_of_block(block),
                             second_partition.get_neighbors_of_block(block))

            # check elements function
            self.assertEqual(first_partition.get_elements_of_block(block),
                             second_partition.get_elements_of_block(block))

            for other_block in range(second_partition.B):
                self.assertEqual(first_partition.get_edge_count(block, other_block),
                                 second_partition.get_edge_count(block, other_block))

        self.assertEqual(list(first_partition.get_degree_distribution_of_blocks()),
                         list(second_partition.get_degree_distribution_of_blocks()))

        self.assertEqual(list(first_partition.get_degree_distribution_of_blocks(probability=False)),
                         list(second_partition.get_degree_distribution_of_blocks(probability=False)))

        self.assertEqual(list(first_partition.get_degree_distribution_of_blocks(probability=True)),
                         list(second_partition.get_degree_distribution_of_blocks(probability=True)))

        self.assertEqual(first_partition.get_number_of_nodes(),
                         second_partition.get_number_of_nodes())

        self.assertEqual(list(first_partition.get_nodes_iter()),
                         list(second_partition.get_nodes_iter()))

        # test block membership retrieval
        for node in second_partition.get_nodes_iter():
            self.assertEqual(first_partition.get_block_of_node(node), second_partition.get_block_of_node(node))

        # test neighbor of node
        for node in second_partition.get_nodes_iter():
            six.assertCountEqual(self, first_partition.get_neighbors_of_node(node),
                                 second_partition.get_neighbors_of_node(node))

    def test_basics(self):
        """"Test simple methods only based on underlying partition"""
        for representation in self.representations:
            self.partition.set_from_representation(representation)

            # check if get and set representation works
            self.assertEqual(self.partition.get_representation(), representation)

            # check total_edge_count
            self.partition.actual_level = len(representation)
            if self.is_weighted:
                self.assertEqual(self.partition.get_edge_count(0, 0), self.graph.size(weight='weight'))
            else:
                self.assertEqual(self.partition.get_edge_count(0, 0), self.graph.size())

            # get block memberships
            block_membership = {}
            for level, level_representation in enumerate(representation):

                if level == 0:
                    partition = NxPartitionGraphBasedHierarchy(graph=self.graph, weighted_graph=self.is_weighted,
                                                               representation=level_representation)

                    # check edge total
                    if self.is_weighted:
                        self.assertEqual(self.graph.size(weight='weight'), self.partition.edge_total)
                    else:
                        self.assertEqual(self.graph.size(), self.partition.edge_total)

                    self.assertEqual(self.partition.partitions[0]._is_weighted, self.is_weighted)
                else:
                    partition = NxPartitionGraphBasedHierarchy(
                        graph=self.partition.partitions[level - 1].get_partition_as_a_graph(), weighted_graph=True)
                    partition.set_from_representation(level_representation)

                block_membership[level] = partition.get_block_memberships()

                self.partition.actual_level = level

                # check if B detection works
                self.assertEqual(self.partition.B, partition.B)

                self._comparator_of_two_partitions(self.partition, partition)

                # test precalc move
                objective_function = TraditionalUnnormalizedLogLikelyhood(self.partition.is_graph_directed())
                for node in self.partition.get_nodes_iter():
                    actual_block = self.partition.get_block_of_node(node)
                    for to_block in range(partition.B):
                        self.assertEqual(
                            self.partition.precalc_move((node, actual_block, to_block), objective_function),
                            partition.precalc_move((node, actual_block, to_block), objective_function))

                # test random node
                every_node_in_a_single_block = False
                if level == 0:
                    for _ in range(10):
                        random_node = self.partition.get_random_node(no_single=False)
                        self.assertTrue(random_node in self.graph)
                        if self.partition.B != len(self.graph):
                            random_node = self.partition.get_random_node(no_single=True)
                            self.assertGreater(
                                self.partition.get_number_of_nodes_in_block(
                                    self.partition.get_block_of_node(random_node)),
                                1)
                        else:
                            every_node_in_a_single_block = True
                            with self.assertRaises(NoFreeNodeException):
                                self.partition.get_random_node(no_single=True)
                else:
                    for _ in range(10):
                        random_node = self.partition.get_random_node(no_single=True)
                        self.assertTrue(random_node in set(range(self.partition.get_number_of_nodes())))
                        self.assertGreater(
                            self.partition.get_number_of_nodes_in_block(self.partition.get_block_of_node(random_node)),
                            1)

                # test get random move
                if level != len(representation) - 1 and not every_node_in_a_single_block:
                    # check if new block is in the same block above
                    for _ in range(10):
                        node, old_block, to_block = self.partition.get_random_move()
                        self.assertEqual(old_block, self.partition.get_block_of_node(node))
                        self.assertEqual(self.partition.partitions[level + 1].get_block_of_node(to_block),
                                         self.partition.partitions[level + 1].get_block_of_node(old_block))

                        # check if valid move
                        self.assertTrue(self.partition._is_valid_move(node, to_block))
                elif every_node_in_a_single_block:
                    with self.assertRaises(NoFreeNodeException):
                        self.partition.get_random_move()
                else:
                    # all moves allowed
                    self.assertTrue(len(self.partition.get_random_move()) == 3)
                    node, _, to_block = self.partition.get_random_move()
                    # check if valid move
                    self.assertTrue(self.partition._is_valid_move(node, to_block))

            # after checking every level possible to compare block membership representation
            self.assertEqual(self.partition.get_block_memberships(),
                             str(block_membership))

        # test that random partition not works
        with self.assertRaises(NotImplementedError):
            self.partition.random_partition()

    def test_move_nodes(self):
        graphs = [nx.DiGraph(), nx.Graph()]
        for g in graphs:
            # create ring graph with 100 nodes
            # and some shortcuts
            # and some selfloops
            if self.is_weighted:
                nx.add_path(g, range(100), weight=.5)
                nx.add_path(g, range(100), weight=.5)

                g.add_edges_from(
                    [(43, 77), (70, 44), (30, 84), (13, 5), (54, 56), (27, 57), (29, 44), (83, 32), (75, 31), (13, 53)],
                    weight=.5)
                g.add_edges_from(
                    [(43, 77), (70, 44), (30, 84), (13, 5), (54, 56), (27, 57), (29, 44), (83, 32), (75, 31), (13, 53)],
                    weight=.5)
                g.add_edges_from([(1, 1), (8, 8), (26, 26), (30, 30), (2, 2), (89, 89), (96, 96), (30, 30)], weight=.5)
                g.add_edges_from([(1, 1), (8, 8), (26, 26), (30, 30), (2, 2), (89, 89), (96, 96), (30, 30)], weight=.5)
            else:
                nx.add_path(g, range(100))
                g.add_edges_from(
                    [(43, 77), (70, 44), (30, 84), (13, 5), (54, 56), (27, 57), (29, 44), (83, 32), (75, 31), (13, 53)])
                g.add_edges_from([(1, 1), (8, 8), (26, 26), (30, 30), (2, 2), (89, 89), (96, 96), (30, 30)])

            hierarchical_partition = NxHierarchicalPartition(g, number_of_blocks=10,
                                                             save_neighbor_edges=True,
                                                             save_neighbor_of_blocks=True,
                                                             calculate_degree_of_blocks=True,
                                                             weighted_graph=self.is_weighted)

            hierarchical_partition.add_level_from_representation({block: block % 2 for block in range(10)})

            hierarchical_partition.actual_level = 0

            # check nonsense move
            with self.assertRaises(ValueError):
                # move any node which belongs to block 0 to block 1 (different block in level 1)
                hierarchical_partition.move_node(
                    hierarchical_partition.partitions[0].get_elements_of_block(0).copy().pop(),
                    1)

            # perform some moves
            for _ in range(100):
                random_move = hierarchical_partition.get_random_move()
                hierarchical_partition.move_node(random_move[0], random_move[2])

            # create now correct partition from the scratch for the top level
            partition = NxPartitionGraphBasedHierarchy(
                graph=hierarchical_partition.partitions[0].get_partition_as_a_graph(),
                weighted_graph=True,
                save_neighbor_edges=True,
                save_neighbor_of_blocks=True,
                calculate_degree_of_blocks=True)

            # with the same groups as below (may have changed)
            partition.set_from_representation(hierarchical_partition.partitions[1].get_representation())
            # compare basics
            hierarchical_partition.actual_level = 1
            self._comparator_of_two_partitions(hierarchical_partition, partition)

            # compare edges
            self._compare_neighbored_edges(hierarchical_partition.partitions[1], partition)

            # simple test for highest level
            hierarchical_partition.move_node(1, 0)
            self.assertEqual(hierarchical_partition.get_block_of_node(1), 0)

            # simple test without edge saving
            hierarchical_partition.set_save_neighbor_edges(False)
            hierarchical_partition.actual_level = 0
            move = hierarchical_partition.get_random_move()
            hierarchical_partition.move_node(move[0], move[2])
            self.assertEqual(hierarchical_partition.get_block_of_node(move[0]), move[2])
            # extend with split
            hierarchical_partition.move_node(move[0], hierarchical_partition.B)
            self.assertEqual(hierarchical_partition.get_block_of_node(move[0]), hierarchical_partition.B - 1)

    def _compare_neighbored_edges(self, first_graph_based_partition, second_graph_based_partition):
        first_partition_as_a_graph = first_graph_based_partition.get_partition_as_a_graph()
        second_partition_as_a_graph = second_graph_based_partition.get_partition_as_a_graph()

        self.assertEqual(len(first_partition_as_a_graph), len(second_partition_as_a_graph))
        self.assertEqual(list(first_partition_as_a_graph.nodes()),
                         list(second_partition_as_a_graph.nodes()))

        for block in first_partition_as_a_graph:
            # implementation specific!
            six.assertCountEqual(self, first_partition_as_a_graph.nodes[block]['edge neighbors'].items,
                                 second_partition_as_a_graph.nodes[block]['edge neighbors'].items)
            # check weights
            for edge in first_partition_as_a_graph.nodes[block]['edge neighbors'].item_to_position:
                # first get both positions
                first_position = first_partition_as_a_graph.nodes[block]['edge neighbors'].item_to_position[edge]
                second_position = second_partition_as_a_graph.nodes[block]['edge neighbors'].item_to_position[edge]
                # then get both weights
                first_weight = first_partition_as_a_graph.nodes[block]['edge neighbors'].weights[first_position]
                second_weight = second_partition_as_a_graph.nodes[block]['edge neighbors'].weights[second_position]
                self.assertEqual(first_weight, second_weight)

    def test_set_save_edge_neighbors(self):
        for representation in self.representations:
            self.partition.set_from_representation(representation)

            # switch on
            self.partition.set_save_neighbor_edges(True)
            self.assertTrue(self.partition._save_neighbor_edges)
            # check for every level
            for level in range(len(representation)):
                self.assertTrue(self.partition.partitions[level]._save_neighbor_edges)

                # basic test if it works
                self.partition.actual_level = level
                possible_blocks = set(range(self.partition.B))
                for block in possible_blocks:
                    edge = self.partition.get_random_neighboring_edge_of_block(block)
                    self.assertEqual(block, self.partition.get_block_of_node(edge[0]))
                    self.assertTrue(self.partition.get_block_of_node(edge[1]) in possible_blocks)

                    # check if one of the edges really exists
                    self.assertTrue(self.partition.partitions[level]._graph.has_edge(edge[0], edge[1]) or
                                    self.partition.partitions[level]._graph.has_edge(edge[1], edge[0]))

            # switch back
            self.partition.set_save_neighbor_edges(False)
            self.assertFalse(self.partition._save_neighbor_edges)
            for level in range(len(representation)):
                self.assertFalse(self.partition.partitions[level]._save_neighbor_edges)

    def test_copy(self):
        graph = nx.complete_graph(10)

        if self.is_weighted:
            # if needed add edge weights
            for edge in graph.edges:
                graph[edge[0]][edge[1]]["weight"] = 1

        representations = [[{8: 0, 1: 1, 3: 2, 5: 3, 6: 4, 7: 5, 9: 6, 0: 8, 4: 7, 2: 9},
                            {2: 0, 7: 1, 4: 0, 1: 1, 8: 0, 9: 1, 5: 1, 3: 1, 6: 0, 0: 0}],
                           [{8: 0, 1: 1, 3: 2, 5: 3, 6: 4, 7: 5, 9: 6, 0: 7, 4: 8, 2: 9},
                            {2: 0, 7: 1, 4: 0, 1: 1, 8: 0, 9: 1, 5: 1, 3: 1, 6: 0, 0: 0}]]
        for representation in representations:
            hierarchical_partition = NxHierarchicalPartition(graph,
                                                             number_of_blocks=10,
                                                             save_neighbor_edges=True,
                                                             save_neighbor_of_blocks=True,
                                                             calculate_degree_of_blocks=True,
                                                             weighted_graph=self.is_weighted)
            hierarchical_partition.set_from_representation(representation)
            # hierarchical_partition.add_level_from_representation({block: block % 2 for block in range(10)})
            hierarchical_partition.actual_level = 0
            new_partition = hierarchical_partition.copy()

            # check if everything is the same
            self.assertEqual(new_partition.max_level, hierarchical_partition.max_level)
            self._comparator_of_two_partitions(hierarchical_partition, new_partition)
            hierarchical_partition.actual_level = 1
            self.assertNotEqual(hierarchical_partition.actual_level, new_partition.actual_level)
            new_partition.actual_level = 1
            self._comparator_of_two_partitions(hierarchical_partition, new_partition)

            # set back on basic level, move node and see what happens
            hierarchical_partition.actual_level = 0
            new_partition.actual_level = 0

            # change one node (remove one block)
            old_block = new_partition.get_block_of_node(0)
            new_block = (new_partition.get_block_of_node(0) + 2) % 10
            new_partition.move_node(0, new_block)

            # check old partition
            self.assertEqual(hierarchical_partition.get_number_of_nodes_in_block(new_block), 1)
            self.assertEqual(hierarchical_partition.get_number_of_nodes_in_block(old_block), 1)
            # catch the case if the new block should be the last block, which then gets relabeled with the old block
            if new_block == 9:
                new_block = old_block
                # test if the move was performed
                self.assertEqual(new_partition.get_block_of_node(0), new_block)
            else:
                # test if the move was performed
                self.assertEqual(new_partition.get_block_of_node(0), new_block)
                # and was not performed in the old partition
                self.assertNotEqual(new_block, hierarchical_partition.get_block_of_node(0))
            # the number of block was changed too
            self.assertNotEqual(new_partition.B, hierarchical_partition.B)
            # and the neighbors of the blocks
            self.assertNotEqual(new_partition.get_neighbors_of_block(new_block),
                                hierarchical_partition.get_neighbors_of_block(new_block))

    def test_merge_blocks(self):
        # set to 2. partition (has two levels)
        self.partition.set_from_representation(self.representations[1])

        new_partition = self.partition.copy()
        # check on level 0
        self.partition.actual_level = 0
        self.partition.merge_blocks({1: 0}, 2)
        representation_after_merge = self.representations[1]
        relabeled_block = max(representation_after_merge[0].values())
        # move all nodes
        for node in representation_after_merge[0]:
            if representation_after_merge[0][node] == 1:
                representation_after_merge[0][node] = 0
            elif representation_after_merge[0][node] == relabeled_block:
                representation_after_merge[0][node] = 1
        representation_after_merge[1][1] = representation_after_merge[1][relabeled_block]
        del representation_after_merge[1][relabeled_block]
        new_partition.set_from_representation(representation_after_merge)
        # compare both partitions
        for level in new_partition.iter_levels():
            self.partition.actual_level = level
            new_partition.actual_level = level
            self._comparator_of_two_partitions(self.partition, new_partition)

        # on top level everything fine
        self.partition.actual_level = 1
        self.partition.merge_blocks({1: 0}, 1)
        self.assertEqual(self.partition.B, 1)

    def test_creation_from_representation(self):
        # simple test
        for representation in self.representations:
            new_partition = NxHierarchicalPartition(self.graph, number_of_blocks=len(self.graph),
                                                    weighted_graph=self.is_weighted,
                                                    representation=representation)
            self.assertEqual(new_partition.get_representation(), representation)

    def test_random_partition(self):
        graphs = [nx.DiGraph(), nx.Graph()]
        for g in graphs:
            # create ring graph with 100 nodes
            # and some shortcuts
            # and some selfloops
            if self.is_weighted:
                nx.add_path(g, range(100), weight=.5)
                nx.add_path(g, range(100), weight=.5)

                g.add_edges_from(
                    [(43, 77), (70, 44), (30, 84), (13, 5), (54, 56), (27, 57), (29, 44), (83, 32), (75, 31), (13, 53)],
                    weight=.5)
                g.add_edges_from(
                    [(43, 77), (70, 44), (30, 84), (13, 5), (54, 56), (27, 57), (29, 44), (83, 32), (75, 31), (13, 53)],
                    weight=.5)
                g.add_edges_from([(1, 1), (8, 8), (26, 26), (30, 30), (2, 2), (89, 89), (96, 96), (30, 30)], weight=.5)
                g.add_edges_from([(1, 1), (8, 8), (26, 26), (30, 30), (2, 2), (89, 89), (96, 96), (30, 30)], weight=.5)
            else:
                nx.add_path(g, range(100))
                g.add_edges_from(
                    [(43, 77), (70, 44), (30, 84), (13, 5), (54, 56), (27, 57), (29, 44), (83, 32), (75, 31), (13, 53)])
                g.add_edges_from([(1, 1), (8, 8), (26, 26), (30, 30), (2, 2), (89, 89), (96, 96), (30, 30)])

            hierarchical_partition = NxHierarchicalPartition(g, number_of_blocks=10,
                                                             save_neighbor_edges=True,
                                                             save_neighbor_of_blocks=True,
                                                             calculate_degree_of_blocks=True,
                                                             weighted_graph=self.is_weighted,
                                                             # representation=[{node: node for node in range(10)}]
                                                             )

            # test error on call without parameters
            with self.assertRaises(NotImplementedError):
                hierarchical_partition.random_partition()

            # test error on call with no number of groups given
            with self.assertRaises(NotImplementedError):
                hierarchical_partition.random_partition(level=0)

            representation = hierarchical_partition.get_representation()
            representation.append({node: node % 5 for node in range(max(representation[0].values()) + 1)})
            representation.append({node: node % 2 for node in range(max(representation[1].values()) + 1)})

            hierarchical_partition.set_from_representation(representation)

            # test error on call with fewer groups then level above
            with self.assertRaises(ValueError):
                hierarchical_partition.random_partition(level=0, number_of_blocks=4)

            # test error on call with more groups then level below
            with self.assertRaises(ValueError):
                hierarchical_partition.random_partition(level=2, number_of_blocks=6)

            for max_level in range(0, 3):
                for level in range(max_level, -1, -1):
                    old_representation = hierarchical_partition.get_representation()
                    number_of_blocks = 20 - level * 5
                    hierarchical_partition.random_partition(number_of_blocks=number_of_blocks, level=level)

                    # first test if number of blocks is correct
                    self.assertEqual(number_of_blocks, hierarchical_partition.get_number_of_blocks_in_level(level))

                    # check validity of random partition
                    # check bottom->top
                    if level == max_level:
                        continue

                    new_representation = hierarchical_partition.get_representation()

                    # second test if number of blocks is correct
                    self.assertEqual(number_of_blocks, max(new_representation[level].values()) + 1)
                    for node in old_representation[level]:
                        old_block = old_representation[level][node]
                        new_block = new_representation[level][node]
                        self.assertEqual(old_representation[level + 1][old_block],
                                         new_representation[level + 1][new_block],
                                         msg="\nNew representation:" \
                                             + str(hierarchical_partition.get_representation()) \
                                             + "\nold representation: " + str(old_representation) \
                                             + "\nnode with error " + str(node) \
                                             + "\nmax level" + str(max_level) \
                                             + "\nlevel " + str(level) \
                                             + "\ncheck level " + str(level))

                # add new level
                representation = hierarchical_partition.get_representation()
                representation.append({block: block % 5 for block in
                                       range(hierarchical_partition.get_number_of_blocks_in_level(max_level))})
                hierarchical_partition.set_from_representation(representation)

    def test_tracking_of_elements_in_block_after_reducing_block_count(self):

        graph = nx.Graph()

        if self.is_weighted:
            nx.add_path(graph, range(10), weight=1)
        else:
            nx.add_path(graph, range(10))

        hierarchical_partition = NxHierarchicalPartition(
            graph,
            number_of_blocks=10,
            save_neighbor_edges=True,
            save_neighbor_of_blocks=True,
            calculate_degree_of_blocks=True,
            weighted_graph=self.is_weighted,
            representation=[{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
                            {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0, 6: 1, 7: 2, 8: 2, 9: 4},
                            {0: 1, 1: 3, 2: 0, 3: 0, 4: 2}])

        hierarchical_partition.actual_level = 1
        # move last node of block 3 into block 2
        hierarchical_partition.move_node(3, 2)

        hierarchical_partition.actual_level = 2
        # check if top block 0 contains only block 2
        self.assertEqual(hierarchical_partition.get_elements_of_block(0), {2})
        for block in range(hierarchical_partition.B):
            self.assertTrue(len(hierarchical_partition.get_elements_of_block(block)) == 1,
                            msg=str(hierarchical_partition.get_elements_of_block(block)))

    def test_delete_level(self):
        other_partition = self.partition.copy()
        for representation in self.representations:
            if len(representation) <= 1:
                continue

            self.partition.set_from_representation(representation)

            self.partition.actual_level = 0
            with self.assertRaises(NotImplementedError):
                self.partition.delete_actual_level()

            while self.partition.max_level > 0:
                old_max_level = self.partition.max_level

                self.partition.actual_level = 1
                self.partition.delete_actual_level()

                self.assertEqual(self.partition.max_level, old_max_level - 1)
                new_representation = self.partition.get_representation()

                self.assertEqual(len(new_representation), old_max_level)

                other_partition.set_from_representation(new_representation)

                for level in other_partition.iter_levels():
                    self.partition.actual_level = level
                    other_partition.actual_level = level
                    self._comparator_of_two_partitions(self.partition, other_partition)

    def test_get_possible_blocks(self):
        for representation in self.representations:
            self.partition.set_from_representation(representation)

            # create correct blocks
            possible_blocks = []
            for top_level in range(1, len(representation)):
                # init lists of blocks by top blocks
                level_possible_blocks_by_top_block = [[] for _ in range(max(representation[top_level].values()) + 1)]
                level_representation = representation[top_level - 1]
                top_level_representation = representation[top_level]

                # fill lists of blocks by top blocks
                for block in top_level_representation:
                    top_block = top_level_representation[block]
                    level_possible_blocks_by_top_block[top_block].append(block)

                # create final data structure of possible blocks by blocks
                possible_blocks_by_blocks = {}
                for node in level_representation:
                    block = level_representation[node]
                    top_block = top_level_representation[block]
                    possible_blocks_by_blocks[block] = level_possible_blocks_by_top_block[top_block]

                possible_blocks.append(possible_blocks_by_blocks)

            # now check the method
            for level in self.partition.iter_levels():
                self.partition.actual_level = level

                for block in range(self.partition.B):
                    if level == self.partition.max_level:
                        self.assertEqual(self.partition.get_possible_blocks(block), range(self.partition.B))
                    else:
                        self.assertEqual(self.partition.get_possible_blocks(block), possible_blocks[level][block])

    def test_add_level_above_actual_level(self):

        # quite simple test
        other_partition = self.partition.copy()

        for representation in self.representations:
            for level in range(len(representation)):
                self.partition.set_from_representation(representation)
                self.partition.actual_level = level

                # check max level before
                self.assertEqual(self.partition.max_level, len(representation) - 1,
                                 msg="\nRepresentation:" + str(representation) + "\n set representation:" + str(
                                     self.partition.get_representation()))

                self.partition.add_level_above_actual_level()

                # check new max level
                self.assertEqual(self.partition.max_level, len(representation))

                other_partition.set_from_representation(self.partition.get_representation())

                # check other things
                for comparision_level in range(self.partition.max_level):
                    self.partition.actual_level = comparision_level
                    other_partition.actual_level = comparision_level
                    self._comparator_of_two_partitions(self.partition, other_partition)

    def test_no_movable_node_in_hierarchy(self):
        number_of_nodes = len(self.graph)
        first_partition = {node: (node - 1) for node in self.graph}
        first_partition[0] = 0
        second_partition = {block: block for block in range(max(first_partition.values()) + 1)}

        self.partition.set_from_representation([first_partition, second_partition])

        self.partition.actual_level = 0

        with self.assertRaises(NoFreeNodeException):
            self.partition.get_random_node()

        with self.assertRaises(NoFreeNodeException):
            self.partition.get_random_move()

    def test_empty_create_assign_B(self):
        """Check that B is not None for empty init"""
        new_partition = NxHierarchicalPartition(self.graph)
        self.assertNotEqual(new_partition.B, None)
        self.assertGreater(new_partition.B, 0)
        self.assertLessEqual(new_partition.B, len(self.graph))


class TestNxHierarchicalPartitionWeighted(TestNxHierarchicalPartition):

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNxHierarchicalPartitionWeighted, self).__init__(methodName)
        self.is_weighted = True
