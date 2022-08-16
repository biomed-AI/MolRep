import random as rd
import unittest as ut

import networkx as nx

from pysbm.sbm.inference import PeixotoInference
from pysbm.sbm.nxpartitiongraphbased import NxPartitionGraphBased
from pysbm.sbm.objective_function_iclex import *
from pysbm.sbm.objective_functions import TraditionalUnnormalizedLogLikelyhood


def general_delta_test_random_move(self, partition, normal_function, delta_function, partition_number, move_number,
                                   move=None):
    """General function to test delta calculations"""
    if move is None:
        move = partition.get_random_move()
    node, old_block, new_block = move
    precalc = partition.precalc_move(move, None)

    # value before the move and before any delta calculation
    start_value = normal_function(partition)

    # calculate delta with given set of parameters
    delta_value = delta_function(partition, old_block, new_block, *precalc)

    # move node to calculate real value
    partition.move_node(node, new_block)
    end_value = normal_function(partition)
    new_representation = partition.get_representation()
    # move node back
    partition.move_node(node, old_block)

    self.assertAlmostEqual(delta_value,
                           end_value - start_value,
                           msg="Difference in partition: " + str(partition_number) + "\n " + str(
                               move_number) + "th move " + "\n actual partition \t" + str(
                               partition.get_representation()) + "\n next partition \t" + str(
                               new_representation) + "\n move " + str(
                               move) + "\n precalc " + str(precalc))


def general_delta_test_new_block_creation(self, partition, normal_function, delta_function, partition_number,
                                          move_number, move=None):
    """Test move of single node into new block"""
    if move is None:
        move = partition.get_random_move()
    node, old_block, new_block = move
    new_block = partition.B
    move = node, old_block, new_block
    precalc = partition.precalc_move(move, None)

    # value before the move and before any delta calculation
    start_value = normal_function(partition)

    # calculate delta with given set of parameters
    delta_value = delta_function(partition, old_block, new_block, *precalc)

    # move node to calculate real value
    partition.move_node(node, new_block)
    end_value = normal_function(partition)
    new_representation = partition.get_representation()
    # move node back
    partition.move_node(node, old_block)

    self.assertAlmostEqual(delta_value,
                           end_value - start_value,
                           msg="Difference in partition: " + str(partition_number) + "\n " + str(
                               move_number) + "th move " + "\n actual partition \t" + str(
                               partition.get_representation()) + "\n next partition \t" + str(
                               new_representation) + "\n move " + str(
                               move) + "\n precalc " + str(precalc))


def general_delta_test_block_with_single_element_deletion(self, partition, normal_function, delta_function,
                                                          partition_number, move_number):
    """Test removal of block with single element"""
    move = partition.get_random_move()
    node, saved_old_block, block_for_merger = move
    new_block = partition.B

    # first create block
    # move node to calculate real value
    partition.move_node(node, new_block)
    representation_with_new_block = partition.get_representation()

    # value before the move and before any delta calculation
    start_value = normal_function(partition)

    # now check move back
    move = node, new_block, block_for_merger
    precalc = partition.precalc_move(move, None)
    # calculate delta with given set of parameters
    delta_value = delta_function(partition, new_block, block_for_merger, *precalc)
    new_representation = partition.get_representation()

    partition.move_node(node, block_for_merger)
    # the end value is the value from the beginning with the node back in its old block
    end_value = normal_function(partition)

    # move node back
    partition.move_node(node, saved_old_block)

    self.assertAlmostEqual(delta_value,
                           end_value - start_value,
                           msg="Difference in partition: " + str(partition_number) + "\n " + str(move_number)
                               + "th move " + "\n actual partition \t" + str(partition.get_representation())
                               + "\n next partition \t" + str(new_representation) + "\n partition with new block "
                               + str(representation_with_new_block) + "\n move " + str(move)
                               + "\n precalc " + str(precalc))


# noinspection PyProtectedMember
def general_delta_test_merge_blocks(self, partition, normal_function, delta_function, partition_number,
                                    move_number, peixoto_inference, from_block=None, to_block=None):
    """Test merger of two blocks"""
    if from_block is None:
        node = partition.get_random_node(no_single=False)
        from_block = partition.get_block_of_node(node)
    if to_block is None:
        possible_blocks = list(partition.get_possible_blocks(from_block))
        possible_blocks.remove(from_block)
        to_block = rd.choice(possible_blocks)

    parameter, reference = peixoto_inference._precalc_block_merge(from_block)
    peixoto_inference._adjustment_delta_pre(reference, from_block, to_block)

    # value before the move and before any delta calculation
    start_value = normal_function(partition)
    starting_partition = partition.get_representation()

    # calculate delta with given set of parameters
    delta_value = delta_function(partition, from_block, to_block, *parameter)

    # move node to calculate real value
    partition.merge_blocks({from_block: to_block}, partition.B - 1)
    end_value = normal_function(partition)
    new_representation = partition.get_representation()
    # set to starting point
    partition.set_from_representation(starting_partition)

    self.assertAlmostEqual(delta_value,
                           end_value - start_value,
                           msg="Difference in partition: " + str(partition_number) + "\n " + str(move_number)
                               + "th move " + "\n actual partition \t" + str(partition.get_representation())
                               + "\n next partition \t" + str(new_representation) + "\n merger "
                               + str({from_block: to_block}) + "\n precalc " + str(parameter))


class TestICLExactDirected(ut.TestCase):
    def setUp(self):
        self.partitions = []

        # ring graph with 4 nodes
        graph = nx.DiGraph()
        nx.add_path(graph, [0, 1, 2, 3, 0])
        self.partitions.append(NxPartitionGraphBased(graph, representation={node: node % 2 for node in graph}))
        # node counts 2, 2
        # edge matrix
        # 0 2
        # 2 0

        # Ring Graph with selfloop
        graph = nx.DiGraph()
        nx.add_path(graph, [0, 1, 2, 3, 0, 0])
        self.partitions.append(NxPartitionGraphBased(graph, representation={node: node % 2 for node in graph}))
        # node counts 2, 2
        # edge matrix
        # 1 2
        # 2 0

        # graph with 3 groups
        graph = nx.DiGraph()
        nx.add_path(graph, [0, 1, 2, 3, 0, 4, 5, 6, 4])
        self.partitions.append(NxPartitionGraphBased(graph, representation={0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}))
        # node counts 1, 3, 3
        # edge matrix
        # 0 1 1
        # 1 2 0
        # 0 0 3

        graph = nx.DiGraph(nx.karate_club_graph())
        self.partitions.append(NxPartitionGraphBased(graph, representation={node: node % 5 for node in graph}))

        self._replace_likelihood = TraditionalUnnormalizedLogLikelyhood(is_directed=True)

    def test_calculate_icl_ex_jeffrey_hyperprior_directed(self):
        # first term
        # 0->1 and 1->0
        # gamma(.5+.5)*gamma(.5+2)*gamma(.5+2)/(gamma(.5*2+4)*gamma(.5)*gamma(.5)) = 3/128
        # 0->0 and 1->1
        # gamma(.5+.5)*gamma(.5+0)*gamma(.5+4)/(gamma(.5*2+4)*gamma(.5)*gamma(.5)) = 35/128
        # second term
        # gamma(.5*2)*gamma(.5+2)*gamma(.5+2)/(gamma(.5*2+4)*gamma(.5)*gamma(.5)) = 3/128
        self.assertAlmostEqual(calculate_icl_ex_jeffrey_hyperprior_directed(self.partitions[0]),
                               2 * math.log(3 / 128) + 2 * math.log(35 / 128) + math.log(3 / 128), places=10)

        # first term
        # 0->1 and 1->0
        # gamma(.5+.5)*gamma(.5+2)*gamma(.5+2)/(gamma(.5*2+4)*gamma(.5)*gamma(.5)) = 3/128
        # 0->0
        # gamma(.5+.5)*gamma(.5+1)*gamma(.5+3)/(gamma(.5*2+4)*gamma(.5)*gamma(.5)) = 5/128
        # 1->1
        # gamma(.5+.5)*gamma(.5+0)*gamma(.5+4)/(gamma(.5*2+4)*gamma(.5)*gamma(.5)) = 35/128
        # second term
        # gamma(.5*2)*gamma(.5+2)*gamma(.5+2)/(gamma(.5*2+4)*gamma(.5)*gamma(.5)) = 3/128
        self.assertAlmostEqual(calculate_icl_ex_jeffrey_hyperprior_directed(self.partitions[1]),
                               2 * math.log(3 / 128) + math.log(5 / 128) + math.log(35 / 128) + math.log(3 / 128),
                               places=10)

        # first term
        # 0->1, 1->0, 0->2
        # gamma(.5+.5)*gamma(.5+1)*gamma(.5+2)/(gamma(.5*2+3)*gamma(.5)*gamma(.5)) = 1/16
        # 0->0
        # gamma(.5+.5)*gamma(.5+0)*gamma(.5+1)/(gamma(.5*2+1)*gamma(.5)*gamma(.5)) = 1/2
        # 1->1
        # gamma(.5+.5)*gamma(.5+2)*gamma(.5+7)/(gamma(.5*2+9)*gamma(.5)*gamma(.5)) = 143/65536
        # 1->2, 2->1
        # gamma(.5+.5)*gamma(.5+0)*gamma(.5+9)/(gamma(.5*2+9)*gamma(.5)*gamma(.5)) = 12155/65536
        # 2->0
        # gamma(.5+.5)*gamma(.5+0)*gamma(.5+3)/(gamma(.5*2+3)*gamma(.5)*gamma(.5)) = 5/16
        # 2->2
        # gamma(.5+.5)*gamma(.5+3)*gamma(.5+6)/(gamma(.5*2+9)*gamma(.5)*gamma(.5)) = 55/65536
        # second term
        # gamma(.5*3)*gamma(.5+1)*gamma(.5+3)*gamma(.5+3)/(gamma(.5*3+7)*gamma(.5)*gamma(.5)*gamma(.5)) = 1/9009
        self.assertEqual(calculate_icl_ex_jeffrey_hyperprior_directed(self.partitions[2]),
                         3 * math.log(1 / 16) + math.log(1 / 2) + math.log(143 / 65536)
                         + 2 * math.log(12155 / 65536) + math.log(5 / 16) + math.log(55 / 65536)
                         + math.log(1 / 9009)
                         )

    def test_calculate_icl_ex_uniform_hyperprior_directed(self):
        # first term
        # 0->1 and 1->0
        # gamma(1+1)*gamma(1+2)*gamma(1+2)/(gamma(1*2+4)*gamma(1)*gamma(1)) = 1/30
        # 0->0 and 1->1
        # gamma(1+1)*gamma(1+0)*gamma(1+4)/(gamma(1*2+4)*gamma(1)*gamma(1)) = 1/5
        # second term
        # gamma(1*2)*gamma(1+2)*gamma(1+2)/(gamma(1*2+4)*gamma(1)*gamma(1)) = 1/30
        self.assertAlmostEqual(calculate_icl_ex_uniform_hyperprior_directed(self.partitions[0]),
                               2 * math.log(1 / 30) + 2 * math.log(1 / 5) + math.log(1 / 30),
                               places=10)

        # first term
        # 0->1 and 1->0
        # gamma(1+1)*gamma(1+2)*gamma(1+2)/(gamma(1*2+4)*gamma(1)*gamma(1)) = 1/30
        # 0->0
        # gamma(1+1)*gamma(1+1)*gamma(1+3)/(gamma(1*2+4)*gamma(1)*gamma(1)) = 1/20
        # 1->1
        # gamma(1+1)*gamma(1+0)*gamma(1+4)/(gamma(1*2+4)*gamma(1)*gamma(1)) = 1/5
        # second term
        # gamma(1*2)*gamma(1+2)*gamma(1+2)/(gamma(1*2+4)*gamma(1)*gamma(1)) = 1/30
        self.assertAlmostEqual(calculate_icl_ex_uniform_hyperprior_directed(self.partitions[1]),
                               2 * math.log(1 / 30) + math.log(1 / 20) + math.log(1 / 5) + math.log(1 / 30),
                               places=10)

        # first term
        # 0->1, 1->0, 0->2
        # gamma(1+1)*gamma(1+1)*gamma(1+2)/(gamma(1*2+3)*gamma(1)*gamma(1)) = 1/12
        # 0->0
        # gamma(1+1)*gamma(1+0)*gamma(1+1)/(gamma(1*2+1)*gamma(1)*gamma(1)) = 1/2
        # 1->1
        # gamma(1+1)*gamma(1+2)*gamma(1+7)/(gamma(1*2+9)*gamma(1)*gamma(1)) = 1/360
        # 1->2, 2->1
        # gamma(1+1)*gamma(1+0)*gamma(1+9)/(gamma(1*2+9)*gamma(1)*gamma(1)) = 1/10
        # 2->0
        # gamma(1+1)*gamma(1+0)*gamma(1+3)/(gamma(1*2+3)*gamma(1)*gamma(1)) = 1/4
        # 2->2
        # gamma(1+1)*gamma(1+3)*gamma(1+6)/(gamma(1*2+9)*gamma(1)*gamma(1)) = 1/840
        # second term
        # gamma(1*3)*gamma(1+1)*gamma(1+3)*gamma(1+3)/(gamma(1*3+7)*gamma(1)*gamma(1)*gamma(1)) = 1/5040
        self.assertAlmostEqual(calculate_icl_ex_uniform_hyperprior_directed(self.partitions[2]),
                               3 * math.log(1 / 12) + math.log(1 / 2) + math.log(1 / 360)
                               + 2 * math.log(1 / 10) + math.log(1 / 4) + math.log(1 / 840)
                               + math.log(1 / 5040),
                               places=10
                               )

    def test_delta_calculate_icl_ex_jeffrey_hyperprior_directed(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_random_move(self, partition, calculate_icl_ex_jeffrey_hyperprior_directed,
                                               delta_calculate_icl_ex_jeffrey_hyperprior_directed, partition_counter,
                                               move_number)

    def test_delta_calculate_icl_ex_uniform_hyperprior_directed(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_random_move(self, partition, calculate_icl_ex_uniform_hyperprior_directed,
                                               delta_calculate_icl_ex_uniform_hyperprior_directed, partition_counter,
                                               move_number)

    def test_delta_calculate_icl_ex_jeffrey_hyperprior_directed_block_creation(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_new_block_creation(self, partition, calculate_icl_ex_jeffrey_hyperprior_directed,
                                                      delta_calculate_icl_ex_jeffrey_hyperprior_directed,
                                                      partition_counter,
                                                      move_number)

    def test_delta_calculate_icl_ex_uniform_hyperprior_directed_block_creation(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_new_block_creation(self, partition, calculate_icl_ex_uniform_hyperprior_directed,
                                                      delta_calculate_icl_ex_uniform_hyperprior_directed,
                                                      partition_counter,
                                                      move_number)

    def test_delta_calculate_icl_ex_jeffrey_hyperprior_directed_single_element_block_removal(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    calculate_icl_ex_jeffrey_hyperprior_directed,
                    delta_calculate_icl_ex_jeffrey_hyperprior_directed,
                    partition_counter,
                    move_number)

    def test_delta_calculate_icl_ex_uniform_hyperprior_directed_single_element_block_removal(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    calculate_icl_ex_uniform_hyperprior_directed,
                    delta_calculate_icl_ex_uniform_hyperprior_directed,
                    partition_counter,
                    move_number)

    def test_delta_calculate_icl_ex_jeffrey_hyperprior_directed_merge_blocks(self):
        for partition_counter, partition in enumerate(self.partitions):
            inference = PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                         no_partition_reset=True)
            for move_number in range(10):
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    calculate_icl_ex_jeffrey_hyperprior_directed,
                    delta_calculate_icl_ex_jeffrey_hyperprior_directed,
                    partition_counter,
                    move_number,
                    inference,
                )

    def test_delta_calculate_icl_ex_uniform_hyperprior_directed_merge_blocks(self):
        for partition_counter, partition in enumerate(self.partitions):
            inference = PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                         no_partition_reset=True)
            for move_number in range(10):
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    calculate_icl_ex_uniform_hyperprior_directed,
                    delta_calculate_icl_ex_uniform_hyperprior_directed,
                    partition_counter,
                    move_number,
                    inference,
                )

    def test_delta_calculate_icl_ex_jeffrey_hyperprior_directed_no_change_move(self):
        # try moving the node which is alone in its block to a new block
        for partition_counter, partition in enumerate(self.partitions):
            # create such a block with only one element
            node = partition.get_random_node(no_single=False)
            # by simply moving any node to a new block
            partition.move_node(node, partition.B)
            # move this node again to a new block
            move = (node, partition.get_block_of_node(node), partition.B)
            general_delta_test_random_move(self, partition, calculate_icl_ex_jeffrey_hyperprior_directed,
                                           delta_calculate_icl_ex_jeffrey_hyperprior_directed, partition_counter,
                                           move_number=0, move=move)

    def test_delta_calculate_icl_ex_uniform_hyperprior_directed_no_change_move(self):
        # try moving the node which is alone in its block to a new block
        for partition_counter, partition in enumerate(self.partitions):
            # create such a block with only one element
            node = partition.get_random_node(no_single=False)
            # by simply moving any node to a new block
            partition.move_node(node, partition.B)
            # move this node again to a new block
            move = (node, partition.get_block_of_node(node), partition.B)
            general_delta_test_random_move(self, partition, calculate_icl_ex_uniform_hyperprior_directed,
                                           delta_calculate_icl_ex_uniform_hyperprior_directed, partition_counter,
                                           move_number=0, move=move)

    def test_error_on_wrong_number_of_parameters(self):
        with self.assertRaises(ValueError):
            delta_calculate_icl_ex_jeffrey_hyperprior_directed(self.partitions[0], 0, 1, None)

        with self.assertRaises(ValueError):
            delta_calculate_icl_ex_uniform_hyperprior_directed(self.partitions[0], 0, 1, None)


class TestICLExactUndirected(ut.TestCase):
    def setUp(self):
        self.partitions = []

        # ring graph with 4 nodes
        graph = nx.Graph()
        nx.add_path(graph, [0, 1, 2, 3, 0])
        self.partitions.append(NxPartitionGraphBased(graph, representation={node: node % 2 for node in graph}))
        # node counts 2, 2
        # edge matrix
        # 0 4
        # 4 0

        # Ring Graph with selfloop
        graph = nx.Graph()
        nx.add_path(graph, [0, 1, 2, 3, 0, 0])
        self.partitions.append(NxPartitionGraphBased(graph, representation={node: node % 2 for node in graph}))
        # node counts 2, 2
        # edge matrix
        # 2 4
        # 4 0

        # graph with 3 groups
        graph = nx.Graph()
        nx.add_path(graph, [0, 1, 2, 3, 0, 4, 5, 6, 4])
        self.partitions.append(NxPartitionGraphBased(graph, representation={0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}))
        # node counts 1, 3, 3
        # edge matrix
        # 0 2 1
        # 2 4 0
        # 1 0 6

        # greater graphs
        graph = nx.karate_club_graph()
        self.partitions.append(NxPartitionGraphBased(graph, representation={node: node % 5 for node in graph}))

        self._replace_likelihood = TraditionalUnnormalizedLogLikelyhood(is_directed=False)

    def test_calculate_icl_ex_jeffrey_hyperprior_undirected(self):
        # first term
        # 0->1
        # gamma(.5+.5)*gamma(.5+4)*gamma(.5+0)/(gamma(.5*2+4)*gamma(.5)*gamma(.5)) = 35/128
        # 0->0 and 1->1
        # gamma(.5+.5)*gamma(.5+0)*gamma(.5+3)/(gamma(.5*2+3)*gamma(.5)*gamma(.5)) = 5/16
        # second term
        # gamma(.5*2)*gamma(.5+2)*gamma(.5+2)/(gamma(.5*2+4)*gamma(.5)*gamma(.5)) = 3/128
        self.assertAlmostEqual(calculate_icl_ex_jeffrey_hyperprior_undirected(self.partitions[0]),
                               math.log(35 / 128) + 2 * math.log(5 / 16) + math.log(3 / 128),
                               places=10)

        # first term
        # 0->1
        # gamma(.5+.5)*gamma(.5+4)*gamma(.5+0)/(gamma(.5*2+4)*gamma(.5)*gamma(.5)) = 35/128
        # 0->0
        # gamma(.5+.5)*gamma(.5+1)*gamma(.5+2)/(gamma(.5*2+3)*gamma(.5)*gamma(.5)) = 1/16
        # 1->1
        # gamma(.5+.5)*gamma(.5+0)*gamma(.5+3)/(gamma(.5*2+3)*gamma(.5)*gamma(.5)) = 5/16
        # second term
        # gamma(.5*2)*gamma(.5+2)*gamma(.5+2)/(gamma(.5*2+4)*gamma(.5)*gamma(.5)) = 3/128
        self.assertEqual(calculate_icl_ex_jeffrey_hyperprior_undirected(self.partitions[1]),
                         math.log(35 / 128) + math.log(1 / 16) + math.log(5 / 16) + math.log(3 / 128))

        # first term
        # 0->1
        # gamma(.5+.5)*gamma(.5+2)*gamma(.5+1)/(gamma(.5*2+3)*gamma(.5)*gamma(.5)) = 1/16
        # 0->2
        # gamma(.5+.5)*gamma(.5+1)*gamma(.5+2)/(gamma(.5*2+3)*gamma(.5)*gamma(.5)) = 1/16
        # 0->0
        # gamma(.5+.5)*gamma(.5+0)*gamma(.5+1)/(gamma(.5*2+1)*gamma(.5)*gamma(.5)) = 1/2
        # 1->2
        # gamma(.5+.5)*gamma(.5+0)*gamma(.5+9)/(gamma(.5*2+9)*gamma(.5)*gamma(.5)) = 12155/65536
        # 1->1
        # gamma(.5+.5)*gamma(.5+2)*gamma(.5+4)/(gamma(.5*2+6)*gamma(.5)*gamma(.5)) = 7/1024
        # 2->2
        # gamma(.5+.5)*gamma(.5+3)*gamma(.5+3)/(gamma(.5*2+6)*gamma(.5)*gamma(.5)) = 5/1024
        # second term
        # gamma(.5*3)*gamma(.5+1)*gamma(.5+3)*gamma(.5+3)/(gamma(.5*3+7)*gamma(.5)*gamma(.5)*gamma(.5)) = 1/9009
        self.assertAlmostEqual(calculate_icl_ex_jeffrey_hyperprior_undirected(self.partitions[2]),
                               2 * math.log(1 / 16) + math.log(1 / 2) + math.log(12155 / 65536) + math.log(7 / 1024)
                               + math.log(5 / 1024)
                               + math.log(1 / 9009),
                               places=10
                               )

    def test_calculate_icl_ex_uniform_hyperprior_undirected(self):
        # first term
        # 0->1
        # gamma(1+1)*gamma(1+4)*gamma(1+0)/(gamma(1*2+4)*gamma(1)*gamma(1)) = 1/5
        # 0->0 and 1->1
        # gamma(1+1)*gamma(1+0)*gamma(1+3)/(gamma(1*2+3)*gamma(1)*gamma(1)) = 1/4
        # second term
        # gamma(1*2)*gamma(1+2)*gamma(1+2)/(gamma(1*2+4)*gamma(1)*gamma(1)) = 1/30
        self.assertEqual(calculate_icl_ex_uniform_hyperprior_undirected(self.partitions[0]),
                         math.log(1 / 5) + 2 * math.log(1 / 4) + math.log(1 / 30))

        # first term
        # 0->1
        # gamma(1+1)*gamma(1+4)*gamma(1+0)/(gamma(1*2+4)*gamma(1)*gamma(1)) = 1/5
        # 0->0
        # gamma(1+1)*gamma(1+1)*gamma(1+2)/(gamma(1*2+3)*gamma(1)*gamma(1)) = 1/12
        # 1->1
        # gamma(1+1)*gamma(1+0)*gamma(1+3)/(gamma(1*2+3)*gamma(1)*gamma(1)) = 1/4
        # second term
        # gamma(1*2)*gamma(1+2)*gamma(1+2)/(gamma(1*2+4)*gamma(1)*gamma(1)) = 1/30
        self.assertAlmostEqual(calculate_icl_ex_uniform_hyperprior_undirected(self.partitions[1]),
                               math.log(1 / 5) + math.log(1 / 12) + math.log(1 / 4) + math.log(1 / 30),
                               places=10)

        # first term
        # 0->1
        # gamma(1+1)*gamma(1+2)*gamma(1+1)/(gamma(1*2+3)*gamma(1)*gamma(1)) = 1/12
        # 0->2
        # gamma(1+1)*gamma(1+1)*gamma(1+2)/(gamma(1*2+3)*gamma(1)*gamma(1)) = 1/12
        # 0->0
        # gamma(1+1)*gamma(1+0)*gamma(1+1)/(gamma(1*2+1)*gamma(1)*gamma(1)) = 1/2
        # 1->2
        # gamma(1+1)*gamma(1+0)*gamma(1+9)/(gamma(1*2+9)*gamma(1)*gamma(1)) = 1/10
        # 1->1
        # gamma(1+1)*gamma(1+2)*gamma(1+4)/(gamma(1*2+6)*gamma(1)*gamma(1)) = 1/105
        # 2->2
        # gamma(1+1)*gamma(1+3)*gamma(1+3)/(gamma(1*2+6)*gamma(1)*gamma(1)) = 1/140
        # second term
        # gamma(1*3)*gamma(1+1)*gamma(1+3)*gamma(1+3)/(gamma(1*3+7)*gamma(1)*gamma(1)*gamma(1)) = 1/5040
        self.assertAlmostEqual(calculate_icl_ex_uniform_hyperprior_undirected(self.partitions[2]),
                               2 * math.log(1 / 12) + math.log(1 / 2) + math.log(1 / 10) + math.log(1 / 105)
                               + math.log(1 / 140)
                               + math.log(1 / 5040),
                               places=10
                               )

    def test_delta_calculate_icl_ex_jeffrey_hyperprior_undirected(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_random_move(self, partition, calculate_icl_ex_jeffrey_hyperprior_undirected,
                                               delta_calculate_icl_ex_jeffrey_hyperprior_undirected, partition_counter,
                                               move_number)

    def test_delta_calculate_icl_ex_uniform_hyperprior_undirected(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_random_move(self, partition, calculate_icl_ex_uniform_hyperprior_undirected,
                                               delta_calculate_icl_ex_uniform_hyperprior_undirected, partition_counter,
                                               move_number)

    def test_delta_calculate_icl_ex_jeffrey_hyperprior_undirected_block_creation(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_new_block_creation(self, partition, calculate_icl_ex_jeffrey_hyperprior_undirected,
                                                      delta_calculate_icl_ex_jeffrey_hyperprior_undirected,
                                                      partition_counter,
                                                      move_number)

    def test_delta_calculate_icl_ex_uniform_hyperprior_undirected_block_creation(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_new_block_creation(self, partition, calculate_icl_ex_uniform_hyperprior_undirected,
                                                      delta_calculate_icl_ex_uniform_hyperprior_undirected,
                                                      partition_counter,
                                                      move_number)

    def test_delta_calculate_icl_ex_jeffrey_hyperprior_undirected_single_element_block_removal(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    calculate_icl_ex_jeffrey_hyperprior_undirected,
                    delta_calculate_icl_ex_jeffrey_hyperprior_undirected,
                    partition_counter,
                    move_number)

    def test_delta_calculate_icl_ex_uniform_hyperprior_undirected_single_element_block_removal(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    calculate_icl_ex_uniform_hyperprior_undirected,
                    delta_calculate_icl_ex_uniform_hyperprior_undirected,
                    partition_counter,
                    move_number)

    def test_delta_calculate_icl_ex_jeffrey_hyperprior_undirected_merge_blocks(self):
        for partition_counter, partition in enumerate(self.partitions):
            inference = PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                         no_partition_reset=True)
            for move_number in range(10):
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    calculate_icl_ex_jeffrey_hyperprior_undirected,
                    delta_calculate_icl_ex_jeffrey_hyperprior_undirected,
                    partition_counter,
                    move_number,
                    inference,
                )

    def test_delta_calculate_icl_ex_uniform_hyperprior_undirected_merge_blocks(self):
        for partition_counter, partition in enumerate(self.partitions):
            inference = PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                         no_partition_reset=True)
            for move_number in range(10):
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    calculate_icl_ex_uniform_hyperprior_undirected,
                    delta_calculate_icl_ex_uniform_hyperprior_undirected,
                    partition_counter,
                    move_number,
                    inference,
                )

    def test_delta_calculate_icl_ex_jeffrey_hyperprior_undirected_no_change_move(self):
        # try moving the node which is alone in its block to a new block
        for partition_counter, partition in enumerate(self.partitions):
            # create such a block with only one element
            node = partition.get_random_node(no_single=False)
            # by simply moving any node to a new block
            partition.move_node(node, partition.B)
            # move this node again to a new block
            move = (node, partition.get_block_of_node(node), partition.B)
            general_delta_test_random_move(self, partition, calculate_icl_ex_jeffrey_hyperprior_undirected,
                                           delta_calculate_icl_ex_jeffrey_hyperprior_undirected, partition_counter,
                                           move_number=0, move=move)

    def test_delta_calculate_icl_ex_uniform_hyperprior_undirected_no_change_move(self):
        # try moving the node which is alone in its block to a new block
        for partition_counter, partition in enumerate(self.partitions):
            # create such a block with only one element
            node = partition.get_random_node(no_single=False)
            # by simply moving any node to a new block
            partition.move_node(node, partition.B)
            # move this node again to a new block
            move = (node, partition.get_block_of_node(node), partition.B)
            general_delta_test_random_move(self, partition, calculate_icl_ex_uniform_hyperprior_undirected,
                                           delta_calculate_icl_ex_uniform_hyperprior_undirected, partition_counter,
                                           move_number=0, move=move)

    def test_error_on_wrong_number_of_parameters(self):
        with self.assertRaises(ValueError):
            delta_calculate_icl_ex_jeffrey_hyperprior_undirected(self.partitions[0], 0, 1, None)

        with self.assertRaises(ValueError):
            delta_calculate_icl_ex_uniform_hyperprior_undirected(self.partitions[0], 0, 1, None)


class TestIntegratedCompleteLikelihoodExact(ut.TestCase):

    def setUp(self):
        self.partitions = []

        graph = nx.Graph()
        nx.add_star(graph, range(20))
        nx.add_path(graph, range(10))
        self.partitions.append(NxPartitionGraphBased(graph, representation={node: node % 6 for node in graph}))

        graph = nx.DiGraph()
        nx.add_star(graph, range(20))
        nx.add_path(graph, range(10))
        self.partitions.append(NxPartitionGraphBased(graph, representation={node: node % 6 for node in graph}))

        self.uniform_icl_ex = IntegratedCompleteLikelihoodExact(
            is_directed=False, hyperprior=IntegratedCompleteLikelihoodExact.UNIFORM_HYPERPRIOR)

        self.jeffrey_icl_ex = IntegratedCompleteLikelihoodExact(
            is_directed=False, hyperprior=IntegratedCompleteLikelihoodExact.JEFFREY_HYPERPRIOR)

    def test_complete(self):
        for partition in self.partitions:
            is_directed = partition.is_graph_directed()

            self.uniform_icl_ex.is_directed = is_directed
            self.jeffrey_icl_ex.is_directed = is_directed

            if is_directed:
                self.assertEqual(self.jeffrey_icl_ex.calculate(partition),
                                 calculate_icl_ex_jeffrey_hyperprior_directed(partition))

                self.assertEqual(self.uniform_icl_ex.calculate(partition),
                                 calculate_icl_ex_uniform_hyperprior_directed(partition))
            else:
                self.assertEqual(self.jeffrey_icl_ex.calculate(partition),
                                 calculate_icl_ex_jeffrey_hyperprior_undirected(partition))

                self.assertEqual(self.uniform_icl_ex.calculate(partition),
                                 calculate_icl_ex_uniform_hyperprior_undirected(partition))

    def test_delta_uniform(self):
        for partition_counter, partition in enumerate(self.partitions):

            # uniform
            self.uniform_icl_ex.is_directed = partition.is_graph_directed()
            inference = PeixotoInference(partition.graph, self.uniform_icl_ex, partition,
                                         no_partition_reset=True)
            for move_number in range(5):
                # normal moves
                general_delta_test_random_move(
                    self,
                    partition,
                    self.uniform_icl_ex.calculate,
                    self.uniform_icl_ex.calculate_delta,
                    partition_counter,
                    move_number)

                # block removals
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    self.uniform_icl_ex.calculate,
                    self.uniform_icl_ex.calculate_delta,
                    partition_counter,
                    move_number)

                # test block merge
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    self.uniform_icl_ex.calculate,
                    self.uniform_icl_ex.calculate_delta,
                    partition_counter,
                    move_number,
                    inference,
                )

    def test_delta_jeffrey(self):
        for partition_counter, partition in enumerate(self.partitions):

            self.jeffrey_icl_ex.is_directed = partition.is_graph_directed()
            inference = PeixotoInference(partition.graph, self.jeffrey_icl_ex, partition,
                                         no_partition_reset=True)
            for move_number in range(5):
                # normal moves
                general_delta_test_random_move(
                    self,
                    partition,
                    self.jeffrey_icl_ex.calculate,
                    self.jeffrey_icl_ex.calculate_delta,
                    partition_counter,
                    move_number)

                # block removals
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    self.jeffrey_icl_ex.calculate,
                    self.jeffrey_icl_ex.calculate_delta,
                    partition_counter,
                    move_number)

                # test block merge
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    self.jeffrey_icl_ex.calculate,
                    self.jeffrey_icl_ex.calculate_delta,
                    partition_counter,
                    move_number,
                    inference,
                )

    def test_default_creation(self):
        likelihood = IntegratedCompleteLikelihoodExact(is_directed=False)
        self.assertEqual(likelihood._hyperprior, IntegratedCompleteLikelihoodExact.JEFFREY_HYPERPRIOR)
