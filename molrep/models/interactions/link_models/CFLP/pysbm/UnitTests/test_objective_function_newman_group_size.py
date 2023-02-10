import math
import random as rd
import unittest as ut

import networkx as nx

from pysbm.sbm import PeixotoInference
from pysbm.sbm.nxpartitiongraphbased import NxPartitionGraphBased
from pysbm.sbm.partition import NxPartition
from pysbm.sbm.objective_function_newman_group_size import *
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
    old_representation = partition.get_representation()
    partition.move_node(node, new_block)
    end_value = normal_function(partition)
    new_representation = partition.get_representation()
    # move node back
    partition.move_node(node, old_block)

    self.assertAlmostEqual(delta_value,
                           end_value - start_value,
                           msg="Difference in partition: " + str(partition_number) + "\n " + str(
                               move_number) + "th move " + "\n actual partition \t" + str(
                               old_representation) + "\n next partition \t" + str(
                               new_representation) + "\n move " + str(
                               move) + "\n precalc " + str(precalc))


def general_delta_test_new_block_creation(self, partition, normal_function, delta_function, partition_number,
                                          move_number):
    """Test move of single node into new block"""
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


def general_delta_empty_block(self, partition, normal_function, delta_function, partition_number):
    """Test creation of an empty block or deletion of one depending on partition"""
    block_to_be_removed = rd.choice(range(partition.B))
    to_block = (block_to_be_removed + 1) % partition.B

    nodes_to_be_moved = []
    old_representation = partition.get_representation()
    for node in old_representation:
        if old_representation[node] == block_to_be_removed:
            nodes_to_be_moved.append(node)

    for node in nodes_to_be_moved:
        move = node, block_to_be_removed, to_block
        parameter = partition.precalc_move(move, None)
        # value before the move and before any delta calculation
        start_value = normal_function(partition)

        # calculate delta with given set of parameters
        delta_value = delta_function(partition, block_to_be_removed, to_block, *parameter)

        # move node to calculate real value
        old_representation = partition.get_representation()
        partition.move_node(node, to_block)
        end_value = normal_function(partition)
        new_representation = partition.get_representation()

        self.assertAlmostEqual(delta_value,
                               end_value - start_value,
                               msg="Difference in partition: " + str(partition_number)
                                   + "\n actual partition \t" + str(old_representation)
                                   + "\n next partition \t" + str(new_representation) + "\n move "
                                   + str(move) + "\n precalc " + str(parameter))


class TestNewmanGroupSizeNonDegreeCorrectedUndirected(ut.TestCase):
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

    def test_newman_reinert_non_degree_corrected_undirected(self):
        # p = 2*4/4^2 = .5
        # P = 1/4*(2-1)!/(4+2-1)!*2!*2!     *4!/(.5*2*2 +1)^(4+1)*0!/(.5*.5*2^2 +1)^(0+1)*0!/(.5*.5*2^2 +1)^(0+1)

        self.assertEqual(newman_reinert_non_degree_corrected_undirected(self.partitions[0]),
                         1 / 4860)

        # p = 2*5/16 = 5/8
        # P = 1/4*(2-1)!/(4+2-1)!*2!*2!     *4!/(5/8*2*2 +1)^(4+1)*2!/(.5*5/8*2^2 +1)^(2+1)*0!/(.5*5/8*2^2 +1)^(0+1)
        self.assertAlmostEqual(newman_reinert_non_degree_corrected_undirected(self.partitions[1]),
                               0.000029715955350507483277951001447555523960588379906119599,
                               delta=.000000000001)

        # p = 2*8/49=16/49
        # P = 1/7*(3-1)!/(7+3-1)!*1!*3!*3!
        # *0!/(.5*16/49*1^2 + 1)^(0+1)*2!/(16/49*1*3+1)^(2+1)*1!/(16/49*1*3+1)^(1+1)
        # *4!/(.5*16/49*3^2+1)^(4+1)*0!/(16/49*3*3+1)^(0+1)*6!/(.5*16/49*3^2+1)^(6+1)
        self.assertAlmostEqual(newman_reinert_non_degree_corrected_undirected(self.partitions[2]),
                               0.00000013679033025618698548528595053317662634807398606783953,
                               delta=.000000000001)

    def test_log_newman_reinert_non_degree_corrected_undirected(self):
        for partition in self.partitions:
            self.assertAlmostEqual(math.log(newman_reinert_non_degree_corrected_undirected(partition)),
                                   log_newman_reinert_non_degree_corrected_undirected(partition),
                                   delta=.000000000001)

    def test_delta_log_newman_reinert_non_degree_corrected_undirected(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_random_move(self, partition, log_newman_reinert_non_degree_corrected_undirected,
                                               delta_log_newman_reinert_non_degree_corrected_undirected,
                                               partition_counter,
                                               move_number)

    def test_delta_log_newman_reinert_non_degree_corrected_undirected_block_creation(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_new_block_creation(self, partition,
                                                      log_newman_reinert_non_degree_corrected_undirected,
                                                      delta_log_newman_reinert_non_degree_corrected_undirected,
                                                      partition_counter,
                                                      move_number)

    def test_delta_log_newman_reinert_non_degree_corrected_undirected_single_element_block_removal(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    log_newman_reinert_non_degree_corrected_undirected,
                    delta_log_newman_reinert_non_degree_corrected_undirected,
                    partition_counter,
                    move_number)

    def test_delta_log_newman_reinert_non_degree_corrected_undirected_merge_blocks(self):
        for partition_counter, partition in enumerate(self.partitions):
            inference = PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                         no_partition_reset=True)
            for move_number in range(10):
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    log_newman_reinert_non_degree_corrected_undirected,
                    delta_log_newman_reinert_non_degree_corrected_undirected,
                    partition_counter,
                    move_number,
                    inference,
                )

    def test_delta_log_newman_reinert_non_degree_corrected_undirected_no_change_move(self):
        # try moving the node which is alone in its block to a new block
        for partition_counter, partition in enumerate(self.partitions):
            # create such a block with only one element
            node = partition.get_random_node(no_single=False)
            # by simply moving any node to a new block
            partition.move_node(node, partition.B)
            # move this node again to a new block
            move = (node, partition.get_block_of_node(node), partition.B)
            general_delta_test_random_move(self, partition, log_newman_reinert_non_degree_corrected_undirected,
                                           delta_log_newman_reinert_non_degree_corrected_undirected, partition_counter,
                                           move_number=0, move=move)

    def test_error_on_wrong_number_of_parameters(self):
        with self.assertRaises(ValueError):
            delta_log_newman_reinert_non_degree_corrected_undirected(self.partitions[0], 0, 1, None)

    def test_general_delta_without_empty_block(self):
        for partition_counter, partition in enumerate(self.partitions):
            general_delta_empty_block(self,
                                      partition,
                                      log_newman_reinert_non_degree_corrected_undirected,
                                      delta_log_newman_reinert_non_degree_corrected_undirected,
                                      partition_counter)

    def test_general_delta_with_empty_block(self):
        for partition_counter, partition in enumerate(self.partitions):
            partition_with_empty_block = NxPartition(partition.graph, representation=partition.get_representation())
            general_delta_empty_block(self,
                                      partition_with_empty_block,
                                      log_newman_reinert_non_degree_corrected_undirected,
                                      delta_log_newman_reinert_non_degree_corrected_undirected,
                                      partition_counter)



class TestNewmanGroupSizeNonDegreeCorrectedDirected(ut.TestCase):
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

    def test_newman_reinert_non_degree_corrected_directed(self):
        # p = 4/4^2 = .25
        # P = 1/4*(2-1)!/(4+2-1)!*2!*2!
        #  *2!/(.25*2*2 +1)^(2+1)*0!/(.25*2^2 +1)^(0+1)*2!/(.25*2*2 +1)^(2+1)*0!/(.25*2^2 +1)^(0+1)

        self.assertEqual(newman_reinert_non_degree_corrected_directed(self.partitions[0]),
                         1 / 7680)

        # p = 5/16 = 5/16
        # P = 1/4*(2-1)!/(4+2-1)!*2!*2!
        # *2!/(5/16*2*2 +1)^(2+1)*1!/(5/16*2^2 +1)^(1+1)*2!/(5/16*2*2 +1)^(2+1)*0!/(5/16*2^2 +1)^(0+1)
        self.assertAlmostEqual(newman_reinert_non_degree_corrected_directed(self.partitions[1]),
                               131072 / 5811307335,
                               delta=.000000000001)

        # p = 8/49
        # P = 1/7*(3-1)!/(7+3-1)!*1!*3!*3!
        # *0!/(8/49*1*1+1)^(0+1)*1!/(8/49*1*3+1)^(1+1)*1!/(8/49*1*3+1)^(1+1)*1!/(8/49*1*3+1)^(1+1)
        # *2!/(8/49*3*3+1)^(2+1)*0!/(8/49*3*3+1)^(0+1)*0!/(8/49*1*3+1)^(0+1)*0!/(8/49*3*3+1)^(0+1)
        # *3!/(8/49*3*3+1)^(3+1)
        self.assertAlmostEqual(newman_reinert_non_degree_corrected_directed(self.partitions[2]),
                               1104427674243920646305299201 / 210065368250185402761297564976886940,
                               delta=.000000000001)

    def test_log_newman_reinert_non_degree_corrected_directed(self):
        for partition in self.partitions:
            self.assertAlmostEqual(math.log(newman_reinert_non_degree_corrected_directed(partition)),
                                   log_newman_reinert_non_degree_corrected_directed(partition),
                                   delta=.000000000001)

    def test_delta_log_newman_reinert_non_degree_corrected_directed(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_random_move(self, partition, log_newman_reinert_non_degree_corrected_directed,
                                               delta_log_newman_reinert_non_degree_corrected_directed,
                                               partition_counter,
                                               move_number)

    def test_delta_log_newman_reinert_non_degree_corrected_directed_block_creation(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_new_block_creation(self, partition,
                                                      log_newman_reinert_non_degree_corrected_directed,
                                                      delta_log_newman_reinert_non_degree_corrected_directed,
                                                      partition_counter,
                                                      move_number)

    def test_delta_log_newman_reinert_non_degree_corrected_directed_single_element_block_removal(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    log_newman_reinert_non_degree_corrected_directed,
                    delta_log_newman_reinert_non_degree_corrected_directed,
                    partition_counter,
                    move_number)

    def test_delta_log_newman_reinert_non_degree_corrected_directed_merge_blocks(self):
        for partition_counter, partition in enumerate(self.partitions):
            inference = PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                         no_partition_reset=True)
            for move_number in range(10):
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    log_newman_reinert_non_degree_corrected_directed,
                    delta_log_newman_reinert_non_degree_corrected_directed,
                    partition_counter,
                    move_number,
                    inference,
                )

    def test_delta_log_newman_reinert_non_degree_corrected_directed_no_change_move(self):
        # try moving the node which is alone in its block to a new block
        for partition_counter, partition in enumerate(self.partitions):
            # create such a block with only one element
            node = partition.get_random_node(no_single=False)
            # by simply moving any node to a new block
            partition.move_node(node, partition.B)
            # move this node again to a new block
            move = (node, partition.get_block_of_node(node), partition.B)
            general_delta_test_random_move(self, partition, log_newman_reinert_non_degree_corrected_directed,
                                           delta_log_newman_reinert_non_degree_corrected_directed, partition_counter,
                                           move_number=0, move=move)

    def test_error_on_wrong_number_of_parameters(self):
        with self.assertRaises(ValueError):
            delta_log_newman_reinert_non_degree_corrected_directed(self.partitions[0], 0, 1, None)


class TestNewmanGroupSizeDegreeCorrectedUndirected(ut.TestCase):
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

    def test_newman_reinert_degree_corrected_undirected(self):
        # p = 2*4/4^2 = .5
        # P = 1/4*(2-1)!/(4+2-1)!*2!*2!     *4!/(.5*2*2 +1)^(4+1)*0!/(.5*.5*2^2 +1)^(0+1)*0!/(.5*.5*2^2 +1)^(0+1)
        # degree correction Term:
        # *2^4*(2-1)!/(2+4-1)!*2^4*(2-1)!/(2+4-1)!
        self.assertEqual(newman_reinert_degree_corrected_undirected(self.partitions[0]),
                         1 / 273375)

        # p = 2*5/16 = 5/8
        # P = 1/4*(2-1)!/(4+2-1)!*2!*2!     *4!/(5/8*2*2 +1)^(4+1)*2!/(.5*5/8*2^2 +1)^(2+1)*0!/(.5*5/8*2^2 +1)^(0+1)
        # degree correction Term:
        # *2^6*(2-1)!/(2+6-1)!*2^4*(2-1)!/(2+4-1)!
        self.assertAlmostEqual(newman_reinert_degree_corrected_undirected(self.partitions[1]),
                               0.000000050312728635779865867430267001152209880361278147927363,
                               delta=.000000000001)

        # p = 2*8/49=16/49
        # P = 1/7*(3-1)!/(7+3-1)!*1!*3!*3!
        # *0!/(.5*16/49*1^2 + 1)^(0+1)*2!/(16/49*1*3+1)^(2+1)*1!/(16/49*1*3+1)^(1+1)
        # *4!/(.5*16/49*3^2+1)^(4+1)*0!/(16/49*3*3+1)^(0+1)*6!/(.5*16/49*3^2+1)^(6+1)
        # degree correction Term:
        # *1^3*(1-1)!/(1+3-1)!*3^6*(3-1)!/(3+6-1)!*3^7*(3-1)!/(3+7-1)!
        self.assertAlmostEqual(newman_reinert_degree_corrected_undirected(self.partitions[2]),
                               0.0000000000099370367064692594486182096957395727646430500481759224,
                               delta=.000000000001)

    def test_log_newman_reinert_degree_corrected_undirected(self):
        for partition in self.partitions:
            self.assertAlmostEqual(math.log(newman_reinert_degree_corrected_undirected(partition)),
                                   log_newman_reinert_degree_corrected_undirected(partition),
                                   delta=.000000000001)

    def test_delta_log_newman_reinert_degree_corrected_undirected(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_random_move(self, partition, log_newman_reinert_degree_corrected_undirected,
                                               delta_log_newman_reinert_degree_corrected_undirected,
                                               partition_counter,
                                               move_number)

    def test_delta_log_newman_reinert_degree_corrected_undirected_block_creation(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_new_block_creation(self, partition,
                                                      log_newman_reinert_degree_corrected_undirected,
                                                      delta_log_newman_reinert_degree_corrected_undirected,
                                                      partition_counter,
                                                      move_number)

    def test_delta_log_newman_reinert_degree_corrected_undirected_single_element_block_removal(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    log_newman_reinert_degree_corrected_undirected,
                    delta_log_newman_reinert_degree_corrected_undirected,
                    partition_counter,
                    move_number)

    def test_delta_log_newman_reinert_degree_corrected_undirected_merge_blocks(self):
        for partition_counter, partition in enumerate(self.partitions):
            inference = PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                         no_partition_reset=True)
            for move_number in range(10):
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    log_newman_reinert_degree_corrected_undirected,
                    delta_log_newman_reinert_degree_corrected_undirected,
                    partition_counter,
                    move_number,
                    inference,
                )

    def test_delta_log_newman_reinert_degree_corrected_undirected_no_change_move(self):
        # try moving the node which is alone in its block to a new block
        for partition_counter, partition in enumerate(self.partitions):
            # create such a block with only one element
            node = partition.get_random_node(no_single=False)
            # by simply moving any node to a new block
            partition.move_node(node, partition.B)
            # move this node again to a new block
            move = (node, partition.get_block_of_node(node), partition.B)
            general_delta_test_random_move(self, partition, log_newman_reinert_degree_corrected_undirected,
                                           delta_log_newman_reinert_degree_corrected_undirected, partition_counter,
                                           move_number=0, move=move)

    def test_error_on_wrong_number_of_parameters(self):
        with self.assertRaises(ValueError):
            delta_log_newman_reinert_degree_corrected_undirected(self.partitions[0], 0, 1, None)


class TestNewmanGroupSizeDegreeCorrectedDirected(ut.TestCase):
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

    def test_newman_reinert_degree_corrected_directed(self):
        # p = 4/4^2 = .25
        # P = 1/4*(2-1)!/(4+2-1)!*2!*2!
        #  *2!/(.25*2*2 +1)^(2+1)*0!/(.25*2^2 +1)^(0+1)*2!/(.25*2*2 +1)^(2+1)*0!/(.25*2^2 +1)^(0+1)
        # degree correction Term:
        # *2^4*(2-1)!/(2+4-1)!*2^4*(2-1)!/(2+4-1)!
        self.assertEqual(newman_reinert_degree_corrected_directed(self.partitions[0]),
                         1 / 432000)

        # p = 5/16 = 5/16
        # P = 1/4*(2-1)!/(4+2-1)!*2!*2!
        # *2!/(5/16*2*2 +1)^(2+1)*1!/(5/16*2^2 +1)^(1+1)*2!/(5/16*2*2 +1)^(2+1)*0!/(5/16*2^2 +1)^(0+1)
        # degree correction Term:
        # *2^6*(2-1)!/(2+6-1)!*2^4*(2-1)!/(2+4-1)!
        self.assertAlmostEqual(newman_reinert_degree_corrected_directed(self.partitions[1]),
                               1048576 / 27458427157875,
                               delta=.000000000001)

        # p = 8/49
        # P = 1/7*(3-1)!/(7+3-1)!*1!*3!*3!
        # *0!/(8/49*1*1+1)^(0+1)*1!/(8/49*1*3+1)^(1+1)*1!/(8/49*1*3+1)^(1+1)*1!/(8/49*1*3+1)^(1+1)
        # *2!/(8/49*3*3+1)^(2+1)*0!/(8/49*3*3+1)^(0+1)*0!/(8/49*1*3+1)^(0+1)*0!/(8/49*3*3+1)^(0+1)
        # *3!/(8/49*3*3+1)^(3+1)
        # degree correction Term:
        # *1^3*(1-1)!/(1+3-1)!*3^6*(3-1)!/(3+6-1)!*3^7*(3-1)!/(3+7-1)!
        self.assertAlmostEqual(newman_reinert_degree_corrected_directed(self.partitions[2]),
                               0.00000000000038193047826203247092278647060938430548855826483198565,
                               delta=.000000000001)

    def test_log_newman_reinert_degree_corrected_directed(self):
        for partition in self.partitions:
            self.assertAlmostEqual(math.log(newman_reinert_degree_corrected_directed(partition)),
                                   log_newman_reinert_degree_corrected_directed(partition),
                                   delta=.000000001)

    def test_delta_log_newman_reinert_degree_corrected_directed(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_random_move(self, partition, log_newman_reinert_degree_corrected_directed,
                                               delta_log_newman_reinert_degree_corrected_directed,
                                               partition_counter,
                                               move_number)

    def test_delta_log_newman_reinert_degree_corrected_directed_block_creation(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_new_block_creation(self, partition,
                                                      log_newman_reinert_degree_corrected_directed,
                                                      delta_log_newman_reinert_degree_corrected_directed,
                                                      partition_counter,
                                                      move_number)

    def test_delta_log_newman_reinert_degree_corrected_directed_single_element_block_removal(self):
        for partition_counter, partition in enumerate(self.partitions):
            for move_number in range(10):
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    log_newman_reinert_degree_corrected_directed,
                    delta_log_newman_reinert_degree_corrected_directed,
                    partition_counter,
                    move_number)

    def test_delta_log_newman_reinert_degree_corrected_directed_merge_blocks(self):
        for partition_counter, partition in enumerate(self.partitions):
            inference = PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                         no_partition_reset=True)
            for move_number in range(10):
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    log_newman_reinert_degree_corrected_directed,
                    delta_log_newman_reinert_degree_corrected_directed,
                    partition_counter,
                    move_number,
                    inference,
                )

    def test_delta_log_newman_reinert_degree_corrected_directed_no_change_move(self):
        # try moving the node which is alone in its block to a new block
        for partition_counter, partition in enumerate(self.partitions):
            # create such a block with only one element
            node = partition.get_random_node(no_single=False)
            # by simply moving any node to a new block
            partition.move_node(node, partition.B)
            # move this node again to a new block
            move = (node, partition.get_block_of_node(node), partition.B)
            general_delta_test_random_move(self, partition, log_newman_reinert_degree_corrected_directed,
                                           delta_log_newman_reinert_degree_corrected_directed, partition_counter,
                                           move_number=0, move=move)

    def test_error_on_wrong_number_of_parameters(self):
        with self.assertRaises(ValueError):
            delta_log_newman_reinert_degree_corrected_directed(self.partitions[0], 0, 1, None)


class TestNewmanReinertObjectiveFunctions(ut.TestCase):

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

        self.non_degree_corrected = NewmanReinertNonDegreeCorrected(is_directed=False)

        self.degree_corrected = NewmanReinertDegreeCorrected(is_directed=False)

    def test_complete(self):
        for partition in self.partitions:
            is_directed = partition.is_graph_directed()

            self.non_degree_corrected.is_directed = is_directed
            self.degree_corrected.is_directed = is_directed

            if is_directed:
                self.assertEqual(self.degree_corrected.calculate(partition),
                                 log_newman_reinert_degree_corrected_directed(partition))

                self.assertEqual(self.non_degree_corrected.calculate(partition),
                                 log_newman_reinert_non_degree_corrected_directed(partition))
            else:
                self.assertEqual(self.degree_corrected.calculate(partition),
                                 log_newman_reinert_degree_corrected_undirected(partition))

                self.assertEqual(self.non_degree_corrected.calculate(partition),
                                 log_newman_reinert_non_degree_corrected_undirected(partition))

    def test_delta_non_degree_corrected(self):
        for partition_counter, partition in enumerate(self.partitions):

            # uniform
            self.non_degree_corrected.is_directed = partition.is_graph_directed()
            inference = PeixotoInference(partition.graph, self.non_degree_corrected, partition,
                                         no_partition_reset=True)
            for move_number in range(5):
                # normal moves
                general_delta_test_random_move(
                    self,
                    partition,
                    self.non_degree_corrected.calculate,
                    self.non_degree_corrected.calculate_delta,
                    partition_counter,
                    move_number)

                # block removals
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    self.non_degree_corrected.calculate,
                    self.non_degree_corrected.calculate_delta,
                    partition_counter,
                    move_number)

                # test block merge
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    self.non_degree_corrected.calculate,
                    self.non_degree_corrected.calculate_delta,
                    partition_counter,
                    move_number,
                    inference,
                )

    def test_delta_degree_corrected(self):
        for partition_counter, partition in enumerate(self.partitions):

            self.degree_corrected.is_directed = partition.is_graph_directed()
            inference = PeixotoInference(partition.graph, self.degree_corrected, partition,
                                         no_partition_reset=True)
            for move_number in range(5):
                # normal moves
                general_delta_test_random_move(
                    self,
                    partition,
                    self.degree_corrected.calculate,
                    self.degree_corrected.calculate_delta,
                    partition_counter,
                    move_number)

                # block removals
                general_delta_test_block_with_single_element_deletion(
                    self,
                    partition,
                    self.degree_corrected.calculate,
                    self.degree_corrected.calculate_delta,
                    partition_counter,
                    move_number)

                # test block merge
                general_delta_test_merge_blocks(
                    self,
                    partition,
                    self.degree_corrected.calculate,
                    self.degree_corrected.calculate_delta,
                    partition_counter,
                    move_number,
                    inference,
                )
