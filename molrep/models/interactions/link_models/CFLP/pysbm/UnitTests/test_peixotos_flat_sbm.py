import math
import random as rd
from unittest import TestCase

import networkx as nx

from pysbm import sbm
from pysbm.sbm.nxpartitiongraphbased import NxPartitionGraphBased
from pysbm.sbm.peixotos_flat_sbm import DeltaModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm
from pysbm.sbm.peixotos_flat_sbm import LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmWrapper
from pysbm.sbm.peixotos_flat_sbm_full import ModelLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm
from pysbm.sbm.peixotos_flat_sbm import ModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm


class TestModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmUndirected(TestCase):
    """Test case of Log Version of ModelLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm (not directed)"""

    # in general a lot easier because we can compare the log values against the non log version
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
        self.log_likelihood = ModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm()

    def test_calculate_p_adjacency_undirected(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(math.log(self.likelihood._calculate_p_adjacency_undirected(partition)),
                                   self.log_likelihood._calculate_p_adjacency_undirected(partition),
                                   msg="Difference in partition:" + str(i))

    def test_calculate_non_degree_corrected_undirected(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(math.log(self.likelihood._calculate_non_degree_corrected_undirected(partition)),
                                   self.log_likelihood._calculate_non_degree_corrected_undirected(partition),
                                   msg="Difference in partition:" + str(i))

    def test_calculate_p_degree_sequence_uniform_undirected(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(math.log(self.likelihood._calculate_p_degree_sequence_uniform_undirected(partition)),
                                   self.log_likelihood._calculate_p_degree_sequence_uniform_undirected(partition),
                                   msg="Difference in partition:" + str(i))

    def test_calculate_p_degree_sequence_uniform_hyperprior_undirected(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(
                math.log(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(partition)),
                self.log_likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected(partition),
                msg="Difference in partition:" + str(i))

    def test_calculate_p_edge_counts_undirected(self):
        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(math.log(self.likelihood._calculate_p_edge_counts_undirected(partition)),
                                   self.log_likelihood._calculate_p_edge_counts_undirected(partition),
                                   msg="Difference in partition:" + str(i) + "\ntrue value" + str(math.log(
                                       self.likelihood._calculate_p_edge_counts_undirected(
                                           partition))) + "\nincorrect value" + str(
                                       self.log_likelihood._calculate_p_edge_counts_undirected(partition)))

    def test_calculate_complete_uniform_hyperprior_undirected(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(
                math.log(self.likelihood.calculate_complete_uniform_hyperprior_undirected(partition)),
                self.log_likelihood.calculate_complete_uniform_hyperprior_undirected(partition),
                msg="Difference in partition:" + str(i))

    def test_calculate_complete_uniform_undirected(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(math.log(self.likelihood.calculate_complete_uniform_undirected(partition)),
                                   self.log_likelihood.calculate_complete_uniform_undirected(partition),
                                   msg="Difference in partition:" + str(i))

    def test_calculate_complete_non_degree_corrected_undirected(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(
                math.log(self.likelihood.calculate_complete_non_degree_corrected_undirected(partition)),
                self.log_likelihood.calculate_complete_non_degree_corrected_undirected(partition),
                msg="Difference in partition:" + str(i))


class TestModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmDirected(TestCase):
    """Test case of Log Version of ModelLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm (directed)"""

    # in general a lot easier because we can compare the log values against the non log version
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
        self.log_likelihood = ModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm()

    def test_calculate_p_adjacency_directed(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(math.log(self.likelihood._calculate_p_adjacency_directed(partition)),
                                   self.log_likelihood._calculate_p_adjacency_directed(partition),
                                   msg="Difference in partition:" + str(i))

    def test_calculate_non_degree_corrected_directed(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(math.log(self.likelihood._calculate_non_degree_corrected_directed(partition)),
                                   self.log_likelihood._calculate_non_degree_corrected_directed(partition),
                                   msg="Difference in partition:" + str(i))

    def test_calculate_p_degree_sequence_uniform_directed(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(math.log(self.likelihood._calculate_p_degree_sequence_uniform_directed(partition)),
                                   self.log_likelihood._calculate_p_degree_sequence_uniform_directed(partition),
                                   msg="Difference in partition:" + str(i))

    def test_calculate_p_degree_sequence_uniform_hyperprior_directed(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(
                math.log(self.likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(partition)),
                self.log_likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed(partition),
                msg="Difference in partition:" + str(i))

    def test_calculate_p_edge_counts_directed(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(math.log(self.likelihood._calculate_p_edge_counts_directed(partition)),
                                   self.log_likelihood._calculate_p_edge_counts_directed(partition),
                                   msg="Difference in partition:" + str(i))

    def test_calculate_complete_uniform_hyperprior_directed(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(math.log(self.likelihood.calculate_complete_uniform_hyperprior_directed(partition)),
                                   self.log_likelihood.calculate_complete_uniform_hyperprior_directed(partition),
                                   msg="Difference in partition:" + str(i))

    def test_calculate_complete_uniform_directed(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(math.log(self.likelihood.calculate_complete_uniform_directed(partition)),
                                   self.log_likelihood.calculate_complete_uniform_directed(partition),
                                   msg="Difference in partition:" + str(i))

    def test_calculate_complete_non_degree_corrected_directed(self):

        for i, partition in enumerate(self.partitions):
            self.assertAlmostEqual(
                math.log(self.likelihood.calculate_complete_non_degree_corrected_directed(partition)),
                self.log_likelihood.calculate_complete_non_degree_corrected_directed(partition),
                msg="Difference in partition:" + str(i))


class TestDeltaModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmUndirected(TestCase):
    """Test case of Log Version of ModelLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm (undirected)"""

    def setUp(self):
        self.graphs = [nx.Graph() for _ in range(8)]
        self.partitions = []

        graph = self.graphs[0]
        self.number_of_nodes = 10
        for i in range(self.number_of_nodes):
            graph.add_edge(i, (i + 1) % self.number_of_nodes)
        # beside complete circle add one edge to connect all blocks
        graph.add_edge(0, 2)
        # here first without selfloops and undirected
        partition = NxPartitionGraphBased(graph=graph,
                                          number_of_blocks=2,
                                          calculate_degree_of_blocks=True,
                                          save_neighbor_edges=True,
                                          save_neighbor_of_blocks=True,
                                          save_degree_distributions=True)
        partition.set_from_representation({node: node % partition.B for node in graph})
        self.partitions.append(partition)

        # now with another block != from, to block
        partition = NxPartitionGraphBased(graph=graph,
                                          number_of_blocks=3,
                                          calculate_degree_of_blocks=True,
                                          save_neighbor_edges=True,
                                          save_neighbor_of_blocks=True,
                                          save_degree_distributions=True)
        partition.set_from_representation({node: node % partition.B for node in graph})
        self.partitions.append(partition)

        # now one with selfloop
        graph = self.graphs[1] = graph.copy()
        graph.add_edge(0, 0)
        partition = NxPartitionGraphBased(graph=graph,
                                          number_of_blocks=3,
                                          calculate_degree_of_blocks=True,
                                          save_neighbor_edges=True,
                                          save_neighbor_of_blocks=True,
                                          save_degree_distributions=True)
        partition.set_from_representation({node: node % partition.B for node in graph})
        self.partitions.append(partition)

        nx.add_path(self.graphs[2], [0, 0, 1, 2, 3])
        nx.add_path(self.graphs[3], [0, 1, 2, 3, 0])
        nx.add_path(self.graphs[4], [0, 1, 2, 3, 0, 0])
        nx.add_path(self.graphs[6], [0, 1, 2, 3, 0, 4])
        self.graphs[5] = self.graphs[4].copy()
        self.graphs[5].add_edge(2, 2)
        nx.add_cycle(self.graphs[7], range(20))

        for graph in self.graphs:
            partition = NxPartitionGraphBased(graph=graph, number_of_blocks=3, save_degree_distributions=True)
            self.partitions.append(partition)
            partition.set_from_representation({node: node % partition.B for node in graph})

        # information about graphs below
        self.log_likelihood = ModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm()
        self.delta_likelihood = DeltaModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm()
        self._replace_likelihood = LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmWrapper(is_directed=False)

    def _general_comparision_single(self, delta_function, normal_function, partition, move=None,
                                    delta_uses_degree=False, delta_uses_kit=False, delta_uses_selfloops=False,
                                    delta_uses_nodes_remaining=False, call_with_args=False,
                                    partition_number=0, move_number=0, delta_uses_to_block_exists=False):
        if move is None:
            move = partition.get_random_move()
        node, old_block, new_block = move
        precalc = kit, selfloops, degree = partition.precalc_move(move, self._replace_likelihood)

        # build parameters for delta call
        parameters = {"partition": partition, "from_block": old_block, "to_block": new_block}
        # always include partition, from and to block
        # other like demanded
        if delta_uses_kit:
            parameters["kit"] = kit
        if delta_uses_selfloops:
            parameters["selfloops"] = selfloops
        if delta_uses_degree:
            parameters["degree"] = degree

        if delta_uses_nodes_remaining:
            parameters["nodes_remaining"] = True

        if delta_uses_to_block_exists:
            parameters["to_block_exists"] = True

        # value before the move and before any delta calculation
        start_value = normal_function(partition)

        # calculate delta with given set of parameters
        if call_with_args:
            delta_value = delta_function(*parameters.values())
        else:
            delta_value = delta_function(**parameters)

        # move node to calculate real value
        partition.move_node(node, new_block)
        end_value = normal_function(partition)
        new_representation = partition.get_representation()
        # move node back
        partition.move_node(node, old_block)

        self.assertAlmostEqual(delta_value,
                               end_value - start_value,
                               msg="Difference in partition: " + str(partition_number) + "\n " + str(
                                   move_number) + "th move " + "\n actual partition " + str(
                                   partition.get_representation()) + "\n next partition" + str(
                                   new_representation) + "\n move " + str(
                                   move) + "\n precalc " + str(precalc))

    def _general_not_existing_to_block(self, delta_function, normal_function, partition, move=None,
                                       delta_uses_degree=False, delta_uses_kit=False, delta_uses_selfloops=False,
                                       delta_uses_nodes_remaining=False, call_with_args=False,
                                       partition_number=0, move_number=0, delta_uses_to_block_exists=True):
        if move is None:
            node = partition.get_random_node()
            move = (node, partition.get_block_of_node(node), partition.B)
        node, old_block, new_block = move
        precalc = kit, selfloops, degree = partition.precalc_move(move, self._replace_likelihood)

        # build parameters for delta call
        parameters = {"partition": partition, "from_block": old_block, "to_block": new_block}
        # always include partition, from and to block
        # other like demanded
        if delta_uses_kit:
            parameters["kit"] = kit
        if delta_uses_selfloops:
            parameters["selfloops"] = selfloops
        if delta_uses_degree:
            parameters["degree"] = degree

        if delta_uses_nodes_remaining:
            parameters["nodes_remaining"] = True

        if delta_uses_to_block_exists:
            parameters["to_block_exists"] = False

        # value before the move and before any delta calculation
        start_value = normal_function(partition)

        # calculate delta with given set of parameters
        if call_with_args:
            delta_value = delta_function(*parameters.values())
        else:
            delta_value = delta_function(**parameters)

        # move node to calculate real value
        partition.move_node(node, new_block)
        end_value = normal_function(partition)
        new_representation = partition.get_representation()
        # move node back
        partition.move_node(node, old_block)

        self.assertAlmostEqual(delta_value,
                               end_value - start_value,
                               msg="Difference in partition: " + str(partition_number) + "\n " + str(
                                   move_number) + "th move " + "\n actual partition " + str(
                                   partition.get_representation()) + "\n next partition" + str(
                                   new_representation) + "\n move " + str(
                                   move) + "\n precalc " + str(precalc))

    def _general_remove_blocks(self, delta_function, normal_function, partition, control_partition, peixoto_inference,
                               from_block=None, to_block=None, delta_uses_degree=False, delta_uses_kit=False,
                               delta_uses_selfloops=False, call_with_args=False, partition_number=0, move_number=0,
                               delta_uses_to_block_exists=False):
        if from_block is None:
            node = partition.get_random_node(no_single=False)
            from_block = partition.get_block_of_node(node)
        if to_block is None:
            possible_blocks = list(partition.get_possible_blocks(from_block))
            possible_blocks.remove(from_block)
            to_block = rd.choice(possible_blocks)

        parameter, reference = peixoto_inference._precalc_block_merge(from_block)
        peixoto_inference._adjustment_delta_pre(reference, from_block, to_block)

        kit, selfloops, degree, nodes_moved, _ = parameter

        # build parameters for delta call
        parameters = {"partition": partition, "from_block": from_block, "to_block": to_block}
        # always include partition, from and to block
        # other like demanded
        if delta_uses_kit:
            parameters["kit"] = kit
        if delta_uses_selfloops:
            parameters["selfloops"] = selfloops
        if delta_uses_degree:
            parameters["degree"] = degree

        if delta_uses_to_block_exists:
            parameters["to_block_exists"] = True

        # new parameter nodes_remaining and nodes moved
        parameters["nodes_moved"] = nodes_moved
        parameters["nodes_remaining"] = False

        # value before the move and before any delta calculation
        start_value = normal_function(control_partition)

        # calculate delta with given set of parameters
        if call_with_args:
            delta_value = delta_function(*parameters.values())
        else:
            delta_value = delta_function(**parameters)

        # perform merge
        control_partition.merge_blocks({from_block: to_block}, control_partition.B - 1)
        end_value = normal_function(control_partition)
        new_representation = control_partition.get_representation()

        # roll back everything
        control_partition.set_from_representation(partition.get_representation())

        self.assertAlmostEqual(delta_value,
                               end_value - start_value,
                               msg="Difference in partition: " + str(partition_number) + "\n " + str(
                                   move_number) + "th move " + "\n actual partition " + str(
                                   partition.get_representation()) + "\n next partition" + str(
                                   new_representation) + "\n move " + str(
                                   from_block) + "->" + str(to_block) + "\n precalc " + str(parameter))

    def test_calculate_p_adjacency_undirected(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood._calculate_delta_p_adjacency_undirected,
                    self.log_likelihood._calculate_p_adjacency_undirected,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_p_adjacency_undirected_not_existing_to_block(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood._calculate_delta_p_adjacency_undirected,
                    self.log_likelihood._calculate_p_adjacency_undirected,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_non_degree_corrected_undirected(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood._calculate_delta_non_degree_corrected_undirected,
                    self.log_likelihood._calculate_non_degree_corrected_undirected,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_non_degree_corrected_undirected_not_existing_to_block(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood._calculate_delta_non_degree_corrected_undirected,
                    self.log_likelihood._calculate_non_degree_corrected_undirected,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_non_degree_corrected_undirected_merge(self):
        # merge two blocks
        for i, partition in enumerate(self.partitions):
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # try random merges
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood._calculate_delta_non_degree_corrected_undirected,
                    self.log_likelihood._calculate_non_degree_corrected_undirected,
                    partition,
                    control_partition,
                    inference,
                    delta_uses_degree=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_p_degree_sequence_uniform_undirected(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_undirected,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_undirected,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_p_degree_sequence_uniform_undirected_not_existing_to_block(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_undirected,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_undirected,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_p_degree_sequence_uniform_undirected_merge(self):
        # merge two blocks
        for i, partition in enumerate(self.partitions):
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # try random merges
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_undirected,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_undirected,
                    partition,
                    control_partition,
                    inference,
                    delta_uses_degree=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_p_degree_sequence_uniform_hyperprior_undirected(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_hyperprior_undirected,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_p_degree_sequence_uniform_hyperprior_undirected_not_existing_to_block(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_hyperprior_undirected,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_p_degree_sequence_uniform_hyperprior_undirected_merge(self):
        # merge two blocks
        for i, partition in enumerate(self.partitions):
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # try random merges
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_hyperprior_undirected,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_hyperprior_undirected,
                    partition,
                    control_partition,
                    inference,
                    delta_uses_degree=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_p_edge_counts_undirected(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood._calculate_delta_p_edge_counts_undirected,
                    self.log_likelihood._calculate_p_edge_counts_undirected,
                    partition,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_p_edge_counts_undirected_not_existing_to_block(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood._calculate_delta_p_edge_counts_undirected,
                    self.log_likelihood._calculate_p_edge_counts_undirected,
                    partition,
                    delta_uses_nodes_remaining=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_p_edge_counts_undirected_merge(self):
        # merge two blocks
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # first classic moves
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood._calculate_delta_p_edge_counts_undirected,
                    self.log_likelihood._calculate_p_edge_counts_undirected,
                    partition,
                    control_partition,
                    inference,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_complete_uniform_hyperprior_undirected(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood.calculate_delta_complete_uniform_hyperprior_undirected,
                    self.log_likelihood.calculate_complete_uniform_hyperprior_undirected,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_complete_uniform_hyperprior_undirected_merge(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # # first classic moves
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood.calculate_delta_complete_uniform_hyperprior_undirected,
                    self.log_likelihood.calculate_complete_uniform_hyperprior_undirected,
                    partition,
                    control_partition,
                    inference,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_complete_uniform_hyperprior_undirected_not_existing_to_block(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood.calculate_delta_complete_uniform_hyperprior_undirected,
                    self.log_likelihood.calculate_complete_uniform_hyperprior_undirected,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    delta_uses_to_block_exists=False,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_complete_uniform_undirected(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood.calculate_delta_complete_uniform_undirected,
                    self.log_likelihood.calculate_complete_uniform_undirected,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_complete_uniform_undirected_merge(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # # first classic moves
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood.calculate_delta_complete_uniform_undirected,
                    self.log_likelihood.calculate_complete_uniform_undirected,
                    partition,
                    control_partition,
                    inference,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_complete_uniform_undirected_not_existing_to_block(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood.calculate_delta_complete_uniform_undirected,
                    self.log_likelihood.calculate_complete_uniform_undirected,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    delta_uses_to_block_exists=False,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_complete_non_degree_corrected_undirected(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood.calculate_delta_complete_non_degree_corrected_undirected,
                    self.log_likelihood.calculate_complete_non_degree_corrected_undirected,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_complete_non_degree_corrected_undirected_merge(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # # first classic moves
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood.calculate_delta_complete_non_degree_corrected_undirected,
                    self.log_likelihood.calculate_complete_non_degree_corrected_undirected,
                    partition,
                    control_partition,
                    inference,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_complete_non_degree_corrected_undirected_not_existing_to_block(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood.calculate_delta_complete_non_degree_corrected_undirected,
                    self.log_likelihood.calculate_complete_non_degree_corrected_undirected,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    delta_uses_to_block_exists=False,
                    partition_number=i,
                    move_number=move_number
                )

    def test_null_move(self):
        """Test moving within the same block"""
        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_non_degree_corrected_undirected(
            self.partitions[0], 0, 0))

        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_uniform_undirected(
            self.partitions[0], 0, 0))

        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_uniform_hyperprior_undirected(
            self.partitions[0], 0, 0))

    def test_creating_and_deleting_block(self):
        """Move last node of a block into a new block"""
        node = self.partitions[0].get_random_node()
        self.partitions[0].move_node(node, self.partitions[0].B)

        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_non_degree_corrected_undirected(
            self.partitions[0], self.partitions[0].B - 1, self.partitions[0].B,
            0, 0, 0))

        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_uniform_undirected(
            self.partitions[0], self.partitions[0].B - 1, self.partitions[0].B,
            0, 0, 0))

        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_uniform_hyperprior_undirected(
            self.partitions[0], self.partitions[0].B - 1, self.partitions[0].B,
            0, 0, 0))


class TestDeltaModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmDirected(TestCase):
    """Test case of Log Version of ModelLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm (directed)"""

    def setUp(self):
        self.graphs = [nx.DiGraph() for _ in range(8)]
        self.partitions = []

        graph = self.graphs[0]
        self.number_of_nodes = 10
        for i in range(self.number_of_nodes):
            graph.add_edge(i, (i + 1) % self.number_of_nodes)
        # beside complete circle add one edge to connect all blocks
        graph.add_edge(0, 2)
        # here first without selfloops and directed
        partition = NxPartitionGraphBased(graph=graph,
                                          number_of_blocks=2,
                                          calculate_degree_of_blocks=True,
                                          save_neighbor_edges=True,
                                          save_neighbor_of_blocks=True,
                                          save_degree_distributions=True)
        partition.set_from_representation({node: node % partition.B for node in graph})
        self.partitions.append(partition)

        # now with another block != from, to block
        partition = NxPartitionGraphBased(graph=graph,
                                          number_of_blocks=3,
                                          calculate_degree_of_blocks=True,
                                          save_neighbor_edges=True,
                                          save_neighbor_of_blocks=True,
                                          save_degree_distributions=True)
        partition.set_from_representation({node: node % partition.B for node in graph})
        self.partitions.append(partition)

        # now one with selfloop
        graph = self.graphs[1] = graph.copy()
        graph.add_edge(0, 0)
        partition = NxPartitionGraphBased(graph=graph,
                                          number_of_blocks=3,
                                          calculate_degree_of_blocks=True,
                                          save_neighbor_edges=True,
                                          save_neighbor_of_blocks=True,
                                          save_degree_distributions=True)
        partition.set_from_representation({node: node % partition.B for node in graph})
        self.partitions.append(partition)

        nx.add_path(self.graphs[2], [0, 0, 1, 2, 3])
        nx.add_path(self.graphs[3], [0, 1, 2, 3, 0])
        nx.add_path(self.graphs[4], [0, 1, 2, 3, 0, 0])
        nx.add_path(self.graphs[6], [0, 1, 2, 3, 0, 4])
        self.graphs[5] = self.graphs[4].copy()
        self.graphs[5].add_edge(2, 2)
        nx.add_cycle(self.graphs[7], range(20))

        for graph in self.graphs:
            partition = NxPartitionGraphBased(graph=graph, number_of_blocks=3, save_degree_distributions=True)
            self.partitions.append(partition)
            partition.set_from_representation({node: node % partition.B for node in graph})

        # information about graphs below
        self.log_likelihood = ModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm()
        self.delta_likelihood = DeltaModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm()
        self._replace_likelihood = LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmWrapper(is_directed=True)

    def _general_comparision_single(self, delta_function, normal_function, partition, move=None,
                                    delta_uses_degree=False, delta_uses_kit=False, delta_uses_selfloops=False,
                                    delta_uses_nodes_remaining=False, delta_uses_to_block_exists=False,
                                    call_with_args=False, partition_number=0, move_number=0):
        if move is None:
            move = partition.get_random_move()
        node, old_block, new_block = move
        precalc = kit, kti, selfloops, in_degree, out_degree = partition.precalc_move(move, self._replace_likelihood)

        # build parameters for delta call
        parameters = {"partition": partition, "from_block": old_block, "to_block": new_block}
        # always include partition, from and to block
        # other like demanded
        if delta_uses_kit:
            parameters["kit"] = kit
            parameters["kti"] = kti
        if delta_uses_selfloops:
            parameters["selfloops"] = selfloops
        if delta_uses_degree:
            parameters["in_degree"] = in_degree
            parameters["out_degree"] = out_degree

        if delta_uses_nodes_remaining:
            parameters["nodes_remaining"] = True

        if delta_uses_to_block_exists:
            parameters["to_block_exists"] = True

        # value before the move and before any delta calculation
        start_value = normal_function(partition)

        # calculate delta with given set of parameters
        if call_with_args:
            delta_value = delta_function(*parameters.values())
        else:
            delta_value = delta_function(**parameters)

        # move node to calculate real value
        partition.move_node(node, new_block)
        end_value = normal_function(partition)
        new_representation = partition.get_representation()
        # move node back
        partition.move_node(node, old_block)

        self.assertAlmostEqual(delta_value,
                               end_value - start_value,
                               msg="Difference in partition: " + str(partition_number) + "\n " + str(
                                   move_number) + "th move " + "\n actual partition " + str(
                                   partition.get_representation()) + "\n next partition" + str(
                                   new_representation) + "\n move " + str(
                                   move) + "\n precalc " + str(precalc))

    def _general_not_existing_to_block(self, delta_function, normal_function, partition, move=None,
                                       delta_uses_degree=False, delta_uses_kit=False, delta_uses_selfloops=False,
                                       delta_uses_nodes_remaining=False, delta_uses_to_block_exists=True,
                                       call_with_args=False, partition_number=0, move_number=0):
        if move is None:
            node = partition.get_random_node()
            move = (node, partition.get_block_of_node(node), partition.B)
        node, old_block, new_block = move
        precalc = kit, kti, selfloops, in_degree, out_degree = partition.precalc_move(move, self._replace_likelihood)

        # build parameters for delta call
        parameters = {"partition": partition, "from_block": old_block, "to_block": new_block}
        # always include partition, from and to block
        # other like demanded
        if delta_uses_kit:
            parameters["kit"] = kit
            parameters["kti"] = kti
        if delta_uses_selfloops:
            parameters["selfloops"] = selfloops
        if delta_uses_degree:
            parameters["in_degree"] = in_degree
            parameters["out_degree"] = out_degree

        if delta_uses_nodes_remaining:
            parameters["nodes_remaining"] = True

        if delta_uses_to_block_exists:
            parameters["to_block_exists"] = False

        # value before the move and before any delta calculation
        start_value = normal_function(partition)

        # calculate delta with given set of parameters
        if call_with_args:
            delta_value = delta_function(*parameters.values())
        else:
            delta_value = delta_function(**parameters)

        # move node to calculate real value
        partition.move_node(node, new_block)
        end_value = normal_function(partition)
        new_representation = partition.get_representation()
        # move node back
        partition.move_node(node, old_block)

        self.assertAlmostEqual(delta_value,
                               end_value - start_value,
                               msg="Difference in partition: " + str(partition_number) + "\n " + str(
                                   move_number) + "th move " + "\n actual partition " + str(
                                   partition.get_representation()) + "\n next partition" + str(
                                   new_representation) + "\n move " + str(
                                   move) + "\n precalc " + str(precalc))

    def _general_remove_blocks(self, delta_function, normal_function, partition, control_partition, peixoto_inference,
                               from_block=None, to_block=None, delta_uses_degree=False, delta_uses_kit=False,
                               delta_uses_selfloops=False, call_with_args=False, partition_number=0, move_number=0,
                               delta_uses_to_block_exists=False):
        if from_block is None:
            node = partition.get_random_node(no_single=False)
            from_block = partition.get_block_of_node(node)
        if to_block is None:
            possible_blocks = list(partition.get_possible_blocks(from_block))
            possible_blocks.remove(from_block)
            to_block = rd.choice(possible_blocks)

        parameter, reference = peixoto_inference._precalc_block_merge(from_block)
        peixoto_inference._adjustment_delta_pre(reference, from_block, to_block)

        kit, kti, selfloops, in_degree, out_degree, nodes_moved, _ = parameter

        # build parameters for delta call
        parameters = {"partition": partition, "from_block": from_block, "to_block": to_block}
        # always include partition, from and to block
        # other like demanded
        if delta_uses_kit:
            parameters["kit"] = kit
            parameters["kti"] = kti
        if delta_uses_selfloops:
            parameters["selfloops"] = selfloops
        if delta_uses_degree:
            parameters["in_degree"] = in_degree
            parameters["out_degree"] = out_degree

        if delta_uses_to_block_exists:
            parameters["to_block_exists"] = True

        # new parameter nodes_remaining and nodes_moved
        parameters["nodes_moved"] = nodes_moved
        parameters["nodes_remaining"] = False

        # value before the move and before any delta calculation
        start_value = normal_function(control_partition)

        # calculate delta with given set of parameters
        if call_with_args:
            delta_value = delta_function(*parameters.values())
        else:
            delta_value = delta_function(**parameters)

        # perform merge
        control_partition.merge_blocks({from_block: to_block}, control_partition.B - 1)
        end_value = normal_function(control_partition)
        new_representation = control_partition.get_representation()
        # roll back everything
        control_partition.set_from_representation(partition.get_representation())

        self.assertAlmostEqual(delta_value,
                               end_value - start_value,
                               msg="Difference in partition: " + str(partition_number) + "\n " + str(
                                   move_number) + "th move " + "\n actual partition " + str(
                                   partition.get_representation()) + "\n next partition" + str(
                                   new_representation) + "\n move " + str(
                                   from_block) + "->" + str(to_block) + "\n precalc " + str(parameter))

    def test_calculate_delta_p_adjacency_directed(self):
        # import random as rd
        # rd.seed(42)
        # np.random.seed(42)

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood._calculate_delta_p_adjacency_directed,
                    self.log_likelihood._calculate_p_adjacency_directed,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_p_adjacency_directed_not_existing_to_block(self):
        # import random as rd
        # rd.seed(42)
        # np.random.seed(42)

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood._calculate_delta_p_adjacency_directed,
                    self.log_likelihood._calculate_p_adjacency_directed,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_non_degree_corrected_directed(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood._calculate_delta_non_degree_corrected_directed,
                    self.log_likelihood._calculate_non_degree_corrected_directed,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_non_degree_corrected_directed_not_existing_to_block(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood._calculate_delta_non_degree_corrected_directed,
                    self.log_likelihood._calculate_non_degree_corrected_directed,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_non_degree_corrected_directed_merge(self):
        # merge two blocks
        for i, partition in enumerate(self.partitions):
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # try random merges
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood._calculate_delta_non_degree_corrected_directed,
                    self.log_likelihood._calculate_non_degree_corrected_directed,
                    partition,
                    control_partition,
                    inference,
                    delta_uses_degree=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_p_degree_sequence_uniform_directed(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_directed,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_directed,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_p_degree_sequence_uniform_directed_not_existing_to_block(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_directed,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_directed,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_p_degree_sequence_uniform_directed_merge(self):
        # merge two blocks
        for i, partition in enumerate(self.partitions):
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # try random merges
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_directed,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_directed,
                    partition,
                    control_partition,
                    inference,
                    delta_uses_degree=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_p_degree_sequence_uniform_hyperprior_directed(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_hyperprior_directed,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_p_degree_sequence_uniform_hyperprior_directed_not_existing_to_block(self):

        for i, partition in enumerate(self.partitions):
            # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_hyperprior_directed,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed,
                    partition,
                    delta_uses_degree=True,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_p_degree_sequence_uniform_hyperprior_directed_merge(self):
        # merge two blocks
        for i, partition in enumerate(self.partitions):
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # try random merges
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood._calculate_delta_p_degree_sequence_uniform_hyperprior_directed,
                    self.log_likelihood._calculate_p_degree_sequence_uniform_hyperprior_directed,
                    partition,
                    control_partition,
                    inference,
                    delta_uses_degree=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_p_edge_counts_directed(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood._calculate_delta_p_edge_counts_directed,
                    self.log_likelihood._calculate_p_edge_counts_directed,
                    partition,
                    delta_uses_nodes_remaining=True,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_p_edge_counts_directed_not_existing_to_block(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood._calculate_delta_p_edge_counts_directed,
                    self.log_likelihood._calculate_p_edge_counts_directed,
                    partition,
                    delta_uses_nodes_remaining=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_p_edge_counts_directed_merge(self):
        # merge two blocks
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # first classic moves
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood._calculate_delta_p_edge_counts_directed,
                    self.log_likelihood._calculate_p_edge_counts_directed,
                    partition,
                    control_partition,
                    inference,
                    delta_uses_to_block_exists=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_complete_uniform_hyperprior_directed(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood.calculate_delta_complete_uniform_hyperprior_directed,
                    self.log_likelihood.calculate_complete_uniform_hyperprior_directed,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_complete_uniform_hyperprior_directed_merge(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # # first classic moves
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood.calculate_delta_complete_uniform_hyperprior_directed,
                    self.log_likelihood.calculate_complete_uniform_hyperprior_directed,
                    partition,
                    control_partition,
                    inference,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_complete_uniform_hyperprior_directed_not_existing_to_block(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood.calculate_delta_complete_uniform_hyperprior_directed,
                    self.log_likelihood.calculate_complete_uniform_hyperprior_directed,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    delta_uses_to_block_exists=False,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_complete_uniform_directed(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood.calculate_delta_complete_uniform_directed,
                    self.log_likelihood.calculate_complete_uniform_directed,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_complete_uniform_directed_merge(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # # first classic moves
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood.calculate_delta_complete_uniform_directed,
                    self.log_likelihood.calculate_complete_uniform_directed,
                    partition,
                    control_partition,
                    inference,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_complete_uniform_directed_not_existing_to_block(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood.calculate_delta_complete_uniform_directed,
                    self.log_likelihood.calculate_complete_uniform_directed,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    delta_uses_to_block_exists=False,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_complete_non_degree_corrected_directed(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_comparision_single(
                    self.delta_likelihood.calculate_delta_complete_non_degree_corrected_directed,
                    self.log_likelihood.calculate_complete_non_degree_corrected_directed,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_complete_non_degree_corrected_directed_merge(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            control_partition = partition.copy()
            inference = sbm.PeixotoInference(partition.graph, self._replace_likelihood, partition,
                                             no_partition_reset=True)
            # # first classic moves
            for move_number in range(10):
                self._general_remove_blocks(
                    self.delta_likelihood.calculate_delta_complete_non_degree_corrected_directed,
                    self.log_likelihood.calculate_complete_non_degree_corrected_directed,
                    partition,
                    control_partition,
                    inference,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    partition_number=i,
                    move_number=move_number
                )

    def test_calculate_delta_complete_non_degree_corrected_directed_not_existing_to_block(self):
        for i, partition in enumerate(self.partitions):
            partition.actual_level = 0
            # # first classic moves
            for move_number in range(10):
                self._general_not_existing_to_block(
                    self.delta_likelihood.calculate_delta_complete_non_degree_corrected_directed,
                    self.log_likelihood.calculate_complete_non_degree_corrected_directed,
                    partition,
                    call_with_args=True,
                    delta_uses_kit=True,
                    delta_uses_selfloops=True,
                    delta_uses_degree=True,
                    delta_uses_to_block_exists=False,
                    partition_number=i,
                    move_number=move_number
                )

    def test_null_move(self):
        """Test moving within the same block"""
        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_non_degree_corrected_directed(
            self.partitions[0], 0, 0))

        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_uniform_directed(
            self.partitions[0], 0, 0))

        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_uniform_hyperprior_directed(
            self.partitions[0], 0, 0))

    def test_creating_and_deleting_block(self):
        """Move last node of a block into a new block"""
        node = self.partitions[0].get_random_node()
        self.partitions[0].move_node(node, self.partitions[0].B)

        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_non_degree_corrected_directed(
            self.partitions[0], self.partitions[0].B - 1, self.partitions[0].B,
            0, 0, 0, 0, 0))

        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_uniform_directed(
            self.partitions[0], self.partitions[0].B - 1, self.partitions[0].B,
            0, 0, 0, 0, 0))

        self.assertEqual(0, self.delta_likelihood.calculate_delta_complete_uniform_hyperprior_directed(
            self.partitions[0], self.partitions[0].B - 1, self.partitions[0].B,
            0, 0, 0, 0, 0))
