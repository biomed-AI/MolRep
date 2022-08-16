from __future__ import division

import math
import random as rd

import networkx as nx
import numpy as np

# from pysbm import sbm
from MolRep.Interactions.link_models.CFLP.pysbm import sbm


class NetworkSuppplier(object):
    """ Create different types of graphs"""
    STANDARD_SEED = 42

    def __init__(self):
        self.partition_class = sbm.NxPartition

    def get_all_test_cases(self, include_directed=True, include_int_weighted=True):
        """
        Returns test cases containing of:
        -
        each consisting at least out of graph and partition.
        Additionally some supply their real division into groups.
        """
        test_cases = []

        test_cases.extend(self.get_karate_network())
        test_cases.extend(self.get_planted_partition(False))
        if include_directed:
            test_cases.extend(self.get_planted_partition(True))

        return test_cases

    def get_karate_network(self):
        """
        Returns Zachary's Karate Club graph
        """
        graph = nx.karate_club_graph()
        partition = self.partition_class(graph)
        # extract "ground truth" given by meta data
        # in the Data their are two groups, Mr. Hi and Officer
        # the information is saved in a node attribute named club
        meta_data_partition = [0 for _ in range(len(graph))]
        for node in graph.nodes_iter():
            if graph.node[node]['club'] == 'Mr. Hi':
                meta_data_partition[node] = 1
        return [(partition, 2, meta_data_partition)]

    def get_planted_partition(self, directed, number_of_instances_per_combination=5):
        """
        Return some planted partition networks
        """
        test_cases = []
        for pout in range(1, 8):
            for pin in range(pout + 1, 11):
                for number_of_vertices in range(2, 11, 2):
                    generator = PlantedPartitionGenerator(4,
                                                          number_of_vertices,
                                                          float(pin) / 10,
                                                          float(pout) / 10)
                    for number_of_different_instances in range(number_of_instances_per_combination):
                        graph, number_of_blocks, real_partition = generator.generate(
                            directed,
                            self.STANDARD_SEED +
                            number_of_different_instances)
                        test_cases.append((self.partition_class(graph),
                                           number_of_blocks,
                                           real_partition))

        return test_cases

    def get_sbm_generated_graphs(self):
        """
        Return generated Graphs based on an SBM
        """


class SBMGenerator(object):
    """Generation of graphs based on given SBM"""
    PROBABILISTIC_EDGES = 'P'
    FIXED_NUMBER_OF_EDGES = 'D'

    def __init__(self,
                 number_of_blocks,
                 nodes_per_block,
                 edge_matrix,
                 type_of_edge_matrix,
                 is_directed_edge_matrix=False):
        self._B = number_of_blocks
        self._nodes_per_block = nodes_per_block
        if len(nodes_per_block) != number_of_blocks:
            raise ValueError()
        self._edge_matrix = edge_matrix
        if type_of_edge_matrix not in {self.FIXED_NUMBER_OF_EDGES, self.PROBABILISTIC_EDGES}:
            raise ValueError()
        self._edge_type = type_of_edge_matrix

        self.random = np.random.RandomState()

        if len(self._edge_matrix) != number_of_blocks:
            raise ValueError()

        for block in range(self._B):
            if is_directed_edge_matrix:
                self._edge_matrix[block][block] = .5 * edge_matrix[block][block]
            if len(self._edge_matrix[block]) != number_of_blocks:
                raise ValueError()

    def generate(self, directed, seed):
        """Generate instance out of given data"""
        old_state = rd.getstate()
        rd.seed(seed)
        self.random = np.random.RandomState(seed)
        if self._edge_type == self.FIXED_NUMBER_OF_EDGES:
            info = self._generate_fixed(directed)
        # elif self._edge_type == self.PROBABILISTIC_EDGES:
        info = self._generate_probabilistic(directed)
        rd.setstate(old_state)

        return info

    def _generate_probabilistic(self, directed):
        """Generate graph with sampling each possible edge"""
        if directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        # determine total number of nodes and add every node to the graph
        total_nodes = sum(self._nodes_per_block)
        graph.add_nodes_from(range(total_nodes))
        real_partition = {}

        # save actual block and count down to next block same for to_block
        from_block = 0
        next_from_block_in = self._nodes_per_block[from_block]

        # print(self._edge_matrix)

        # for each possible edge test the probability
        for node_from in range(total_nodes):
            # check at the beginning and decrease at the end
            #   to ensure that in the last loop no index error is raised
            if next_from_block_in == 0:
                from_block += 1
                next_from_block_in = self._nodes_per_block[from_block]

            # create real partition
            real_partition[node_from] = from_block

            # reset to block info
            # in undirected case only sample one way
            if directed:
                to_nodes = range(total_nodes)
                to_block = 0
                next_to_block_in = self._nodes_per_block[to_block]
            else:
                to_nodes = range(node_from, total_nodes)
                to_block = from_block
                next_to_block_in = next_from_block_in

            for node_to in to_nodes:
                # as above check here and decrease add the end of the cycle
                if next_to_block_in == 0:
                    to_block += 1
                    next_to_block_in = self._nodes_per_block[to_block]

                # add random edge
                if rd.random() < self._get_edge_probability(node_from,
                                                            node_to,
                                                            from_block,
                                                            to_block):
                    graph.add_edge(node_from, node_to)

                next_to_block_in -= 1

            next_from_block_in -= 1

        return graph, self._B, real_partition

    def _get_edge_probability(self, node_from, node_to, from_block, to_block):
        """
        Return the probability of a certain edge between two nodes of two blocks
        in the probabilistic version
        """
        # additional arguments are needed for degree corrected version
        return self._edge_matrix[from_block][to_block]

    def _generate_fixed(self, directed):
        """
        Generate graph by distributing a fixed number of edges between
        randomly chosen block participants.
        """
        if directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        # initiate data structure for sampling and add all nodes to the graph
        block_info = self._create_block_info(graph)

        # create real partition
        real_partition = {}
        # for block, number_of_nodes in enumerate(self._nodes_per_block):
        #    real_partition.extend(number_of_nodes * [block])

        # test if loop below works like loop above (just for dicts instead of lists

        counter = 0
        for block in range(len(self._nodes_per_block)):
            for number_of_nodes in self._nodes_per_block[block]:
                real_partition[counter] = [block]
                counter += 1

        # now sample all block -> block combinations and distribute randomly the edges
        # by choosing two random ends for each edge between two blocks
        for from_block in range(self._B):
            for to_block in range(self._B):
                # short way in undirected case
                if not directed and to_block > from_block:
                    break
                # check value
                if (directed or from_block != to_block) \
                        and self._edge_matrix[from_block][to_block] > self._nodes_per_block[from_block] * \
                        self._nodes_per_block[to_block]:
                    raise ValueError("Too many edges demanded. \nDemanded numeber: "
                                     + str(self._edge_matrix[from_block][to_block]) + "\npossible edges "
                                     + str(self._nodes_per_block[from_block] * self._nodes_per_block[to_block])
                                     + "\nbetween blocks " + str(from_block) + "->" + str(to_block))
                elif not directed and from_block == to_block \
                        and self._edge_matrix[from_block][to_block] > self._nodes_per_block[
                    from_block] * (self._nodes_per_block[to_block] + 1) / 2:
                    raise ValueError("Too many edges demanded. \nDemanded numeber: "
                                     + str(self._edge_matrix[from_block][to_block]) + "\npossible edges "
                                     + str(self._nodes_per_block[from_block] * \
                                           (self._nodes_per_block[to_block] + 1) / 2)
                                     + "\nbetween blocks " + str(from_block) + "->" + str(to_block))
                # create exact the number of edges
                for _ in range(self._edge_matrix[from_block][to_block]):
                    while True:
                        from_node = self._get_random_block_element(from_block, block_info)
                        to_node = self._get_random_block_element(to_block, block_info)
                        # check if edge already exists else choose new!
                        if graph.has_edge(from_node, to_node):
                            graph.add_edge(from_node, to_node)
                            break
        return graph, self._B, real_partition

    def _create_block_info(self, graph):
        """Prepare information for each block and add all nodes to the graph"""
        # first create for each block the right number of nodes and add them to the graph
        # and save a list of nodes belonging to each block
        block_to_node = []
        actual_start_value = 0
        for number_of_nodes in self._nodes_per_block:
            block_to_node.append(range(actual_start_value,
                                       actual_start_value + number_of_nodes))
            graph.add_nodes_from(range(actual_start_value,
                                       actual_start_value + number_of_nodes))
            actual_start_value += number_of_nodes
        return block_to_node

    # in degree corrected self reference is needed
    # pylint: disable=no-self-use
    def _get_random_block_element(self, block, block_info):
        """Return random element of a certain block"""
        return rd.choice(block_info[block])


class SBMGeneratorDegreeCorrected(SBMGenerator):
    """Generation of SBM with the degree corrected model"""

    _SAMPLE_SIZE = 50

    def __init__(self, number_of_blocks, degree_distribution_per_block,
                 edge_matrix, type_of_edge_matrix, is_directed_edge_matrix=False):
        self._degree_distribution_per_block = degree_distribution_per_block
        # flat degree distribution and create node_per_block array
        self._flattened_block_distribution = []
        nodes_per_block = []
        for block_distribution in degree_distribution_per_block:
            # if needed calculate probability
            if sum(block_distribution) != 1:
                total = sum(block_distribution)
                for i, value in enumerate(block_distribution):
                    block_distribution[i] = value * 1.0 / total
            self._flattened_block_distribution.append(block_distribution)
            nodes_per_block.append(len(block_distribution))
        # call old init method to do the rest
        super(SBMGeneratorDegreeCorrected, self).__init__(
            number_of_blocks,
            nodes_per_block,
            edge_matrix,
            type_of_edge_matrix,
            is_directed_edge_matrix)

    def _get_edge_probability(self, node_from, node_to, from_block, to_block):
        """
        In degree corrected case take care of the probability of each node too.
        """
        return self._edge_matrix[from_block][to_block] \
               * self._flattened_block_distribution[node_from] \
               * self._flattened_block_distribution[node_to]

    def _create_block_info(self, graph):
        # add second field for saving the samples
        node_list = super(SBMGeneratorDegreeCorrected, self)._create_block_info(graph)
        block_info = []
        for nodes in node_list:
            block_info.append((nodes, []))
        return block_info

    # in degree corrected self reference is needed
    # pylint: disable=no-self-use
    def _get_random_block_element(self, block, block_info):
        """Return random element of a certain block"""
        nodes, samples = block_info[block]
        # check if samples is empty
        if not samples:
            #  draw multiple samples because this should speed up the generation
            samples.extend(self.random.choice(nodes,
                                              size=self._SAMPLE_SIZE,
                                              replace=True,
                                              p=self._degree_distribution_per_block[block]))

        return samples.pop()


class PlantedPartitionGenerator(SBMGenerator):
    """Generator for Planted Partition Networks based on SBMGenerator"""

    def __init__(self,
                 number_of_groups,
                 number_of_vertices_in_each_group,
                 edge_probability_in_group,
                 edge_probability_between_groups):
        #        create nodes per block array
        nodes_per_block = [number_of_vertices_in_each_group for _ in range(number_of_groups)]
        edge_matrix = [[edge_probability_between_groups if i != j else edge_probability_in_group
                        for i in range(number_of_groups)] for j in range(number_of_groups)]

        # call old init method to do the rest
        super(PlantedPartitionGenerator, self).__init__(
            number_of_groups,
            nodes_per_block,
            edge_matrix,
            self.PROBABILISTIC_EDGES,
            is_directed_edge_matrix=False)


class Comparator(object):
    """
    Execute the different inference algorithm on a given test cases
    """

    def __init__(self, test_cases, inference_algorithms, objective_function):
        self.test_cases = test_cases
        self.inference_algorithms = inference_algorithms
        self.objective_function = objective_function

    def execute_single_test(self, starting_partition, inference_class):
        test_partition = starting_partition.copy()
        inference = inference_class(starting_partition.graph,
                                    self.objective_function,
                                    test_partition)
        inference.infer_stochastic_block_model()
        return test_partition

    def execute_all_tests(self, repetitions=1, evaluate_only_objective_function=True):
        evaluated_results = []
        partitions = []
        for test_case in self.test_cases:
            if len(test_case) == 3:
                partition, number_of_blocks, real_partition = test_case
                result_evaluator = NormalizedMutualInformation(real_partition)
            elif len(test_case) == 1:
                partition = test_case[0]
                result_evaluator = ObjectiveFunctionEvaluator(self.objective_function)
                number_of_blocks = 8
            else:
                raise ValueError("Too short test case")
            if evaluate_only_objective_function:
                result_evaluator = ObjectiveFunctionEvaluator(self.objective_function)
            for inference_class in self.inference_algorithms:
                for _ in range(repetitions):
                    partition.random_partition(number_of_blocks)
                    resulting_partition = self.execute_single_test(
                        partition,
                        inference_class)

                    result = result_evaluator.evaluate(resulting_partition)
                    evaluated_results.append(result)
                    partitions.append(resulting_partition)

        return evaluated_results, partitions


class SimplePartition(object):
    """
    Simple partition representation without any graph and edge information
    only save both node to block and block to node relationship
    """

    def __init__(self, partition_representation):
        # only empty initial values, content is given by setter
        self._node_to_block = {}
        self._number_of_nodes = 0
        self._number_of_blocks = 0
        self._block_to_node = []
        self._number_of_nodes_per_block = []
        # logic is implemented in the setter of node to block
        self.node_to_block = partition_representation

    @property
    def node_to_block(self):
        """
        Array displaying the relationship of node to block, i.e.
        node_to_block[i] gives the block membership of the i-th node
        """
        return self._node_to_block

    @node_to_block.setter
    def node_to_block(self, new_node_to_block):
        """
        Set membership of nodes to block.
        :param new_node_to_block: dict of memberships
        :type new_node_to_block: list(int)
        """
        self._node_to_block = new_node_to_block
        self._number_of_nodes = len(new_node_to_block)
        self._number_of_blocks = max(new_node_to_block.values()) + 1
        self._block_to_node = [set() for _ in range(self._number_of_blocks)]
        self._number_of_nodes_per_block = [0 for _ in range(self._number_of_blocks)]
        for node in new_node_to_block:
            block = new_node_to_block[node]
            self._block_to_node[block].add(node)
            self._number_of_nodes_per_block[block] += 1

    @property
    def number_of_nodes(self):
        """Total number of nodes"""
        return self._number_of_nodes

    @property
    def number_of_blocks(self):
        """Total number of blocks"""
        return self._number_of_blocks

    def get_number_of_nodes_in_block(self, block):
        """Return number of nodes belonging to the given block"""
        return self._number_of_nodes_per_block[block]

    def get_nodes_of_block(self, block):
        """Return the set of nodes belonging to the given block"""
        return self._block_to_node[block]

    def __str__(self):
        return str(self._node_to_block)

    def __repr__(self):
        return "%s(%r)" % ("SimplePartition", self._node_to_block)


class EvaluateResult(object):
    def evaluate(self, partition):
        """Evaluate a given partition"""
        raise NotImplementedError()


class NormalizedMutualInformation(EvaluateResult):
    def __init__(self, real_partition):
        self._real_partition = SimplePartition(real_partition)
        self._real_partition_entropy = self.entropy(self._real_partition)

    @staticmethod
    def entropy(simple_partition):
        """
        Calculate the entropy of a partition, i.e. the formula
        H(partition)=-sum_{i=1}^{B}p_i log p_i,
        where p_i = (size of block i)/(total number of nodes)
        """
        # cast to shorter variable name
        partition = simple_partition

        entropy = 0.0
        for block in range(partition.number_of_blocks):
            p_i = partition.get_number_of_nodes_in_block(block) \
                  / partition.number_of_nodes
            if p_i != 0:
                entropy -= p_i * math.log(p_i)
        return entropy

    def mutual_information(self, simple_partition):
        """
        Calculate the mutual information between the partition
        (described by the partition representation) and the real partition.
        The mutual information is given by:
        I(partition, real_partition) = sum_{i=1}^{B_partition} sum_{j=1}^{B_real_partition}
        where p_i is given like described in entropy and
        p_{ij} log(p_{ij}/(p_i*p_j))
        with p_ij = number of elements of intersection of the nodes belonging
        to block i of partition and block j of real_partition divided
        by number of nodes
        """
        # cast to shorter variable name
        partition = simple_partition

        # if partition.number_of_nodes != self._real_partition.number_of_nodes:
        #     raise ValueError()

        mutual_information = 0.0
        for i in range(partition.number_of_blocks):
            p_i = partition.get_number_of_nodes_in_block(i) \
                  / partition.number_of_nodes
            for j in range(self._real_partition.number_of_blocks):

                p_ij = len(partition.get_nodes_of_block(i).intersection(
                    self._real_partition.get_nodes_of_block(j))) \
                       / partition.number_of_nodes

                if p_ij != 0:
                    p_j = self._real_partition.get_number_of_nodes_in_block(j) \
                          / self._real_partition.number_of_nodes

                    mutual_information += p_ij * math.log(p_ij / (p_i * p_j))

        return mutual_information

    def evaluate(self, partition):
        """
        Calculate the normalized mutual information comparing the given partition
        to the known real partition.
        The formula is
        normalized_mutual_information(partition, real_partition) =
        mutual_information(partition, real_partition)/
        sqrt(entropy(partition)*entropy(real_partition))

        (For the definitions of mutual information and entropy take a look at
        the corresponding methods implemented in this class)
        """
        if isinstance(partition, SimplePartition):
            simple_partition = partition
        else:
            simple_partition = SimplePartition(partition.get_representation())
        denominator = math.sqrt(self.entropy(simple_partition) * self._real_partition_entropy)
        # defined in that way
        if denominator == 0:
            return 0
        return self.mutual_information(simple_partition) / denominator


class ObjectiveFunctionEvaluator(EvaluateResult):
    """Simple wrapper around objective function"""

    def __init__(self, objective_function):
        self._objective_function = objective_function

    def evaluate(self, partition):
        """Calculate the objective function value of the given partition"""
        return self._objective_function.calculate(partition)
