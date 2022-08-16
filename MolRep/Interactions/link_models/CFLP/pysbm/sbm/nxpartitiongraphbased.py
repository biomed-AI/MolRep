""" Partition of NetworkX Graphs based on internal representation as NetworkX Graph"""

import random as rd

import networkx as nx
import numpy as np

# from pysbm import additional_types
from MolRep.Interactions.link_models.CFLP.pysbm import additional_types
from .partition import Partition
from .exceptions import NoFreeNodeException


class NxPartitionGraphBased(Partition):

    def __init__(self, graph=nx.DiGraph(), number_of_blocks=None, calculate_degree_of_blocks=True,
                 save_neighbor_of_blocks=True, fill_random=True, save_neighbor_edges=False,
                 weighted_graph=False, save_degree_distributions=False, representation=None,
                 with_covariate=False):
        self._graph = graph
        self.partition = {}  # Link of node to groups
        self._is_directed = graph.is_directed()
        self._is_weighted = weighted_graph
        self._with_covariate = with_covariate
        if self._is_directed:
            self._partition_graph_representation = nx.DiGraph()
        else:
            self._partition_graph_representation = nx.Graph()
        self._WA = 'weight'
        self._NODES_A = 'nodes'
        self._NEIGHBORS_A = 'neighbors'
        self._NEIGHBORING_EDGES_A = 'edge neighbors'
        self._DEGREES = 'degrees'
        self._COVARIATE = 'covariate'

        self._save_degree_distributions = save_degree_distributions

        if weighted_graph:
            self._neighbor_edge_class = additional_types.WeightedListDict
        else:
            self._neighbor_edge_class = additional_types.ListDict

        # create node saving data structure
        self.node_neighbors = {}
        self._refresh_node_neighbors()

        super(NxPartitionGraphBased, self).__init__(graph,
                                                    number_of_blocks=number_of_blocks,
                                                    calculate_degree_of_blocks=calculate_degree_of_blocks,
                                                    save_neighbor_of_blocks=save_neighbor_of_blocks,
                                                    fill_random=fill_random,
                                                    save_neighbor_edges=save_neighbor_edges,
                                                    representation=representation,
                                                    with_covariate=with_covariate)
        # Handling of degree method for directed graphs
        if self._is_directed:
            self.get_degree_of_block = self.get_out_degree_of_block

    def is_graph_directed(self):
        return self._is_directed

    def move_node(self, node, to_block):
        # safe old block
        old_block = self.partition[node]
        # fast exit for no movement
        if old_block == to_block:
            return

        # check if null move (only relabeling
        if len(self._partition_graph_representation.nodes[old_block][self._NODES_A]) == 1:
            # quick check if not a one element is moved to another one element block
            if to_block not in self._partition_graph_representation:
                return

        # refresh partition values
        # assign new block
        self.partition[node] = to_block
        if to_block in self._partition_graph_representation:
            # if blocks were not deleted may insert into empty block and have to increase block count
            if self.with_empty_blocks:
                # if node set is empty
                if not self._partition_graph_representation.nodes[to_block][self._NODES_A]:
                    self.B += 1
            self._partition_graph_representation.nodes[to_block][self._NODES_A].add(node)
        else:
            self._partition_graph_representation.add_node(to_block)
            self._partition_graph_representation.nodes[to_block][self._NODES_A] = {node}
            self.B += 1
            if self._save_neighbor_edges:
                self._partition_graph_representation.nodes[to_block][
                    self._NEIGHBORING_EDGES_A] = self._neighbor_edge_class()
            if self._save_neighbor_of_blocks:
                self._partition_graph_representation.nodes[to_block][self._NEIGHBORS_A] = set()
            if self._save_degree_distributions:
                self._partition_graph_representation.nodes[to_block][self._DEGREES] = {}

        # refresh degree distributions if demanded
        if self._save_degree_distributions:
            if self._is_directed:
                if self._is_weighted:
                    degree = (
                        self._graph.in_degree(node, weight=self._WA), self._graph.out_degree(node, weight=self._WA))
                else:
                    degree = (self._graph.in_degree(node), self._graph.out_degree(node))
            else:
                if self._is_weighted:
                    degree = self._graph.degree(node, weight=self._WA)
                else:
                    degree = self._graph.degree(node)
            # add new
            self._partition_graph_representation.nodes[to_block][self._DEGREES][degree] = \
                self._partition_graph_representation.nodes[to_block][self._DEGREES].get(degree, 0) + 1
            # remove old
            if self._partition_graph_representation.nodes[old_block][self._DEGREES][degree] == 1:
                # to only store those entries which are non zero
                del self._partition_graph_representation.nodes[old_block][self._DEGREES][degree]
            else:
                self._partition_graph_representation.nodes[old_block][self._DEGREES][degree] -= 1

        # refresh edges
        if self._is_directed:
            for _, to_node, data in self._graph.out_edges(node, data=True):
                # check default value
                if not self._is_weighted:
                    weight = 1
                else:
                    weight = data[self._WA]
                if node != to_node:
                    self._change_edge_count(node, to_node, weight, data,
                                            old_from_block=old_block)
                else:
                    #  if selfloop concern that from and to block is changed
                    self._change_edge_count(node, node, weight, data,
                                            old_from_block=old_block,
                                            old_to_block=old_block)
            weight = 1
            for from_node, _, data in self._graph.in_edges(node, data=True):
                #  for in edges NetworkX support only full data access therefore this workaround
                if self._is_weighted:
                    weight = data[self._WA]
                if node != from_node:
                    self._change_edge_count(from_node, node, weight, data,
                                            old_to_block=old_block)
                    #  else no change because already changed in successors!
        else:
            for _, to_node, data in self._graph.edges(node, data=True):
                # check default value
                if not self._is_weighted:
                    weight = 1
                else:
                    weight = data[self._WA]
                if node != to_node:
                    self._change_edge_count(node, to_node, weight, data,
                                            old_from_block=old_block)
                else:
                    #  if selfloop concern that from and to block is changed
                    self._change_edge_count(node, node, weight, data,
                                            old_from_block=old_block,
                                            old_to_block=old_block)

        # remove old block
        if len(self._partition_graph_representation.nodes[old_block][self._NODES_A]) > 1:
            self._partition_graph_representation.nodes[old_block][self._NODES_A].discard(node)
        else:
            self.B -= 1
            if not self.with_empty_blocks:
                self._partition_graph_representation.remove_node(old_block)
                # relabel partition to keep block numbers between 0 and B-1
                # first check if anything is to do
                if old_block != self.B:
                    if self._save_neighbor_of_blocks:
                        # if demanded refresh saved neighbors
                        for block in self._partition_graph_representation.nodes[self.B][self._NEIGHBORS_A]:
                            # avoid changing deleted block or relabeling of item in container
                            # while it is used for iterating
                            if block != self.B and block != old_block:
                                self._partition_graph_representation.nodes[block][self._NEIGHBORS_A].discard(self.B)
                                self._partition_graph_representation.nodes[block][self._NEIGHBORS_A].add(old_block)

                        # refresh self reference
                        if self.B in self._partition_graph_representation.nodes[self.B][self._NEIGHBORS_A]:
                            self._partition_graph_representation.nodes[self.B][self._NEIGHBORS_A].discard(self.B)
                            self._partition_graph_representation.nodes[self.B][self._NEIGHBORS_A].add(old_block)

                    nx.relabel_nodes(self._partition_graph_representation, {self.B: old_block}, copy=False)
                    for node in self._partition_graph_representation.nodes[old_block][self._NODES_A]:
                        self.partition[node] = old_block
            else:
                self._partition_graph_representation.nodes[old_block][self._NODES_A].discard(node)

    def _change_edge_count(self, from_node, to_node, weight, edge_data,
                           old_from_block=None, old_to_block=None):
        """ Change value of block edges """
        self._decrease_edge_count(from_node, to_node, weight, edge_data, old_from_block, old_to_block)
        self._add_edge_count(from_node, to_node, weight, edge_data)

    def _decrease_edge_count(self, from_node, to_node, weight, edge_data,
                             old_from_block=None, old_to_block=None):
        # get block membership which were not given
        if old_from_block is None:
            old_from_block = self.partition[from_node]
        if old_to_block is None:
            old_to_block = self.partition[to_node]

        # decrease edge count
        self._partition_graph_representation[old_from_block][old_to_block][self._WA] -= weight
        if self._partition_graph_representation[old_from_block][old_to_block][self._WA] == 0:
            self._partition_graph_representation.remove_edge(old_from_block, old_to_block)
            # if now no edge exist remove neighbor
            if not self._partition_graph_representation.has_edge(old_to_block,
                                                                 old_from_block) and self._save_neighbor_of_blocks:
                self._partition_graph_representation.nodes[old_from_block][self._NEIGHBORS_A].discard(old_to_block)
                self._partition_graph_representation.nodes[old_to_block][self._NEIGHBORS_A].discard(old_from_block)
        elif self._with_covariate:
            # only if edge is not removed decrease edge covariate value
            self._partition_graph_representation[old_from_block][old_to_block][self._COVARIATE] -= edge_data[
                self._COVARIATE]

        # if demanded refresh neighbor edge variables
        if self._save_neighbor_edges:
            if self._is_weighted:
                # remove old ones
                self._partition_graph_representation.nodes[old_to_block][self._NEIGHBORING_EDGES_A].remove(
                    (to_node, from_node),
                    weight)
                self._partition_graph_representation.nodes[old_from_block][self._NEIGHBORING_EDGES_A].remove(
                    (from_node, to_node),
                    weight)
            else:
                # remove old ones
                self._partition_graph_representation.nodes[old_to_block][self._NEIGHBORING_EDGES_A].remove(
                    (to_node, from_node))
                self._partition_graph_representation.nodes[old_from_block][self._NEIGHBORING_EDGES_A].remove(
                    (from_node, to_node))

    def _add_edge_count(self, from_node, to_node, weight, edge_data):
        # add new edge count
        new_from_block = self.partition[from_node]
        new_to_block = self.partition[to_node]

        if self._partition_graph_representation.has_edge(new_from_block, new_to_block):
            self._partition_graph_representation[new_from_block][new_to_block][self._WA] += weight
            if self._with_covariate:
                self._partition_graph_representation[new_from_block][new_to_block][self._COVARIATE] += edge_data[
                    self._COVARIATE]
        else:
            # if new connection of blocks add neighborhood relationship
            if self._save_neighbor_of_blocks and not self._partition_graph_representation.has_edge(new_to_block,
                                                                                                   new_from_block):
                self._partition_graph_representation.nodes[new_from_block][self._NEIGHBORS_A].add(new_to_block)
                self._partition_graph_representation.nodes[new_to_block][self._NEIGHBORS_A].add(new_from_block)
            # then add edge
            self._partition_graph_representation.add_edge(new_from_block, new_to_block)
            self._partition_graph_representation[new_from_block][new_to_block][self._WA] = weight
            if self._with_covariate:
                self._partition_graph_representation[new_from_block][new_to_block][self._COVARIATE] = edge_data[
                    self._COVARIATE]

        # if demanded refresh neighbor edge variables
        if self._save_neighbor_edges:
            if self._is_weighted:
                self._partition_graph_representation.nodes[new_to_block][self._NEIGHBORING_EDGES_A].add(
                    (to_node, from_node),
                    weight)
                self._partition_graph_representation.nodes[new_from_block][self._NEIGHBORING_EDGES_A].add(
                    (from_node, to_node),
                    weight)
            else:
                self._partition_graph_representation.nodes[new_to_block][self._NEIGHBORING_EDGES_A].add(
                    (to_node, from_node))
                self._partition_graph_representation.nodes[new_from_block][self._NEIGHBORING_EDGES_A].add(
                    (from_node, to_node))

    def get_number_of_nodes_in_block(self, block_number):
        return len(self._partition_graph_representation.nodes[block_number][self._NODES_A])

    def get_degree_of_block(self, block_number):
        return self._partition_graph_representation.degree(block_number, weight=self._WA)

    def get_in_degree_of_block(self, block_number):
        return self._partition_graph_representation.in_degree(block_number, weight=self._WA)

    def get_out_degree_of_block(self, block_number):
        return self._partition_graph_representation.out_degree(block_number, weight=self._WA)

    def get_edge_count(self, from_block, to_block):
        if self._partition_graph_representation.has_edge(from_block, to_block):
            if self._is_directed:
                return self._partition_graph_representation[from_block][to_block][self._WA]
            if from_block != to_block:
                return self._partition_graph_representation[from_block][to_block][self._WA]
            return 2 * self._partition_graph_representation[from_block][to_block][self._WA]
        return 0

    def get_sum_of_covariates(self, from_block, to_block):
        """
        Return summed value of covariates between group from_block and to_block.
        Unlike get_edge_count, this method returns the sum in all cases and not twice that value
        if the graph is not directed and from_block is the same as to_block.
        :param from_block: from block
        :param to_block: to block
        :return: sum of covariates between the two blocks
        """
        return self._partition_graph_representation[from_block][to_block][self._COVARIATE]

    def random_partition(self, number_of_blocks=None):
        # First create representation and fill partition with representation afterwards
        representation = {}
        permuted_nodes = np.random.permutation(list(self._graph.nodes.keys()))
        blocks_filled = 0
        if number_of_blocks is None:
            number_of_blocks = rd.randint(1, self._graph.number_of_nodes())
        self.B = number_of_blocks
        # the permutation ensures that not always the same nodes are assigned
        # in the first iteration of the loop to the blocks
        for node in permuted_nodes:
            #  ensure that every block has a node
            if blocks_filled < number_of_blocks:
                representation[node] = blocks_filled
                blocks_filled += 1
            else:
                representation[node] = rd.randint(0, self.B - 1)
        self._fill_from_representation(representation)

    def _fill_from_representation(self, representation):
        # determine number of blocks
        number_of_blocks = max(representation.values()) + 1

        self.B = number_of_blocks
        self._partition_graph_representation.clear()
        self.partition = {}

        for block in range(number_of_blocks):
            self._partition_graph_representation.add_node(block)
            self._partition_graph_representation.nodes[block][self._NODES_A] = set()
            if self._save_neighbor_edges:
                self._partition_graph_representation.nodes[block][
                    self._NEIGHBORING_EDGES_A] = self._neighbor_edge_class()
            if self._save_neighbor_of_blocks:
                self._partition_graph_representation.nodes[block][self._NEIGHBORS_A] = set()
            if self._save_degree_distributions:
                self._partition_graph_representation.nodes[block][self._DEGREES] = {}

        for node in self._graph:
            self.partition[node] = representation[node]
            self._partition_graph_representation.nodes[representation[node]][self._NODES_A].add(node)

            if self._save_degree_distributions:
                if self._is_directed:
                    if self._is_weighted:
                        degree = (
                            self._graph.in_degree(node, weight=self._WA),
                            self._graph.out_degree(node, weight=self._WA))
                    else:
                        degree = (self._graph.in_degree(node), self._graph.out_degree(node))
                else:
                    if self._is_weighted:
                        degree = self._graph.degree(node, weight=self._WA)
                    else:
                        degree = self._graph.degree(node)
                self._partition_graph_representation.nodes[self.partition[node]][self._DEGREES][degree] = \
                    self._partition_graph_representation.nodes[self.partition[node]][self._DEGREES].get(degree, 0) + 1

        # fill edge counts
        for from_node, to_node, data in self._graph.edges(data=True):
            if not self._is_weighted:
                weight = 1
            else:
                weight = data[self._WA]
            self._add_edge_count(from_node, to_node, weight, data)

    def get_random_node(self, no_single=True):
        # first try only random with check
        node = rd.choice(list(self._graph.nodes.keys()))
        if not no_single:
            return node
        if len(self._partition_graph_representation.nodes[self.partition[node]][self._NODES_A]) == 1:
            #    after first try slower
            unchecked_nodes = list(self._graph.nodes.keys())
            unchecked_nodes.remove(node)
            while unchecked_nodes:
                node = rd.choice(unchecked_nodes)
                unchecked_nodes.remove(node)
                if len(self._partition_graph_representation.nodes[self.partition[node]][self._NODES_A]) > 1:
                    break
            else:
                # all nodes in a block with size 1
                raise NoFreeNodeException()
        return node

    def get_random_move(self):
        node = self.get_random_node()
        old_block = self.partition[node]

        # Get random new block and ensure that it differs from that before
        new_block = rd.randint(0, self.B - 1)
        if new_block == old_block:
            new_block += 1
            if new_block == self.B:
                new_block = 0

        return node, old_block, new_block

    def get_random_neighboring_edge_of_block(self, block_number):
        return self._partition_graph_representation.nodes[block_number][self._NEIGHBORING_EDGES_A].choose_random()

    def precalc_move(self, move_candidate, objective_function, covariate=None):

        if covariate is None:
            edge_attribute = self._WA
            use_edge_information = self._is_weighted
        else:
            use_edge_information = True
            edge_attribute = covariate

        node, from_block, to_block = move_candidate
        # Need to calculate:
        # neighbor_blocks: blocks neighboring the node (each block only once)
        # kit: dict key block t value number of edges to block t
        #    -----> both of the above in one dict neighbor_block_info
        # selfloops: number of selfloops
        # degree: degree of node (to move)
        neighbor_block_info = {from_block: 0, to_block: 0}
        selfloops = 0
        for _, neighbor, data in self._graph.edges(node, data=True):
            if use_edge_information:
                weight = data[edge_attribute]
            else:
                weight = 1
            if neighbor == node:
                selfloops += weight
            else:
                block = self.partition[neighbor]
                neighbor_block_info[block] = neighbor_block_info.get(block, 0) + weight
        if self.is_graph_directed():
            # in neighbor_block_info only the information about the successors is saved but
            # for the delta calculation of directed graphs the information about the predecessors is needed too
            predecessor_block_info = {from_block: 0, to_block: 0}
            for neighbor, _, data in self._graph.in_edges(node, data=True):
                # in edges only supply full edge data access therefor cast value or default
                if use_edge_information:
                    weight = data[edge_attribute]
                else:
                    weight = 1
                # selfloops already counted
                if neighbor != node:
                    block = self.partition[neighbor]
                    predecessor_block_info[block] = predecessor_block_info.get(block, 0) + weight

            if use_edge_information:
                parameters = (neighbor_block_info,
                              predecessor_block_info,
                              selfloops,
                              self._graph.in_degree(node, weight=edge_attribute),
                              self._graph.out_degree(node, weight=edge_attribute))
            else:
                parameters = (neighbor_block_info,
                              predecessor_block_info,
                              selfloops,
                              self._graph.in_degree(node),
                              self._graph.out_degree(node))
        else:
            if use_edge_information:
                parameters = (neighbor_block_info, selfloops, self._graph.degree(node, weight=edge_attribute))
            else:
                parameters = (neighbor_block_info, selfloops, self._graph.degree(node))

        if self._with_covariate and covariate is None:
            # if partition includes covariate information get and return the additional information
            return parameters + self.precalc_move(move_candidate, objective_function, covariate=self._COVARIATE)

        return parameters

    def get_neighbors_of_block(self, block_number):
        return self._partition_graph_representation.nodes[block_number][self._NEIGHBORS_A]

    def get_neighbors_of_node(self, node):
        return self.node_neighbors[node]

    def _refresh_node_neighbors(self):
        """
        Refresh internal storage of node neighbors
        """
        if self.is_graph_directed():
            for node in self._graph.nodes:
                # create new list with both entries
                # and for faster access save this data into dictionary
                neighbors = list(self._graph[node].keys())
                for neighbor in self._graph.predecessors(node):
                    # add all nodes which are not already included (successors too)
                    if neighbor not in self._graph[node]:
                        neighbors.append(neighbor)
                self.node_neighbors[node] = neighbors
        else:
            for node in self._graph.nodes:
                self.node_neighbors[node] = list(self._graph[node].keys())

    def get_representation(self):
        return self.partition.copy()

    def copy(self):
        new = self.__class__(self._graph,
                             self.B,
                             self._calculate_degree_of_blocks,
                             self._save_neighbor_of_blocks,
                             fill_random=False,
                             save_neighbor_edges=self._save_neighbor_edges,
                             weighted_graph=self._is_weighted,
                             with_covariate=self._with_covariate)

        new.partition = self.partition.copy()
        new._partition_graph_representation = self._partition_graph_representation.copy()

        # copy membership saving
        for block in self._partition_graph_representation.nodes:
            new._partition_graph_representation.nodes[block][self._NODES_A] = \
                self._partition_graph_representation.nodes[block][self._NODES_A].copy()

        if self._save_neighbor_edges:
            for block in self._partition_graph_representation.nodes:
                new._partition_graph_representation.nodes[block][self._NEIGHBORING_EDGES_A] = \
                    self._partition_graph_representation.nodes[block][self._NEIGHBORING_EDGES_A].copy()

        if self._save_neighbor_of_blocks:
            for block in self._partition_graph_representation.nodes:
                new._partition_graph_representation.nodes[block][self._NEIGHBORS_A] = \
                    self._partition_graph_representation.nodes[block][self._NEIGHBORS_A].copy()

        return new

    def get_nodes_iter(self):
        return self._graph.nodes

    def get_block_of_node(self, node):
        return self.partition[node]

    def get_block_memberships(self):
        return [self.partition[node] for node in sorted(self.partition.keys())]

    def set_save_neighbor_edges(self, save_neighbor_edges):
        # if present before delete edges and save parameter
        if self._save_neighbor_edges and not save_neighbor_edges:
            self._save_neighbor_edges = save_neighbor_edges
            for block in self._partition_graph_representation.nodes:
                del self._partition_graph_representation.nodes[block][self._NEIGHBORING_EDGES_A]
        # if new create list
        if not self._save_neighbor_edges and save_neighbor_edges:
            self._save_neighbor_edges = save_neighbor_edges

            for block in self._partition_graph_representation.nodes:
                self._partition_graph_representation.nodes[block][
                    self._NEIGHBORING_EDGES_A] = self._neighbor_edge_class()

            if self._is_weighted:
                for edge in self._graph.edges(data=self._WA, default=1):
                    # add to_node in list of block of from_node
                    self._partition_graph_representation.nodes[self.partition[edge[0]]][self._NEIGHBORING_EDGES_A].add(
                        (edge[0], edge[1]), edge[2])
                    # to include the "incoming" edges too:
                    self._partition_graph_representation.nodes[self.partition[edge[1]]][self._NEIGHBORING_EDGES_A].add(
                        (edge[1], edge[0]), edge[2])
            else:
                for edge in self._graph.edges():
                    # add to_node in list of block of from_node
                    self._partition_graph_representation.nodes[self.partition[edge[0]]][self._NEIGHBORING_EDGES_A].add(
                        (edge[0], edge[1]))
                    # to include the "incoming" edges too:
                    self._partition_graph_representation.nodes[self.partition[edge[1]]][self._NEIGHBORING_EDGES_A].add(
                        (edge[1], edge[0]))

    def merge_blocks(self, merges, new_block_count):
        # retrieve nodes in right order
        node_mergers = []
        for block in sorted(merges.keys(), reverse=True):
            node_mergers.append(
                (self._partition_graph_representation.nodes[block][self._NODES_A].copy(), merges[block]))

        for nodes, to_block in node_mergers:
            for node in nodes:
                self.move_node(node, to_block)

    def get_number_of_nodes(self):
        return len(self._graph)

    def get_degree_distribution_of_blocks(self, probability=True):

        def create_degree_distribution(degree_iter):
            """local function which creates the degree distribution for a given degree iterator"""
            degree_distributions = [{} for _ in range(self.B)]

            for node, degree in degree_iter:
                degree_distributions[self.partition[node]][degree] = degree_distributions[self.partition[node]].get(
                    degree,
                    0) + 1

            if probability:
                return transform_into_probabilities(degree_distributions)
            return degree_distributions

        def transform_into_probabilities(raw_degree_distributions):
            for block, degree_distribution in enumerate(raw_degree_distributions):
                block_size = 1.0 / len(self._partition_graph_representation.nodes[block][self._NODES_A])
                for degree in degree_distribution:
                    degree_distribution[degree] *= block_size

            return raw_degree_distributions

        if self._graph.is_directed():
            if self._is_weighted:
                return create_degree_distribution(self._graph.in_degree(weight=self._WA)), create_degree_distribution(
                    self._graph.out_degree(weight=self._WA))
            return create_degree_distribution(self._graph.in_degree()), create_degree_distribution(
                self._graph.out_degree())
        # else...
        if self._save_degree_distributions and not self._is_directed:
            saved_degree_distributions = []
            for saved_block in range(self.B):
                saved_degree_distributions.append(
                    self._partition_graph_representation.nodes[saved_block][self._DEGREES].copy())

            if probability:
                return transform_into_probabilities(saved_degree_distributions)
            # else:
            return saved_degree_distributions
        # else:
        if self._is_weighted:
            return create_degree_distribution(self._graph.degree(weight=self._WA))
        return create_degree_distribution(self._graph.degree())

    def get_graph_matrix_representation(self, with_weights=True):
        if with_weights:
            return nx.to_numpy_matrix(self._graph, weight=self._WA)
        return nx.to_numpy_matrix(self._graph)

    @staticmethod
    def _degree_iter(node_and_degree_iter):
        for node, degree in node_and_degree_iter:
            yield degree

    def get_degree_iter(self):
        if self._is_weighted:
            return self._degree_iter(self._graph.degree(weight=self._WA))
        return self._degree_iter(self._graph.degree())

    def get_in_degree_iter(self):
        if self._is_weighted:
            return self._degree_iter(self._graph.in_degree(weight=self._WA))
        return self._degree_iter(self._graph.in_degree())

    def get_out_degree_iter(self):
        if self._is_weighted:
            return self._degree_iter(self._graph.out_degree(weight=self._WA))
        return self._degree_iter(self._graph.out_degree())

    def get_joint_in_out_degree_distribution_of_blocks(self):
        degree_distributions = [{} for _ in range(self.B)]

        # quicker if saving done
        if self._save_degree_distributions and self._is_directed:
            for block in range(self.B):
                degree_distributions[block] = self._partition_graph_representation.nodes[block][self._DEGREES].copy()
            return degree_distributions

        for node in self._graph.nodes:
            if self._is_weighted:
                joint_degrees = (self._graph.in_degree(node, weight=self._WA),
                                 self._graph.out_degree(node, weight=self._WA))
            else:
                joint_degrees = (self._graph.in_degree(node), self._graph.out_degree(node))
            degree_distributions[self.partition[node]][joint_degrees] = degree_distributions[self.partition[node]].get(
                joint_degrees, 0) + 1

        return degree_distributions

    def set_from_representation(self, representation):
        saved_number_of_blocks = self.B
        old_with_empty_blocks = self.with_empty_blocks
        self.with_empty_blocks = True
        super(NxPartitionGraphBased, self).set_from_representation(representation)
        self.with_empty_blocks = old_with_empty_blocks
        # check number of empty blocks
        if self.B != len(self._partition_graph_representation) or self.partition != representation:
            # else only possible if representation contains information about every node
            if len(representation) != len(self._graph):
                raise ValueError()
            # every node is assigned to another block, first assign all nodes block 0
            self.with_empty_blocks = True
            for node in self._graph:
                self.move_node(node, 0)

            # remove nodes and reset counter
            self._partition_graph_representation.remove_nodes_from(
                range(1, max(len(self._partition_graph_representation), saved_number_of_blocks) + 1))

            self.B = 1
            super(NxPartitionGraphBased, self).set_from_representation(representation)
            self.with_empty_blocks = old_with_empty_blocks
            # now the correct number of blocks should be present
            if self.B != len(self._partition_graph_representation):
                raise ValueError()

    def __str__(self):
        output = ""
        for node in sorted(self.partition.keys()):
            if output:
                output += ", "
            output += str(node) + ": " + str(self.partition[node])
        return "{" + output + "}"

    def get_partition_as_a_graph(self):
        """
        Get a reference to the internal graph
        :return: reference to the complete partition information represented as a graph
        :rtype nx.DiGraph()
        """
        return self._partition_graph_representation

    def get_number_of_nodes_with_same_degree_in_block(self, block, degree):
        return self._partition_graph_representation.nodes[block][self._DEGREES].get(degree, 0)

    def get_possible_blocks(self, block, with_in_operation=False):
        return range(self.B)

    def get_degree_distribution_of_single_block(self, block):
        """
        Returns degree distribution of a block. For a directed graph returns the joint degree distribution of a block.
        """
        degree_distribution = {}

        # quicker if saving done
        if self._save_degree_distributions:
            degree_distribution = self._partition_graph_representation.nodes[block][self._DEGREES].copy()
            return degree_distribution

        if self._is_directed:
            for node in self._partition_graph_representation.nodes[block][self._NODES_A]:
                if self._is_weighted:
                    joint_degrees = (self._graph.in_degree(node, weight=self._WA),
                                     self._graph.out_degree(node, weight=self._WA))
                else:
                    joint_degrees = (self._graph.in_degree(node), self._graph.out_degree(node))
                degree_distribution[joint_degrees] = degree_distribution.get(joint_degrees, 0) + 1
        else:
            for node in self._partition_graph_representation.nodes[block][self._NODES_A]:
                if self._is_weighted:
                    degree = self._graph.degree(node, weight=self._WA)
                else:
                    degree = self._graph.degree(node)
                degree_distribution[degree] = degree_distribution.get(degree, 0) + 1

        return degree_distribution

    def get_number_of_edges(self):
        if self._is_weighted:
            return self._graph.size(weight=self._WA)
        return self._graph.size()

    def get_edge_iter_with_covariate(self):
        return self._graph.edges(data=self._COVARIATE)


class NxPartitionGraphBasedWithMoveCounter(NxPartitionGraphBased):
    """Partition including counter of node moves"""

    def __init__(self, graph=nx.DiGraph(), number_of_blocks=None, calculate_degree_of_blocks=True,
                 save_neighbor_of_blocks=True, fill_random=True, save_neighbor_edges=False,
                 weighted_graph=False, save_degree_distributions=False, representation=None,
                 with_covariate=False):
        self.node_moves = 0
        super(NxPartitionGraphBasedWithMoveCounter, self).__init__(
            graph=graph,
            number_of_blocks=number_of_blocks,
            calculate_degree_of_blocks=calculate_degree_of_blocks,
            save_neighbor_of_blocks=save_neighbor_of_blocks,
            fill_random=fill_random,
            save_neighbor_edges=save_neighbor_edges,
            weighted_graph=weighted_graph,
            save_degree_distributions=save_degree_distributions,
            representation=representation,
            with_covariate=with_covariate)

    def move_node(self, node, to_block):
        # only count real moves
        if self.partition[node] != to_block:
            self.node_moves += 1
            super(NxPartitionGraphBasedWithMoveCounter, self).move_node(node, to_block)


class NxPartitionGraphBasedHierarchy(NxPartitionGraphBased):

    def __init__(self, graph=nx.DiGraph(), number_of_blocks=None, calculate_degree_of_blocks=True,
                 save_neighbor_of_blocks=True, fill_random=True, save_neighbor_edges=False, weighted_graph=False,
                 save_degree_distributions=False, representation=None, with_covariate=False):
        super(NxPartitionGraphBasedHierarchy, self).__init__(graph, number_of_blocks, calculate_degree_of_blocks,
                                                             save_neighbor_of_blocks, fill_random,
                                                             save_neighbor_edges, weighted_graph,
                                                             save_degree_distributions,
                                                             representation=representation,
                                                             with_covariate=with_covariate
                                                             )

    def move_node(self, node, to_block, return_edge_changes=False):
        # safe old block
        old_block = self.partition[node]
        # fast exit for no movement
        if old_block == to_block:
            return

        new_edges = []
        old_edges = []
        change_of_number_of_blocks = False
        # check if block will be removed
        if len(self._partition_graph_representation.nodes[old_block][self._NODES_A]) == 1:
            change_of_number_of_blocks = True
            # quick check if not a one element is moved to another one element block
            if to_block not in self._partition_graph_representation:
                return

        # refresh partition values
        # assign new block
        self.partition[node] = to_block
        if to_block in self._partition_graph_representation:
            # if blocks were not deleted may insert into empty block and have to increase block count
            if self.with_empty_blocks:
                # if node set is empty
                if not self._partition_graph_representation.nodes[to_block][self._NODES_A]:
                    self.B += 1
                    change_of_number_of_blocks = True
            self._partition_graph_representation.nodes[to_block][self._NODES_A].add(node)
        else:
            self._partition_graph_representation.add_node(to_block)
            self._partition_graph_representation.nodes[to_block][self._NODES_A] = {node}
            self.B += 1
            change_of_number_of_blocks = True
            if self._save_neighbor_edges:
                self._partition_graph_representation.nodes[to_block][
                    self._NEIGHBORING_EDGES_A] = self._neighbor_edge_class()
            if self._save_neighbor_of_blocks:
                self._partition_graph_representation.nodes[to_block][self._NEIGHBORS_A] = set()
            if self._save_degree_distributions:
                self._partition_graph_representation.nodes[to_block][self._DEGREES] = {}

        # refresh degree distributions if demanded
        if self._save_degree_distributions:
            if self._is_directed:
                if self._is_weighted:
                    degree = (
                        self._graph.in_degree(node, weight=self._WA), self._graph.out_degree(node, weight=self._WA))
                else:
                    degree = (self._graph.in_degree(node), self._graph.out_degree(node))
            else:
                if self._is_weighted:
                    degree = self._graph.degree(node, weight=self._WA)
                else:
                    degree = self._graph.degree(node)
            # add new
            self._partition_graph_representation.nodes[to_block][self._DEGREES][degree] = \
                self._partition_graph_representation.nodes[to_block][self._DEGREES].get(degree, 0) + 1
            # remove old
            if self._partition_graph_representation.nodes[old_block][self._DEGREES][degree] == 1:
                # to only store those entries which are non zero
                del self._partition_graph_representation.nodes[old_block][self._DEGREES][degree]
            else:
                self._partition_graph_representation.nodes[old_block][self._DEGREES][degree] -= 1

        # refresh edges
        if self._is_directed:
            for _, to_node, data in self._graph.out_edges(node, data=True):
                # check default value
                if self._is_weighted:
                    weight = data[self._WA]
                else:
                    weight = 1

                if node != to_node:
                    self._change_edge_count(node, to_node, weight, data,
                                            old_from_block=old_block)
                    # if needed save block edges
                    if change_of_number_of_blocks or return_edge_changes:
                        old_edges.append((old_block, self.partition[to_node], weight))
                        new_edges.append((to_block, self.partition[to_node], weight))
                else:
                    #  if selfloop concern that from and to block is changed
                    self._change_edge_count(node, node, weight, data,
                                            old_from_block=old_block,
                                            old_to_block=old_block)
                    # if needed save block edges
                    if change_of_number_of_blocks or return_edge_changes:
                        old_edges.append((old_block, old_block, weight))
                        new_edges.append((to_block, to_block, weight))

            for from_node, _, data in self._graph.in_edges(node, data=True):
                #  for in edges NetworkX support only full data access therefore this workaround
                if self._is_weighted:
                    weight = data[self._WA]
                else:
                    weight = 1
                if node != from_node:
                    #  else no change because already changed in successors!
                    self._change_edge_count(from_node, node, weight, data,
                                            old_to_block=old_block)
                    # if needed save block edges
                    if change_of_number_of_blocks or return_edge_changes:
                        old_edges.append((self.partition[from_node], old_block, weight))
                        new_edges.append((self.partition[from_node], to_block, weight))
        else:
            for _, to_node, data in self._graph.edges(node, data=True):
                # check default value
                if self._is_weighted:
                    weight = data[self._WA]
                else:
                    weight = 1

                if node != to_node:
                    self._change_edge_count(node, to_node, weight, data,
                                            old_from_block=old_block)
                    # if needed save edges
                    if change_of_number_of_blocks or return_edge_changes:
                        old_edges.append((old_block, self.partition[to_node], weight))
                        new_edges.append((to_block, self.partition[to_node], weight))
                else:
                    #  if selfloop concern that from and to block is changed
                    self._change_edge_count(node, node, weight, data,
                                            old_from_block=old_block,
                                            old_to_block=old_block)
                    # if needed save edges
                    if change_of_number_of_blocks or return_edge_changes:
                        old_edges.append((old_block, old_block, weight))
                        new_edges.append((to_block, to_block, weight))

        # remove old block
        if len(self._partition_graph_representation.nodes[old_block][self._NODES_A]) > 1:
            self._partition_graph_representation.nodes[old_block][self._NODES_A].discard(node)
        else:
            self.B -= 1
            if not self.with_empty_blocks:
                self._partition_graph_representation.remove_node(old_block)
                # relabel partition to keep block numbers between 0 and B-1
                if old_block != self.B:
                    old_relabeled_edges = []
                    new_relabeled_edges = []

                    if self._is_directed:
                        for _, to_block, weight in self._partition_graph_representation.out_edges(self.B,
                                                                                                  data=self._WA):
                            old_relabeled_edges.append((self.B, to_block, weight))
                            if to_block != self.B:
                                new_relabeled_edges.append((old_block, to_block, weight))
                            else:
                                new_relabeled_edges.append((old_block, old_block, weight))
                        for from_block, _, data in self._partition_graph_representation.in_edges(self.B,
                                                                                                 data=True):
                            #  for in edges NetworkX support only full data access therefore this workaround
                            weight = data[self._WA]
                            if from_block != self.B:
                                old_relabeled_edges.append((from_block, self.B, weight))
                                new_relabeled_edges.append((from_block, old_block, weight))
                            # else:
                            #     new_relabeled_edges.append((old_block, old_block, weight))
                    else:
                        for from_block, to_block, weight in self._partition_graph_representation.edges(self.B,
                                                                                                       data=self._WA):
                            old_relabeled_edges.append((self.B, to_block, weight))
                            if to_block != self.B:
                                new_relabeled_edges.append((old_block, to_block, weight))
                            else:
                                new_relabeled_edges.append((old_block, old_block, weight))

                    if self._save_neighbor_of_blocks:
                        # if demanded refresh saved neighbors
                        for block in self._partition_graph_representation.nodes[self.B][self._NEIGHBORS_A]:
                            # handle reference to own block later to avoid changing container while iterating
                            if block != self.B and block != old_block:
                                self._partition_graph_representation.nodes[block][self._NEIGHBORS_A].discard(self.B)
                                self._partition_graph_representation.nodes[block][self._NEIGHBORS_A].add(old_block)

                        # refresh self reference
                        if self.B in self._partition_graph_representation.nodes[self.B][self._NEIGHBORS_A]:
                            self._partition_graph_representation.nodes[self.B][self._NEIGHBORS_A].discard(self.B)
                            self._partition_graph_representation.nodes[self.B][self._NEIGHBORS_A].add(old_block)

                    nx.relabel_nodes(self._partition_graph_representation, {self.B: old_block}, copy=False)
                    for node in self._partition_graph_representation.nodes[old_block][self._NODES_A]:
                        self.partition[node] = old_block

                    return False, old_block, old_edges, new_edges, old_relabeled_edges, new_relabeled_edges
                return False, old_block, old_edges, new_edges, None, None
            else:
                self._partition_graph_representation.nodes[old_block][self._NODES_A].discard(node)
                return False, old_block, old_edges, new_edges, None, None

        if change_of_number_of_blocks:
            return True, old_block, to_block, old_edges, new_edges
        if return_edge_changes:
            return old_edges, new_edges

    def split_node(self, old_node, new_node, old_edges, new_edges):
        if new_node != len(self._graph) - 1:
            raise ValueError()
        # same block assignment
        self.partition[new_node] = self.partition[old_node]
        self._partition_graph_representation.nodes[self.partition[new_node]][self._NODES_A].add(new_node)

        # refresh edge counts
        self.change_saved_edges(old_edges, new_edges)

    def merge_node(self, removed_node, old_edges, new_edges, old_relabeled_edges, new_relabeled_edges):
        # refresh edge counts
        self.change_saved_edges(old_edges, new_edges)

        if removed_node != len(self._graph):
            # cleanup mess from relabeling last node to keep node (block) numbers between 0 ... n (B-1)
            relabeled_node = len(self._graph)
            # delete node in container of nodes belonging to a block
            self._partition_graph_representation.nodes[self.partition[removed_node]][self._NODES_A].discard(
                removed_node)
            # update element info of relabeled node
            self._partition_graph_representation.nodes[self.partition[relabeled_node]][self._NODES_A].discard(
                relabeled_node)
            self._partition_graph_representation.nodes[self.partition[relabeled_node]][self._NODES_A].add(
                removed_node)
            # keep track of block membership: update removed node with the information from the relabeled node
            self.partition[removed_node] = self.partition[relabeled_node]
            # relabel saved edges
            self.change_saved_edges(old_relabeled_edges, new_relabeled_edges)
            # finally delete last entry of partition
            del self.partition[relabeled_node]
        else:
            self._partition_graph_representation.nodes[self.partition[removed_node]][self._NODES_A].discard(
                removed_node)
            del self.partition[removed_node]

    def change_saved_edges(self, old_edges, new_edges):
        """
        Remove the old edges and insert the new edges in the corresponding blocks,
        if saving neighboring edges is enabled.
        :param old_edges: edges to be deleted
        :param new_edges: edges to be inserted
        """
        if self._save_neighbor_edges:
            # distinguish need of weight
            if self._is_weighted:
                # insert new
                for from_node, to_node, weight in new_edges:
                    self._partition_graph_representation.nodes[self.partition[from_node]][
                        self._NEIGHBORING_EDGES_A].add((from_node, to_node), weight)

                    self._partition_graph_representation.nodes[self.partition[to_node]][
                        self._NEIGHBORING_EDGES_A].add((to_node, from_node), weight)

                # remove old
                for from_node, to_node, weight in old_edges:
                    self._partition_graph_representation.nodes[self.partition[from_node]][
                        self._NEIGHBORING_EDGES_A].remove((from_node, to_node), weight)

                    self._partition_graph_representation.nodes[self.partition[to_node]][
                        self._NEIGHBORING_EDGES_A].remove((to_node, from_node), weight)

            else:
                # make no sense for unweighted network because higher level in hierarchy are always weighted
                raise NotImplementedError()
                # # insert new
                # for from_node, to_node, weight in new_edges:
                #     self._partition_graph_representation.nodes[self.partition[from_node]][
                #         self._NEIGHBORING_EDGES_A].add((from_node, to_node))
                #
                #     self._partition_graph_representation.nodes[self.partition[to_node]][
                #         self._NEIGHBORING_EDGES_A].add((to_node, from_node))
                #
                # # remove old
                # for from_node, to_node, weight in old_edges:
                #     self._partition_graph_representation.nodes[self.partition[from_node]][
                #         self._NEIGHBORING_EDGES_A].remove((from_node, to_node))
                #
                #     self._partition_graph_representation.nodes[self.partition[to_node]][
                #         self._NEIGHBORING_EDGES_A].remove((to_node, from_node))

    def handle_change_information(self, *args):
        """
        Determine which of the methods: change_saved_edges, merge_node or split_node is to adress and forward the call
        :param args: parameter returned from move node
        """
        if len(args) == 2:
            self.change_saved_edges(*args)
        elif len(args) == 6:
            self.merge_node(*args[1:])
        elif len(args) == 5:
            self.split_node(*args[1:])
        else:
            raise ValueError()

    # add the moment no extra implementation is needed because first its not needed and secondly not designed to be sure
    #  that all the moves are allowed or a way to update levels above
    # def merge_blocks(self, merges, new_block_count, return_edge_changes=False):
    #     # retrieve nodes in right order
    #     node_mergers = []
    #     for block in sorted(merges.keys(), reverse=True):
    #         node_mergers.append(
    #             (self._partition_graph_representation.nodes[block][self._NODES_A].copy(), merges[block]))
    #
    #     if return_edge_changes:
    #         changes = []
    #         for nodes, to_block in node_mergers:
    #             for node in nodes:
    #                 changes.append(self.move_node(node, to_block, return_edge_changes=True))
    #         return changes
    #     else:
    #         for nodes, to_block in node_mergers:
    #             for node in nodes:
    #                 self.move_node(node, to_block, return_edge_changes=False)

    def get_elements_of_block(self, block):
        return self._partition_graph_representation.nodes[block][self._NODES_A]

    def replace_graph_reference(self, new_reference):
        """
        Replace reference of graph. Still the new graph should contain the same information as the new one.
        No checks are performed.
        """
        self._graph = new_reference

    def switch_graph_delete_lower_level(self, new_graph_reference, old_lower_partition):
        """
        Exchange graph and all belonging information in this partition.
        (All edge counters stays the same. Only works for removal of level direct below this level!)
        :param new_graph_reference: reference to new graph, which is the basis for this partition
        :type new_graph_reference nx.Graph
        :param old_lower_partition: old partition one level below
        :type old_lower_partition NxPartitionGraphBasedHierarchy
        :return:
        """
        save_neighbor_edges = self._save_neighbor_edges
        if save_neighbor_edges:
            # have to create these information from the scratch afterwards
            self.set_save_neighbor_edges(False)

        new_representation = {}
        for block in old_lower_partition.get_nodes_iter():
            # create new partition by stepping over the one level
            new_representation[block] = self.partition[old_lower_partition.get_block_of_node(block)]

        self.replace_graph_reference(new_graph_reference)
        self._fill_from_representation(new_representation, only_update_nodes=True)

        if save_neighbor_edges:
            self.set_save_neighbor_edges(save_neighbor_edges)

    def _fill_from_representation(self, representation, only_update_nodes=False):
        """
        Extended version with additional parameters to only care about the nodes.
        :param representation: New representation
        :param only_update_nodes:
        :return:
        """
        # determine number of blocks
        number_of_blocks = max(representation.values()) + 1

        self.B = number_of_blocks
        if not only_update_nodes:
            self._partition_graph_representation.clear()
        self.partition = {}

        for block in range(number_of_blocks):
            self._partition_graph_representation.add_node(block)
            self._partition_graph_representation.nodes[block][self._NODES_A] = set()
            if self._save_neighbor_edges:
                self._partition_graph_representation.nodes[block][
                    self._NEIGHBORING_EDGES_A] = self._neighbor_edge_class()
            if self._save_neighbor_of_blocks and not only_update_nodes:
                self._partition_graph_representation.nodes[block][self._NEIGHBORS_A] = set()
            if self._save_degree_distributions:
                self._partition_graph_representation.nodes[block][self._DEGREES] = {}

        for node in self._graph:
            self.partition[node] = representation[node]
            self._partition_graph_representation.nodes[representation[node]][self._NODES_A].add(node)

            if self._save_degree_distributions:
                if self._is_directed:
                    if self._is_weighted:
                        degree = (
                            self._graph.in_degree(node, weight=self._WA),
                            self._graph.out_degree(node, weight=self._WA))
                    else:
                        degree = (self._graph.in_degree(node), self._graph.out_degree(node))
                else:
                    if self._is_weighted:
                        degree = self._graph.degree(node, weight=self._WA)
                    else:
                        degree = self._graph.degree(node)
                self._partition_graph_representation.nodes[self.partition[node]][self._DEGREES][degree] = \
                    self._partition_graph_representation.nodes[self.partition[node]][self._DEGREES].get(degree,
                                                                                                        0) + 1

        # fill edge counts
        if not only_update_nodes:
            for from_node, to_node, data in self._graph.edges(data=True):
                if not self._is_weighted:
                    weight = 1
                else:
                    weight = data[self._WA]
                self._add_edge_count(from_node, to_node, weight, data)


class NxHierarchicalPartition(Partition):

    def __init__(self, graph=nx.DiGraph(), number_of_blocks=None, calculate_degree_of_blocks=True,
                 save_neighbor_of_blocks=True, fill_random=False, save_neighbor_edges=False,
                 weighted_graph=False, save_degree_distributions=False, representation=None,
                 with_covariate=False):
        self._is_weighted = weighted_graph
        self._WA = 'weight'
        self._graph = graph
        self._single_partition_parameters = {
            # "number_of_blocks": number_of_blocks,
            "calculate_degree_of_blocks": calculate_degree_of_blocks,
            "save_neighbor_of_blocks": save_neighbor_of_blocks,
            "save_neighbor_edges": save_neighbor_edges,
            "weighted_graph": weighted_graph,
            "save_degree_distributions": save_degree_distributions,
            "with_covariate": with_covariate
        }
        if representation is None:
            if number_of_blocks is None:
                self.partitions = [
                    NxPartitionGraphBasedHierarchy(graph=self._graph,
                                                   fill_random=True,
                                                   **self._single_partition_parameters)]
                number_of_blocks = self.partitions[0].B
            else:
                self.partitions = [
                    NxPartitionGraphBasedHierarchy(graph=self._graph,
                                                   number_of_blocks=number_of_blocks,
                                                   fill_random=True,
                                                   **self._single_partition_parameters)]
        else:
            self.partitions = [
                NxPartitionGraphBasedHierarchy(graph=self._graph,
                                               representation=representation[0],
                                               fill_random=False,
                                               **self._single_partition_parameters)]
            number_of_blocks = max(representation[0].values()) + 1

        # higher levels all weighted
        self._single_partition_parameters["weighted_graph"] = True
        self._single_partition_parameters["fill_random"] = False

        self.actual_level = 0
        self.max_level = 0
        self._is_directed = graph.is_directed()

        if self._is_weighted:
            self.edge_total = self._graph.size(weight=self._WA)
        else:
            self.edge_total = self._graph.size()
        # if self._is_directed:
        #     # in undirected case diagonal values of edge count matrix are counted twice
        #     self.edge_total *= 2

        self._with_covariate = with_covariate
        self._COVARIATE = 'covariate'
        if with_covariate:
            self.covariate_total = self._graph.size(weight=self._COVARIATE)

        super(NxHierarchicalPartition, self).__init__(graph, number_of_blocks, calculate_degree_of_blocks,
                                                      save_neighbor_of_blocks, fill_random,
                                                      save_neighbor_edges, representation=representation)

    @property
    def B(self):
        """Number of blocks"""
        return self.partitions[self.actual_level].B

    @B.setter
    def B(self, number_of_blocks):
        self.partitions[self.actual_level].B = number_of_blocks

    def is_graph_directed(self):
        return self._is_directed

    def move_node(self, node, to_block, check_move=True):
        # if move is allowed, no additional information need to be updated
        if check_move:
            if not self._is_valid_move(node, to_block):
                raise ValueError("node" + str(node) + "of block" + str(
                    self.partitions[self.actual_level].partition[node]) + "->" + str(to_block))
        if self.actual_level == self.max_level:
            # on max level no need to care about moves
            self.partitions[self.actual_level].move_node(node, to_block)
        elif self._save_neighbor_edges:
            # if not max level
            return_value = self.partitions[self.actual_level].move_node(node, to_block, return_edge_changes=True)
            if return_value is not None:
                self.partitions[self.actual_level + 1].handle_change_information(*return_value)
        else:
            return_value = self.partitions[self.actual_level].move_node(node, to_block, return_edge_changes=False)
            if return_value is not None:
                self.partitions[self.actual_level + 1].handle_change_information(*return_value)

    def _is_valid_move(self, node, to_block):
        # on maximal level all moves are allowed
        if self.actual_level == self.max_level:
            return True
        # new block
        if to_block == self.partitions[self.actual_level].B:
            return True
        # else check if from and to block are in the same block
        from_block = self.partitions[self.actual_level].get_block_of_node(node)
        if self.partitions[self.actual_level + 1].get_block_of_node(from_block) != self.partitions[
            self.actual_level + 1].get_block_of_node(to_block):
            return False
        return True

    def get_number_of_nodes_in_block(self, block_number):
        return self.partitions[self.actual_level].get_number_of_nodes_in_block(block_number)

    def get_degree_of_block(self, block_number):
        return self.partitions[self.actual_level].get_degree_of_block(block_number)

    def get_in_degree_of_block(self, block_number):
        return self.partitions[self.actual_level].get_in_degree_of_block(block_number)

    def get_out_degree_of_block(self, block_number):
        return self.partitions[self.actual_level].get_out_degree_of_block(block_number)

    def get_edge_count(self, from_block, to_block):
        if self.actual_level == self.max_level + 1 and from_block == to_block == 0:
            return self.edge_total
        return self.partitions[self.actual_level].get_edge_count(from_block, to_block)

    def get_sum_of_covariates(self, from_block, to_block):
        """
        Return summed value of covariates between group from_block and to_block.
        Unlike get_edge_count, this method returns the sum in all cases and not twice that value
        if the graph is not directed and from_block is the same as to_block.
        :param from_block: from block
        :param to_block: to block
        :return: sum of covariates between the two blocks
        """
        return self.partitions[self.actual_level].get_sum_of_covariates(from_block, to_block)

    def random_partition(self, number_of_blocks=None, level=-1):
        # random hierarchical partition is not needed
        if level == -1 or number_of_blocks is None:
            raise NotImplementedError()
        else:
            saved_level = self.actual_level
            self.actual_level = level
            # generate random partition of a level which is consistent with above partition
            if number_of_blocks < self.get_number_of_blocks_in_level(level + 1):
                # need at least the number of blocks of the level above
                raise ValueError()
            if number_of_blocks > self.get_number_of_blocks_in_level(level - 1):
                # do not generate partitions with empty blocks!
                raise ValueError()

            # in top partition no restrictions
            if level == self.max_level:
                # no restrictions -> simply use normal procedure
                self.partitions[level].random_partition(number_of_blocks=number_of_blocks)
                return

            candidates = additional_types.WeightedListDict()
            partition_grouped_by_top_block = []
            nodes_by_top_block = []
            possible_node_merge_by_top_block = []
            nodes_in_new_blocks_by_top_block = []

            # determine for each node all other nodes in the same block
            #  and move all blocks of a top block into one block
            for top_block in range(self.partitions[level + 1].B):
                nodes = []

                min_block = float('inf')

                for block in self.partitions[level + 1].get_elements_of_block(top_block):
                    nodes.extend(self.partitions[level].get_elements_of_block(block))

                    # determine smallest block number
                    if block < min_block:
                        min_block = block

                # move all nodes into smallest block
                for node in nodes:
                    # do not need to check move, because we know everything is fine
                    self.move_node(node, min_block, check_move=False)

                # initialise all needed data structures
                # create init partition
                partition_grouped_by_top_block.append({node: i for i, node in enumerate(nodes)})
                # fill structure with nodes per block
                nodes_in_new_blocks_by_top_block.append({i: [node] for i, node in enumerate(nodes)})
                # insert nodes of top block
                nodes_by_top_block.append(nodes)
                # insert start value in random select structure
                possible_node_merge_by_top_block.append(-1)
                # fill structure for random weighted selection
                candidates.add(top_block, len(nodes) - 1)

            number_of_merges = self.get_number_of_blocks_in_level(level - 1) - number_of_blocks

            # generate new partition by merging two nodes of a same top block
            for try_counter in range(10 * self.partitions[level].get_number_of_nodes()):
                # check end condition
                if number_of_merges == 0:
                    break
                # choose random top block
                chosen_block = candidates.choose_random()

                # choose random node which former block is a member of top block
                node = rd.choice(nodes_by_top_block[chosen_block])

                if possible_node_merge_by_top_block[chosen_block] != -1:

                    # check if got different node
                    if node == possible_node_merge_by_top_block[chosen_block]:
                        continue

                    second_node = possible_node_merge_by_top_block[chosen_block]

                    if partition_grouped_by_top_block[chosen_block][node] < \
                            partition_grouped_by_top_block[chosen_block][second_node]:
                        lower_block = partition_grouped_by_top_block[chosen_block][node]
                        higher_block = partition_grouped_by_top_block[chosen_block][second_node]

                    elif partition_grouped_by_top_block[chosen_block][node] > \
                            partition_grouped_by_top_block[chosen_block][second_node]:
                        lower_block = partition_grouped_by_top_block[chosen_block][second_node]
                        higher_block = partition_grouped_by_top_block[chosen_block][node]
                    else:
                        # both are equal -> skip
                        continue

                    # update all members of higher block
                    for node_in_higher_block in nodes_in_new_blocks_by_top_block[chosen_block][higher_block]:
                        partition_grouped_by_top_block[chosen_block][node_in_higher_block] = lower_block

                    # update node list
                    nodes_in_new_blocks_by_top_block[chosen_block][lower_block].extend(
                        nodes_in_new_blocks_by_top_block[chosen_block][higher_block])

                    # for security and memory saving delete higher block
                    del nodes_in_new_blocks_by_top_block[chosen_block][higher_block]

                    # decrease count by 1 only if merge happens or new to node was selected
                    candidates.remove(chosen_block, weight=1)

                    # clean up
                    possible_node_merge_by_top_block[chosen_block] = -1
                    number_of_merges -= 1
                else:
                    # perform merge only after two hits on same block
                    possible_node_merge_by_top_block[chosen_block] = node
            else:
                message = "Aimed number of merges: " + str(self.get_number_of_blocks_in_level(level - 1)
                                                           - number_of_blocks) + "\n calculated number of merges: " \
                          + str(self.get_number_of_blocks_in_level(level - 1) - number_of_blocks - number_of_merges) \
                          + "\n missing number of mergers" + str(number_of_merges) \
                          + "\n actual number of blocks: " + str(self.partitions[level].B)
                if level < self.max_level:
                    message += "\nNumber of blocks above: " + str(self.partitions[level + 1].B)
                if level > 0:
                    message += "\nNumber of blocks below: " + str(self.partitions[level - 1].B)
                message += "\n demanded level: " + str(level)
                message += "\n candidates:" + str(candidates)
                message += "\n number of tries: " + str(candidates)
                message += "\nActual representation: " + str(self.get_representation())

                # did not succeed in generating merges
                raise RuntimeError(message)

            next_free_block = self.partitions[level].B
            # next step move all nodes
            # set start point

            # assert next_free_block == self.partitions[level + 1].B

            for blocks_by_top_block in nodes_in_new_blocks_by_top_block:
                for block in blocks_by_top_block:
                    if block == 0:
                        # all nodes in block 0 can remain in the block
                        continue
                    for node in blocks_by_top_block[block]:
                        self.move_node(node, next_free_block, check_move=False)
                    # set new block number
                    next_free_block += 1

            # clean up
            self.actual_level = saved_level

    def _fill_from_representation(self, representation, call_from_init=True):
        # first level already cared, add others to the partition
        for i, level_representation in enumerate(representation):
            if i == 0 and call_from_init:
                continue
            self.add_level_from_representation(level_representation)

    def get_random_node(self, no_single=True, only_movable_nodes=True):
        node = self.partitions[self.actual_level].get_random_node(no_single)
        if not only_movable_nodes or self.actual_level == self.max_level:
            return node
        # check if node can not be moved
        if self.partitions[self.actual_level + 1].get_number_of_nodes_in_block(
                self.partitions[self.actual_level + 1].get_block_of_node(
                    self.partitions[self.actual_level].get_block_of_node(node))) == 1:
            # singleton found try others
            # retrieve all possible nodes
            if self.actual_level == 0:
                unchecked_nodes = list(self.partitions[0].get_nodes_iter())
            else:
                unchecked_nodes = list(range(self.partitions[self.actual_level - 1].B))
            unchecked_nodes.remove(node)
            # check all nodes and select every time a new random one
            while unchecked_nodes:
                node = rd.choice(unchecked_nodes)
                actual_block = self.partitions[self.actual_level].get_block_of_node(node)
                # first check no single
                if no_single and self.partitions[self.actual_level].get_number_of_nodes_in_block(actual_block) == 1:
                    unchecked_nodes.remove(node)
                    continue
                # check if node can be moved
                if self.partitions[self.actual_level + 1].get_number_of_nodes_in_block(
                        self.partitions[self.actual_level + 1].get_block_of_node(
                            actual_block)) > 1:
                    break
                # remove node to only select from possible nodes
                unchecked_nodes.remove(node)
            else:
                raise NoFreeNodeException()
        return node

    def get_random_move(self):
        # returns only allowed moves
        if self.actual_level != self.max_level:
            # if not max level restrict to only allowed moves
            # no node which is only element in block can be moved, so restrict choice
            node = self.get_random_node(no_single=True)
            # retrieve possible blocks and select another one
            actual_block = self.partitions[self.actual_level].get_block_of_node(node)
            # retrieve other blocks in that top level block
            possible_blocks = self.partitions[self.actual_level + 1].get_elements_of_block(
                self.partitions[self.actual_level + 1].get_block_of_node(actual_block))
            # first greedy
            next_block = rd.choice(list(possible_blocks))
            if next_block == actual_block:
                # if not successful remove old block and choose new random
                reduced_possible_blocks = possible_blocks.copy()
                reduced_possible_blocks.discard(actual_block)
                next_block = rd.choice(list(reduced_possible_blocks))
            return node, actual_block, next_block
        else:
            # in maximal level all moves are allowed, so proceed like usual
            return self.partitions[self.actual_level].get_random_move()

    def get_random_neighboring_edge_of_block(self, block_number):
        return self.partitions[self.actual_level].get_random_neighboring_edge_of_block(block_number)

    def precalc_move(self, move_candidate, objective_function):
        # if correct move no change above
        return self.partitions[self.actual_level].precalc_move(move_candidate, objective_function)

    def get_neighbors_of_block(self, block_number):
        return self.partitions[self.actual_level].get_neighbors_of_block(block_number)

    def get_neighbors_of_node(self, node):
        if self.actual_level == 0:
            return self.partitions[self.actual_level].get_neighbors_of_node(node)
        return list(self.partitions[self.actual_level - 1].get_neighbors_of_block(node))

    def get_representation(self):
        representation = []
        for partition in self.partitions:
            representation.append(partition.get_representation())
        return representation

    def set_from_representation(self, representation):
        # like above in get, set in the same way
        # delete all but not first partition to later have correct graphs below
        same_levels = 1
        for same_levels in range(1, min(self.max_level + 1, len(representation))):
            if self.partitions[same_levels].get_representation() != representation[same_levels]:
                break
        del self.partitions[same_levels:]
        self.max_level = same_levels - 1

        # set lowest partitions
        self.partitions[0].set_from_representation(representation[0])

        # create and set missing partitions
        for i in range(same_levels, len(representation)):
            self.add_level_from_representation(representation[i])

    def add_level_from_representation(self, level_representation):
        """Create new partition level at the top of the hierarchy"""
        self.partitions.append(
            NxPartitionGraphBasedHierarchy(graph=self.partitions[self.max_level].get_partition_as_a_graph(),
                                           representation=level_representation,
                                           **self._single_partition_parameters))
        # refresh counter
        self.max_level += 1
        # reset actual level
        self.actual_level = 0

    def copy(self):
        new = NxHierarchicalPartition(self._graph, number_of_blocks=self.B,
                                      calculate_degree_of_blocks=self._calculate_degree_of_blocks,
                                      save_neighbor_of_blocks=self._save_neighbor_of_blocks,
                                      save_neighbor_edges=self._save_neighbor_edges,
                                      fill_random=False,
                                      weighted_graph=self._is_weighted,
                                      with_covariate=self._with_covariate)
        partitions = []
        for level, partition in enumerate(self.partitions):
            new_partition = partition.copy()
            if level > 0:
                # be sure that the internal graphs of the new partitions of the higher levels point to the right
                # graphs below
                new_partition.replace_graph_reference(partitions[level - 1].get_partition_as_a_graph())
            partitions.append(new_partition)
        new.partitions = partitions
        new.actual_level = self.actual_level
        new.max_level = self.max_level
        return new

    def get_nodes_iter(self):
        return self.partitions[self.actual_level].get_nodes_iter()

    def get_block_of_node(self, node):
        return self.partitions[self.actual_level].get_block_of_node(node)

    def get_block_memberships(self):
        output = {}
        for level, partition in enumerate(self.partitions):
            output[level] = partition.get_block_memberships()
        return str(output)

    def set_save_neighbor_edges(self, save_neighbor_edges):
        # change own state
        self._save_neighbor_edges = save_neighbor_edges
        # change value in parameters
        self._single_partition_parameters["save_neighbor_edges"] = save_neighbor_edges
        # and the state of all partitions in the right order
        for partition in self.partitions:
            partition.set_save_neighbor_edges(save_neighbor_edges)

    def merge_blocks(self, merges, new_block_count):
        # retrieve nodes in right order
        node_mergers = []
        for block in sorted(merges.keys(), reverse=True):
            node_mergers.append(
                (self.partitions[self.actual_level].get_elements_of_block(block).copy(), merges[block]))

        for nodes, to_block in node_mergers:
            for node in nodes:
                self.move_node(node, to_block)

    def get_number_of_nodes(self):
        return self.partitions[self.actual_level].get_number_of_nodes()

    def get_degree_distribution_of_blocks(self, probability=True):
        return self.partitions[self.actual_level].get_degree_distribution_of_blocks(probability)

    def get_graph_matrix_representation(self, with_weights=True):
        return self.partitions[self.actual_level].get_graph_matrix_representation()

    def get_degree_iter(self):
        return self.partitions[self.actual_level].get_degree_iter()

    def get_in_degree_iter(self):
        return self.partitions[self.actual_level].get_in_degree_iter()

    def get_out_degree_iter(self):
        return self.partitions[self.actual_level].get_out_degree_iter()

    def get_number_of_edges(self):
        return self.edge_total

    def get_edge_iter_with_covariate(self):
        return self.partitions[self.actual_level].get_edge_iter_with_covariate()

    def get_joint_in_out_degree_distribution_of_blocks(self):
        return self.partitions[self.actual_level].get_joint_in_out_degree_distribution_of_blocks()

    def get_number_of_nodes_with_same_degree_in_block(self, block, degree):
        return self.partitions[self.actual_level].get_number_of_nodes_with_same_degree_in_block(block, degree)

    def get_elements_of_block(self, block):
        return self.partitions[self.actual_level].get_elements_of_block(block)

    def get_degree_distribution_of_single_block(self, block):
        """
        Returns degree distribution of a block. For a directed graph returns the joint degree distribution of a block.
        """
        return self.partitions[self.actual_level].get_degree_distribution_of_single_block(block)

    def iter_levels(self):
        """Return level iterator"""
        return range(len(self.partitions))

    def get_number_of_blocks_in_level(self, level):
        if level == -1:
            return len(self.graph)
        elif level == self.max_level + 1:
            return 1
        else:
            return self.partitions[level].B

    def get_possible_blocks(self, block, with_in_operation=False):
        if self.actual_level == self.max_level:
            return range(self.partitions[self.actual_level].B)
        else:
            if with_in_operation:
                return self.partitions[self.actual_level + 1].get_elements_of_block(
                    self.partitions[self.actual_level + 1].get_block_of_node(block))
            return list(self.partitions[self.actual_level + 1].get_elements_of_block(
                self.partitions[self.actual_level + 1].get_block_of_node(block)))

    def delete_actual_level(self):
        if self.actual_level == 0:
            raise NotImplementedError()
        if self.actual_level == self.max_level:
            del self.partitions[self.actual_level]
            self.max_level -= 1
            self.actual_level -= 1

        else:
            # # need to perform the changes
            # for node in self.partitions[self.actual_level].get_nodes_iter():
            #     if self.partitions[self.actual_level].get_number_of_nodes_in_block(
            #             self.partitions[self.actual_level].get_block_of_node(node)) > 1:
            #         # move to new node
            #         self.move_node(node, self.partitions[self.actual_level].B, check_move=False)
            #
            # self.partitions[self.actual_level + 1]._partition_graph_representation = self.partitions[
            #     self.actual_level].get_partition_as_a_graph()

            self.partitions[self.actual_level + 1].switch_graph_delete_lower_level(
                self.partitions[self.actual_level - 1].get_partition_as_a_graph(), self.partitions[self.actual_level])

            # perform cleaning
            del self.partitions[self.actual_level]
            self.max_level -= 1
            self.actual_level -= 1

    def add_level_above_actual_level(self):
        # create new partition based on below graph and each block as a singleton
        new_level_partition = NxPartitionGraphBasedHierarchy(
            graph=self.partitions[self.actual_level].get_partition_as_a_graph(),
            representation={block: block for block in range(self.partitions[self.actual_level].B)},
            **self._single_partition_parameters)

        if self.actual_level != self.max_level:
            # if above level is present, exchange graph reference
            self.partitions[self.actual_level + 1].replace_graph_reference(
                new_level_partition.get_partition_as_a_graph())

        self.partitions.insert(self.actual_level + 1, new_level_partition)
        self.max_level += 1
