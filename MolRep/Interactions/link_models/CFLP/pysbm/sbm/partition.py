"""
Partition of a graph into different groups with saving a lot of additional information
"""
import random as rd

import networkx as nx
import numpy as np

# from pysbm import additional_types
from MolRep.Interactions.link_models.CFLP.pysbm import additional_types

from .exceptions import NoFreeNodeException


class Partition(object):
    """ Partition of a graph """

    def __init__(self, graph, number_of_blocks=None, calculate_degree_of_blocks=True,
                 save_neighbor_of_blocks=True, fill_random=True, save_neighbor_edges=False, representation=None,
                 with_covariate=False):
        self._graph = graph
        self.B = number_of_blocks
        self._calculate_degree_of_blocks = calculate_degree_of_blocks
        self._save_neighbor_of_blocks = save_neighbor_of_blocks
        self._save_neighbor_edges = save_neighbor_edges
        self.with_empty_blocks = False
        if fill_random:
            self.random_partition(number_of_blocks)
        if representation is not None:
            self._fill_from_representation(representation)

        self._with_covariate = with_covariate

    @property
    def with_covariate(self):
        return self._with_covariate

    @property
    def graph(self):
        """Get the graph over which the partition is"""
        return self._graph

    def is_graph_directed(self):
        """ Return True if Graph is directed """
        raise NotImplementedError()

    def move_node(self, node, to_block):
        """ Change Block Membership of a single node """
        raise NotImplementedError()


    def get_number_of_nodes_in_block(self, block_number):
        """ Get Number of nodes """
        raise NotImplementedError()

    def get_degree_of_block(self, block_number):
        """
        Return summed degree of nodes in block
            - only works with calculate_degree_of_blocks = True
            In directed case returns out degree of block. For in degree of Block
            use get_in_degree_of_block
        """
        raise NotImplementedError()

    def get_in_degree_of_block(self, block_number):
        """
        Return summed in degree of nodes in block
            - only works with calculate_degree_of_blocks = True
        """
        raise NotImplementedError()

    def get_out_degree_of_block(self, block_number):
        """
        Return summed out degree of nodes in block
            - only works with calculate_degree_of_blocks = True
        """
        raise NotImplementedError()

    def get_edge_count(self, from_block, to_block):
        """Get number of edges between the two given blocks"""
        raise NotImplementedError()

    def random_partition(self, number_of_blocks=None):
        """ Fill partition with random partition """
        raise NotImplementedError()

    def get_random_node(self, no_single=True):
        """
        Return random node of the graph.
        If no_single is True check, if node is only node of block.
        """
        raise NotImplementedError()

    def get_random_move(self):
        """ Return triple of random node, old block and random new block """
        raise NotImplementedError()

    def get_random_neighboring_edge_of_block(self, block_number):
        """ 
        Return a random edge adjacent to given block. 
        Important: on directed includes links pointing to a block too.
        For easier handling all edges are stored in the format
        (node of block_number, node of other block)
        even in the case the real edge in a directed graph is the other way round.
        """
        raise NotImplementedError()

    def precalc_move(self, move_candidate, objective_function):
        """ Calculate arguments for delta of objective function """
        raise NotImplementedError()

    def get_neighbors_of_block(self, block_number):
        """ Get neighboring blocks of given block """
        raise NotImplementedError()

    def get_neighbors_of_node(self, node):
        """ Get neighbors of node """
        raise NotImplementedError()

    def get_representation(self):
        """ Return in short form partition of nodes """
        raise NotImplementedError()

    def set_from_representation(self, representation):
        """ Set block membership from representation, which is a dict of nodes (key) an corresponding block (value)."""
        for node in representation:
            self.move_node(node, representation[node])

    def copy(self):
        """ Return a copy of the partition """
        raise NotImplementedError()

    def get_nodes_iter(self):
        """ Return iteration over all nodes """
        raise NotImplementedError()

    def get_block_of_node(self, node):
        """ Return the block of a node """
        raise NotImplementedError()

    def get_block_memberships(self):
        """ Return list with all block memberships.
            Needed e.g. for drawing.
        """
        raise NotImplementedError()

    def set_save_neighbor_edges(self, save_neighbor_edges):
        """ Change flag save_neighbor_edges """
        raise NotImplementedError()

    def merge_blocks(self, merges, new_block_count):
        """
        Change block membership of nodes according to merges.
            :merges: sorted dict of node merges (ascending)
        """
        raise NotImplementedError()

    def get_number_of_nodes(self):
        """Return number of nodes in graph"""
        raise NotImplementedError()

    def get_degree_distribution_of_blocks(self, probability=True):
        """
        Return degree distribution {p_i}_i for each block,
        i.e. a node in the given block has the degree i with the probability p_i.
        """
        raise NotImplementedError()

    def get_graph_matrix_representation(self, with_weights=True):
        """
        Return adjacency matrix representation as numpy matrix
        :param with_weights: include edge weights
        :return: (weighted) adjacency matrix
        :type with_weights bool
        :rtype np.matrix
        """
        raise NotImplementedError()

    def get_degree_iter(self):
        """
        Return iterator of node degrees of the graph
        :return: iterator of node degrees
        """
        raise NotImplementedError()

    def get_in_degree_iter(self):
        """
        Return iterator of node in degrees of the graph
        :return: iterator of node in degrees
        """
        raise NotImplementedError()

    def get_out_degree_iter(self):
        """
        Return iterator of node out degrees of the graph
        :return: iterator of node out degrees
        """
        raise NotImplementedError()

    def get_joint_in_out_degree_distribution_of_blocks(self):
        """
        Return iterator of joint in-out-degree distribution of the blocks
        :return: list of dicts with joint distribution of each block
        """
        raise NotImplementedError()

    def _fill_from_representation(self, representation):
        """
        Fill Partition from the given representation
        """
        raise NotImplementedError()

    def get_possible_blocks(self, block, with_in_operation=False):
        """
        Return all possible blocks for a given block
        :param block:
        :param with_in_operation: return object which supports x in object operation
        :return: list of possible blocks
        """
        raise NotImplementedError()

    def get_number_of_edges(self):
        """
        Returns the number of edges of the graph.
        If the partition is weighted the sum of all edge weights is returned.
        :return: (weighted) number of edges
        """
        raise NotImplementedError()

    def get_sum_of_covariates(self, from_block, to_block):
        """
        Return summed value of covariates between group from_block and to_block.
        Unlike get_edge_count, this method returns the sum in all cases and not twice that value
        if the graph is not directed and from_block is the same as to_block.
        :param from_block: from block
        :param to_block: to block
        :return: sum of covariates between the two blocks
        """
        raise NotImplementedError()


class NxPartition(Partition):
    """
    Partition of NetworkX Graph
    At the moment only weights no covariates are implemented
    and beside block merging, empty blocks will be kept
    """

    def __init__(self, graph=nx.DiGraph(), number_of_blocks=None, calculate_degree_of_blocks=True,
                 save_neighbor_of_blocks=True, fill_random=True, save_neighbor_edges=False,
                 weighted_graph=False, representation=None):
        self._is_weighted = weighted_graph
        # centrally store name of weight attribute (=WA)
        self._WA = 'weight'
        self._graph = graph
        self.partition = {}
        self.number_of_nodes_per_block = np.zeros(1, dtype=int)
        self.block_edges = np.zeros((1, 1))
        if calculate_degree_of_blocks:
            self.block_degrees = np.zeros(1)
            if self.is_graph_directed():
                self.block_in_degrees = np.zeros(1)
        if save_neighbor_of_blocks:
            self.list_of_block_neighbors = [set()]
        if save_neighbor_edges:
            if self._is_weighted:
                self.neighboring_edges = [additional_types.WeightedListDict()]
            else:
                self.neighboring_edges = [additional_types.ListDict()]
        super(NxPartition, self).__init__(graph, number_of_blocks, calculate_degree_of_blocks,
                                          save_neighbor_of_blocks, fill_random,
                                          save_neighbor_edges,
                                          representation=representation)
        # in the moment only implemented with empty blocks
        self.with_empty_blocks = True

        if self.is_graph_directed():
            # data structure for node saving because NetworkX only supplies
            # predecessors or successor access and not the unity of both
            self.node_neighbors = {}

    def get_out_degree_of_block(self, block_number):
        return self.block_degrees[block_number]

    def _add_node(self, node, block):
        """
        Add node to block. Careful use no edge count refreshing.
        """
        self.partition[node] = block
        self.number_of_nodes_per_block[block] += 1
        if self._calculate_degree_of_blocks:
            if self._graph.is_directed():
                self.block_degrees[block] += self._graph.out_degree(node,
                                                                    weight=self._WA)
            else:
                self.block_degrees[block] += self._graph.degree(node, weight=self._WA)

    def _add_edge_count(self, from_node, to_node, weight=1):
        """ Increase value of block_edges
        """
        from_block = self.partition[from_node]
        to_block = self.partition[to_node]

        # if needed refresh block neighbor
        if self._save_neighbor_of_blocks:
            if self.block_edges[from_block][to_block] == 0:
                self.list_of_block_neighbors[from_block].add(to_block)
                #  neighboring block in direct case in both directions
                if self.is_graph_directed():
                    self.list_of_block_neighbors[to_block].add(from_block)

                    # if needed refresh outgoing edges
        if self._save_neighbor_edges:
            # save all edges even those with aim to the same block
            if self._is_weighted:
                # noinspection PyArgumentList
                self.neighboring_edges[from_block].add((from_node, to_node), weight)
                if self.is_graph_directed():
                    # noinspection PyArgumentList
                    self.neighboring_edges[to_block].add((to_node, from_node), weight)
            else:
                self.neighboring_edges[from_block].add((from_node, to_node))
                if self.is_graph_directed():
                    self.neighboring_edges[to_block].add((to_node, from_node))

        self.block_edges[from_block][to_block] += weight

        # if directed change in degree too
        if self.is_graph_directed() and self._calculate_degree_of_blocks:
            self.block_in_degrees[to_block] += weight

    def _remove_edge_count(self, from_node, to_node, weight,
                           from_node_block=None, to_node_block=None):
        """ Decrease value of block_edges """
        # get block membership which were not given
        if from_node_block is None:
            from_node_block = self.partition[from_node]
        if to_node_block is None:
            to_node_block = self.partition[to_node]

        self.block_edges[from_node_block][to_node_block] -= weight

        # if needed refresh block neighbor
        if self._save_neighbor_of_blocks:
            # double check for directed graphs
            if self.block_edges[from_node_block][to_node_block] == 0 and \
                    self.block_edges[to_node_block][from_node_block] == 0:
                self.list_of_block_neighbors[from_node_block].discard(to_node_block)
                self.list_of_block_neighbors[to_node_block].discard(from_node_block)

                # if needed refresh outgoing edges
        if self._save_neighbor_edges:
            if self._is_weighted:
                # noinspection PyArgumentList
                self.neighboring_edges[from_node_block].remove((from_node, to_node), weight)
                if self.is_graph_directed():
                    # noinspection PyArgumentList
                    self.neighboring_edges[to_node_block].remove((to_node, from_node), weight)
            else:
                self.neighboring_edges[from_node_block].remove((from_node, to_node))
                if self.is_graph_directed():
                    self.neighboring_edges[to_node_block].remove((to_node, from_node))

                    # if directed change in degree too
        if self.is_graph_directed() and self._calculate_degree_of_blocks:
            self.block_in_degrees[to_node_block] -= weight

    def _change_edge_count(self, from_node, to_node, weight=1,
                           old_from_block=None, old_to_block=None):
        """ Change value of block edges """
        # check default value
        if not self._is_weighted:
            weight = 1
        # add to new and remove from old
        self._remove_edge_count(from_node,
                                to_node,
                                weight,
                                from_node_block=old_from_block,
                                to_node_block=old_to_block)
        self._add_edge_count(from_node, to_node, weight)

    def is_graph_directed(self):
        return self._graph.is_directed()

    def move_node(self, node, to_block):
        # save old block and set new block
        old_block = self.partition[node]
        # quick exit for no change
        if old_block == to_block:
            return
        self.partition[node] = to_block

        # refresh number of nodes in blocks (actual empty blocks are allowed)
        self.number_of_nodes_per_block[old_block] -= 1
        self.number_of_nodes_per_block[to_block] += 1

        # refresh block degree
        if self._calculate_degree_of_blocks:
            if self._graph.is_directed():
                self.block_degrees[old_block] -= self._graph.out_degree(node, weight=self._WA)
                self.block_degrees[to_block] += self._graph.out_degree(node, weight=self._WA)
            else:
                self.block_degrees[old_block] -= self._graph.degree(node, weight=self._WA)
                self.block_degrees[to_block] += self._graph.degree(node, weight=self._WA)

                # refresh values of edge counts
        if self._graph.is_directed():
            #  if selfloop concern that from and to block is changed
            for _, to_node, weight in self._graph.out_edges(node, data=self._WA):
                if node != to_node:
                    self._change_edge_count(node, to_node, weight,
                                            old_from_block=old_block)
                else:
                    self._change_edge_count(node, node, weight,
                                            old_from_block=old_block,
                                            old_to_block=old_block)
            weight = 1
            for from_node, _, data in self._graph.in_edges(node, data=True):
                #  for in edges NetworkX support only full data access therefore this workaround
                if self._is_weighted:
                    weight = data[self._WA]
                if node != from_node:
                    self._change_edge_count(from_node, node, weight,
                                            old_to_block=old_block)
                    #  else no change because already changed in successors!
        else:
            for _, to_node, weight in self._graph.edges(node, data=self._WA):
                # for to_node in self._graph.neighbors(node):
                #  if selfloop concern that from and to block is changed
                if node != to_node:
                    self._change_edge_count(node, to_node, weight,
                                            old_from_block=old_block)
                    self._change_edge_count(to_node, node, weight,
                                            old_to_block=old_block)
                else:
                    self._change_edge_count(node, node, weight,
                                            old_from_block=old_block,
                                            old_to_block=old_block)
                    self._change_edge_count(node, node, weight,
                                            old_from_block=old_block,
                                            old_to_block=old_block)

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

        # create internal data structures
        self.B = number_of_blocks
        self.number_of_nodes_per_block = np.zeros(number_of_blocks, dtype=int)
        self.block_edges = np.zeros((number_of_blocks, number_of_blocks))

        # if needed create extra data structure for degree, neighbors and neighboring edges
        if self._calculate_degree_of_blocks:
            self.block_degrees = np.zeros(number_of_blocks)
            if self.is_graph_directed():
                self.block_in_degrees = np.zeros(number_of_blocks)

        if self._save_neighbor_of_blocks:
            self.list_of_block_neighbors = [set() for _ in range(self.B)]

        if self._save_neighbor_edges:
            if self._is_weighted:
                self.neighboring_edges = [additional_types.WeightedListDict() for _ in range(self.B)]
            else:
                self.neighboring_edges = [additional_types.ListDict() for _ in range(self.B)]

        # assign nodes like given in the representation
        for node in self._graph:
            self._add_node(node, representation[node])

        # fill edge counts
        for from_node, to_node, weight in self._graph.edges(data=self._WA, default=1):
            self._add_edge_count(from_node, to_node, weight)
            # in undirected case create symmetric matrix (with double the value on the diagonal)
            if not self._graph.is_directed():
                self._add_edge_count(to_node, from_node, weight)

    def get_number_of_nodes_in_block(self, block_number):
        return self.number_of_nodes_per_block[block_number]

    def get_degree_of_block(self, block_number):
        return self.block_degrees[block_number]

    def get_in_degree_of_block(self, block_number):
        return self.block_in_degrees[block_number]

    def get_edge_count(self, from_block, to_block):
        return self.block_edges[from_block][to_block]

    def get_neighbors_of_block(self, block_number):
        return self.list_of_block_neighbors[block_number]

    def get_neighbors_of_node(self, node):
        # in directed case return predecessors and successors
        if self.is_graph_directed():
            # create new list with both entries
            # and for faster access save this data into dictionary
            if node in self.node_neighbors:
                return self.node_neighbors[node]
            neighbors = list(self._graph[node].keys())
            for neighbor in self._graph.predecessors(node):
                # add all nodes which are not already included (successors too)
                if neighbor not in self._graph[node]:
                    neighbors.append(neighbor)
            self.node_neighbors[node] = neighbors
            return neighbors
        # return list(set(nx.all_neighbors(self._graph, node)))
        else:
            return list(self._graph[node].keys())

    def get_random_node(self, no_single=True):
        # first try only random with check
        node = rd.choice(list(self._graph.nodes.keys()))
        if not no_single:
            return node
        if self.number_of_nodes_per_block[self.partition[node]] == 1:
            #    after first try slower
            unchecked_nodes = list(self._graph.nodes.keys())
            unchecked_nodes.remove(node)
            while unchecked_nodes:
                node = rd.choice(unchecked_nodes)
                unchecked_nodes.remove(node)
                if self.number_of_nodes_per_block[self.partition[node]] > 1:
                    break
            else:
                # all nodes in a block with size 1
                raise NoFreeNodeException()
        return node

    def get_random_move(self):
        """ Return triple of random node, old block and random new block """
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
        return self.neighboring_edges[block_number].choose_random()

    def precalc_move(self, move_candidate, objective_function):
        """ Calculate arguments for delta of objective function """
        node, from_block, to_block = move_candidate
        # Need to calculate:
        # neighbor_blocks: blocks neighboring the node (each block only once)
        # kit: dict key block t value number of edges to block t
        #    -----> both of the above in one dict neighbor_block_info
        # selfloops: number of selfloops
        # degree: degree of node (to move)
        neighbor_block_info = {from_block: 0, to_block: 0}
        selfloops = 0
        for _, neighbor, weight in self._graph.edges(node, data=self._WA, default=1):
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
                if self._is_weighted:
                    weight = data[self._WA]
                else:
                    weight = 1
                # selfloops already counted
                if neighbor != node:
                    block = self.partition[neighbor]
                    predecessor_block_info[block] = predecessor_block_info.get(block, 0) \
                                                    + weight

            parameters = (neighbor_block_info,
                          predecessor_block_info,
                          selfloops,
                          self._graph.in_degree(node, weight=self._WA),
                          self._graph.out_degree(node, weight=self._WA))
        else:
            parameters = (neighbor_block_info, selfloops, self._graph.degree(node, weight=self._WA))

        return parameters

    def get_representation(self):
        return self.partition.copy()

    def __str__(self):
        output = ""
        for node in sorted(self.partition.keys()):
            if output:
                output += ", "
            output += str(node) + ": " + str(self.partition[node])
        return "{" + output + "}"

    def copy(self):
        new = NxPartition(self._graph,
                          self.B,
                          self._calculate_degree_of_blocks,
                          self._save_neighbor_of_blocks,
                          fill_random=False,
                          save_neighbor_edges=self._save_neighbor_edges,
                          weighted_graph=self._is_weighted)

        new.partition = self.partition.copy()
        new.block_edges = self.block_edges.copy()
        new.number_of_nodes_per_block = self.number_of_nodes_per_block.copy()

        if self._calculate_degree_of_blocks:
            new.block_degrees = self.block_degrees.copy()
            if self.is_graph_directed():
                new.block_in_degrees = self.block_in_degrees.copy()

        if self._save_neighbor_of_blocks:
            # to get a real copy of the sets, copying is needed
            new.list_of_block_neighbors = []
            for neighbor in self.list_of_block_neighbors:
                new.list_of_block_neighbors.append(neighbor.copy())

        if self._save_neighbor_edges:
            new.neighboring_edges = []
            for edges in self.neighboring_edges:
                new.neighboring_edges.append(edges.copy())

        return new

    def get_nodes_iter(self):
        return iter(self._graph.nodes)

    def get_block_of_node(self, node):
        return self.partition[node]

    def get_block_memberships(self):
        return [self.partition[node] for node in sorted(self.partition.keys())]

    def set_save_neighbor_edges(self, save_neighbor_edges):
        # if present before delete edges and save parameter
        if self._save_neighbor_edges and not save_neighbor_edges:
            self._save_neighbor_edges = save_neighbor_edges
            del self.neighboring_edges
        # if new create list
        if not self._save_neighbor_edges and save_neighbor_edges:
            self._save_neighbor_edges = save_neighbor_edges

            if self._is_weighted:
                self.neighboring_edges = [additional_types.WeightedListDict() for _ in range(self.B)]
                for edge in self._graph.edges(data=self._WA, default=1):
                    # add to_node in list of block of from_node
                    self.neighboring_edges[self.partition[edge[0]]].add((edge[0], edge[1]), edge[2])
                    # to include the "incoming" edges too:
                    self.neighboring_edges[self.partition[edge[1]]].add((edge[1], edge[0]), edge[2])
            else:
                self.neighboring_edges = [additional_types.ListDict() for _ in range(self.B)]
                for edge in self._graph.edges():
                    # add to_node in list of block of from_node
                    self.neighboring_edges[self.partition[edge[0]]].add((edge[0], edge[1]))
                    # to include the "incoming" edges too:
                    self.neighboring_edges[self.partition[edge[1]]].add((edge[1], edge[0]))

    def merge_blocks(self, merges, new_block_count):
        # no checks!
        for node in self.partition:
            if self.partition[node] in merges:
                self.move_node(node, merges[self.partition[node]])
                # refresh block count
        self.B = new_block_count

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
                for block, degree_distribution in enumerate(degree_distributions):
                    block_size = 1.0 / self.number_of_nodes_per_block[block]
                    for degree in degree_distribution:
                        degree_distribution[degree] *= block_size

            return degree_distributions

        if self._graph.is_directed():
            if self._is_weighted:
                return create_degree_distribution(self._graph.in_degree(weight=self._WA)), create_degree_distribution(
                    self._graph.out_degree(weight=self._WA))
            return create_degree_distribution(self._graph.in_degree()), create_degree_distribution(
                self._graph.out_degree())
        # else...
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

        for node in self._graph.nodes:
            if self._is_weighted:
                joint_degrees = (self._graph.in_degree(node, weight=self._WA),
                                 self._graph.out_degree(node, weight=self._WA))
            else:
                joint_degrees = (self._graph.in_degree(node), self._graph.out_degree(node))
            degree_distributions[self.partition[node]][joint_degrees] = degree_distributions[self.partition[node]].get(
                joint_degrees, 0) + 1

        return degree_distributions

    def get_possible_blocks(self, block, with_in_operation=False):
        return range(self.B)

    def get_number_of_edges(self):
        if self._is_weighted:
            return self._graph.size(weight=self._WA)
        return self._graph.size()

    def get_sum_of_covariates(self, from_block, to_block):
        raise NotImplementedError()


class NxPartitionWithMoveCounter(NxPartition):
    """Partition which counts the number of performed node moves"""

    def __init__(self, graph, number_of_blocks=None, calculate_degree_of_blocks=True, save_neighbor_of_blocks=True,
                 fill_random=True, save_neighbor_edges=False, weighted_graph=False, representation=None):
        self.node_moves = 0
        super(NxPartitionWithMoveCounter, self).__init__(graph, number_of_blocks, calculate_degree_of_blocks,
                                                         save_neighbor_of_blocks, fill_random,
                                                         save_neighbor_edges, weighted_graph, representation)

    def move_node(self, node, to_block):
        # only count real moves
        if self.partition[node] != to_block:
            self.node_moves += 1
            super(NxPartitionWithMoveCounter, self).move_node(node, to_block)

    def get_sum_of_covariates(self, from_block, to_block):
        raise NotImplementedError()
