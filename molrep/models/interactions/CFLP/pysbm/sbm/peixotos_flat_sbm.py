# Flat versions of objective functions described in Tiago P. Peixoto
# Nonparametric Bayesian inference of the microcanonical stochastic block model
# for more see in peixotos_hierarchical_sbm.py
# @formatter:off

import math


from .objective_functions import ObjectiveFunction
from .peixotos_hierarchical_sbm_tools import get_log_number_of_restricted_partitions
from .peixotos_hierarchical_sbm_tools import log_binom

class ModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm:

    def __init__(self):
        super(ModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm, self).__init__()

        self._saved_xi = None
        self._graph_belonging_to_xi = None

    def calculate_complete_uniform_hyperprior_undirected(self, partition):
        probability = 0
        probability += self._calculate_p_adjacency_undirected(partition)
        probability += self._calculate_p_degree_sequence_uniform_hyperprior_undirected(partition)
        probability += self._calculate_p_edge_counts_undirected(partition)
        return probability

    def calculate_complete_uniform_undirected(self, partition):
        probability = 0
        probability += self._calculate_p_adjacency_undirected(partition)
        probability += self._calculate_p_degree_sequence_uniform_undirected(partition)
        probability += self._calculate_p_edge_counts_undirected(partition)
        return probability

    def calculate_complete_non_degree_corrected_undirected(self, partition):
        probability = 0
        probability += self._calculate_non_degree_corrected_undirected(partition)
        probability += self._calculate_p_edge_counts_undirected(partition)
        return probability

    def _calculate_p_adjacency_undirected(self, partition):
        """
        Calculate \log P(A|k,e,b) = \log \Xi(A) - \log \Omega(e)
        with
        \log \Omega(e) = \sum_r \log e_r!- \sum_{r<s}\log e_{rs}! -\sum_r \log e_{rr}!!
        with e_r = \sum_s e_{rs} and (2m)!! = 2^m m!
        and
        \log \Xi(A) = \sum_i \log k_i! - \sum_{i<j}\log A_{ij}! - \sum_i \log A_{ii}!!

        Value depends on adjacency matrix, degree sequence, edge counts and partition

        Xi depends only on the adjacency matrix and is only computed once
        :type partition Partition
        :return: probability \log P(A|k,e,b)
        """

        # check if xi was already computed
        if self._saved_xi is None or self._graph_belonging_to_xi != partition.graph:
            # if xi is still initial compute xi and therefore get information from partition

            adjacency_matrix = partition.get_graph_matrix_representation(with_weights=False)
            degree_sequence = partition.get_degree_iter()
            self._saved_xi = 0
            for degree in degree_sequence:
                self._saved_xi += math.lgamma(degree + 1)

            # denominator = 1
            for i in range(len(adjacency_matrix)):
                self._saved_xi -= math.log(2) * adjacency_matrix[i, i] + math.lgamma(adjacency_matrix[i, i] + 1)
                for j in range(i + 1, len(adjacency_matrix)):
                    self._saved_xi -= math.lgamma(adjacency_matrix[i, j] + 1)

            # save graph reference
            self._graph_belonging_to_xi = partition.graph

        omega = 0
        denominator = 0
        for r in range(partition.B):
            omega += math.lgamma(1 + partition.get_degree_of_block(r))
            denominator += math.log(2) * partition.get_edge_count(r, r) / 2 \
                + math.lgamma(1 + partition.get_edge_count(r, r) / 2)

            for s in range(r + 1, partition.B):
                denominator += math.lgamma(1 + partition.get_edge_count(r, s))

        omega -= denominator
        return self._saved_xi - omega

    @staticmethod
    def _calculate_non_degree_corrected_undirected(partition, full=True):
        """
        Combination of probability of degree sequence and probability of adjacency matrix.
        Formula
        \log P(A|e,b) = \sum_{r<s} \log e_{rs}! + \sum_r \log e_{rr}!! - \sum_r e_r\log n_r
                    - \sum_{i<j}\log A_{ij}! -\sum_i \log A_{ii}!!
        :param partition: Partition of a graph with saved degree
        :type partition Partition
        :param full: Include calculation of adjacency or skip for faster calculation
        :type full Boolean
        :return: probability \log P(A|e,b)
        """
        probability = 0

        denominator = 0
        for r in range(partition.B):
            probability += math.log(2) * partition.get_edge_count(r, r) / 2 \
                           + math.lgamma(1 + partition.get_edge_count(r, r) / 2)

            number_of_nodes = partition.get_number_of_nodes_in_block(r)
            if number_of_nodes > 0:
                denominator += math.log(number_of_nodes) * partition.get_degree_of_block(r)
            for s in range(r + 1, partition.B):
                probability += math.lgamma(1 + partition.get_edge_count(r, s))

        probability -= denominator

        if full:
            adjacency_matrix = partition.get_graph_matrix_representation(with_weights=False)

            denominator = 0
            for i in range(len(adjacency_matrix)):
                denominator += math.log(2) * adjacency_matrix[i, i] + math.lgamma(1 + adjacency_matrix[i, i])
                for j in range(i + 1, len(adjacency_matrix)):
                    denominator += math.lgamma(1 + adjacency_matrix[i, j])

            probability -= denominator

        return probability

    @staticmethod
    def _calculate_p_degree_sequence_uniform_undirected(partition):
        """
        Calculate probability of degree sequence using a uniform prior
        \log P(k|e,b) = -\sum_r \log ((n_r e_r))
        :param partition: partition of a graph
        :type partition Partition
        :return: probability of degree sequence given the edge counts and groups under a uniform prior
        :rtype float
        """
        probability = 0
        for r in range(partition.B):
            n_r = partition.get_number_of_nodes_in_block(r)
            if n_r > 0:
                e_r = partition.get_degree_of_block(r)
                probability -= log_binom(n_r + e_r - 1, e_r)
        return probability

    @staticmethod
    def _calculate_p_degree_sequence_uniform_hyperprior_undirected(partition):

        # Formulas:
        # \log P(k|e,b) = \log P(k|\eta)+ \log P(\eta|e,b)
        # \log P(k|\eta) = \sum_r \sum_k \log \eta_k^r-\log n_r!
        # \log P(\eta|e,b) = -\sum_r \log q(e_r,n_r)
        # Together
        # \log P(k|e,b) = \sum_r (-\log q(e_r, n_r)-\log n_r!) + \sum_k \log \eta_k^r
        degree_counts = partition.get_degree_distribution_of_blocks(probability=False)
        probability = 0
        for r in range(partition.B):
            probability -= get_log_number_of_restricted_partitions(int(partition.get_degree_of_block(r)),
                                                                   int(partition.get_number_of_nodes_in_block(r))) \
                           + math.lgamma(1 + partition.get_number_of_nodes_in_block(r))
            for degree_count in degree_counts[r].values():
                probability += math.lgamma(1 + int(degree_count))
        return probability

    @staticmethod
    def _calculate_p_edge_counts_undirected(partition):
        """
        Formulas
        P(e) = (( B(B+1)/2 E ))^{-1}
        P(b) = \frac{\prod_r n_r!}{N!} (N-1 B-1)^{-1} 1/N
        :param partition:
        :type partition Partition
        :return: log P(e) + log P(b)
        """
        probability = 0

        number_of_blocks = partition.B
        edge_total = partition.get_number_of_edges()
        probability -= log_binom(number_of_blocks * (number_of_blocks + 1) / 2 + edge_total - 1, edge_total)

        # part for P(b)
        for r in range(number_of_blocks):
            n_r = partition.get_number_of_nodes_in_block(r)

            probability += math.lgamma(1 + n_r)

        # group independent terms of last product
        number_of_nodes = partition.get_number_of_nodes()
        probability -= math.lgamma(1 + number_of_nodes)

        probability -= log_binom(number_of_nodes - 1, number_of_blocks - 1)

        probability -= math.log(number_of_nodes)
        return probability

    # ----------------------------------
    # directed Versions
    # ----------------------------------
    def calculate_complete_uniform_hyperprior_directed(self, partition):
        probability = 0
        probability += self._calculate_p_adjacency_directed(partition)
        probability += self._calculate_p_degree_sequence_uniform_hyperprior_directed(partition)
        probability += self._calculate_p_edge_counts_directed(partition)
        return probability

    def calculate_complete_uniform_directed(self, partition):
        probability = 0
        probability += self._calculate_p_adjacency_directed(partition)
        probability += self._calculate_p_degree_sequence_uniform_directed(partition)
        probability += self._calculate_p_edge_counts_directed(partition)
        return probability

    def calculate_complete_non_degree_corrected_directed(self, partition):
        probability = 0
        probability += self._calculate_non_degree_corrected_directed(partition)
        probability += self._calculate_p_edge_counts_directed(partition)
        return probability

    def _calculate_p_adjacency_directed(self, partition, calculate_complete=True):
        """
        Calculate \log P(A|k,e,b) = \log \Xi(A) - \log \Omega(e)
        with
        \log \Omega(e) = \sum_r \log e_r^+! \log e_r^-! - \sum_{r,s}\log e_{rs}!
        with e_r = \sum_s e_{rs} and (2m)!! = 2^m m!
        and
        \log \Xi(A) = \sum_i \log k_i^+! \log k_i^-!-\sum_{i,j}\log A_{ij}!

        Value depends on adjacency matrix, degree sequence, edge counts and partition

        Xi depends only on the adjacency matrix and is only computed once
        :type partition Partition
        :return: probability \log P(A|k,e,b)
        """

        # check if xi was already computed
        if self._saved_xi is None or calculate_complete:
            # if xi is still initial compute xi and therefore get information from partition
            adjacency_matrix = partition.get_graph_matrix_representation(with_weights=False)
            in_degree_sequence = partition.get_in_degree_iter()
            out_degree_sequence = partition.get_out_degree_iter()
            self._saved_xi = 0
            for in_degree in in_degree_sequence:
                self._saved_xi += math.lgamma(1 + in_degree)

            for out_degree in out_degree_sequence:
                self._saved_xi += math.lgamma(1 + out_degree)

            denominator = 0
            for i in range(len(adjacency_matrix)):
                for j in range(len(adjacency_matrix)):
                    denominator += math.lgamma(1 + adjacency_matrix[i, j])

            self._saved_xi -= denominator

        omega = 0
        denominator = 0
        for r in range(partition.B):
            omega += math.lgamma(1 + partition.get_in_degree_of_block(r))
            omega += math.lgamma(1 + partition.get_out_degree_of_block(r))

            for s in range(partition.B):
                denominator += math.lgamma(1 + partition.get_edge_count(r, s))

        omega -= denominator
        return self._saved_xi - omega

    @staticmethod
    def _calculate_non_degree_corrected_directed(partition, full=True):
        """
        Combination of probability of degree sequence and probability of adjacency matrix.
        Formula
        \log P(A|e,b) = \log e_{rs}!-\sum_r e_r^+\log n_r - e_r^-\log n_r
                    -\sum_{ij}\log A_{ij}!
        :param partition: Partition of a graph with saved degree
        :type partition Partition
        :param full: Include calculation of adjacency or skip for faster calculation
        :type full Boolean
        :return: log probability
        """
        probability = 0

        denominator = 0
        for r in range(partition.B):
            denominator += math.log(partition.get_number_of_nodes_in_block(r)) * partition.get_in_degree_of_block(r)
            denominator += math.log(partition.get_number_of_nodes_in_block(r)) * partition.get_out_degree_of_block(r)

            for s in range(partition.B):
                probability += math.lgamma(1 + partition.get_edge_count(r, s))

        probability -= denominator

        if full:
            adjacency_matrix = partition.get_graph_matrix_representation(with_weights=False)

            denominator = 0
            for i in range(len(adjacency_matrix)):
                for j in range(len(adjacency_matrix)):
                    denominator += math.lgamma(1 + adjacency_matrix[i, j])

            probability -= denominator

        return probability

    @staticmethod
    def _calculate_p_degree_sequence_uniform_directed(partition):
        """
        Calculate probability of degree sequence using a uniform prior
        \log P(k|e,b) = -\sum_r \log ((n_r e_r^+)) +\log ((n_r e_r^-))
        :param partition: partition of a graph
        :type partition Partition
        :return: log probability of degree sequence given the edge counts and groups under a uniform prior
        :rtype float
        """
        probability = 0
        for r in range(partition.B):
            n_r = partition.get_number_of_nodes_in_block(r)
            if n_r > 0:
                e_r_p = partition.get_in_degree_of_block(r)
                probability -= log_binom(n_r + e_r_p - 1, e_r_p)

                e_r_m = partition.get_out_degree_of_block(r)
                probability -= log_binom(n_r + e_r_m - 1, e_r_m)
        return probability

    @staticmethod
    def _calculate_p_degree_sequence_uniform_hyperprior_directed(partition):
        """
        Formulas:
        \log P(k|e,b) = \log P(k|\eta) + \log P(\eta|e,b)
        \log P(k|\eta) = \sum_r \sum_{k^+,k^-} r*\log \eta_{k^+,k^-}!-\log n_r!
        with the joint (in, out)-degree distribution
        \log P(\eta|e,b) = -\sum_r \log q(e_r^+,n_r) + \log q(e_r^-,n_r)
        """
        degree_counts = partition.get_joint_in_out_degree_distribution_of_blocks()
        probability = 0
        for r in range(partition.B):
            probability -= get_log_number_of_restricted_partitions(int(partition.get_in_degree_of_block(r)),
                                                                   int(partition.get_number_of_nodes_in_block(r))) \
                           + get_log_number_of_restricted_partitions(int(partition.get_out_degree_of_block(r)),
                                                                     int(partition.get_number_of_nodes_in_block(r))) \
                           + math.lgamma(1 + partition.get_number_of_nodes_in_block(r))
            for degree_count in degree_counts[r].values():
                probability += math.lgamma(1 + int(degree_count))
        return probability

    @staticmethod
    def _calculate_p_edge_counts_directed(partition):
        """
        Formulas
        P(e) = (( B*B E ))^{-1}
        P(b) = \frac{\prod_r n_r!}{N!} (N-1 B-1)^{-1} 1/N
        :param partition:
        :type partition Partition
        :return: log P(e) + log P(b)
        """
        probability = 0

        number_of_blocks = partition.B
        edge_total = partition.get_number_of_edges()
        probability -= log_binom(number_of_blocks * number_of_blocks + edge_total - 1, edge_total)

        # part for P(b)
        for r in range(number_of_blocks):
            n_r = partition.get_number_of_nodes_in_block(r)

            probability += math.lgamma(1 + n_r)

        # group independent terms of last product
        number_of_nodes = partition.get_number_of_nodes()
        probability -= math.lgamma(1 + number_of_nodes)

        probability -= log_binom(number_of_nodes - 1, number_of_blocks - 1)

        probability -= math.log(number_of_nodes)
        return probability


class DeltaModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm:

    def calculate_delta_complete_uniform_hyperprior_undirected(self, partition, from_block, to_block, *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 3:
            kit, selfloops, degree = args
            nodes_moved = 1
            nodes_remaining = not (partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 5:
            kit, selfloops, degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        delta += self._calculate_delta_p_adjacency_undirected(
            partition, from_block, to_block, kit, selfloops, degree, to_block_exists)
        delta += self._calculate_delta_p_degree_sequence_uniform_hyperprior_undirected(
            partition, from_block, to_block, degree,
            nodes_remaining, to_block_exists, nodes_moved
        )

        delta += self._calculate_delta_p_edge_counts_undirected(
            partition, from_block, to_block, nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    def calculate_delta_complete_uniform_undirected(self, partition, from_block, to_block, *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 3:
            kit, selfloops, degree = args
            nodes_moved = 1
            nodes_remaining = not (partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 5:
            kit, selfloops, degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        delta += self._calculate_delta_p_adjacency_undirected(
            partition, from_block, to_block, kit, selfloops, degree, to_block_exists)
        delta += self._calculate_delta_p_degree_sequence_uniform_undirected(
            partition, from_block, to_block, degree,
            nodes_remaining, to_block_exists, nodes_moved
        )

        delta += self._calculate_delta_p_edge_counts_undirected(
            partition, from_block, to_block, nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    def calculate_delta_complete_non_degree_corrected_undirected(self, partition, from_block, to_block,
                                                                 *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 3:
            kit, selfloops, degree = args
            nodes_moved = 1
            nodes_remaining = not (partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 5:
            kit, selfloops, degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        delta += self._calculate_delta_non_degree_corrected_undirected(
            partition, from_block, to_block, kit, selfloops, degree,
            nodes_remaining, to_block_exists, nodes_moved)
        delta += self._calculate_delta_p_edge_counts_undirected(
            partition, from_block, to_block, nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    @staticmethod
    def _calculate_delta_p_adjacency_undirected(partition, from_block, to_block, kit, selfloops, degree, to_block_exists):
        """
        Calculate \Delta \log P(A|k,e,b)
        """

        delta = 0
        # delta of first sum \sum_r \log e_r!
        # old block
        delta += math.lgamma(1 + partition.get_degree_of_block(from_block)) - math.lgamma(
            1 + partition.get_degree_of_block(from_block) - degree)
        # new block
        if to_block_exists:
            degree_of_to_block = partition.get_degree_of_block(to_block)
        else:
            degree_of_to_block = 0
        delta += math.lgamma(1 + degree_of_to_block) - math.lgamma(
            1 + degree_of_to_block + degree)
        # delta of second sum \sum_{r<s}\log e_{rs}!
        for block in kit:
            if block != from_block and block != to_block:
                delta += math.lgamma(1 + partition.get_edge_count(from_block, block) - kit[block]) \
                            - math.lgamma(1 + partition.get_edge_count(from_block, block)) \
                            + math.lgamma(1 + partition.get_edge_count(to_block, block) + kit[block]) \
                            - math.lgamma(1 + partition.get_edge_count(to_block, block))
        # handle the term with both from and to block
        delta += math.lgamma(1 + partition.get_edge_count(from_block, to_block) + kit[from_block] - kit[to_block]) \
            - math.lgamma(1 + partition.get_edge_count(from_block, to_block))

        # handle last sum \sum_r \log e_{rr}!!
        delta += math.log(2) * (partition.get_edge_count(from_block, from_block) / 2 - kit[from_block] - selfloops)\
            + math.lgamma(1 + partition.get_edge_count(from_block, from_block) / 2 - kit[from_block] - selfloops) \
            - math.log(2) * partition.get_edge_count(from_block, from_block) / 2 \
            - math.lgamma(1 + partition.get_edge_count(from_block, from_block) / 2) \
            + math.log(2) * (partition.get_edge_count(to_block, to_block) / 2 + kit[to_block] + selfloops)\
            + math.lgamma(1 + partition.get_edge_count(to_block, to_block) / 2 + kit[to_block] + selfloops) \
            - math.log(2) * partition.get_edge_count(to_block, to_block) / 2 \
            - math.lgamma(1 + partition.get_edge_count(to_block, to_block) / 2)

        return delta

    @staticmethod
    def _calculate_delta_non_degree_corrected_undirected(partition, from_block, to_block, kit, selfloops, degree,
                                                         nodes_remaining, to_block_exists, nodes_moved=1):
        """
        Combination of probability of degree sequence and probability of adjacency matrix.
        Formula
        \Delta \log P(A|e,b)
        """
        delta = 0

        # delta of sum \sum_{r<s}\log e_{rs}!
        for block in kit:
            if block != from_block and block != to_block:
                delta += math.lgamma(1 + partition.get_edge_count(from_block, block) - kit[block]) \
                            - math.lgamma(1 + partition.get_edge_count(from_block, block)) \
                            + math.lgamma(1 + partition.get_edge_count(to_block, block) + kit[block]) \
                            - math.lgamma(1 + partition.get_edge_count(to_block, block))
        # handle the term with both from and to block
        delta += math.lgamma(1 + partition.get_edge_count(from_block, to_block) + kit[from_block] - kit[to_block]) \
            - math.lgamma(1 + partition.get_edge_count(from_block, to_block))

        # handle sum \sum_r \log e_{rr}!!
        delta += math.log(2) * (partition.get_edge_count(from_block, from_block) / 2 - kit[from_block] - selfloops)\
            + math.lgamma(1 + partition.get_edge_count(from_block, from_block) / 2 - kit[from_block] - selfloops) \
            - math.log(2) * partition.get_edge_count(from_block, from_block) / 2 \
            - math.lgamma(1 + partition.get_edge_count(from_block, from_block) / 2) \
            + math.log(2) * (partition.get_edge_count(to_block, to_block) / 2 + kit[to_block] + selfloops)\
            + math.lgamma(1 + partition.get_edge_count(to_block, to_block) / 2 + kit[to_block] + selfloops) \
            - math.log(2) * partition.get_edge_count(to_block, to_block) / 2 \
            - math.lgamma(1 + partition.get_edge_count(to_block, to_block) / 2)

        if to_block_exists:
            number_of_nodes_in_to_block = partition.get_number_of_nodes_in_block(to_block)
            degree_of_to_block = partition.get_degree_of_block(to_block)
        else:
            number_of_nodes_in_to_block = 0
            degree_of_to_block = 0

        delta += math.log(partition.get_number_of_nodes_in_block(from_block)) \
            * partition.get_degree_of_block(from_block) \
            - math.log(number_of_nodes_in_to_block + nodes_moved) * (
                 degree_of_to_block + degree)

        if number_of_nodes_in_to_block > 0:
            delta += math.log(partition.get_number_of_nodes_in_block(to_block)) \
                     * degree_of_to_block

        if nodes_remaining:
            delta -= math.log(partition.get_number_of_nodes_in_block(from_block) - nodes_moved) * (
                partition.get_degree_of_block(from_block) - degree)

        return delta

    @staticmethod
    def _calculate_delta_p_degree_sequence_uniform_undirected(partition, from_block, to_block, degree,
                                                              nodes_remaining, to_block_exists, nodes_moved=1):
        """
        Calculate probability of degree sequence using a uniform prior
        \Delta \log P(k|e,b)
        """

        delta = 0
        values = [
            (True, partition.get_number_of_nodes_in_block(from_block), partition.get_degree_of_block(from_block)),
                  ]

        if to_block_exists:
            values.append((True, partition.get_number_of_nodes_in_block(to_block),
                           partition.get_degree_of_block(to_block)))
            values.append((False, partition.get_number_of_nodes_in_block(to_block) + nodes_moved,
                           partition.get_degree_of_block(to_block) + degree))
        else:
            values.append((False, nodes_moved, degree))

        if nodes_remaining:
            values.append((False, partition.get_number_of_nodes_in_block(from_block) - nodes_moved,
                          partition.get_degree_of_block(from_block) - degree))

        for add, number_of_nodes, degree_of_block in values:
            if number_of_nodes > 0:
                if add:
                    delta += log_binom(number_of_nodes + degree_of_block - 1, degree_of_block)
                else:
                    delta -= log_binom(number_of_nodes + degree_of_block - 1, degree_of_block)

        return delta

    @staticmethod
    def _calculate_delta_p_degree_sequence_uniform_hyperprior_undirected(partition, from_block, to_block, degree,
                                                                         nodes_remaining, to_block_exists,
                                                                         nodes_moved=1):

        # Formulas:
        # \log P(k|e,b) = \log P(k|\eta)+ \log P(\eta|e,b)
        # \log P(k|\eta) = \sum_r \sum_k \log \eta_k^r!-\log n_r!
        # \log P(\eta|e,b) = -\sum_r \log q(e_r,n_r)
        # Together
        # \log P(k|e,b) = \sum_r (-\log q(e_r, n_r)-\log n_r!) + \sum_k \log \eta_k^r!
        if nodes_moved != 1 and nodes_remaining:
            # in that case need degree distributions of all moved blocks
            raise NotImplementedError()

        delta = 0

        # delta of \sum_k \log \eta_k^r!
        if nodes_remaining:  # because of check at the beginning equivalent to nodes_moved == 1
            delta += math.lgamma(0 + partition.get_number_of_nodes_with_same_degree_in_block(from_block, degree)) \
                - math.lgamma(1 + partition.get_number_of_nodes_with_same_degree_in_block(from_block, degree))
            if to_block_exists:
                delta += math.lgamma(2 + partition.get_number_of_nodes_with_same_degree_in_block(to_block, degree)) \
                    - math.lgamma(1 + partition.get_number_of_nodes_with_same_degree_in_block(to_block, degree))

            # critical part of \sum_r -\log q(e_r, n_r)
            delta -= get_log_number_of_restricted_partitions(int(partition.get_degree_of_block(from_block) - degree),
                                                             int(partition.get_number_of_nodes_in_block(from_block)
                                                             - nodes_moved))
            # critical part of  \sum_r -\log n_r!
            delta -= math.lgamma(1 + partition.get_number_of_nodes_in_block(from_block) - nodes_moved)
        else:
            # move of complete block
            degree_distribution = partition.get_degree_distribution_of_single_block(from_block)

            for node_degree in degree_distribution:
                delta += - math.lgamma(
                    1 + partition.get_number_of_nodes_with_same_degree_in_block(from_block, node_degree)) \
                    + math.lgamma(1 + partition.get_number_of_nodes_with_same_degree_in_block(to_block, node_degree)
                              + degree_distribution[node_degree]) \
                    - math.lgamma(1 + partition.get_number_of_nodes_with_same_degree_in_block(to_block, node_degree))

        # delta of \sum_r -\log q(e_r, n_r)
        delta -= - get_log_number_of_restricted_partitions(int(partition.get_degree_of_block(from_block)),
                                                           int(partition.get_number_of_nodes_in_block(from_block)))
        if to_block_exists:
            delta -= get_log_number_of_restricted_partitions(int(partition.get_degree_of_block(to_block) + degree),
                                                             int(partition.get_number_of_nodes_in_block(to_block)
                                                             + nodes_moved))  \
                - get_log_number_of_restricted_partitions(int(partition.get_degree_of_block(to_block)),
                                                          int(partition.get_number_of_nodes_in_block(to_block)))
        else:
            delta -= get_log_number_of_restricted_partitions(degree, nodes_moved)

        # delta of \sum_r -\log n_r!
        delta -= - math.lgamma(1 + partition.get_number_of_nodes_in_block(from_block))

        if to_block_exists:
            delta -= math.lgamma(1 + partition.get_number_of_nodes_in_block(to_block) + nodes_moved)\
                - math.lgamma(1 + partition.get_number_of_nodes_in_block(to_block))
        else:
            delta -= math.lgamma(1 + nodes_moved)

        return delta

    @staticmethod
    def _calculate_delta_p_edge_counts_undirected(partition, from_block, to_block, nodes_remaining, to_block_exists,
                                                  nodes_moved=1):
        """
        Calculate the delta of log P(e) + log P(b)
        :param partition: Partition
        :type partition Partition
        :param from_block: node(s) moved from this block
        :type from_block int
        :param to_block: node(s) moved to this block
        :type to_block int
        :param nodes_remaining: True or False if from_block is empty after move
        :type nodes_remaining Boolean
        :param nodes_moved: number of nodes moved
        :type nodes_moved int
        :return: delta of log P(e) + log P(b)
        """

        delta = 0

        # handle term -\log B_{l-1}! - \log \nCr{B_{l-1}-1, B_l-1)-\log B_{l-1}
        new_block_count = -1
        if to_block == partition.B:
            # new block
            new_block_count = to_block + 1
        if partition.get_number_of_nodes_in_block(from_block) == nodes_moved and from_block != to_block:
            # block will be removed
            new_block_count = partition.B - 1
        if new_block_count != -1:

            # term - \log \nCr{N-1, B-1)
            # terms for change B_l
            delta -= log_binom(partition.get_number_of_nodes() - 1, new_block_count - 1)

            delta += log_binom(partition.get_number_of_nodes() - 1, partition.B - 1)

            # deal with (( B*(B+1)/2 E)) term
            number_of_edges = partition.get_number_of_edges()
            # new term
            delta -= log_binom(new_block_count * (new_block_count + 1) / 2 + number_of_edges - 1, number_of_edges)
            # old term
            delta += log_binom(partition.B * (partition.B + 1) / 2 + number_of_edges - 1, number_of_edges)

        # delta for \sum_r \log n^l_r!
        delta += - math.lgamma(1 + partition.get_number_of_nodes_in_block(from_block))
        if to_block_exists:
            delta += math.lgamma(1 + partition.get_number_of_nodes_in_block(to_block) + nodes_moved) \
                - math.lgamma(1 + partition.get_number_of_nodes_in_block(to_block))
        else:
            delta += math.lgamma(1 + nodes_moved)

        if nodes_remaining:
            delta += math.lgamma(1 + partition.get_number_of_nodes_in_block(from_block) - nodes_moved)

        return delta

    # ----------------------------------
    # directed Versions
    # ----------------------------------
    def calculate_delta_complete_uniform_hyperprior_directed(self, partition, from_block, to_block, *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 5:
            kit, kti, selfloops, in_degree, out_degree = args
            nodes_moved = 1
            nodes_remaining = not (partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 7:
            kit, kti, selfloops, in_degree, out_degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        delta += self._calculate_delta_p_adjacency_directed(
            partition, from_block, to_block, kit, kti, selfloops, in_degree, out_degree, to_block_exists)
        delta += self._calculate_delta_p_degree_sequence_uniform_hyperprior_directed(
            partition, from_block, to_block, in_degree, out_degree,
            nodes_remaining, to_block_exists, nodes_moved
        )

        delta += self._calculate_delta_p_edge_counts_directed(
            partition, from_block, to_block, nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    def calculate_delta_complete_uniform_directed(self, partition, from_block, to_block, *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 5:
            kit, kti, selfloops, in_degree, out_degree = args
            nodes_moved = 1
            nodes_remaining = not (partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 7:
            kit, kti, selfloops, in_degree, out_degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        delta += self._calculate_delta_p_adjacency_directed(
            partition, from_block, to_block, kit, kti, selfloops, in_degree, out_degree, to_block_exists)
        delta += self._calculate_delta_p_degree_sequence_uniform_directed(
            partition, from_block, to_block, in_degree, out_degree,
            nodes_remaining, to_block_exists, nodes_moved)

        delta += self._calculate_delta_p_edge_counts_directed(
            partition, from_block, to_block, nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    def calculate_delta_complete_non_degree_corrected_directed(self, partition, from_block, to_block, *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 5:
            kit, kti, selfloops, in_degree, out_degree = args
            nodes_moved = 1
            nodes_remaining = not (partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 7:
            kit, kti, selfloops, in_degree, out_degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        delta += self._calculate_delta_non_degree_corrected_directed(
            partition, from_block, to_block, kit, kti, selfloops, in_degree, out_degree,
            nodes_remaining, to_block_exists, nodes_moved)

        delta += self._calculate_delta_p_edge_counts_directed(
            partition, from_block, to_block, nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    @staticmethod
    def _calculate_delta_p_adjacency_directed(partition, from_block, to_block, kit, kti, selfloops, in_degree,
                                              out_degree, to_block_exists):
        """
        Calculate \Delta \log P(A|k,e,b)
        :type partition Partition
        :return: delta of probability \log P(A|k,e,b)
        """

        delta = 0
        # delta of first sum \sum_r \log e_r!
        # old block
        delta += math.lgamma(1 + partition.get_in_degree_of_block(from_block)) \
            - math.lgamma(1 + partition.get_in_degree_of_block(from_block) - in_degree)\
            + math.lgamma(1 + partition.get_out_degree_of_block(from_block))\
            - math.lgamma(1 + partition.get_out_degree_of_block(from_block) - out_degree)
        # new block
        if to_block_exists:
            in_degree_of_to_block = partition.get_in_degree_of_block(to_block)
            out_degree_of_to_block = partition.get_out_degree_of_block(to_block)
        else:
            in_degree_of_to_block = 0
            out_degree_of_to_block = 0
        delta += math.lgamma(1 + in_degree_of_to_block) \
            - math.lgamma(1 + in_degree_of_to_block + in_degree)\
            + math.lgamma(1 + out_degree_of_to_block)\
            - math.lgamma(1 + out_degree_of_to_block + out_degree)

        # delta of second sum \sum_{rs}\log e_{rs}!
        for block in kit:
            if block != from_block and block != to_block:
                delta += math.lgamma(1 + partition.get_edge_count(from_block, block) - kit[block]) \
                            - math.lgamma(1 + partition.get_edge_count(from_block, block)) \
                            + math.lgamma(1 + partition.get_edge_count(to_block, block) + kit[block]) \
                            - math.lgamma(1 + partition.get_edge_count(to_block, block))

        for block in kti:
            if block != from_block and block != to_block:
                delta += math.lgamma(1 + partition.get_edge_count(block, from_block) - kti[block]) \
                            - math.lgamma(1 + partition.get_edge_count(block, from_block)) \
                            + math.lgamma(1 + partition.get_edge_count(block, to_block) + kti[block]) \
                            - math.lgamma(1 + partition.get_edge_count(block, to_block))

        # handle the term with both from and to block
        delta += math.lgamma(1 + partition.get_edge_count(from_block, to_block) + kti[from_block] - kit[to_block]) \
            - math.lgamma(1 + partition.get_edge_count(from_block, to_block)) \
            + math.lgamma(1 + partition.get_edge_count(to_block, from_block) + kit[from_block] - kti[to_block]) \
            - math.lgamma(1 + partition.get_edge_count(to_block, from_block))

        # handle terms with only from_block or to_block
        delta += math.lgamma(1 + partition.get_edge_count(from_block, from_block)
                         - kti[from_block] - kit[from_block] - selfloops) \
            - math.lgamma(1 + partition.get_edge_count(from_block, from_block)) \
            + math.lgamma(1 + partition.get_edge_count(to_block, to_block) + kit[to_block] + kti[to_block] + selfloops) \
            - math.lgamma(1 + partition.get_edge_count(to_block, to_block))

        return delta

    @staticmethod
    def _calculate_delta_non_degree_corrected_directed(partition, from_block, to_block, kit, kti, selfloops,
                                                       in_degree, out_degree, nodes_remaining, to_block_exists,
                                                       nodes_moved=1):
        """
        Combination of probability of degree sequence and probability of adjacency matrix.
        Formula
        \log P(A|e,b) = \log e_{rs}!-\sum_r e_r^+\log n_r - e_r^-\log n_r
                    -\sum_{ij}\log A_{ij}!
        :param partition: Partition of a graph with saved degree
        :type partition Partition
        :return: delta of log probability
        """

        delta = 0

        # # delta of second sum \sum_{rs}\log e_{rs}!
        for block in kit:
            if block != from_block and block != to_block:
                delta += math.lgamma(1 + partition.get_edge_count(from_block, block) - kit[block]) \
                            - math.lgamma(1 + partition.get_edge_count(from_block, block)) \
                            + math.lgamma(1 + partition.get_edge_count(to_block, block) + kit[block]) \
                            - math.lgamma(1 + partition.get_edge_count(to_block, block))

        for block in kti:
            if block != from_block and block != to_block:
                delta += math.lgamma(1 + partition.get_edge_count(block, from_block) - kti[block]) \
                            - math.lgamma(1 + partition.get_edge_count(block, from_block)) \
                            + math.lgamma(1 + partition.get_edge_count(block, to_block) + kti[block]) \
                            - math.lgamma(1 + partition.get_edge_count(block, to_block))

        # handle the term with both from and to block
        delta += math.lgamma(1 + partition.get_edge_count(from_block, to_block) + kti[from_block] - kit[to_block]) \
            - math.lgamma(1 + partition.get_edge_count(from_block, to_block)) \
            + math.lgamma(1 + partition.get_edge_count(to_block, from_block) + kit[from_block] - kti[to_block]) \
            - math.lgamma(1 + partition.get_edge_count(to_block, from_block))

        # handle terms with only from_block or to_block
        delta += math.lgamma(1 + partition.get_edge_count(from_block, from_block)
                         - kti[from_block] - kit[from_block] - selfloops) \
            - math.lgamma(1 + partition.get_edge_count(from_block, from_block)) \
            + math.lgamma(1 + partition.get_edge_count(to_block, to_block) + kit[to_block] + kti[to_block] + selfloops) \
            - math.lgamma(1 + partition.get_edge_count(to_block, to_block))

        # handle term \sum_r e_r^+\log n_r - e_r^-\log n_r
        if to_block_exists:
            number_of_nodes_in_to_block = partition.get_number_of_nodes_in_block(to_block)
            in_degree_of_to_block = partition.get_in_degree_of_block(to_block)
            out_degree_of_to_block = partition.get_out_degree_of_block(to_block)

            if number_of_nodes_in_to_block > 0:
                delta += math.log(number_of_nodes_in_to_block) * in_degree_of_to_block \
                         + math.log(number_of_nodes_in_to_block) * out_degree_of_to_block
        else:
            number_of_nodes_in_to_block = 0
            in_degree_of_to_block = 0
            out_degree_of_to_block = 0

        delta -= - math.log(partition.get_number_of_nodes_in_block(from_block)) \
            * partition.get_in_degree_of_block(from_block) \
            + math.log(number_of_nodes_in_to_block + nodes_moved) * (
                in_degree_of_to_block + in_degree) \
            - math.log(partition.get_number_of_nodes_in_block(from_block)) \
            * partition.get_out_degree_of_block(from_block) \
            + math.log(number_of_nodes_in_to_block + nodes_moved) * (
                out_degree_of_to_block + out_degree)

        if nodes_remaining:
            delta -= math.log(partition.get_number_of_nodes_in_block(from_block) - nodes_moved) * (
                partition.get_in_degree_of_block(from_block) - in_degree) \
                + math.log(partition.get_number_of_nodes_in_block(from_block) - nodes_moved) * (
                            partition.get_out_degree_of_block(from_block) - out_degree)

        return delta

    @staticmethod
    def _calculate_delta_p_degree_sequence_uniform_directed(partition, from_block, to_block, in_degree, out_degree,
                                                            nodes_remaining, to_block_exists, nodes_moved=1):
        """
        Calculate probability of degree sequence using a uniform prior
        \log P(k|e,b) = -\sum_r \log ((n_r e_r^+)) +\log ((n_r e_r^-))
        :param partition: partition of a graph
        :type partition Partition
        :return: log probability of degree sequence given the edge counts and groups under a uniform prior
        :rtype float
        """

        delta = 0
        values = [
            (True, partition.get_number_of_nodes_in_block(from_block), partition.get_in_degree_of_block(from_block)),

            (True, partition.get_number_of_nodes_in_block(from_block), partition.get_out_degree_of_block(from_block)),
                  ]

        if to_block_exists:
            values.extend(
                [
                    (True, partition.get_number_of_nodes_in_block(to_block),
                     partition.get_in_degree_of_block(to_block)),
                    (False, partition.get_number_of_nodes_in_block(to_block) + nodes_moved,
                     partition.get_in_degree_of_block(to_block) + in_degree),

                    (True, partition.get_number_of_nodes_in_block(to_block),
                     partition.get_out_degree_of_block(to_block)),
                    (False, partition.get_number_of_nodes_in_block(to_block) + nodes_moved,
                     partition.get_out_degree_of_block(to_block) + out_degree)
                ]
            )
        else:
            values.extend(
                [
                    (False, nodes_moved, in_degree),

                    (False, nodes_moved, out_degree)
                ]
            )

        if nodes_remaining:
            values.append((False, partition.get_number_of_nodes_in_block(from_block) - nodes_moved,
                           partition.get_in_degree_of_block(from_block) - in_degree))

            values.append((False, partition.get_number_of_nodes_in_block(from_block) - nodes_moved,
                           partition.get_out_degree_of_block(from_block) - out_degree))

        for add, number_of_nodes, degree_of_block in values:
            if number_of_nodes > 0:
                if add:
                    delta += log_binom(number_of_nodes + degree_of_block - 1, degree_of_block)
                else:
                    delta -= log_binom(number_of_nodes + degree_of_block - 1, degree_of_block)
        return delta

    @staticmethod
    def _calculate_delta_p_degree_sequence_uniform_hyperprior_directed(partition, from_block, to_block, in_degree,
                                                                       out_degree, nodes_remaining, to_block_exists,
                                                                       nodes_moved=1):
        """
        Formulas:
        \log P(k|e,b) = \log P(k|\eta) + \log P(\eta|e,b)
        \log P(k|\eta) = \sum_r \sum_{k^+,k^-} *\log \eta_{k^+,k^-}-\log n_r!
        with the joint (in, out)-degree distribution
        \log P(\eta|e,b) = -\sum_r \log q(e_r^+,n_r) + \log q(e_r^-,n_r)
        """
        if nodes_moved != 1 and nodes_remaining:
            # in that case need degree distributions of all moved blocks
            raise NotImplementedError()

        delta = 0

        # delta of \sum_k \log \eta_k^r!
        if nodes_remaining:  # because of check at the beginning equivalent to nodes_moved == 1
            delta += math.lgamma(
                0 + partition.get_number_of_nodes_with_same_degree_in_block(from_block, (in_degree, out_degree))) \
                - math.lgamma(
                1 + partition.get_number_of_nodes_with_same_degree_in_block(from_block, (in_degree, out_degree)))
            if to_block_exists:
                delta += math.lgamma(
                    2 + partition.get_number_of_nodes_with_same_degree_in_block(to_block, (in_degree, out_degree))) \
                    - math.lgamma(
                    1 + partition.get_number_of_nodes_with_same_degree_in_block(to_block, (in_degree, out_degree)))

            # critical part of \sum_r -\log q(e_r, n_r)
            delta -= get_log_number_of_restricted_partitions(
                int(partition.get_in_degree_of_block(from_block) - in_degree),
                int(partition.get_number_of_nodes_in_block(from_block) - nodes_moved)) \
                + get_log_number_of_restricted_partitions(
                int(partition.get_out_degree_of_block(from_block) - out_degree),
                int(partition.get_number_of_nodes_in_block(from_block) - nodes_moved))
            # critical part of  \sum_r -\log n_r!
            delta -= math.lgamma(1 + partition.get_number_of_nodes_in_block(from_block) - nodes_moved)
        else:
            # move of complete block
            degree_distribution = partition.get_degree_distribution_of_single_block(from_block)

            for node_degree in degree_distribution:
                delta += - math.lgamma(1 +
                                   partition.get_number_of_nodes_with_same_degree_in_block(from_block, node_degree)) \
                    + math.lgamma(1 + partition.get_number_of_nodes_with_same_degree_in_block(to_block, node_degree)
                              + degree_distribution[node_degree]) \
                    - math.lgamma(1 + partition.get_number_of_nodes_with_same_degree_in_block(to_block, node_degree))

        # delta of -\sum_r \log q(e_r^+,n_r) + \log q(e_r^-,n_r)
        delta -= - get_log_number_of_restricted_partitions(int(partition.get_in_degree_of_block(from_block)),
                                                           int(partition.get_number_of_nodes_in_block(from_block)))  \
            - get_log_number_of_restricted_partitions(int(partition.get_out_degree_of_block(from_block)),
                                                      int(partition.get_number_of_nodes_in_block(from_block)))

        if to_block_exists:
            delta -= get_log_number_of_restricted_partitions(
                int(partition.get_in_degree_of_block(to_block) + in_degree),
                int(partition.get_number_of_nodes_in_block(to_block) + nodes_moved))  \
                - get_log_number_of_restricted_partitions(int(partition.get_in_degree_of_block(to_block)),
                                                          int(partition.get_number_of_nodes_in_block(to_block))) \
                + get_log_number_of_restricted_partitions(
                int(partition.get_out_degree_of_block(to_block) + out_degree),
                int(partition.get_number_of_nodes_in_block(to_block) + nodes_moved))  \
                - get_log_number_of_restricted_partitions(int(partition.get_out_degree_of_block(to_block)),
                                                          int(partition.get_number_of_nodes_in_block(to_block)))
        else:
            delta -= get_log_number_of_restricted_partitions(in_degree, nodes_moved)  \
                            + get_log_number_of_restricted_partitions(out_degree, nodes_moved)

        # delta of \sum_r -\log n_r!
        delta -= - math.lgamma(1 + partition.get_number_of_nodes_in_block(from_block))

        if to_block_exists:
            delta -= math.lgamma(1 + partition.get_number_of_nodes_in_block(to_block) + nodes_moved)\
                - math.lgamma(1 + partition.get_number_of_nodes_in_block(to_block))
        else:
            delta -= math.lgamma(nodes_moved)

        return delta

    @staticmethod
    def _calculate_delta_p_edge_counts_directed(partition, from_block, to_block, nodes_remaining, to_block_exists,
                                                nodes_moved=1):
        """
        Calculate the delta of log P(e) + log P(b)
        :param partition: Partition
        :type partition Partition
        :param from_block: node(s) moved from this block
        :type from_block int
        :param to_block: node(s) moved to this block
        :type to_block int
        :param nodes_remaining: True or False if from_block is empty after move
        :type nodes_remaining Boolean
        :param nodes_moved: number of nodes moved
        :type nodes_moved int
        :return: delta of log P(e) + log P(b)
        """

        delta = 0

        # handle term -\log B_{l-1}! - \log \nCr{B_{l-1}-1, B_l-1)-\log B_{l-1}
        new_block_count = -1
        if to_block == partition.B:
            # new block
            new_block_count = to_block + 1
        if partition.get_number_of_nodes_in_block(from_block) == nodes_moved and from_block != to_block:
            # block will be removed
            new_block_count = partition.B - 1
        if new_block_count != -1:

            # term - \log \nCr{N-1, B-1)
            # terms for change B_l
            delta -= log_binom(partition.get_number_of_nodes() - 1, new_block_count - 1)

            delta += log_binom(partition.get_number_of_nodes() - 1, partition.B - 1)

            # deal with (( B*B E )) term
            number_of_edges = partition.get_number_of_edges()
            # new term
            delta -= log_binom(new_block_count * new_block_count + number_of_edges - 1, number_of_edges)
            # old term
            delta += log_binom(partition.B * partition.B + number_of_edges - 1, number_of_edges)

        # delta for \sum_r \log n^l_r!
        delta += - math.lgamma(1 + partition.get_number_of_nodes_in_block(from_block))
        if to_block_exists:
            delta += math.lgamma(1 + partition.get_number_of_nodes_in_block(to_block) + nodes_moved) \
                - math.lgamma(1 + partition.get_number_of_nodes_in_block(to_block))
        else:
            delta += math.lgamma(1 + nodes_moved)

        if nodes_remaining:
            delta += math.lgamma(1 + partition.get_number_of_nodes_in_block(from_block) - nodes_moved)

        return delta

# ----------------------------------------------
# As Objective Function Class
# ----------------------------------------------


class LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmWrapper(ObjectiveFunction):
    # class constants
    UNIFORM_HYPERPRIOR = 'uniform hyperprior'
    UNIFORM = 'uniform'
    NON_DEGREE_CORRECTED = 'non degree corrected'

    title = "LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmWrapper"
    short_title = title

    def __init__(self, is_directed, function_type=None):
        self._delta = DeltaModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm()
        self._complete = ModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm()

        if function_type is None:
            function_type = self.UNIFORM_HYPERPRIOR

        if function_type == self.UNIFORM_HYPERPRIOR:
            calculate_complete_directed = self._complete.calculate_complete_uniform_hyperprior_directed
            calculate_delta_directed = self._delta.calculate_delta_complete_uniform_hyperprior_directed
            calculate_complete_undirected = self._complete.calculate_complete_uniform_hyperprior_undirected
            calculate_delta_undirected = self._delta.calculate_delta_complete_uniform_hyperprior_undirected
        elif function_type == self.UNIFORM:
            calculate_complete_directed = self._complete.calculate_complete_uniform_directed
            calculate_delta_directed = self._delta.calculate_delta_complete_uniform_directed
            calculate_complete_undirected = self._complete.calculate_complete_uniform_undirected
            calculate_delta_undirected = self._delta.calculate_delta_complete_uniform_undirected
        else:
            calculate_complete_directed = self._complete.calculate_complete_non_degree_corrected_directed
            calculate_delta_directed = self._delta.calculate_delta_complete_non_degree_corrected_directed
            calculate_complete_undirected = self._complete.calculate_complete_non_degree_corrected_undirected
            calculate_delta_undirected = self._delta.calculate_delta_complete_non_degree_corrected_undirected

        super(LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmWrapper, self).__init__(
            is_directed,
            calculate_complete_undirected,
            calculate_complete_directed,
            calculate_delta_undirected,
            calculate_delta_directed
        )


class LogLikelihoodOfFlatMicrocanonicalNonDegreeCorrected(LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmWrapper):

    title = "LogLikelihoodOfFlatMicrocanonicalNonDegreeCorrected"
    short_title = "SPC"

    def __init__(self, is_directed):
        super(LogLikelihoodOfFlatMicrocanonicalNonDegreeCorrected, self).__init__(
            is_directed, function_type=self.NON_DEGREE_CORRECTED)


class LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedUniform(LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmWrapper):

    title = "LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedUniform"
    short_title = "DCPU"

    def __init__(self, is_directed):
        super(LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedUniform, self).__init__(
            is_directed, function_type=self.UNIFORM)


class LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedUniformHyperprior(LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmWrapper):

    title = "LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedUniformHyperprior"
    short_title = "DCPUH"

    def __init__(self, is_directed):
        super(LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedUniformHyperprior, self).__init__(
            is_directed, function_type=self.UNIFORM)
