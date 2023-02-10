# Flat versions of objective functions described in Tiago P. Peixoto
# Nonparametric Bayesian inference of the microcanonical stochastic block model
# for more see in peixotos_hierarchical_sbm.py
# with usage of scipy and only non log version
# @formatter:off

import math

from scipy.special import binom

from .peixotos_hierarchical_sbm_tools_full import BINOMIALS
from .peixotos_hierarchical_sbm_tools_full import get_number_of_restricted_partitions


class ModelLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm:
    def __init__(self):
        self._saved_xi = None
        self._graph_belonging_to_xi = None

    def calculate_complete_uniform_hyperprior_undirected(self, partition):
        probability = 1
        probability *= self._calculate_p_adjacency_undirected(partition)
        probability *= self._calculate_p_degree_sequence_uniform_hyperprior_undirected(partition)
        probability *= self._calculate_p_edge_counts_undirected(partition)
        return probability

    def calculate_complete_uniform_undirected(self, partition):
        probability = 1
        probability *= self._calculate_p_adjacency_undirected(partition)
        probability *= self._calculate_p_degree_sequence_uniform_undirected(partition)
        probability *= self._calculate_p_edge_counts_undirected(partition)
        return probability

    def calculate_complete_non_degree_corrected_undirected(self, partition):
        probability = 1
        probability *= self._calculate_non_degree_corrected_undirected(partition)
        probability *= self._calculate_p_edge_counts_undirected(partition)
        return probability

    def _calculate_p_adjacency_undirected(self, partition):
        """
        Calculate P(A|k,e,b) = \Xi(A)/\Omega(e)
        with
        \Omega(e) = \frac{\prod_r e_r!}{\prod_{r<s}e_{rs}! \prod_r e_{rr}!!}
        with e_r = \sum_s e_{rs} and (2m)!! = 2^m m!
        and
        \Xi(A) = \frac{\prod_i k_i!}{\prod_{i<j}A_{ij}! \prod_i A_{ii}!!}

        Value depends on adjacency matrix, degree sequence, edge counts and partition

        Xi depends only on the adjacency matrix and is only computed once
        :type partition Partition
        :return: probability P(A|k,e,b)
        """

        # check if xi was already computed
        if self._saved_xi is None or self._graph_belonging_to_xi != partition.graph:
            # if xi is still initial compute xi and therefore get information from partition

            adjacency_matrix = partition.get_graph_matrix_representation(with_weights=False)
            degree_sequence = partition.get_degree_iter()
            self._saved_xi = 1
            for degree in degree_sequence:
                self._saved_xi *= math.factorial(degree)

            denominator = 1
            for i in range(len(adjacency_matrix)):
                denominator *= math.pow(2, adjacency_matrix[i, i]) * math.factorial(adjacency_matrix[i, i])
                for j in range(i + 1, len(adjacency_matrix)):
                    denominator *= math.factorial(adjacency_matrix[i, j])

            self._saved_xi /= denominator
            # save graph of xi
            self._graph_belonging_to_xi = partition.graph

        omega = 1
        denominator = 1
        for r in range(partition.B):
            omega *= math.factorial(partition.get_degree_of_block(r))
            denominator *= math.pow(2, partition.get_edge_count(r, r) / 2) \
                * math.factorial(partition.get_edge_count(r, r) / 2)

            for s in range(r + 1, partition.B):
                denominator *= math.factorial(partition.get_edge_count(r, s))

        omega /= denominator
        return self._saved_xi / omega

    @staticmethod
    def _calculate_non_degree_corrected_undirected(partition, full=True):
        """
        Combination of probability of degree sequence and probability of adjacency matrix.
        Formula
        P(A|e,b) = \frac{\prod_{r<s} e_{rs}! \prod_r e_{rr}!!}{\prod_r n_r^{e_r}}
                    * \frac{1}{\prod_{i<j}A_{ij}! \prod_i A_{ii}!!}
        :param partition: Partition of a graph with saved degree
        :type partition Partition
        :param full: Include calculation of adjacency or skip for faster calculation
        :type full Boolean
        :return: probability
        """
        probability = 1

        denominator = 1
        for r in range(partition.B):
            probability *= math.pow(2, partition.get_edge_count(r, r) / 2) \
                           * math.factorial(partition.get_edge_count(r, r) / 2)

            denominator *= math.pow(partition.get_number_of_nodes_in_block(r), partition.get_degree_of_block(r))
            for s in range(r + 1, partition.B):
                probability *= math.factorial(partition.get_edge_count(r, s))

        probability /= denominator

        if full:
            adjacency_matrix = partition.get_graph_matrix_representation(with_weights=False)

            denominator = 1
            for i in range(len(adjacency_matrix)):
                denominator *= math.pow(2, adjacency_matrix[i, i]) * math.factorial(adjacency_matrix[i, i])
                for j in range(i + 1, len(adjacency_matrix)):
                    denominator *= math.factorial(adjacency_matrix[i, j])

            probability /= denominator

        return probability

    @staticmethod
    def _calculate_p_degree_sequence_uniform_undirected(partition):
        """
        Calculate probability of degree sequence using a uniform prior
        P(k|e,b) = \prod_r ((n_r e_r))^{-1}
        :param partition: partition of a graph
        :type partition Partition
        :return: probability of degree sequence given the edge counts and groups under a uniform prior
        :rtype float
        """
        probability = 1
        for r in range(partition.B):
            n_r = partition.get_number_of_nodes_in_block(r)
            e_r = partition.get_degree_of_block(r)
            binomial_values = (n_r + e_r - 1, e_r)
            if binomial_values not in BINOMIALS:
                BINOMIALS[binomial_values] = binom(*binomial_values)
            probability /= BINOMIALS[binomial_values]
        return probability

    @staticmethod
    def _calculate_p_degree_sequence_uniform_hyperprior_undirected(partition):

        # Formulas:
        # P(k|e,b) = P(k|\eta) P(\eta|e,b)
        # P(k|\eta) = \prod_r \frac{\prod_k \eta_k^r!}{n_r!}
        # P(\eta|e,b) = prod_r q(e_r,n_r)^{-1}
        # Together
        # P(k|e,b) = \prod_r \frac{q(e_r, n_r)^{-1}}{n_r!} \prod_k \eta_k^r!
        degree_counts = partition.get_degree_distribution_of_blocks(probability=False)
        probability = 1
        for r in range(partition.B):
            probability /= get_number_of_restricted_partitions(int(partition.get_degree_of_block(r)),
                                                               int(partition.get_number_of_nodes_in_block(r))) \
                           * math.factorial(partition.get_number_of_nodes_in_block(r))
            for degree_count in degree_counts[r].values():
                probability *= math.factorial(int(degree_count))
        return probability

    @staticmethod
    def _calculate_p_edge_counts_undirected(partition):
        """
        Formulas
        P(e) = (( B(B+1)/2 E ))^{-1}
        P(b) = \frac{\prod_r n_r!}{N!} (N-1 B-1)^{-1} 1/N
        :param partition:
        :type partition Partition
        :return: P(e) * P(b)
        """
        probability = 1

        number_of_blocks = partition.B
        edge_total = partition.get_number_of_edges()
        binomial_values = (number_of_blocks * (number_of_blocks + 1) / 2 + edge_total - 1, edge_total)
        if binomial_values not in BINOMIALS:
            BINOMIALS[binomial_values] = binom(*binomial_values)
        probability /= BINOMIALS[binomial_values]

        # part for P(b)
        for r in range(number_of_blocks):
            n_r = partition.get_number_of_nodes_in_block(r)

            probability *= math.factorial(n_r)

        # group independent terms of last product
        number_of_nodes = partition.get_number_of_nodes()
        probability /= math.factorial(number_of_nodes)

        binomial_values = (number_of_nodes - 1, number_of_blocks - 1)
        if binomial_values not in BINOMIALS:
            BINOMIALS[binomial_values] = binom(*binomial_values)
        probability /= BINOMIALS[binomial_values]

        probability /= number_of_nodes

        return probability

    # ----------------------------------
    # directed Versions
    # ----------------------------------

    def calculate_complete_uniform_hyperprior_directed(self, partition):
        probability = 1
        probability *= self._calculate_p_adjacency_directed(partition)
        probability *= self._calculate_p_degree_sequence_uniform_hyperprior_directed(partition)
        probability *= self._calculate_p_edge_counts_directed(partition)
        return probability

    def calculate_complete_uniform_directed(self, partition):
        probability = 1
        probability *= self._calculate_p_adjacency_directed(partition)
        probability *= self._calculate_p_degree_sequence_uniform_directed(partition)
        probability *= self._calculate_p_edge_counts_directed(partition)
        return probability

    def calculate_complete_non_degree_corrected_directed(self, partition):
        probability = 1
        probability *= self._calculate_non_degree_corrected_directed(partition)
        probability *= self._calculate_p_edge_counts_directed(partition)
        return probability

    def _calculate_p_adjacency_directed(self, partition):
        """
        Calculate P(A|k,e,b) = \Xi(A)/\Omega(e)
        with
        \Omega(e) = \frac{\prod_r e_r^+! e_r^-!}{\prod_{r,s}e_{rs}!}
        with e_r = \sum_s e_{rs} and (2m)!! = 2^m m!
        and
        \Xi(A) = \frac{\prod_i k_i^+! k_i^-!}{\prod_{i,j}A_{ij}!}

        Value depends on adjacency matrix, degree sequence, edge counts and partition

        Xi depends only on the adjacency matrix and is only computed once
        :type partition Partition
        :return: probability P(A|k,e,b)
        """

        # check if xi was already computed
        if self._saved_xi is None or self._graph_belonging_to_xi != partition.graph:
            # if xi is still initial compute xi and therefore get information from partition
            adjacency_matrix = partition.get_graph_matrix_representation(with_weights=False)
            in_degree_sequence = partition.get_in_degree_iter()
            out_degree_sequence = partition.get_out_degree_iter()
            self._saved_xi = 1
            for in_degree in in_degree_sequence:
                self._saved_xi *= math.factorial(in_degree)

            for out_degree in out_degree_sequence:
                self._saved_xi *= math.factorial(out_degree)

            denominator = 1
            for i in range(len(adjacency_matrix)):
                for j in range(len(adjacency_matrix)):
                    denominator *= math.factorial(adjacency_matrix[i, j])

            self._saved_xi /= denominator
            # save graph of xi
            self._graph_belonging_to_xi = partition.graph

        omega = 1
        denominator = 1
        for r in range(partition.B):
            omega *= math.factorial(partition.get_in_degree_of_block(r))
            omega *= math.factorial(partition.get_out_degree_of_block(r))

            for s in range(partition.B):
                denominator *= math.factorial(partition.get_edge_count(r, s))

        omega /= denominator
        return self._saved_xi / omega

    @staticmethod
    def _calculate_non_degree_corrected_directed(partition, full=True):
        """
        Combination of probability of degree sequence and probability of adjacency matrix.
        Formula
        P(A|e,b) = \frac{e_{rs}!}{\prod_r n_r^{e_r^+} n_r^{e_r^-}}
                    * \frac{1}{\prod_{ij}A_{ij}!}
        :param partition: Partition of a graph with saved degree
        :type partition Partition
        :param full: Include calculation of adjacency or skip for faster calculation
        :type full Boolean
        :return: probability
        """
        probability = 1

        denominator = 1
        for r in range(partition.B):
            denominator *= math.pow(partition.get_number_of_nodes_in_block(r), partition.get_in_degree_of_block(r))
            denominator *= math.pow(partition.get_number_of_nodes_in_block(r), partition.get_out_degree_of_block(r))

            for s in range(partition.B):
                probability *= math.factorial(partition.get_edge_count(r, s))

        probability /= denominator

        if full:
            adjacency_matrix = partition.get_graph_matrix_representation(with_weights=False)

            denominator = 1
            for i in range(len(adjacency_matrix)):
                for j in range(len(adjacency_matrix)):
                    denominator *= math.factorial(adjacency_matrix[i, j])

            probability /= denominator

        return probability

    @staticmethod
    def _calculate_p_degree_sequence_uniform_directed(partition):
        """
        Calculate probability of degree sequence using a uniform prior
        P(k|e,b) = \prod_r ((n_r e_r^+))^{-1} ((n_r e_r^-))^{-1}
        :param partition: partition of a graph
        :type partition Partition
        :return: probability of degree sequence given the edge counts and groups under a uniform prior
        :rtype float
        """
        probability = 1
        for r in range(partition.B):
            n_r = partition.get_number_of_nodes_in_block(r)
            e_r_p = partition.get_in_degree_of_block(r)
            binomial_values = (n_r + e_r_p - 1, e_r_p)
            if binomial_values not in BINOMIALS:
                BINOMIALS[binomial_values] = binom(*binomial_values)
            probability /= BINOMIALS[binomial_values]

            e_r_m = partition.get_out_degree_of_block(r)
            binomial_values = (n_r + e_r_m - 1, e_r_m)
            if binomial_values not in BINOMIALS:
                BINOMIALS[binomial_values] = binom(*binomial_values)
            probability /= BINOMIALS[binomial_values]
        return probability

    @staticmethod
    def _calculate_p_degree_sequence_uniform_hyperprior_directed(partition):
        """
        Formulas:
        P(k|e,b) = P(k|\eta) P(\eta|e,b)
        P(k|\eta) = \prod_r \frac{\prod_{k^+,k^-} \eta_{k^+,k^-}^r!}{n_r!}
        with the joint (in, out)-degree distribution
        P(\eta|e,b) = \prod_r q(e_r^+,n_r)^{-1} q(e_r^-,n_r)^{-1}
        Together
        P(k|e,b) = \prod_r \frac{q(e_r, n_r)^{-1}}{n_r!} \prod_k \eta_k^r!
        """
        degree_counts = partition.get_joint_in_out_degree_distribution_of_blocks()
        probability = 1
        for r in range(partition.B):
            probability /= get_number_of_restricted_partitions(int(partition.get_in_degree_of_block(r)),
                                                               int(partition.get_number_of_nodes_in_block(r))) \
                           * get_number_of_restricted_partitions(int(partition.get_out_degree_of_block(r)),
                                                                 int(partition.get_number_of_nodes_in_block(r))) \
                           * math.factorial(partition.get_number_of_nodes_in_block(r))
            for degree_count in degree_counts[r].values():
                probability *= math.factorial(int(degree_count))
        return probability

    @staticmethod
    def _calculate_p_edge_counts_directed(partition):
        """
        Formulas
        P(e) = (( B*B E ))^{-1}
        P(b) = \frac{\prod_r n_r!}{N!} (N-1 B-1)^{-1} 1/N
        :param partition:
        :type partition Partition
        :return: P(e) * P(b)
        """
        probability = 1

        number_of_blocks = partition.B
        edge_total = partition.get_number_of_edges()
        binomial_values = (number_of_blocks * number_of_blocks + edge_total - 1, edge_total)
        if binomial_values not in BINOMIALS:
            BINOMIALS[binomial_values] = binom(*binomial_values)
        probability /= BINOMIALS[binomial_values]

        # part for P(b)
        for r in range(number_of_blocks):
            n_r = partition.get_number_of_nodes_in_block(r)

            probability *= math.factorial(n_r)

        # group independent terms of last product
        number_of_nodes = partition.get_number_of_nodes()
        probability /= math.factorial(number_of_nodes)

        binomial_values = (number_of_nodes - 1, number_of_blocks - 1)
        if binomial_values not in BINOMIALS:
            BINOMIALS[binomial_values] = binom(*binomial_values)
        probability /= BINOMIALS[binomial_values]

        probability /= number_of_nodes

        return probability
