# Implementation based on the following publication of Tiago P. Peixoto
# Nonparametric Bayesian inference of the microcanonical stochastic block model
# full version with scipy


import math

from scipy.special import binom

from .peixotos_flat_sbm_full import ModelLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm
from .peixotos_hierarchical_sbm_tools_full import BINOMIALS
from .nxpartitiongraphbased import NxHierarchicalPartition


# ---------------------------
# Hierarchical Version
# ---------------------------
# @formatter:off
class ModelLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbm(
        ModelLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm):

    def calculate_complete_uniform_hyperprior_undirected(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 1
        probability *= self._calculate_p_adjacency_undirected(hierarchy_partition)
        probability *= self._calculate_p_degree_sequence_uniform_hyperprior_undirected(hierarchy_partition)
        probability *= self._calculate_p_edge_counts_hierarchy_undirected(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    def calculate_complete_uniform_undirected(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 1
        probability *= self._calculate_p_adjacency_undirected(hierarchy_partition)
        probability *= self._calculate_p_degree_sequence_uniform_undirected(hierarchy_partition)
        probability *= self._calculate_p_edge_counts_hierarchy_undirected(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    def calculate_complete_non_degree_corrected_undirected(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 1
        probability *= self._calculate_non_degree_corrected_undirected(hierarchy_partition)
        probability *= self._calculate_p_edge_counts_hierarchy_undirected(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    @staticmethod
    def _calculate_p_edge_counts_hierarchy_undirected(hierarchy_partition):
        """
        Formulas
        P({e_l}|{b_l}) = \prod_{l=1}^L P(e_l| e_{l+1}, b_l)

        P(e_l| e_{l+1}, b_l) = \prod_{r<s} (( n_r^l*n_s^l  e_{rs}^{l+1} ))^{-1}
                                    * \prod_r (( n_r^l*(n_r^l + 1)/2  e_{rr}^{l+1}/2 ))^{-1}

        P({b_l}) = \prod_{l=1}^L P(b_l)
        with P(b_l) as above
            P(b_l) = \prod_r n^l_r! / B_{l-1}! * \nCr{B_{l-1}-1, B_l-1)^{-1} * 1/B_{l-1}
        and boundary condition B_0 = N
        :param hierarchy_partition:
        :type hierarchy_partition NxHierarchicalPartition
        :return:
        """
        # Notation with e_{rs}^{l+1} a little bit confusing, it is the number of edges between group n_r^l and n_s^l
        #  which we save here in level l at edge_count(r,s)
        probability = 1
        number_of_blocks = 1
        for level in hierarchy_partition.iter_levels():
            hierarchy_partition.actual_level = level
            number_of_blocks = hierarchy_partition.B
            for r in range(number_of_blocks):
                n_r = hierarchy_partition.get_number_of_nodes_in_block(r)
                for s in range(r + 1, number_of_blocks):
                    n_s = hierarchy_partition.get_number_of_nodes_in_block(s)
                    temp = hierarchy_partition.get_edge_count(r, s)
                    binomial_values = (n_r * n_s + temp - 1, temp)
                    if binomial_values not in BINOMIALS:
                        BINOMIALS[binomial_values] = binom(*binomial_values)
                    probability /= BINOMIALS[binomial_values]
                # second product
                temp = hierarchy_partition.get_edge_count(r, r) / 2
                binomial_values = (n_r * (n_r + 1) / 2 + temp - 1, temp)
                if binomial_values not in BINOMIALS:
                    BINOMIALS[binomial_values] = binom(*binomial_values)
                probability /= BINOMIALS[binomial_values]

                probability *= math.factorial(n_r)

            # group independent terms of last product
            number_of_blocks_below = hierarchy_partition.get_number_of_blocks_in_level(level - 1)
            probability /= math.factorial(number_of_blocks_below)

            binomial_values = (number_of_blocks_below - 1, number_of_blocks - 1)
            if binomial_values not in BINOMIALS:
                BINOMIALS[binomial_values] = binom(*binomial_values)
            probability /= BINOMIALS[binomial_values]

            probability /= number_of_blocks_below

        # add last hierarchy level (singleton B_L = 1 not included in hierarchy partition)
        number_of_blocks_below = number_of_blocks

        binomial_values = (
            number_of_blocks_below * (number_of_blocks_below + 1) / 2 + hierarchy_partition.edge_total - 1,
            hierarchy_partition.edge_total)
        if binomial_values not in BINOMIALS:
            BINOMIALS[binomial_values] = binom(*binomial_values)
        probability /= BINOMIALS[binomial_values]

        # next factor \prod_r n_r^L!/B_{L-1}! is always one (n_r^L = n_1^L = B_{L-1})
        # next factor simplifies to 1 too, because binomial above 0 is always 1
        # and last factor
        probability /= number_of_blocks_below
        return probability

    # ----------------------------------
    # directed Versions
    # ----------------------------------
    def calculate_complete_uniform_hyperprior_directed(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 1
        probability *= self._calculate_p_adjacency_directed(hierarchy_partition)
        probability *= self._calculate_p_degree_sequence_uniform_hyperprior_directed(hierarchy_partition)
        probability *= self._calculate_p_edge_counts_hierarchy_directed(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    def calculate_complete_uniform_directed(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 1
        probability *= self._calculate_p_adjacency_directed(hierarchy_partition)
        probability *= self._calculate_p_degree_sequence_uniform_directed(hierarchy_partition)
        probability *= self._calculate_p_edge_counts_hierarchy_directed(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    def calculate_complete_non_degree_corrected_directed(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 1
        probability *= self._calculate_non_degree_corrected_directed(hierarchy_partition)
        probability *= self._calculate_p_edge_counts_hierarchy_directed(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    @staticmethod
    def _calculate_p_edge_counts_hierarchy_directed(hierarchy_partition):
        """
        Formulas
        P({e_l}|{b_l}) = \prod_{l=1}^L P(e_l| e_{l+1}, b_l)

        P(e_l| e_{l+1}, b_l) = \prod_{r,s} (( n_r^l*n_s^l  e_{rs}^{l+1} ))^{-1}

        P({b_l}) = \prod_{l=1}^L P(b_l)
        with P(b_l) as above
            P(b_l) = \prod_r n^l_r! / B_{l-1}! * \nCr{B_{l-1}-1, B_l-1)^{-1} * 1/B_{l-1}
        and boundary condition B_0 = N
        :param hierarchy_partition:
        :type hierarchy_partition NxHierarchicalPartition
        :return: probability
        """
        probability = 1
        number_of_blocks = 1
        for level in hierarchy_partition.iter_levels():
            hierarchy_partition.actual_level = level
            number_of_blocks = hierarchy_partition.B
            for r in range(number_of_blocks):
                n_r = hierarchy_partition.get_number_of_nodes_in_block(r)
                for s in range(number_of_blocks):
                    n_s = hierarchy_partition.get_number_of_nodes_in_block(s)
                    temp = hierarchy_partition.get_edge_count(r, s)
                    binomial_values = (n_r * n_s + temp - 1, temp)
                    if binomial_values not in BINOMIALS:
                        BINOMIALS[binomial_values] = binom(*binomial_values)
                    probability /= BINOMIALS[binomial_values]

                probability *= math.factorial(n_r)

            number_of_blocks_below = hierarchy_partition.get_number_of_blocks_in_level(level - 1)
            probability /= math.factorial(number_of_blocks_below)

            binomial_values = (number_of_blocks_below - 1, number_of_blocks - 1)
            if binomial_values not in BINOMIALS:
                BINOMIALS[binomial_values] = binom(*binomial_values)
            probability /= BINOMIALS[binomial_values]

            probability /= number_of_blocks_below

        # include last hierarchy step
        number_of_blocks_below = number_of_blocks

        binomial_values = (
            number_of_blocks_below * number_of_blocks_below + hierarchy_partition.edge_total - 1,
            hierarchy_partition.edge_total)
        if binomial_values not in BINOMIALS:
            BINOMIALS[binomial_values] = binom(*binomial_values)
        probability /= BINOMIALS[binomial_values]

        # next factor \prod_r n_r^L!/B_{L-1}! is always one (n_r^L = n_1^L = B_{L-1})
        # next factor simplifies to 1 too, because binomial above 0 is always 1
        # and last factor
        probability /= number_of_blocks_below
        return probability