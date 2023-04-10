# Implementation based on the following publication of Tiago P. Peixoto
# Nonparametric Bayesian inference of the microcanonical stochastic block model
#


import math


from .peixotos_flat_sbm import ModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm
from .peixotos_flat_sbm import DeltaModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm
from .peixotos_hierarchical_sbm_tools import log_binom
from .nxpartitiongraphbased import NxHierarchicalPartition
from .objective_functions import ObjectiveFunction


# larger limit e.g. 10^4 need more time to calculate 45s vs. <1s and
#   permanent save of this data e.g. with pickle would need 1,87Gb


# see below NUMBER_OF_RESTRICTED_PARTITIONS_MATRIX

#
# List of Variables
#  N number of nodes
#  b = {b_i} partition of nodes into B groups
#  B number of groups, i.e. b_i \in [1,B]
#  k = {k_i} degree sequence
#  e = {e_{rs}} matrix of edge counts between groups r and s
#    (in undirected case e_rr twice the number of edges inside the group)
#  A adjacency matrix
#
#  \hat{e_{rs}}(A,b) = \sum_{ij} A_{ij}\delta_{b_i,r}\delta_{b_j,s}
#  \hat{k_i}(A,b) = \sum_j A_{ij}
#
#  P(A,k,e,b) = P(A|k,e,b)P(k|e,b)P(e|b)P(b) joint distribution
#   = P(A,b) marginal distribution
#
#  \Sigma = -log_2 P(A,k,e,b) = \mathfrac{S} + \mathfrac{L} description length
#   where \mathfrac{S} = -log_2 P(A|k,e,b) number of bits to precisely describe the network,
#                                           if the model parameters are known
#    \mathfrac{L} = -log_2 P(k,e,b) number of bits necessary to describe the model parameters
#
#   n = {n_r} group sizes
#   H(n) = -\sum_r(n_r/N) ln(n_r/N) entropy of group size distribution
#   \eta = {\eta_k^r} degree distribution with \eta_k^r being the number of nodes with degree k in group r
#   q(m,n) number of restricted partitions of the integer m into at most n parts
#
#   L number of levels
#   {e_l} weighted adjacency matrix at level l

# How to get to the final likelihood?
# A Prior on partitions
#
#    bad choice for prior on partitions:
#     P(b|B) = B^{-N} - every equal likely
#    because all groups have approximately the same size
#
#    better choice conditioned on group sizes n = {n_r}
#     P(b|n) = \prod_r n_r! / N!
#
#   #  hyperprior for group size distribution:
#   #    P(n|B) = ((B N))^{-1}
#   #   with ((m n)) = \nCr{n+m-1, m}
#   #
#   #  take logarithm and use Stirling's factorial approximation as well as B<<N
#   #  ln P(b,n|B) \approx -N H(n) - B ln(N)
#   #     with H(n) = -\sum_r(n_r/N) ln(n_r/N) entropy of group size distribution
#
#    to have no empty groups we use
#     P(n|B) = \nCr{N-1, B-1)^{-1}
#
#    Prior on B: P(B) = 1/N
#
#   ! in total:
#     P(b) = P(b|n)P(n|B)P(B)
#          = \prod_r n_r! / N! * \nCr{N-1, B-1)^{-1} * 1/N
#
#
#
# B Prior for the degrees
#   I Non degree corrected model
#    P(k|e,b) = \prod_r \frac{e_r!}{n_r^{e_r}\prod_{i\in b_r}k_i!}
#    P(A|e,b) = \frac{\prod_{r<s}e_{rs}! \prod_r e_{rr}!!}{\prod_r n_r^{e_r}} * 1/(\prod_{i<j}A_{ij}! \prod_i A_{ii}!!)
#      with x!! = 2^x x!
#
#
#   II Degree Corrected Uniform Distribution
#    P(k|e,b) = \prod_r ((n_r e_r))^{-1}
#
#   III Degree Corrected with hyperprior
#    P(k|e,b) = P(k|\eta) P(\eta|e,b)
#    P(k|\eta) = \prod_r \frac{\prod_k \eta_k^r}{n_r!}
#    P(\eta|e,b) = \prod_r q(e_r, n_r)^{-1}
#
#
# C Prior for the edge counts
#  P(e) = (( ((B 2)) E ))^{-1}  - undirected case!
#
#
# D Hierarchies
#  now multiple partitions one for each level, enforce B_L = 1
#  P({e_l}|{b_l}) = \prod_{l=1}^L P(e_l| e_{l+1}, b_l)
#
#  P(e_l| e_{l+1}, b_l) = \prod_{r<s} (( n_r^l*n_s^l  e_{rs}^{l+1} ))^{-1}
#                           * \prod_r (( n_r^l*(n_r^l + 1)/2  e_{rr}^{l+1}/2 ))^{-1}
#
#  P({b_l}) = \prod_{l=1}^L P(b_l)
#  with P(b_l) as above
#   P(b_l) = \prod_r n^l_r! / B_{l-1}! * \nCr{B_{l-1}-1, B_l-1)^{-1} * 1/B_{l-1}
#  and boundary condition B_0 = N
#
#
#  Last choose prior on depth L of hierarchy, e.g. P(L) = 1/L_max with e.g. L_max=N
#  but because this would only add an overall multiplicative constant, it can be omitted
#

# Appendix #
# q(m,n)
#   Recursive formula: q(m,n) = q(m,n-1) + q(m-n,n)
#
#  Approx for n \ge m^{1/6}
#   q(m,n) \approx f(u)/m exp(\sqrt{m} g(u)) with u = n/\sqrt(m) and
#    f(u) = \frac{v(u)}{2^{3/2} \pi u} [1- (1+u^2/2)exp(-v(u))]^{-1/2}
#    g(u) = \frac{2v(u)}{u} - u ln(1-exp(-v(u)))
#
#  v(u) is implicitly given by solving
#    v = u sqrt(-v^2/2 - Li_2 (1-exp(v)))
#   with Li_2(z) = -\int_0^z \frac{ln(1-t)}{t} dt the dilogarithm function
#
# for smaller values n << m^{1/3} we have
#  q(m,n) \approx \frac{ (m-1 n-1) }{m!}
#


# ---------------------------
# Hierarchical Version
# ---------------------------
# @formatter:off


class ModelLogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbm(
        ModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm):
    """Log Version of Likelihood of Hierarchical Microcanonical Degree Corrected SBM"""

    def calculate_complete_uniform_hyperprior_undirected(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 0
        probability += self._calculate_p_adjacency_undirected(hierarchy_partition)
        probability += self._calculate_p_degree_sequence_uniform_hyperprior_undirected(hierarchy_partition)
        probability += self._calculate_p_edge_counts_hierarchy_undirected(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    def calculate_complete_uniform_undirected(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 0
        probability += self._calculate_p_adjacency_undirected(hierarchy_partition)
        probability += self._calculate_p_degree_sequence_uniform_undirected(hierarchy_partition)
        probability += self._calculate_p_edge_counts_hierarchy_undirected(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    def calculate_complete_non_degree_corrected_undirected(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 0
        probability += self._calculate_non_degree_corrected_undirected(hierarchy_partition)
        probability += self._calculate_p_edge_counts_hierarchy_undirected(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    @staticmethod
    def _calculate_p_edge_counts_hierarchy_undirected(hierarchy_partition):
        """
        Formulas
        \log P({e_l}|{b_l}) = \sum_{l=1}^L \log P(e_l| e_{l+1}, b_l)

        \log P(e_l| e_{l+1}, b_l) = \sum_{r<s}\log  (( n_r^l*n_s^l  e_{rs}^{l+1} ))^{-1}
                                    + \sum_r \log (( n_r^l*(n_r^l + 1)/2  e_{rr}^{l+1}/2 ))^{-1}

        \log P({b_l}) = \sum_{l=1}^L \log P(b_l)
        with \log P(b_l) as above
            \log P(b_l) = \sum_r \log n^l_r! - \log B_{l-1}! - \log \nCr{B_{l-1}-1, B_l-1)-\log B_{l-1}
        and boundary condition B_0 = N
        :param hierarchy_partition:
        :type hierarchy_partition NxHierarchicalPartition
        :return:
        """
        # Notation with e_{rs}^{l+1} a little bit confusing, it is the number of edges between group n_r^l and n_s^l
        #  which we save here in level l at edge_count(r,s)
        probability = 0
        number_of_blocks = 1
        saved_level = hierarchy_partition.actual_level
        for level in hierarchy_partition.iter_levels():
            hierarchy_partition.actual_level = level
            number_of_blocks = hierarchy_partition.B
            for r in range(number_of_blocks):
                n_r = hierarchy_partition.get_number_of_nodes_in_block(r)
                for s in range(r + 1, number_of_blocks):
                    n_s = hierarchy_partition.get_number_of_nodes_in_block(s)
                    temp = hierarchy_partition.get_edge_count(r, s)
                    probability -= log_binom(n_r * n_s + temp - 1, temp)
                # second product
                temp = hierarchy_partition.get_edge_count(r, r) / 2
                probability -= log_binom(n_r * (n_r + 1) / 2 + temp - 1, temp)

                probability += math.lgamma(1 + n_r)

            # group independent terms of last product
            number_of_blocks_below = hierarchy_partition.get_number_of_blocks_in_level(level - 1)
            probability -= math.lgamma(1 + number_of_blocks_below)

            probability -= log_binom(number_of_blocks_below - 1, number_of_blocks - 1)

            probability -= math.log(number_of_blocks_below)

        # add last hierarchy level (singleton B_L = 1 not included in hierarchy partition)
        number_of_blocks_below = number_of_blocks

        probability -= log_binom(
            number_of_blocks_below * (number_of_blocks_below + 1) / 2 + hierarchy_partition.edge_total - 1,
            hierarchy_partition.edge_total)

        # next two factors as log always 0
        # and last factor
        probability -= math.log(number_of_blocks_below)
        hierarchy_partition.actual_level = saved_level
        return probability

    # ----------------------------------
    # directed Versions
    # ----------------------------------
    def calculate_complete_uniform_hyperprior_directed(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 0
        probability += self._calculate_p_adjacency_directed(hierarchy_partition)
        probability += self._calculate_p_degree_sequence_uniform_hyperprior_directed(hierarchy_partition)
        probability += self._calculate_p_edge_counts_hierarchy_directed(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    def calculate_complete_uniform_directed(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 0
        probability += self._calculate_p_adjacency_directed(hierarchy_partition)
        probability += self._calculate_p_degree_sequence_uniform_directed(hierarchy_partition)
        probability += self._calculate_p_edge_counts_hierarchy_directed(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    def calculate_complete_non_degree_corrected_directed(self, hierarchy_partition):
        saved_level = hierarchy_partition.actual_level
        hierarchy_partition.actual_level = 0
        probability = 0
        probability += self._calculate_non_degree_corrected_directed(hierarchy_partition)
        probability += self._calculate_p_edge_counts_hierarchy_directed(hierarchy_partition)
        hierarchy_partition.actual_level = saved_level
        return probability

    @staticmethod
    def _calculate_p_edge_counts_hierarchy_directed(hierarchy_partition):
        """
        Formulas
        \log P({e_l}|{b_l}) = \sum_{l=1}^L \log P(e_l| e_{l+1}, b_l)

        \log P(e_l| e_{l+1}, b_l) = -\sum_{r,s} \log (( n_r^l*n_s^l  e_{rs}^{l+1} ))

        \log P({b_l}) = \sum_{l=1}^L \log P(b_l)
        with \log P(b_l) as above
            \log P(b_l) = \sum_r \log n^l_r! - \log B_{l-1}! - \log \nCr{B_{l-1}-1, B_l-1)-\log B_{l-1}
        and boundary condition B_0 = N
        :param hierarchy_partition:
        :type hierarchy_partition NxHierarchicalPartition
        :return: log probability
        """
        probability = 0
        number_of_blocks = 0
        saved_level = hierarchy_partition.actual_level
        for level in hierarchy_partition.iter_levels():
            hierarchy_partition.actual_level = level
            number_of_blocks = hierarchy_partition.B
            for r in range(number_of_blocks):
                n_r = hierarchy_partition.get_number_of_nodes_in_block(r)
                for s in range(number_of_blocks):
                    n_s = hierarchy_partition.get_number_of_nodes_in_block(s)
                    temp = hierarchy_partition.get_edge_count(r, s)
                    if n_r > 0 and n_s > 0 or temp > 0:
                        probability -= log_binom(n_r * n_s + temp - 1, temp)

                probability += math.lgamma(1 + n_r)

            number_of_blocks_below = hierarchy_partition.get_number_of_blocks_in_level(level - 1)
            probability -= math.lgamma(1 + number_of_blocks_below)

            probability -= log_binom(number_of_blocks_below - 1, number_of_blocks - 1)

            probability -= math.log(number_of_blocks_below)

        # include last hierarchy step
        number_of_blocks_below = number_of_blocks

        probability -= log_binom(
            number_of_blocks_below * number_of_blocks_below + hierarchy_partition.edge_total - 1,
            hierarchy_partition.edge_total)

        # next two factors always 0 (as log)
        # and last factor
        probability -= math.log(number_of_blocks_below)

        hierarchy_partition.actual_level = saved_level
        return probability


class DeltaModelLogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbm(
        DeltaModelLogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbm):
    """Delta Log Version of Likelihood of Hierarchical Microcanonical Degree Corrected SBM"""

    def calculate_delta_complete_uniform_hyperprior_undirected(self, hierarchy_partition, from_block, to_block, *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 3:
            kit, selfloops, degree = args
            nodes_moved = 1
            nodes_remaining = not (hierarchy_partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 5:
            kit, selfloops, degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != hierarchy_partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        if hierarchy_partition.actual_level == 0:
            delta += self._calculate_delta_p_adjacency_undirected(
                hierarchy_partition, from_block, to_block, kit, selfloops, degree, to_block_exists)
            delta += self._calculate_delta_p_degree_sequence_uniform_hyperprior_undirected(
                hierarchy_partition, from_block, to_block, degree,
                nodes_remaining, to_block_exists, nodes_moved
            )
        delta += self._calculate_delta_p_edge_counts_hierarchy_undirected(
            hierarchy_partition, hierarchy_partition.actual_level, from_block, to_block, kit, selfloops,
            nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    def calculate_delta_complete_uniform_undirected(self, hierarchy_partition, from_block, to_block, *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 3:
            kit, selfloops, degree = args
            nodes_moved = 1
            nodes_remaining = not (hierarchy_partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 5:
            kit, selfloops, degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != hierarchy_partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        if hierarchy_partition.actual_level == 0:
            delta += self._calculate_delta_p_adjacency_undirected(
                hierarchy_partition, from_block, to_block, kit, selfloops, degree, to_block_exists)
            delta += self._calculate_delta_p_degree_sequence_uniform_undirected(
                hierarchy_partition, from_block, to_block, degree,
                nodes_remaining, to_block_exists, nodes_moved
            )
        delta += self._calculate_delta_p_edge_counts_hierarchy_undirected(
            hierarchy_partition, hierarchy_partition.actual_level, from_block, to_block, kit, selfloops,
            nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    def calculate_delta_complete_non_degree_corrected_undirected(self, hierarchy_partition, from_block, to_block,
                                                                 *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 3:
            kit, selfloops, degree = args
            nodes_moved = 1
            nodes_remaining = not (hierarchy_partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 5:
            kit, selfloops, degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != hierarchy_partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        if hierarchy_partition.actual_level == 0:
            delta += self._calculate_delta_non_degree_corrected_undirected(
                hierarchy_partition, from_block, to_block, kit, selfloops, degree,
                nodes_remaining, to_block_exists, nodes_moved)
        delta += self._calculate_delta_p_edge_counts_hierarchy_undirected(
            hierarchy_partition, hierarchy_partition.actual_level, from_block, to_block, kit, selfloops,
            nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    @staticmethod
    def calculate_delta_actual_level_removed_undirected(hierarchy_partition):
        if hierarchy_partition.actual_level == 0:
            # would change others too
            raise ValueError()
        level = hierarchy_partition.actual_level
        delta = 0

        new_number_of_nodes_per_block = {block: 0 for block in range(
            hierarchy_partition.get_number_of_blocks_in_level(level + 1))}

        # code from above for one level
        number_of_blocks = hierarchy_partition.B
        for r in range(number_of_blocks):
            n_r = hierarchy_partition.get_number_of_nodes_in_block(r)
            for s in range(r + 1, number_of_blocks):
                n_s = hierarchy_partition.get_number_of_nodes_in_block(s)
                temp = hierarchy_partition.get_edge_count(r, s)
                if n_r > 0 and n_s > 0 or temp > 0:
                    delta -= log_binom(n_r * n_s + temp - 1, temp)
            # second product
            temp = hierarchy_partition.get_edge_count(r, r) / 2
            if n_r > 0 or temp > 0:
                delta -= log_binom(n_r * (n_r + 1) / 2 + temp - 1, temp)

            delta += math.lgamma(1 + n_r)
            if level != hierarchy_partition.max_level:
                new_number_of_nodes_per_block[hierarchy_partition.partitions[level + 1].partition[r]] += n_r
            else:
                new_number_of_nodes_per_block[0] += n_r

        number_of_blocks_below = hierarchy_partition.get_number_of_blocks_in_level(level - 1)

        # only multiset coefficient for level above
        if level != hierarchy_partition.max_level:
            # only in this case (else the term \prod n_r^l/B_{l-1}! = 1 -> log(..) = 0)
            delta -= math.lgamma(1 + number_of_blocks)

            hierarchy_partition.actual_level += 1
            for r in range(hierarchy_partition.B):
                n_r = hierarchy_partition.get_number_of_nodes_in_block(r)
                for s in range(r + 1, hierarchy_partition.B):
                    n_s = hierarchy_partition.get_number_of_nodes_in_block(s)
                    temp = hierarchy_partition.get_edge_count(r, s)
                    if n_r > 0 and n_s > 0 or temp > 0:
                        delta -= log_binom(n_r * n_s + temp - 1, temp)
                # second product
                temp = hierarchy_partition.get_edge_count(r, r) / 2
                if n_r > 0 or temp > 0:
                    delta -= log_binom(n_r * (n_r + 1) / 2 + temp - 1, temp)

                delta += math.lgamma(1 + n_r)

            hierarchy_partition.actual_level -= 1
        else:
            # in this case include fraction of other block which is in the second step included in the 0 term
            delta -= math.lgamma(1 + number_of_blocks_below)

            delta -= log_binom(
                number_of_blocks * (number_of_blocks + 1) / 2 + hierarchy_partition.edge_total - 1,
                hierarchy_partition.edge_total)

        # terms for binomial coefficient (B_{l-1}-1 B_l - 1)
        delta -= log_binom(number_of_blocks_below - 1, number_of_blocks - 1)

        delta -= math.log(number_of_blocks)  # !! change to actual block number

        # # add term for level above
        delta -= log_binom(number_of_blocks - 1, hierarchy_partition.get_number_of_blocks_in_level(level + 1) - 1)

        # switch sign of delta
        delta *= -1

        # care about new term
        # first term for level above
        if level != hierarchy_partition.max_level:

            hierarchy_partition.actual_level += 1
            for r in range(hierarchy_partition.B):
                n_r = new_number_of_nodes_per_block[r]
                for s in range(r + 1, hierarchy_partition.B):
                    n_s = new_number_of_nodes_per_block[s]
                    temp = hierarchy_partition.get_edge_count(r, s)
                    if n_r > 0 and n_s > 0 or temp > 0:
                        delta -= log_binom(n_r * n_s + temp - 1, temp)

                # second product
                temp = hierarchy_partition.get_edge_count(r, r) / 2
                if n_r > 0 or temp > 0:
                    delta -= log_binom(n_r * (n_r + 1) / 2 + temp - 1, temp)

                delta += math.lgamma(1 + n_r)

            hierarchy_partition.actual_level -= 1

            # only binomial coefficient is affected after the multiset binomial coefficients
            delta -= log_binom(number_of_blocks_below - 1,
                               hierarchy_partition.get_number_of_blocks_in_level(level + 1) - 1)
        else:
            delta -= log_binom(
                new_number_of_nodes_per_block[0] * (new_number_of_nodes_per_block[0] + 1) / 2
                + hierarchy_partition.edge_total - 1,
                hierarchy_partition.edge_total)

        return delta

    @staticmethod
    def _calculate_delta_p_edge_counts_hierarchy_undirected(hierarchy_partition, level, from_block, to_block,
                                                            kit, selfloops, nodes_remaining, to_block_exists,
                                                            nodes_moved=1):
        """

        :param hierarchy_partition:
        :type hierarchy_partition NxHierarchicalPartition
        :return:
        """

        delta = 0

        # handle term -\log B_{l-1}! - \log \nCr{B_{l-1}-1, B_l-1)-\log B_{l-1}
        new_block_count = -1
        top_block_node_change = 0
        if to_block == hierarchy_partition.B:
            # new block
            new_block_count = to_block + 1
            top_block_node_change = 1
        if hierarchy_partition.get_number_of_nodes_in_block(from_block) == nodes_moved and from_block != to_block:
            # block will be removed
            new_block_count = hierarchy_partition.B - 1
            top_block_node_change = -1
        if new_block_count != -1:
            # term - \log B_{l-1}
            delta -= math.log(new_block_count) - math.log(hierarchy_partition.B)

            # term - \log \nCr{B_{l-1}-1, B_l-1)
            # terms for change B_l
            delta -= log_binom(hierarchy_partition.get_number_of_blocks_in_level(level - 1) - 1,
                               new_block_count - 1)

            delta += log_binom(hierarchy_partition.get_number_of_blocks_in_level(level - 1) - 1,
                               hierarchy_partition.B - 1)

            # terms for change B_{l-1}
            delta -= log_binom(new_block_count - 1,
                               hierarchy_partition.get_number_of_blocks_in_level(level + 1) - 1)

            delta += log_binom(hierarchy_partition.B - 1,
                               hierarchy_partition.get_number_of_blocks_in_level(level + 1) - 1)

            # deal with changes of nodes_in_block of top_block belonging to from_block
            if hierarchy_partition.actual_level < hierarchy_partition.max_level:
                # handle term -\log B_{l-1}!
                delta -= math.lgamma(1 + new_block_count) - math.lgamma(1 + hierarchy_partition.B)

                # switch to level above actual level
                hierarchy_partition.actual_level += 1
                top_block = hierarchy_partition.get_block_of_node(from_block)
                old_n_top_block = hierarchy_partition.get_number_of_nodes_in_block(top_block)
                new_n_top_block = old_n_top_block + top_block_node_change

                # handle all (( n_r e_r )) terms
                for block in hierarchy_partition.get_neighbors_of_block(top_block):
                    if block != top_block:
                        n_block = hierarchy_partition.get_number_of_nodes_in_block(block)
                        # old value
                        edge_count = hierarchy_partition.get_edge_count(top_block, block)
                        # neighbor -> edge_count > 0
                        delta += log_binom(old_n_top_block * n_block + edge_count - 1, edge_count)
                        # new value
                        delta -= log_binom(new_n_top_block * n_block + edge_count - 1, edge_count)

                # deal with entry with both top_block
                # old term
                edge_count = hierarchy_partition.get_edge_count(top_block, top_block) / 2
                delta += log_binom(old_n_top_block * (old_n_top_block + 1) / 2 + edge_count - 1, edge_count)
                # new term
                delta -= log_binom(new_n_top_block * (new_n_top_block + 1) / 2 + edge_count - 1, edge_count)

                # handle n_r! term
                delta += math.lgamma(1 + new_n_top_block) - math.lgamma(1 + old_n_top_block)

                # switch back to actual level
                hierarchy_partition.actual_level -= 1
            else:
                # only one term!
                # no need of extra term -\log B_{l-1}! because effects cancel out
                # new term
                delta -= log_binom(
                    new_block_count * (new_block_count + 1) / 2 + hierarchy_partition.edge_total - 1,
                    hierarchy_partition.edge_total)
                # old term
                delta += log_binom(
                    hierarchy_partition.B * (hierarchy_partition.B + 1) / 2 + hierarchy_partition.edge_total - 1,
                    hierarchy_partition.edge_total)

        # delta of second sum \sum_{r<s}\log e_{rs}!
        n_from_block_old = hierarchy_partition.get_number_of_nodes_in_block(from_block)
        n_from_block_new = n_from_block_old - nodes_moved
        if to_block_exists:
            n_to_block_old = hierarchy_partition.get_number_of_nodes_in_block(to_block)
        else:
            n_to_block_old = 0
        n_to_block_new = n_to_block_old + nodes_moved
        # important loop over all neighbors of the block because number of nodes in block effects all
        neighbors = hierarchy_partition.get_neighbors_of_block(from_block)
        if to_block_exists:
            neighbors = neighbors.union(hierarchy_partition.get_neighbors_of_block(to_block))
        # for those with no kit value available use 0 instead
        for block in neighbors:
            if block != from_block and block != to_block:
                n_block = hierarchy_partition.get_number_of_nodes_in_block(block)
                # old value from block
                edge_count = hierarchy_partition.get_edge_count(from_block, block)
                # neighbor -> edge_count > 0
                delta += log_binom(n_from_block_old * n_block + edge_count - 1, edge_count)

                if nodes_remaining:
                    # new value from block
                    if block in kit:
                        edge_count -= kit[block]
                    delta -= log_binom(n_from_block_new * n_block + edge_count - 1, edge_count)

                # old value to block
                if to_block_exists:
                    edge_count = hierarchy_partition.get_edge_count(to_block, block)
                    delta += log_binom(n_to_block_old * n_block + edge_count - 1, edge_count)
                else:
                    edge_count = 0
                # new value to block
                if block in kit:
                    edge_count += kit[block]
                delta -= log_binom(n_to_block_new * n_block + edge_count - 1, edge_count)

        # handle the term with both from and to block
        # old e_{from_block, to_block} term
        if to_block_exists:
            edge_count = hierarchy_partition.get_edge_count(from_block, to_block)
            delta += log_binom(n_from_block_old * n_to_block_old + edge_count - 1, edge_count)
        else:
            edge_count = 0

        if nodes_remaining:
            # new term
            edge_count += kit[from_block] - kit[to_block]
            delta -= log_binom(n_from_block_new * n_to_block_new + edge_count - 1, edge_count)

        # handle sum -\sum_r \log (( n_r^l*(n_r^l + 1)/2  e_{rr}^{l+1}/2 ))
        # old e_{from_block, from_block} term
        edge_count = hierarchy_partition.get_edge_count(from_block, from_block) / 2
        delta += log_binom(n_from_block_old * (n_from_block_old + 1) / 2 + edge_count - 1, edge_count)

        if nodes_remaining:
            # new term
            edge_count -= kit[from_block] + selfloops
            delta -= log_binom(n_from_block_new * (n_from_block_new + 1) / 2 + edge_count - 1, edge_count)

        # old e_{to_block, to_block} term
        if to_block_exists:
            edge_count = hierarchy_partition.get_edge_count(to_block, to_block) / 2
            delta += log_binom(n_to_block_old * (n_to_block_old + 1) / 2 + edge_count - 1, edge_count)
        else:
            edge_count = 0
        # new term
        edge_count += kit[to_block] + selfloops
        delta -= log_binom(n_to_block_new * (n_to_block_new + 1) / 2 + edge_count - 1, edge_count)

        # delta for \sum_r \log n^l_r!
        delta += - math.lgamma(1 + n_from_block_old) \
            + math.lgamma(1 + n_to_block_new) \
            - math.lgamma(1 + n_to_block_old)

        if nodes_remaining:
            delta += math.lgamma(1 + n_from_block_new)

        return delta

    # ----------------------------------
    # directed Versions
    # ----------------------------------
    def calculate_delta_complete_uniform_hyperprior_directed(self, hierarchy_partition, from_block, to_block, *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 5:
            kit, kti, selfloops, in_degree, out_degree = args
            nodes_moved = 1
            nodes_remaining = not (hierarchy_partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 7:
            kit, kti, selfloops, in_degree, out_degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != hierarchy_partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        if hierarchy_partition.actual_level == 0:
            delta += self._calculate_delta_p_adjacency_directed(
                hierarchy_partition, from_block, to_block, kit, kti, selfloops, in_degree, out_degree, to_block_exists)
            delta += self._calculate_delta_p_degree_sequence_uniform_hyperprior_directed(
                hierarchy_partition, from_block, to_block, in_degree, out_degree,
                nodes_remaining, to_block_exists, nodes_moved
            )
        delta += self._calculate_delta_p_edge_counts_hierarchy_directed(
            hierarchy_partition, hierarchy_partition.actual_level, from_block, to_block, kit, kti, selfloops,
            nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    def calculate_delta_complete_uniform_directed(self, hierarchy_partition, from_block, to_block, *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 5:
            kit, kti, selfloops, in_degree, out_degree = args
            nodes_moved = 1
            nodes_remaining = not (hierarchy_partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 7:
            kit, kti, selfloops, in_degree, out_degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != hierarchy_partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        if hierarchy_partition.actual_level == 0:
            delta += self._calculate_delta_p_adjacency_directed(
                hierarchy_partition, from_block, to_block, kit, kti, selfloops, in_degree, out_degree, to_block_exists)
            delta += self._calculate_delta_p_degree_sequence_uniform_directed(
                hierarchy_partition, from_block, to_block, in_degree, out_degree,
                nodes_remaining, to_block_exists, nodes_moved)
        delta += self._calculate_delta_p_edge_counts_hierarchy_directed(
            hierarchy_partition, hierarchy_partition.actual_level, from_block, to_block, kit, kti, selfloops,
            nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    def calculate_delta_complete_non_degree_corrected_directed(self, hierarchy_partition, from_block, to_block, *args):
        if from_block == to_block:
            return 0
        # handle input variables
        if len(args) == 5:
            kit, kti, selfloops, in_degree, out_degree = args
            nodes_moved = 1
            nodes_remaining = not (hierarchy_partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
        elif len(args) == 7:
            kit, kti, selfloops, in_degree, out_degree, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        to_block_exists = to_block != hierarchy_partition.B

        # quick exit for 'no move'
        if not to_block_exists and not nodes_remaining:
            return 0

        delta = 0
        if hierarchy_partition.actual_level == 0:
            delta += self._calculate_delta_non_degree_corrected_directed(
                hierarchy_partition, from_block, to_block, kit, kti, selfloops, in_degree, out_degree,
                nodes_remaining, to_block_exists, nodes_moved)
        delta += self._calculate_delta_p_edge_counts_hierarchy_directed(
            hierarchy_partition, hierarchy_partition.actual_level, from_block, to_block, kit, kti, selfloops,
            nodes_remaining, to_block_exists, nodes_moved
        )
        return delta

    @staticmethod
    def calculate_delta_actual_level_removed_directed(hierarchy_partition):
        if hierarchy_partition.actual_level == 0:
            # would change others too
            raise ValueError()
        level = hierarchy_partition.actual_level
        delta = 0

        new_number_of_nodes_per_block = {block: 0 for block in range(
            hierarchy_partition.get_number_of_blocks_in_level(level + 1))}

        # code from above for one level
        number_of_blocks = hierarchy_partition.B
        for r in range(number_of_blocks):
            n_r = hierarchy_partition.get_number_of_nodes_in_block(r)
            for s in range(number_of_blocks):
                n_s = hierarchy_partition.get_number_of_nodes_in_block(s)
                temp = hierarchy_partition.get_edge_count(r, s)
                if n_r > 0 and n_s > 0 or temp > 0:
                    delta -= log_binom(n_r * n_s + temp - 1, temp)

            delta += math.lgamma(1 + n_r)
            if level != hierarchy_partition.max_level:
                new_number_of_nodes_per_block[hierarchy_partition.partitions[level + 1].partition[r]] += n_r
            else:
                new_number_of_nodes_per_block[0] += n_r

        number_of_blocks_below = hierarchy_partition.get_number_of_blocks_in_level(level - 1)

        # only multiset coefficient for level above
        if level != hierarchy_partition.max_level:
            # only in this case (else the term \prod n_r^l/B_{l-1}! = 1 -> log(..) = 0)
            delta -= math.lgamma(1 + number_of_blocks)

            hierarchy_partition.actual_level += 1
            for r in range(hierarchy_partition.B):
                n_r = hierarchy_partition.get_number_of_nodes_in_block(r)
                for s in range(hierarchy_partition.B):
                    n_s = hierarchy_partition.get_number_of_nodes_in_block(s)
                    temp = hierarchy_partition.get_edge_count(r, s)
                    if n_r > 0 and n_s > 0 or temp > 0:
                        delta -= log_binom(n_r * n_s + temp - 1, temp)

                delta += math.lgamma(1 + n_r)

            hierarchy_partition.actual_level -= 1
        else:
            # in this case include fraction of other block which is in the second step included in the 0 term
            delta -= math.lgamma(1 + number_of_blocks_below)

            delta -= log_binom(
                number_of_blocks * number_of_blocks + hierarchy_partition.edge_total - 1,
                hierarchy_partition.edge_total)

        # terms for binomial coefficient (B_{l-1}-1 B_l - 1)
        delta -= log_binom(number_of_blocks_below - 1, number_of_blocks - 1)

        delta -= math.log(number_of_blocks)  # !! change to actual block number

        # # add term for level above
        delta -= log_binom(number_of_blocks - 1, hierarchy_partition.get_number_of_blocks_in_level(level + 1) - 1)

        # switch sign of delta
        delta *= -1

        # care about new term
        # first term for level above
        if level != hierarchy_partition.max_level:

            hierarchy_partition.actual_level += 1
            for r in range(hierarchy_partition.B):
                n_r = new_number_of_nodes_per_block[r]
                for s in range(hierarchy_partition.B):
                    n_s = new_number_of_nodes_per_block[s]
                    temp = hierarchy_partition.get_edge_count(r, s)
                    if n_r > 0 and n_s > 0 or temp > 0:
                        delta -= log_binom(n_r * n_s + temp - 1, temp)

                delta += math.lgamma(1 + n_r)

            hierarchy_partition.actual_level -= 1

            # only binomial coefficient is affected after the multiset binomial coefficients
            delta -= log_binom(number_of_blocks_below - 1,
                               hierarchy_partition.get_number_of_blocks_in_level(level + 1) - 1)
        else:
            delta -= log_binom(
                new_number_of_nodes_per_block[0] * new_number_of_nodes_per_block[0]
                + hierarchy_partition.edge_total - 1,
                hierarchy_partition.edge_total)

        return delta

    @staticmethod
    def _calculate_delta_p_edge_counts_hierarchy_directed(hierarchy_partition, level, from_block, to_block, kit, kti,
                                                          selfloops, nodes_remaining, to_block_exists, nodes_moved=1):
        """
        Formulas
        \log P({e_l}|{b_l}) = \sum_{l=1}^L \log P(e_l| e_{l+1}, b_l)

        \log P(e_l| e_{l+1}, b_l) = -\sum_{r,s} \log (( n_r^l*n_s^l  e_{rs}^{l+1} ))

        \log P({b_l}) = \sum_{l=1}^L \log P(b_l)
        with \log P(b_l) as above
            \log P(b_l) = \sum_r \log n^l_r! - \log B_{l-1}! - \log \nCr{B_{l-1}-1, B_l-1)-\log B_{l-1}
        and boundary condition B_0 = N
        :param hierarchy_partition:
        :type hierarchy_partition NxHierarchicalPartition
        :return: log probability
        """
        delta = 0

        # handle term -\log B_{l-1}! - \log \nCr{B_{l-1}-1, B_l-1)-\log B_{l-1}
        new_block_count = -1
        top_block_node_change = 0
        if to_block == hierarchy_partition.B:
            # new block
            new_block_count = to_block + 1
            top_block_node_change = 1
        if hierarchy_partition.get_number_of_nodes_in_block(from_block) == nodes_moved and from_block != to_block:
            # block will be removed
            new_block_count = hierarchy_partition.B - 1
            top_block_node_change = -1
        if new_block_count != -1:
            # term - \log B_{l-1}
            delta -= math.log(new_block_count) - math.log(hierarchy_partition.B)

            # term - \log \nCr{B_{l-1}-1, B_l-1)
            # terms for change B_l
            delta -= log_binom(hierarchy_partition.get_number_of_blocks_in_level(level - 1) - 1,
                               new_block_count - 1)

            delta += log_binom(hierarchy_partition.get_number_of_blocks_in_level(level - 1) - 1,
                               hierarchy_partition.B - 1)

            # terms for change B_{l-1}
            delta -= log_binom(new_block_count - 1,
                               hierarchy_partition.get_number_of_blocks_in_level(level + 1) - 1)

            delta += log_binom(hierarchy_partition.B - 1,
                               hierarchy_partition.get_number_of_blocks_in_level(level + 1) - 1)

            # deal with changes of nodes_in_block of top_block belonging to from_block
            if hierarchy_partition.actual_level < hierarchy_partition.max_level:
                # handle term -\log B_{l-1}!
                delta -= math.lgamma(1 + new_block_count) - math.lgamma(1 + hierarchy_partition.B)

                # switch to level above actual level
                hierarchy_partition.actual_level += 1
                top_block = hierarchy_partition.get_block_of_node(from_block)
                old_n_top_block = hierarchy_partition.get_number_of_nodes_in_block(top_block)
                new_n_top_block = old_n_top_block + top_block_node_change

                # handle all (( n_r e_r )) terms
                for block in hierarchy_partition.get_neighbors_of_block(top_block):
                    if block != top_block:
                        n_block = hierarchy_partition.get_number_of_nodes_in_block(block)
                        # old value
                        edge_count = hierarchy_partition.get_edge_count(top_block, block)
                        delta += log_binom(old_n_top_block * n_block + edge_count - 1, edge_count)
                        # new value
                        delta -= log_binom(new_n_top_block * n_block + edge_count - 1, edge_count)

                        # old value
                        edge_count = hierarchy_partition.get_edge_count(block, top_block)
                        delta += log_binom(old_n_top_block * n_block + edge_count - 1, edge_count)
                        # new value
                        delta -= log_binom(new_n_top_block * n_block + edge_count - 1, edge_count)

                # entry for both top block
                edge_count = hierarchy_partition.get_edge_count(top_block, top_block)
                delta += log_binom(old_n_top_block * old_n_top_block + edge_count - 1, edge_count)
                # new value
                delta -= log_binom(new_n_top_block * new_n_top_block + edge_count - 1, edge_count)

                # handle n_r! term
                delta += math.lgamma(1 + new_n_top_block) - math.lgamma(1 + old_n_top_block)

                # switch back to actual level
                hierarchy_partition.actual_level -= 1
            else:
                # only one term!
                # no need of extra term -\log B_{l-1}! because effects cancel out
                # new term
                delta -= log_binom(
                    new_block_count * new_block_count + hierarchy_partition.edge_total - 1,
                    hierarchy_partition.edge_total)
                # old term
                delta += log_binom(
                    hierarchy_partition.B * hierarchy_partition.B + hierarchy_partition.edge_total - 1,
                    hierarchy_partition.edge_total)

        # # delta of second sum \sum_{rs}\log e_{rs}!
        n_from_block_old = hierarchy_partition.get_number_of_nodes_in_block(from_block)
        n_from_block_new = n_from_block_old - nodes_moved
        if to_block_exists:
            n_to_block_old = hierarchy_partition.get_number_of_nodes_in_block(to_block)
        else:
            n_to_block_old = 0
        n_to_block_new = n_to_block_old + nodes_moved
        # important loop over all neighbors of the block because number of nodes in block effects all
        neighbors = hierarchy_partition.get_neighbors_of_block(from_block)
        if to_block_exists:
            neighbors = neighbors.union(hierarchy_partition.get_neighbors_of_block(to_block))
        # for those with no kit value available use 0 instead
        for block in neighbors:
            if block != from_block and block != to_block:
                n_block = hierarchy_partition.get_number_of_nodes_in_block(block)
                # old value from block -> block
                edge_count = hierarchy_partition.get_edge_count(from_block, block)
                delta += log_binom(n_from_block_old * n_block + edge_count - 1, edge_count)
                if nodes_remaining:
                    # new value from block -> block
                    if block in kit:
                        edge_count -= kit[block]
                    delta -= log_binom(n_from_block_new * n_block + edge_count - 1, edge_count)

                # old value to block -> block
                if to_block_exists:
                    edge_count = hierarchy_partition.get_edge_count(to_block, block)
                    delta += log_binom(n_to_block_old * n_block + edge_count - 1, edge_count)
                else:
                    edge_count = 0
                # new value to block -> block
                if block in kit:
                    edge_count += kit[block]
                delta -= log_binom(n_to_block_new * n_block + edge_count - 1, edge_count)

                # other directions
                # old value from block <- block
                edge_count = hierarchy_partition.get_edge_count(block, from_block)
                delta += log_binom(n_from_block_old * n_block + edge_count - 1, edge_count)
                if nodes_remaining:
                    # new value from block <- block
                    if block in kti:
                        edge_count -= kti[block]
                    delta -= log_binom(n_from_block_new * n_block + edge_count - 1, edge_count)

                # old value to block <- block
                if to_block_exists:
                    edge_count = hierarchy_partition.get_edge_count(block, to_block)
                    delta += log_binom(n_to_block_old * n_block + edge_count - 1, edge_count)
                else:
                    edge_count = 0
                # new value to block <- block
                if block in kti:
                    edge_count += kti[block]
                delta -= log_binom(n_to_block_new * n_block + edge_count - 1, edge_count)

        # handle the term with both from and to block
        # old e_{from_block, to_block} term
        if to_block_exists:
            edge_count = hierarchy_partition.get_edge_count(from_block, to_block)
            delta += log_binom(n_from_block_old * n_to_block_old + edge_count - 1, edge_count)
        else:
            edge_count = 0
        if nodes_remaining:
            # new term
            edge_count += kti[from_block] - kit[to_block]
            delta -= log_binom(n_from_block_new * n_to_block_new + edge_count - 1, edge_count)

        # other direction
        if to_block_exists:
            edge_count = hierarchy_partition.get_edge_count(to_block, from_block)
            delta += log_binom(n_from_block_old * n_to_block_old + edge_count - 1, edge_count)
        else:
            edge_count = 0
        if nodes_remaining:
            # new term
            edge_count += kit[from_block] - kti[to_block]
            delta -= log_binom(n_from_block_new * n_to_block_new + edge_count - 1, edge_count)

        # handle term with both from_block or to_block
        # old e_{from_block, from_block} term
        edge_count = hierarchy_partition.get_edge_count(from_block, from_block)
        delta += log_binom(n_from_block_old * n_from_block_old + edge_count - 1, edge_count)
        if nodes_remaining:
            # new term
            edge_count -= kit[from_block] + kti[from_block] + selfloops
            delta -= log_binom(n_from_block_new * n_from_block_new + edge_count - 1, edge_count)

        # old e_{to_block, to_block} term
        if to_block_exists:
            edge_count = hierarchy_partition.get_edge_count(to_block, to_block)
            delta += log_binom(n_to_block_old * n_to_block_old + edge_count - 1, edge_count)
        else:
            edge_count = 0
        # new term
        edge_count += kit[to_block] + kti[to_block] + selfloops
        delta -= log_binom(n_to_block_new * n_to_block_new + edge_count - 1, edge_count)

        # delta for \sum_r \log n^l_r!
        delta += - math.lgamma(1 + n_from_block_old) \
            + math.lgamma(1 + n_to_block_new) \
            - math.lgamma(1 + n_to_block_old) \
            + math.lgamma(1 + n_from_block_new)

        return delta


# @formatter:on

# ----------------------------------------------
# As Objective Function Class
# ----------------------------------------------


class LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmWrapper(ObjectiveFunction):
    # class constants
    UNIFORM_HYPERPRIOR = 'uniform hyperprior'
    UNIFORM = 'uniform'
    NON_DEGREE_CORRECTED = 'non degree corrected'

    def __init__(self, is_directed, function_type=None):
        self._delta = DeltaModelLogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbm()
        self._complete = ModelLogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbm()

        if function_type is None:
            function_type = self.UNIFORM_HYPERPRIOR

        if is_directed:
            self.calculate_delta_actual_level_removed = self._delta.calculate_delta_actual_level_removed_directed
        else:
            self.calculate_delta_actual_level_removed = self._delta.calculate_delta_actual_level_removed_undirected

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

        super(LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmWrapper, self).__init__(
            is_directed,
            calculate_complete_undirected,
            calculate_complete_directed,
            calculate_delta_undirected,
            calculate_delta_directed
        )


# @formatter:off
class LogLikelihoodOfHierarchicalMicrocanonicalNonDegreeCorrected(
        LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmWrapper):
    title = "LogLikelihoodOfHierarchicalMicrocanonicalNonDegreeCorrected"
    # noinspection SpellCheckingInspection
    short_title = "HSPC"

    def __init__(self, is_directed):
        super(LogLikelihoodOfHierarchicalMicrocanonicalNonDegreeCorrected, self).__init__(
            is_directed, function_type=self.NON_DEGREE_CORRECTED)


class LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedUniform(
        LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmWrapper):
    title = "LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedUniform"
    # noinspection SpellCheckingInspection
    short_title = "HDCPU"

    def __init__(self, is_directed):
        super(LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedUniform, self).__init__(
            is_directed, function_type=self.UNIFORM)


class LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedUniformHyperprior(
        LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmWrapper):
    title = "LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedUniformHyperprior"
    # noinspection SpellCheckingInspection
    short_title = "HDCPUH"

    def __init__(self, is_directed):
        super(LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedUniformHyperprior, self).__init__(
            is_directed, function_type=self.UNIFORM)
