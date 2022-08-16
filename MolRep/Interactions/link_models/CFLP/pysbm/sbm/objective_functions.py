"""
Different Objective Functions evaluating the probability of a certain partition
"""

# real non truncating division (same behaviour as in python 3)
from __future__ import division

import math


class ObjectiveFunction(object):
    """Measures the fitting of the SBM to the given graph"""

    title = "Forgot to replace short_title"
    short_title = "Forgot to replace short short_title"

    def __init__(self, is_directed, function_calculate_undirected, function_calculate_directed,
                 function_calculate_delta_undirected, function_calculate_delta_directed):
        self.old_value = 0
        self._number_of_calculated_deltas = 0
        self.number_of_calculate_full = 0
        self._is_directed = is_directed

        self.function_calculate_undirected = function_calculate_undirected
        self.function_calculate_directed = function_calculate_directed
        self.function_calculate_delta_undirected = function_calculate_delta_undirected
        self.function_calculate_delta_directed = function_calculate_delta_directed

        if is_directed:
            self._calculate = function_calculate_directed
            self._calculate_delta = function_calculate_delta_directed
        else:
            self._calculate = function_calculate_undirected
            self._calculate_delta = function_calculate_delta_undirected

    @property
    def is_directed(self):
        return self._is_directed

    @is_directed.setter
    def is_directed(self, new_value):
        if new_value != self._is_directed:
            self._is_directed = new_value
            if new_value:
                self._calculate = self.function_calculate_directed
                self._calculate_delta = self.function_calculate_delta_directed
            else:
                self._calculate = self.function_calculate_undirected
                self._calculate_delta = self.function_calculate_delta_undirected

    def calculate_delta(self, partition, from_block, to_block, *args):
        """
        Calculate the difference of the objective function after one node move
        """
        assert partition.is_graph_directed() == self.is_directed  # comment in only for checking
        self._number_of_calculated_deltas += 1
        #       ensure from_block != to_block, else use this fast way
        if from_block == to_block:
            return 0
        return self._calculate_delta(partition, from_block, to_block, *args)

    def calculate(self, partition):
        """ Calculate the value of the objective function """
        assert partition.is_graph_directed() == self.is_directed  # comment in only for checking
        self.number_of_calculate_full += 1
        return self._calculate(partition)

    # def calculate_from_inference(self, inference):
    #     """ Wrapper around calculate which gets partition from inference """
    #     return self.calculate(inference.partition)

    @property
    def number_of_calculated_deltas(self):
        """ Return number of calculated deltas """
        return self._number_of_calculated_deltas


def _help_function_h(x):
    if x != 0:
        return x * math.log(x)
    # else:..
    return 0


# @formatter:off
def calculate_complete_traditional_unnormalized_log_likelyhood(partition):
    likelyhood = 0.0
    for r in range(partition.B):
        for s in range(partition.B):
            value = partition.get_edge_count(r, s)

            if value != 0:
                value /= (partition.get_number_of_nodes_in_block(r)
                          * partition.get_number_of_nodes_in_block(s))
                likelyhood += partition.get_edge_count(r, s) \
                    * math.log(value)
    return likelyhood


def calculate_delta_traditional_unnormalized_log_likelyhood_undirected(partition, from_block, to_block, *args):
    #     distinguish by given parameters
    if len(args) == 3:
        kit, selfloops, _ = args
        nodes_moved = 1
        nodes_remaining = partition.get_number_of_nodes_in_block(from_block) != nodes_moved
    elif len(args) == 5:
        kit, selfloops, _, nodes_moved, nodes_remaining = args
    else:
        raise ValueError()

    delta = 0.0
    h = _help_function_h

    def g(x):
        return 2 * h(x)

    for block in kit:
        if block != from_block and block != to_block:
            delta += g(partition.get_edge_count(from_block, block) - kit[block]) \
                     - g(partition.get_edge_count(from_block, block)) \
                     + g(partition.get_edge_count(to_block, block) + kit[block]) \
                     - g(partition.get_edge_count(to_block, block))

    delta += g(partition.get_edge_count(from_block, to_block)
               + kit[from_block] - kit[to_block]) \
        - g(partition.get_edge_count(from_block, to_block)) \
        + h(partition.get_edge_count(from_block, from_block)
            - 2 * (kit[from_block] + selfloops)) \
        - h(partition.get_edge_count(from_block, from_block)) \
        + h(partition.get_edge_count(to_block, to_block)
            + 2 * (kit[to_block] + selfloops)) \
        - h(partition.get_edge_count(to_block, to_block))

    if partition.get_number_of_nodes_in_block(to_block) != 0:
        delta += 2 * partition.get_edge_count(to_block, to_block) \
            * math.log(partition.get_number_of_nodes_in_block(to_block)) \
            + 2 * partition.get_edge_count(from_block, to_block) \
                * math.log(partition.get_number_of_nodes_in_block(from_block)
                           * partition.get_number_of_nodes_in_block(to_block))

    delta += 2 * partition.get_edge_count(from_block, from_block) \
        * math.log(partition.get_number_of_nodes_in_block(from_block)) \
        - 2 * (partition.get_edge_count(to_block, to_block)
               + 2 * (kit[to_block] + selfloops)) \
        * math.log(partition.get_number_of_nodes_in_block(to_block) + nodes_moved)

    if nodes_remaining:
        delta -= 2 * (partition.get_edge_count(from_block, from_block)
                      - 2 * (kit[from_block] + selfloops)) \
                 * math.log(partition.get_number_of_nodes_in_block(from_block) - nodes_moved) \
                 + 2 * (partition.get_edge_count(from_block, to_block)
                        + kit[from_block] - kit[to_block]) \
                   * math.log((partition.get_number_of_nodes_in_block(from_block) - nodes_moved)
                              * (partition.get_number_of_nodes_in_block(to_block) + nodes_moved))

    for block in partition.get_neighbors_of_block(from_block):
        if block != from_block and block != to_block:
            delta -= -2 * partition.get_edge_count(from_block, block) \
                     * math.log(partition.get_number_of_nodes_in_block(from_block)
                                * partition.get_number_of_nodes_in_block(block))
            if nodes_remaining:
                delta -= 2 * (partition.get_edge_count(from_block, block) - kit.get(block, 0)) \
                         * math.log((partition.get_number_of_nodes_in_block(from_block)
                                     - nodes_moved) * partition.get_number_of_nodes_in_block(block))

                # make a copy of kit because we want to discard all managed and else
                #  the given list is manipulated
    kit_copy = kit.copy()

    for block in partition.get_neighbors_of_block(to_block):
        # delete all managed to later care about the rest
        value = kit_copy.pop(block, 0)
        if block != from_block and block != to_block:
            delta -= -2 * partition.get_edge_count(to_block, block) \
                     * math.log(partition.get_number_of_nodes_in_block(to_block)
                                * partition.get_number_of_nodes_in_block(block)) \
                     + 2 * (partition.get_edge_count(to_block, block) + value) \
                     * math.log((partition.get_number_of_nodes_in_block(to_block) + nodes_moved)
                                * partition.get_number_of_nodes_in_block(block))

            # care about new neighbors of the to block
    for block in kit_copy:
        if block != from_block and block != to_block:
            delta -= 2 * (partition.get_edge_count(to_block, block) + kit[block]) \
                     * math.log((partition.get_number_of_nodes_in_block(to_block) + nodes_moved)
                                * partition.get_number_of_nodes_in_block(block))
    return delta


def calculate_delta_traditional_unnormalized_log_likelyhood_directed(partition, from_block, to_block, *args):
    #     distinguish by given parameters
    if len(args) == 5:
        kit, kti, selfloops, _, _ = args
        nodes_moved = 1
        nodes_remaining = partition.get_number_of_nodes_in_block(from_block) != nodes_moved
    elif len(args) == 7:
        kit, kti, selfloops, _, _, nodes_moved, nodes_remaining = args
    else:
        raise ValueError()

    delta = 0.0
    h = _help_function_h

    for block in kit:
        if block != from_block and block != to_block:
            delta += h(partition.get_edge_count(from_block, block) - kit[block]) \
                     - h(partition.get_edge_count(from_block, block)) \
                     + h(partition.get_edge_count(to_block, block) + kit[block]) \
                     - h(partition.get_edge_count(to_block, block))
            # other direction
    for block in kti:
        if block != from_block and block != to_block:
            delta += h(partition.get_edge_count(block, from_block) - kti[block]) \
                     - h(partition.get_edge_count(block, from_block)) \
                     + h(partition.get_edge_count(block, to_block) + kti[block]) \
                     - h(partition.get_edge_count(block, to_block))

    delta += h(partition.get_edge_count(from_block, to_block)
               + kti[from_block] - kit[to_block]) \
        - h(partition.get_edge_count(from_block, to_block)) \
        + h(partition.get_edge_count(to_block, from_block)
            + kit[from_block] - kti[to_block]) \
        - h(partition.get_edge_count(to_block, from_block)) \
        + h(partition.get_edge_count(from_block, from_block)
            - (kit[from_block] + kti[from_block] + selfloops)) \
        - h(partition.get_edge_count(from_block, from_block)) \
        + h(partition.get_edge_count(to_block, to_block)
            + (kit[to_block] + kti[to_block] + selfloops)) \
        - h(partition.get_edge_count(to_block, to_block))

    if partition.get_number_of_nodes_in_block(to_block) != 0:
        delta += 2 * partition.get_edge_count(to_block, to_block) \
            * math.log(partition.get_number_of_nodes_in_block(to_block)) \
            + (partition.get_edge_count(from_block, to_block)
               + partition.get_edge_count(to_block, from_block)) \
            * math.log(partition.get_number_of_nodes_in_block(from_block)
                       * partition.get_number_of_nodes_in_block(to_block))

    delta += 2 * partition.get_edge_count(from_block, from_block) \
        * math.log(partition.get_number_of_nodes_in_block(from_block)) \
        - 2 * (partition.get_edge_count(to_block, to_block)
               + (kit[to_block] + kti[to_block] + selfloops)) \
        * math.log(partition.get_number_of_nodes_in_block(to_block) + nodes_moved)

    if nodes_remaining:
        delta -= 2 * (partition.get_edge_count(from_block, from_block)
                      - (kit[from_block] + kti[from_block] + selfloops)) \
                 * math.log(partition.get_number_of_nodes_in_block(from_block) - nodes_moved) \
                 + (partition.get_edge_count(from_block, to_block)
                    + kti[from_block] - kit[to_block]
                    + partition.get_edge_count(to_block, from_block)
                    + kit[from_block] - kti[to_block]) \
                   * math.log((partition.get_number_of_nodes_in_block(from_block) - nodes_moved)
                              * (partition.get_number_of_nodes_in_block(to_block) + nodes_moved))

    for block in partition.get_neighbors_of_block(from_block):
        if block != from_block and block != to_block:
            delta += (partition.get_edge_count(from_block, block)
                      + partition.get_edge_count(block, from_block)) \
                     * math.log(partition.get_number_of_nodes_in_block(from_block)
                                * partition.get_number_of_nodes_in_block(block))
            if nodes_remaining:
                delta -= (partition.get_edge_count(from_block, block) - kti.get(block, 0)
                          + partition.get_edge_count(block, from_block) - kit.get(block, 0)) \
                         * math.log((partition.get_number_of_nodes_in_block(from_block)
                                     - nodes_moved) * partition.get_number_of_nodes_in_block(block))

                # make a copy of kit because we want to discard all managed and else
                #  the given list is manipulated
    kit = kit.copy()
    kti = kti.copy()

    for block in partition.get_neighbors_of_block(to_block):
        # delete all managed to later care about the rest
        kit_value = kit.pop(block, 0)
        kti_value = kti.pop(block, 0)
        if block != from_block and block != to_block:
            delta += (partition.get_edge_count(to_block, block)
                      + partition.get_edge_count(block, to_block)) \
                     * math.log(partition.get_number_of_nodes_in_block(to_block)
                                * partition.get_number_of_nodes_in_block(block)) \
                     - (partition.get_edge_count(to_block, block) + kti_value
                        + partition.get_edge_count(block, to_block) + kit_value) \
                     * math.log((partition.get_number_of_nodes_in_block(to_block) + nodes_moved)
                                * partition.get_number_of_nodes_in_block(block))

            # care about all blocks which are new neighbors
    for block in kit:
        if block != from_block and block != to_block:
            delta -= (partition.get_edge_count(block, to_block) + kit[block]) \
                     * math.log((partition.get_number_of_nodes_in_block(to_block) + nodes_moved)
                                * partition.get_number_of_nodes_in_block(block))

    for block in kti:
        if block != from_block and block != to_block:
            delta -= (partition.get_edge_count(to_block, block) + kti[block]) \
                     * math.log((partition.get_number_of_nodes_in_block(to_block) + nodes_moved)
                                * partition.get_number_of_nodes_in_block(block))

    return delta


class TraditionalUnnormalizedLogLikelyhood(ObjectiveFunction):
    """Non degree corrected unnormalized Log Likelyhood from Karrer and Newman"""
    title = "Traditional Unnormalized Log Likelyhood"
    short_title = "SKN"

    def __init__(self, is_directed):
        super(TraditionalUnnormalizedLogLikelyhood, self).__init__(
            is_directed,
            calculate_complete_traditional_unnormalized_log_likelyhood,
            calculate_complete_traditional_unnormalized_log_likelyhood,
            calculate_delta_traditional_unnormalized_log_likelyhood_undirected,
            calculate_delta_traditional_unnormalized_log_likelyhood_directed)
        # same formula as in undirected case (but different delta)


def calculate_complete_degree_corrected_unnormalized_log_likelyhood_undirected(partition):
    likelyhood = 0.0
    # formula with zero checks to ensure right result
    for r in range(partition.B):
        kappa_r = partition.get_degree_of_block(r)
        if kappa_r == 0:
            continue
        for s in range(partition.B):
            kappa_s = partition.get_degree_of_block(s)
            if kappa_s == 0:
                continue
            value = partition.get_edge_count(r, s) \
                / (kappa_r * kappa_s)
            if value != 0:
                likelyhood += partition.get_edge_count(r, s) \
                              * math.log(value)
    return likelyhood


def calculate_complete_degree_corrected_unnormalized_log_likelyhood_directed(partition):
    likelyhood = 0.0
    # formula with zero checks to ensure right result
    for r in range(partition.B):
        kappa_r = partition.get_out_degree_of_block(r)
        if kappa_r == 0:
            continue
        for s in range(partition.B):
            kappa_s = partition.get_in_degree_of_block(s)
            if kappa_s == 0:
                continue
            value = partition.get_edge_count(r, s) \
                / (kappa_r * kappa_s)
            if value != 0:
                likelyhood += partition.get_edge_count(r, s) \
                              * math.log(value)
    return likelyhood


def calculate_delta_degree_corrected_unnormalized_log_likelyhood_undirected(partition, from_block, to_block, *args):
    if len(args) == 3:
        kit, selfloops, degree = args
    else:
        raise ValueError()
    delta = 0.0
    h = _help_function_h

    def g(x):
        return 2 * h(x)

    kappa_r = partition.get_degree_of_block(from_block)
    kappa_s = partition.get_degree_of_block(to_block)

    for block in kit:
        if block != from_block and block != to_block:
            delta += g(partition.get_edge_count(from_block, block) - kit[block]) \
                     - g(partition.get_edge_count(from_block, block)) \
                     + g(partition.get_edge_count(to_block, block) + kit[block]) \
                     - g(partition.get_edge_count(to_block, block))

    delta += g(partition.get_edge_count(from_block, to_block)
               + kit[from_block] - kit[to_block]) \
        - g(partition.get_edge_count(from_block, to_block)) \
        + h(partition.get_edge_count(from_block, from_block)
            - 2 * (kit[from_block] + selfloops)) \
        - h(partition.get_edge_count(from_block, from_block)) \
        + h(partition.get_edge_count(to_block, to_block)
            + 2 * (kit[to_block] + selfloops)) \
        - h(partition.get_edge_count(to_block, to_block)) \
        - g(kappa_r - degree) + g(kappa_r) \
        - g(kappa_s + degree) + g(kappa_s)

    return delta


def calculate_delta_degree_corrected_unnormalized_log_likelyhood_directed(partition, from_block, to_block, *args):
    if len(args) == 5:
        kit, kti, selfloops, in_degree, out_degree = args
    else:
        raise ValueError()
    delta = 0.0
    h = _help_function_h

    for block in kit:
        if block != from_block and block != to_block:
            delta += h(partition.get_edge_count(from_block, block) - kit[block]) \
                     - h(partition.get_edge_count(from_block, block)) \
                     + h(partition.get_edge_count(to_block, block) + kit[block]) \
                     - h(partition.get_edge_count(to_block, block))

    for block in kti:
        if block != from_block and block != to_block:
            delta += h(partition.get_edge_count(block, from_block) - kti[block]) \
                     - h(partition.get_edge_count(block, from_block)) \
                     + h(partition.get_edge_count(block, to_block) + kti[block]) \
                     - h(partition.get_edge_count(block, to_block))

    delta += h(partition.get_edge_count(from_block, to_block)
               + kti[from_block] - kit[to_block]) \
        - h(partition.get_edge_count(from_block, to_block)) \
        + h(partition.get_edge_count(to_block, from_block)
            + kit[from_block] - kti[to_block]) \
        - h(partition.get_edge_count(to_block, from_block)) \
        + h(partition.get_edge_count(from_block, from_block)
            - (kit[from_block] + kti[from_block] + selfloops)) \
        - h(partition.get_edge_count(from_block, from_block)) \
        + h(partition.get_edge_count(to_block, to_block)
            + (kit[to_block] + kti[to_block] + selfloops)) \
        - h(partition.get_edge_count(to_block, to_block))
    # now the part contributing to the changes in the block in and out degree
    delta += - h(partition.get_out_degree_of_block(from_block) - out_degree) \
        + h(partition.get_out_degree_of_block(from_block)) \
        - h(partition.get_in_degree_of_block(from_block) - in_degree) \
        + h(partition.get_in_degree_of_block(from_block)) \
        - h(partition.get_out_degree_of_block(to_block) + out_degree) \
        + h(partition.get_out_degree_of_block(to_block)) \
        - h(partition.get_in_degree_of_block(to_block) + in_degree) \
        + h(partition.get_in_degree_of_block(to_block))

    return delta


class DegreeCorrectedUnnormalizedLogLikelyhood(ObjectiveFunction):
    """Degree corrected unnormalized Log Likelyhood from Karrer and Newman"""
    title = "Degree Corrected Unnormalized Log Likelyhood"
    short_title = "DCKN"

    def __init__(self, is_directed):
        super(DegreeCorrectedUnnormalizedLogLikelyhood, self).__init__(
            is_directed,
            calculate_complete_degree_corrected_unnormalized_log_likelyhood_undirected,
            calculate_complete_degree_corrected_unnormalized_log_likelyhood_directed,
            calculate_delta_degree_corrected_unnormalized_log_likelyhood_undirected,
            calculate_delta_degree_corrected_unnormalized_log_likelyhood_directed)


def calculate_delta_traditional_microcanonical_entropy_undirected(partition, from_block, to_block, *args):
    # delta function only implemented for sparse graphs
    return .5 * calculate_delta_traditional_unnormalized_log_likelyhood_undirected(partition,
                                                                                   from_block,
                                                                                   to_block,
                                                                                   *args)


class TraditionalMicrocanonicalEntropy(ObjectiveFunction):
    """
    Non degree corrected microcanonical Entropy (with changed sign) from Peixoto

    Sparse and faster formula from Supplemental Material: Parsimonious Module Inference in Large Networks
    """
    title = "Traditional Microcanonical Entropy Sparse Approximation"
    short_title = "SPS"

    def __init__(self, is_directed, fixed_value=0):
        super(TraditionalMicrocanonicalEntropy, self).__init__(
            is_directed,
            self.calculate_complete_traditional_microcanonical_entropy_undirected,
            self.calculate_complete_traditional_microcanonical_entropy_directed,
            calculate_delta_traditional_microcanonical_entropy_undirected,
            calculate_delta_traditional_unnormalized_log_likelyhood_directed,
        )
        # fixed value = number of edges
        self.fixed_value = float(fixed_value)

    def calculate_complete_traditional_microcanonical_entropy_undirected(self, partition):
        entropy = self.fixed_value
        intermediate_sum = 0.0
        for r in range(partition.B):
            entropy += partition.get_degree_of_block(r) * \
                       math.log(partition.get_number_of_nodes_in_block(r))
            for s in range(partition.B):
                value = partition.get_edge_count(r, s)
                if value != 0:
                    intermediate_sum += value * math.log(value)
        entropy -= .5 * intermediate_sum
        entropy = - entropy
        # Different SIGN!
        return entropy

    def calculate_complete_traditional_microcanonical_entropy_directed(self, partition):
        entropy = self.fixed_value
        intermediate_sum = 0.0
        for r in range(partition.B):
            entropy += ( partition.get_out_degree_of_block(r) + partition.get_in_degree_of_block(r)) * \
                       math.log(partition.get_number_of_nodes_in_block(r))
            for s in range(partition.B):
                value = partition.get_edge_count(r, s)
                if value != 0:
                    intermediate_sum += value * math.log(value)
        # without .5*
        entropy -= intermediate_sum
        entropy = - entropy
        # Different SIGN!
        return entropy


class TraditionalMicrocanonicalEntropyDense(ObjectiveFunction):
    """
    Non degree corrected microcanonical Entropy (with changed sign) from Peixoto
    Complete formula without approximation, which delta calculation needs more time
    """
    title = "Traditional Microcanonical Entropy Dense/Complete"
    short_title = "SPD"

    def __init__(self, is_directed, fixed_value=0):
        super(TraditionalMicrocanonicalEntropyDense, self).__init__(
            is_directed,
            self.calculate_complete_traditional_microcanonical_entropy_dense_undirected,
            self.calculate_complete_traditional_microcanonical_entropy_dense_directed,
            self.calculate_delta_traditional_microcanonical_entropy_dense_undirected,
            self.calculate_delta_traditional_microcanonical_entropy_dense_directed,
        )

        def binary_entropy(x):
            try:
                if 0 < x < 1:
                    return -x * math.log(x, 2) - (1 - x) * math.log(1 - x, 2)
                return 0
            except ValueError:
                print(x)

        self._binary_entropy = binary_entropy
        # fixed value = number of edges
        self.fixed_value = float(fixed_value)

    def calculate_complete_traditional_microcanonical_entropy_dense_undirected(self, partition):
        entropy = 0.0
        for r in range(partition.B):
            for s in range(partition.B):
                value = partition.get_edge_count(r, s) \
                        / (partition.get_number_of_nodes_in_block(r)
                           * partition.get_number_of_nodes_in_block(s))
                if value != 0:
                    entropy += partition.get_edge_count(r, s) * self._binary_entropy(value)
        entropy = -.5 * entropy
        # Different SIGN!
        return entropy

    def calculate_complete_traditional_microcanonical_entropy_dense_directed(self, partition):
        entropy = 0.0
        for r in range(partition.B):
            for s in range(partition.B):
                value = partition.get_edge_count(r, s) \
                        / (partition.get_number_of_nodes_in_block(r)
                           * partition.get_number_of_nodes_in_block(s))
                if value != 0:
                    entropy += partition.get_edge_count(r, s) * self._binary_entropy(value)
        # according to SM of Parsimonious Module Inference in Large Networks only difference no .5 term
        entropy = - entropy
        # Different SIGN!
        return entropy

    def calculate_delta_traditional_microcanonical_entropy_dense_undirected(self, partition, from_block, to_block, *args):
        if len(args) == 3:
            kit, selfloops, _ = args
            nodes_moved = 1
            nodes_remaining = partition.get_number_of_nodes_in_block(from_block) != nodes_moved
        elif len(args) == 5:
            kit, selfloops, _, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        delta = 0

        old_from_block_node_count = partition.get_number_of_nodes_in_block(from_block)
        old_to_block_node_count = partition.get_number_of_nodes_in_block(to_block)
        new_from_block_node_count = old_from_block_node_count - nodes_moved
        new_to_block_node_count = old_to_block_node_count + nodes_moved

        for block in range(partition.B):
            if from_block != block != to_block:
                # old value from_block -> block
                edge_count = partition.get_edge_count(from_block, block)
                node_count = partition.get_number_of_nodes_in_block(block)
                if edge_count != 0:
                    delta += edge_count * self._binary_entropy(edge_count / (old_from_block_node_count * node_count))

                # new value from_block -> block
                if nodes_remaining:
                    if block in kit:
                        edge_count -= kit[block]
                    if edge_count != 0:
                        delta -= edge_count * self._binary_entropy(
                            edge_count / (new_from_block_node_count * node_count))

                # old value to_block -> block
                edge_count = partition.get_edge_count(to_block, block)
                node_count = partition.get_number_of_nodes_in_block(block)
                if edge_count != 0:
                    delta += edge_count * self._binary_entropy(edge_count / (old_to_block_node_count * node_count))

                # new value to_block -> block
                if block in kit:
                    edge_count += kit[block]
                if edge_count != 0:
                    delta -= edge_count * self._binary_entropy(
                        edge_count / (new_to_block_node_count * node_count))

        # now handle terms for from_block, to_block
        # from_block -> from_block
        edge_count = partition.get_edge_count(from_block, from_block)
        if edge_count != 0:
            delta += .5 * edge_count * self._binary_entropy(
                edge_count / (old_from_block_node_count * old_from_block_node_count))

        if nodes_remaining:
            edge_count -= 2 * (kit[from_block] + selfloops)
            if edge_count != 0:
                delta -= .5 * edge_count * self._binary_entropy(
                    edge_count / (new_from_block_node_count * new_from_block_node_count))

        # to_block <-> to_block
        edge_count = partition.get_edge_count(to_block, to_block)
        if edge_count != 0:
            delta += .5 * edge_count * self._binary_entropy(
                edge_count / (old_to_block_node_count * old_to_block_node_count))

        edge_count += 2*(kit[to_block] + selfloops)
        if edge_count != 0:
            delta -= .5 * edge_count * self._binary_entropy(
                edge_count / (new_to_block_node_count * new_to_block_node_count))

        # from_block -> to_block
        edge_count = partition.get_edge_count(from_block, to_block)
        if edge_count != 0 :
            delta += edge_count * self._binary_entropy(
                edge_count / (old_from_block_node_count * old_to_block_node_count))

        if nodes_remaining:
            edge_count += kit[from_block] - kit[to_block]
            if edge_count != 0 :
                delta -= edge_count * self._binary_entropy(
                    edge_count / (new_from_block_node_count * new_to_block_node_count))

        return delta

    def calculate_delta_traditional_microcanonical_entropy_dense_directed(self, partition, from_block, to_block, *args):
        if len(args) == 5:
            kit, kti, selfloops, _, _ = args
            nodes_moved = 1
            nodes_remaining = partition.get_number_of_nodes_in_block(from_block) != nodes_moved
        elif len(args) == 7:
            kit, kti, selfloops, _, _, nodes_moved, nodes_remaining = args
        else:
            raise ValueError()

        delta = 0

        old_from_block_node_count = partition.get_number_of_nodes_in_block(from_block)
        old_to_block_node_count = partition.get_number_of_nodes_in_block(to_block)
        new_from_block_node_count = old_from_block_node_count - nodes_moved
        new_to_block_node_count = old_to_block_node_count + nodes_moved

        for block in range(partition.B):
            if from_block != block != to_block:
                # old value from_block -> block
                edge_count = partition.get_edge_count(from_block, block)
                node_count = partition.get_number_of_nodes_in_block(block)
                if edge_count != 0:
                    delta += edge_count * self._binary_entropy(edge_count / (old_from_block_node_count * node_count))

                # new value from_block -> block
                if nodes_remaining:
                    if block in kit:
                        edge_count -= kit[block]
                    if edge_count != 0:
                        delta -= edge_count * self._binary_entropy(
                            edge_count / (new_from_block_node_count * node_count))

                # old value to_block -> block
                edge_count = partition.get_edge_count(to_block, block)
                node_count = partition.get_number_of_nodes_in_block(block)
                if edge_count != 0:
                    delta += edge_count * self._binary_entropy(edge_count / (old_to_block_node_count * node_count))

                # new value to_block -> block
                if block in kit:
                    edge_count += kit[block]
                if edge_count != 0:
                    delta -= edge_count * self._binary_entropy(
                        edge_count / (new_to_block_node_count * node_count))

                # other directions:
                # old value block -> from_block
                edge_count = partition.get_edge_count(block, from_block)
                node_count = partition.get_number_of_nodes_in_block(block)
                if edge_count != 0:
                    delta += edge_count * self._binary_entropy(edge_count / (old_from_block_node_count * node_count))

                # new value block -> from_block
                if nodes_remaining:
                    if block in kti:
                        edge_count -= kti[block]
                    if edge_count != 0:
                        delta -= edge_count * self._binary_entropy(
                            edge_count / (new_from_block_node_count * node_count))

                # old value block -> to_block
                edge_count = partition.get_edge_count(block, to_block)
                node_count = partition.get_number_of_nodes_in_block(block)
                if edge_count != 0:
                    delta += edge_count * self._binary_entropy(edge_count / (old_to_block_node_count * node_count))

                # new value block -> to_block
                if block in kti:
                    edge_count += kti[block]
                if edge_count != 0:
                    delta -= edge_count * self._binary_entropy(
                        edge_count / (new_to_block_node_count * node_count))

        # now handle terms for from_block, to_block
        # from_block -> from_block
        edge_count = partition.get_edge_count(from_block, from_block)
        if edge_count != 0:
            delta += edge_count * self._binary_entropy(
                edge_count / (old_from_block_node_count * old_from_block_node_count))

        if nodes_remaining:
            edge_count -= kit[from_block] + kti[from_block] + selfloops
            if edge_count != 0:
                delta -= edge_count * self._binary_entropy(
                    edge_count / (new_from_block_node_count * new_from_block_node_count))

        # to_block <-> to_block
        edge_count = partition.get_edge_count(to_block, to_block)
        if edge_count != 0:
            delta += edge_count * self._binary_entropy(
                edge_count / (old_to_block_node_count * old_to_block_node_count))

        edge_count += kit[to_block] + kti[to_block] + selfloops
        if edge_count != 0:
            delta -= edge_count * self._binary_entropy(
                edge_count / (new_to_block_node_count * new_to_block_node_count))

        # from_block -> to_block
        edge_count = partition.get_edge_count(from_block, to_block)
        if edge_count != 0 :
            delta += edge_count * self._binary_entropy(
                edge_count / (old_from_block_node_count * old_to_block_node_count))

        if nodes_remaining:
            edge_count += kti[from_block] - kit[to_block]
            if edge_count != 0 :
                delta -= edge_count * self._binary_entropy(
                    edge_count / (new_from_block_node_count * new_to_block_node_count))

        # to_block -> from_block
        edge_count = partition.get_edge_count(to_block, from_block)
        if edge_count != 0 :
            delta += edge_count * self._binary_entropy(
                edge_count / (old_from_block_node_count * old_to_block_node_count))

        if nodes_remaining:
            edge_count += kit[from_block] - kti[to_block]
            if edge_count != 0 :
                delta -= edge_count * self._binary_entropy(
                    edge_count / (new_from_block_node_count * new_to_block_node_count))

        return delta


def calculate_delta_degree_corrected_microcanonical_entropy_undirected(partition, from_block, to_block, *args):
    #    Different SIGN than original!
    return .5 * calculate_delta_degree_corrected_unnormalized_log_likelyhood_undirected(
        partition, from_block, to_block, *args)


class DegreeCorrectedMicrocanonicalEntropy(ObjectiveFunction):
    """
    Degree corrected microcanonical entropy from Peixoto.

    Because the similarities to the one from Karrer and Newman, it uses the class
    DegreeCorrectedUnnormalizedLogLikelyhood.
    Formulas e.g. (5) and (6) from SM: Parsimonious Module Inference in Large Networks
    """
    title = "Degree Corrected Microcanonical Entropy"
    short_title = "DCP"

    def _calculate_directed(self, partition):
        raise NotImplementedError()

    def _calculate_delta_directed(self, partition, from_block, to_block, *args):
        raise NotImplementedError()

    def __init__(self, is_directed, fixed_value=0):
        def stub(*args):
            raise NotImplementedError()
        super(DegreeCorrectedMicrocanonicalEntropy, self).__init__(
            is_directed,
            self.calculate_complete_degree_corrected_microcanonical_entropy_undirected,
            self.calculate_complete_degree_corrected_microcanonical_entropy_directed,
            calculate_delta_degree_corrected_microcanonical_entropy_undirected,
            calculate_delta_degree_corrected_unnormalized_log_likelyhood_directed,
        )
        # If the absolute value is needed fixed_value := -E-\sum N_k ln(k!)
        self.fixed_value = float(fixed_value)

    def calculate_complete_degree_corrected_microcanonical_entropy_undirected(self, partition):
        #    Different SIGN! - original without the minus
        return -(self.fixed_value
                 - .5 * calculate_complete_degree_corrected_unnormalized_log_likelyhood_undirected(partition))

    def calculate_complete_degree_corrected_microcanonical_entropy_directed(self, partition):
        #    Different SIGN! - original without the minus
        return -(self.fixed_value
                 - calculate_complete_degree_corrected_unnormalized_log_likelyhood_directed(partition))
