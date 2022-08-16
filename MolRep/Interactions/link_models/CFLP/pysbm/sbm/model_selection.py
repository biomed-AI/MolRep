# real non truncating division (same behaviour as in python 3)
from __future__ import division

import math

from .peixotos_hierarchical_sbm_tools import log_binom

# from pysbm import sbm
from MolRep.Interactions.link_models.CFLP.pysbm import sbm
import networkx as nx


def minimum_description_length_peixoto_first(partition_representation,
                                             number_of_nodes,
                                             number_of_edges,
                                             is_directed,
                                             is_degree_corrected):
    """
    from Peixoto 2013 Parsimonious Module Inference in Large Networks
    :param partition_representation: representation of a partition
    :param number_of_edges: number of edges in graph
    :param number_of_nodes: number of nodes in graph
    :param is_directed: is the described graph directed?
    :param is_degree_corrected: is the model a degree corrected variant of SBM
    :type partition_representation dict
    :type number_of_nodes int
    :type number_of_edges int
    :type is_directed bool
    :type is_degree_corrected bool
    :return: correction of objective function value
    """

    def h(x):
        """binary entropy function"""
        return (1 + x) * math.log(1 + x) - x * math.log(x)

    number_of_blocks = max(partition_representation.values()) + 1

    if is_directed:
        value = number_of_edges * h(number_of_blocks * number_of_blocks / number_of_edges) \
                + number_of_nodes * math.log(number_of_blocks)
    else:
        value = number_of_edges * h(number_of_blocks * (number_of_blocks + 1) / (2 * number_of_edges)) \
                + number_of_nodes * math.log(number_of_blocks)

    if is_degree_corrected:
        # both needs iterator over degrees, but don't depend on the partition!
        if is_directed:
            # correct formula: for node i: value += log deg( node i)
            value += 0
        else:
            # correct formula: for node i: value += log in deg( node i) + log out deg (node i)
            value += 0
    return value


def minimum_description_length_traditional(partition_representation,
                                           number_of_edges,
                                           is_directed,
                                           is_degree_corrected):
    """
    own formula according to general description of MDL
    :param partition_representation: representation of a partition
    :param number_of_edges: number of edges in graph
    :param is_directed: is the described graph directed?
    :param is_degree_corrected: is the model a degree corrected variant of SBM
    :type partition_representation dict
    :type number_of_edges int
    :type is_directed bool
    :type is_degree_corrected bool
    :return: correction of objective function value
    """
    number_of_blocks = max(partition_representation.values()) + 1
    value = number_of_edges * math.log(number_of_blocks)
    if is_directed:
        value += number_of_blocks * number_of_blocks * math.log(number_of_edges)
    else:
        value += number_of_blocks * (number_of_blocks + 1) / 2 * math.log(number_of_edges)
    if is_degree_corrected:
        # both needs iterator over degrees, but don't depend on the partition!
        if is_directed:
            # correct formula: for node i: value += log deg( node i)
            value += 0
        else:
            # correct formula: for node i: value += log in deg( node i) + log out deg (node i)
            value += 0

    return value


def minimum_description_length_peixoto(partition,
                                       is_degree_corrected):
    """
    From Peixoto 2014 ﻿Hierarchical block structures and high-resolution model selection in large networks eq. 9 and 10
    :param partition: complete partition with all information about graph and node partition
    :param is_degree_corrected: is the model a degree corrected variant of SBM
    :type partition sbm.Partition
    :return: correction of objective function value
    """
    number_of_blocks = partition.B
    value = log_binom(number_of_blocks + partition.get_number_of_nodes() - 1, partition.get_number_of_nodes())
    # left out partition independent term log |V|!
    for block in range(number_of_blocks):
        value -= math.lgamma(partition.get_number_of_nodes_in_block(block) + 1)

    if partition.is_graph_directed():
        value += log_binom(number_of_blocks * number_of_blocks + partition.get_number_of_edges() - 1,
                           partition.get_number_of_edges())
    else:
        value += log_binom(number_of_blocks * (number_of_blocks + 1) / 2 + partition.get_number_of_edges() - 1,
                           partition.get_number_of_edges())

    if is_degree_corrected:
        def calculate_weighted_entropys(block_distributions):
            """
            Calculate the sum of the entropy's of the given distribution for each block
            and weight them accordingly to the number of nodes in the block.
            """
            total = 0
            for block_counter, distribution in enumerate(block_distributions):
                total += partition.get_number_of_nodes_in_block(block_counter) * calculate_entropy(
                    distribution)
            return total

        if partition.is_graph_directed():
            in_degree_distributions, out_degree_distributions = partition.get_degree_distribution_of_blocks()
            value -= calculate_weighted_entropys(in_degree_distributions)
            value -= calculate_weighted_entropys(out_degree_distributions)
        else:
            degree_distributions = partition.get_degree_distribution_of_blocks()
            value -= calculate_weighted_entropys(degree_distributions)
    return value


def akaike_information_criterion(partition_representation,
                                 is_directed,
                                 is_degree_corrected):
    """
    own formula according to general description of AIC
    :param partition_representation: representation of a partition
    :param is_directed: is the described graph directed?
    :param is_degree_corrected: is the model a degree corrected variant of SBM
    :type partition_representation dict
    :type is_directed bool
    :type is_degree_corrected bool
    :return: correction of objective function value
    """
    number_of_blocks = max(partition_representation.values()) + 1
    if is_directed:
        value = number_of_blocks * number_of_blocks
    else:
        value = number_of_blocks * (number_of_blocks + 1) / 2
    value += number_of_blocks
    if is_degree_corrected:
        value += number_of_blocks
    return value


def bayesian_information_criterion(partition_representation,
                                   number_of_nodes,
                                   is_directed,
                                   is_degree_corrected,
                                   sparse=True):
    """
    from Yan 2016 ﻿Bayesian model selection of stochastic block models
    :param partition_representation: representation of a partition
    :param number_of_nodes: number of nodes in graph
    :param is_directed: is the described graph directed?
    :param is_degree_corrected: is the model a degree corrected variant of SBM
    :param sparse: is the described graph sparse?
    :type partition_representation dict
    :type number_of_nodes int
    :type is_directed bool
    :type is_degree_corrected bool
    :type sparse bool
    :return: correction of objective function value
    """
    number_of_blocks = max(partition_representation.values()) + 1
    if sparse:
        edge_term = number_of_nodes * number_of_nodes * number_of_nodes
    else:
        edge_term = number_of_nodes * number_of_nodes
    if is_directed:
        value = number_of_blocks * number_of_blocks * math.log(edge_term)
    else:
        value = number_of_blocks * (number_of_blocks + 1) / 2 * math.log(edge_term)
    if is_degree_corrected:
        if is_directed:
            # correct but partition independent +=2*math.log(2*number_of_nodes)
            value += 0
        else:
            # correct but partition independent +=2*math.log(number_of_nodes)
            value += 0
    return .5 * value


def integrated_complete_likelihood(partition_representation,
                                   number_of_nodes,
                                   is_directed,
                                   is_degree_corrected):
    """
    model selection according to ICL
    :param partition_representation: representation of a partition
    :param number_of_nodes: number of nodes in graph
    :param is_directed: is the described graph directed?
    :param is_degree_corrected: is the model a degree corrected variant of SBM
    :type partition_representation dict
    :type number_of_nodes int
    :type is_directed bool
    :type is_degree_corrected bool
    :return: correction of objective function value
    """
    number_of_blocks = max(partition_representation.values()) + 1
    if is_degree_corrected:
        raise NotImplementedError()
    value = (number_of_blocks - 1) / 2 * math.log(number_of_nodes)

    if is_directed:
        value += number_of_blocks * number_of_blocks / 2 * math.log(number_of_nodes * number_of_nodes)
    else:
        value += number_of_blocks * (number_of_blocks + 1) / 4 * math.log(
            number_of_nodes * (number_of_nodes - 1) / 2)

    return value


class ModelSelectionByFunction(object):
    """Infer number of blocks via model selection (partition independent values are ignored!)"""

    def __init__(self, objective_function, number_of_edges, number_of_nodes, double_objective_function=False):
        self._objective_function = objective_function
        self._number_of_edges = number_of_edges
        self._number_of_nodes = number_of_nodes
        self._double_objective_function = double_objective_function

    def minimum_description_length_peixoto_first(self, partition, is_degree_corrected):
        return minimum_description_length_peixoto_first(partition.get_representation(), self._number_of_nodes,
                                                        self._number_of_edges, partition.is_graph_directed(),
                                                        is_degree_corrected)

    def minimum_description_length_traditional(self, partition, is_degree_corrected):
        return minimum_description_length_traditional(partition.get_representation(), self._number_of_edges,
                                                      partition.is_graph_directed(), is_degree_corrected)

    @staticmethod
    def minimum_description_length_peixoto(partition, is_degree_corrected):
        return minimum_description_length_peixoto(partition, is_degree_corrected)

    @staticmethod
    def akaike_information_criterion(partition, is_degree_corrected):
        return akaike_information_criterion(partition.get_representation(), partition.is_graph_directed(),
                                            is_degree_corrected)

    def bayesian_information_criterion(self, partition, is_degree_corrected, sparse=True):
        return bayesian_information_criterion(partition.get_representation(), self._number_of_nodes,
                                              partition.is_graph_directed(), is_degree_corrected, sparse)

    def integrated_complete_likelihood(self, partition, is_degree_corrected):
        return integrated_complete_likelihood(partition.get_representation(), self._number_of_nodes,
                                              partition.is_graph_directed(), is_degree_corrected)


def calculate_entropy(distribution):
    total = 0
    for degree in distribution:
        total += distribution[degree] * math.log(distribution[degree])
    return total


class ExponentialSearch:
    def __init__(self, value_container):
        self.values = value_container
        self._changing_point_to_linear_search = 10
        if len(self.values) == 0:
            raise ValueError

    def search_minimum(self, lower_boundary=0, upper_boundary=None):
        moving_lower_bound = 1
        middle_point = 1
        moving_upper_bound = 2
        if upper_boundary is None:
            upper_bound = len(self.values) - lower_boundary
        else:
            upper_bound = upper_boundary - lower_boundary
        # if only single element this is the minimum
        if upper_bound == 1:
            return 0
        elif upper_bound == 2:
            if self.values[0] < self.values[1]:
                return 0
            else:
                return 1

        # move further until the decrease stops
        while moving_upper_bound <= upper_bound and self.values[moving_upper_bound - 1 + lower_boundary] < self.values[
            moving_lower_bound - 1 + lower_boundary]:
            moving_lower_bound = middle_point
            middle_point = moving_upper_bound
            moving_upper_bound *= 2

        if moving_upper_bound > upper_bound:
            moving_upper_bound = upper_bound

        # at this moment we knew that the minimum lies between moving_lower_bound and moving_upper_bound
        if moving_upper_bound - moving_lower_bound > self._changing_point_to_linear_search:
            return self.search_minimum(moving_lower_bound, moving_upper_bound)
        else:
            return self.search_minimum_linear(moving_lower_bound - 1, moving_upper_bound - 1)

    def search_minimum_linear(self, lower_boundary=0, upper_boundary=None):
        """
        Simple linear search for minimum. If supplied only in the boundaries given.
        :param lower_boundary: Lower offset for searching.
        :param upper_boundary: Upper offset for searching.
            If not supplied calculated with len(self.values)-1-lower_boundary
        :type lower_boundary: int
        :type upper_boundary: int
        :return:
        :rtype int
        """
        if upper_boundary is None:
            top_bound = len(self.values) - 1 - lower_boundary
        else:
            top_bound = upper_boundary

        minimal_index = lower_boundary
        minimal_value = self.values[lower_boundary]

        for i in range(lower_boundary + 1, top_bound + 1):
            if self.values[i] < minimal_value:
                minimal_value = self.values[i]
                minimal_index = i

        return minimal_index


class AbstractModelSelection:

    def __init__(self, model_selection_function):
        self._model_selection_function = model_selection_function

    def create_parameters_for_model_selection_function(self, graph, partition_representation):
        raise NotImplementedError()

    def select_number_of_groups(self, graph, objective_function_values, partition_representations,
                                return_complete_information=False):
        new_values = {}
        for number_of_blocks in objective_function_values:
            new_values[number_of_blocks] = objective_function_values[number_of_blocks] - \
                                           self._model_selection_function(
                                               **self.create_parameters_for_model_selection_function(
                                                   graph, partition_representations[number_of_blocks]
                                               )
                                           )
        selected_number_of_groups = max(new_values, key=lambda x: new_values[x])
        if return_complete_information:
            return selected_number_of_groups, new_values[selected_number_of_groups], new_values
        return selected_number_of_groups, new_values[selected_number_of_groups]


class ModelSelectionByObjectiveFunctionValueOnly(AbstractModelSelection):
    """Model Selection is included in the objective function"""

    title = "Model Selection by Objective Function Value Only"

    def __init__(self):
        def dummy_model_selection_function():
            """Replacement for additional model selection function"""
            return 0

        super(ModelSelectionByObjectiveFunctionValueOnly, self).__init__(dummy_model_selection_function)

    def create_parameters_for_model_selection_function(self, graph, partition_representation):
        return {}


class ModelSelectionWithExtraFunction(AbstractModelSelection):

    def __init__(self,
                 model_selection_function,
                 is_degree_corrected,
                 title,
                 function_needs_partition=False,
                 function_needs_partition_representation=True,
                 function_needs_number_of_nodes=False,
                 function_needs_number_of_edges=False,
                 function_needs_is_directed=True,
                 additional_fixed_key_arguments=None
                 ):
        super(ModelSelectionWithExtraFunction, self).__init__(model_selection_function)
        self.title = title
        self.is_degree_corrected = is_degree_corrected
        self.function_needs_partition = function_needs_partition
        self.function_needs_partition_representation = function_needs_partition_representation
        self.function_needs_number_of_nodes = function_needs_number_of_nodes
        self.function_needs_number_of_edges = function_needs_number_of_edges
        self.function_needs_is_directed = function_needs_is_directed
        self.additional_fixed_key_arguments = additional_fixed_key_arguments

    def create_parameters_for_model_selection_function(self, graph, partition_representation):
        """
        Creates from the given information the parameters for the model selection function
        :param graph: graph which nodes are clustered
        :param partition_representation: dictionary containing the clustering of the nodes, with the nodes as key and
        the communities labeled starting from 0
        :type graph: nx.Graph
        :type partition_representation dict
        :return:
        """
        parameters = {"is_degree_corrected": self.is_degree_corrected}

        if self.function_needs_partition:
            parameters["partition"] = sbm.NxPartition(graph, representation=partition_representation)

        if self.function_needs_partition_representation:
            parameters["partition_representation"] = partition_representation

        if self.function_needs_number_of_nodes:
            parameters["number_of_nodes"] = len(graph.nodes())

        if self.function_needs_number_of_edges:
            parameters["number_of_edges"] = len(graph.edges())

        if self.function_needs_is_directed:
            parameters["is_directed"] = graph.is_directed()

        if self.additional_fixed_key_arguments is not None:
            parameters.update(**self.additional_fixed_key_arguments)

        return parameters


def get_possible_model_selection(objective_function_class):
    """
    Returns a list of corresponding model selection classes
    :param objective_function_class:
    :type objective_function_class class sbm.ObjectiveFunction
    :return: list of model selections
    """
    model_selections = []

    if objective_function_class in [
        sbm.IntegratedCompleteLikelihoodExactJeffrey,
        sbm.IntegratedCompleteLikelihoodExactUniform,
        sbm.NewmanReinertDegreeCorrected,
        sbm.NewmanReinertNonDegreeCorrected,
        sbm.LogLikelihoodOfFlatMicrocanonicalNonDegreeCorrected,
        sbm.LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedUniform,
        sbm.LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedUniformHyperprior,
    ]:
        model_selections.append(ModelSelectionByObjectiveFunctionValueOnly())
    elif objective_function_class in [
        sbm.TraditionalUnnormalizedLogLikelyhood,
        sbm.DegreeCorrectedUnnormalizedLogLikelyhood,
        sbm.TraditionalMicrocanonicalEntropy,
        sbm.TraditionalMicrocanonicalEntropyDense,
        sbm.DegreeCorrectedMicrocanonicalEntropy,
    ]:
        if objective_function_class in [
            sbm.DegreeCorrectedUnnormalizedLogLikelyhood,
            sbm.DegreeCorrectedMicrocanonicalEntropy,
        ]:
            is_degree_corrected = True
        else:
            is_degree_corrected = False

            model_selections.append(ModelSelectionWithExtraFunction(integrated_complete_likelihood,
                                                                    is_degree_corrected,
                                                                    "ICL",
                                                                    function_needs_number_of_nodes=True,
                                                                    ))

        model_selections.append(ModelSelectionWithExtraFunction(akaike_information_criterion,
                                                                is_degree_corrected,
                                                                "AIC"))
        model_selections.append(ModelSelectionWithExtraFunction(minimum_description_length_peixoto,
                                                                is_degree_corrected,
                                                                "MDL",
                                                                function_needs_partition=True,
                                                                function_needs_partition_representation=False,
                                                                function_needs_is_directed=False))
        model_selections.append(ModelSelectionWithExtraFunction(bayesian_information_criterion,
                                                                is_degree_corrected,
                                                                "BIC sparse",
                                                                function_needs_number_of_nodes=True))
        model_selections.append(ModelSelectionWithExtraFunction(bayesian_information_criterion,
                                                                is_degree_corrected,
                                                                "BIC",
                                                                function_needs_number_of_nodes=True,
                                                                additional_fixed_key_arguments={'sparse': False}))

        model_selections.append(ModelSelectionWithExtraFunction(minimum_description_length_traditional,
                                                                is_degree_corrected,
                                                                "MDLt",
                                                                function_needs_number_of_edges=True))

        model_selections.append(ModelSelectionWithExtraFunction(minimum_description_length_peixoto_first,
                                                                is_degree_corrected,
                                                                "MDL Peixoto first",
                                                                function_needs_number_of_edges=True,
                                                                function_needs_number_of_nodes=True))

    return model_selections
