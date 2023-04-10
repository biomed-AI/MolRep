import unittest as ut
from pysbm.sbm.model_selection import ModelSelectionByObjectiveFunctionValueOnly
from pysbm.sbm.model_selection import minimum_description_length_peixoto_first
from pysbm.sbm.model_selection import minimum_description_length_traditional
from pysbm.sbm.model_selection import minimum_description_length_peixoto
from pysbm.sbm.model_selection import akaike_information_criterion
from pysbm.sbm.model_selection import bayesian_information_criterion
from pysbm.sbm.model_selection import integrated_complete_likelihood
from pysbm.sbm.model_selection import ModelSelectionByFunction
from pysbm.sbm.model_selection import ModelSelectionWithExtraFunction
from pysbm.sbm.model_selection import get_possible_model_selection
from pysbm.sbm.model_selection import AbstractModelSelection
import networkx as nx
from pysbm.sbm import NxPartition
from pysbm.sbm import IntegratedCompleteLikelihoodExactUniform
from pysbm.sbm import TraditionalMicrocanonicalEntropy
from pysbm.sbm import DegreeCorrectedMicrocanonicalEntropy
from math import log


class TestModelSelectionByObjectiveFunctionValueOnly(ut.TestCase):
    def setUp(self):
        self.model_selection = ModelSelectionByObjectiveFunctionValueOnly()
        self.graph = nx.Graph()
        nx.add_path(self.graph, range(4))
        self.partition = {node: node % 2 for node in range(4)}

    def test_create_parameters_for_model_selection_function(self):
        self.assertEqual(
            self.model_selection.create_parameters_for_model_selection_function(self.graph, self.partition), {})

    def test_select_number_of_groups(self):
        objective_functions_values = {x: x for x in range(10)}
        partitions = {x: self.partition for x in range(10)}

        self.assertEqual(
            self.model_selection.select_number_of_groups(self.graph, objective_functions_values, partitions), (9, 9))

        objective_functions_values = {x: -x for x in range(10)}
        self.assertEqual(
            self.model_selection.select_number_of_groups(self.graph, objective_functions_values, partitions), (0, 0))

        objective_functions_values[5] = 10
        self.assertEqual(
            self.model_selection.select_number_of_groups(self.graph, objective_functions_values, partitions), (5, 10))

        objective_functions_values[5] = 10
        self.assertEqual(
            self.model_selection.select_number_of_groups(self.graph, objective_functions_values, partitions, True),
            (5, 10, objective_functions_values))

    def test_get_possible_model_selection(self):
        model_selections = get_possible_model_selection(IntegratedCompleteLikelihoodExactUniform)
        self.assertEqual(len(model_selections), 1)
        self.assertTrue(isinstance(model_selections[0], ModelSelectionByObjectiveFunctionValueOnly))

        graph = nx.Graph()
        nx.add_path(graph, range(4))
        values = {2: 0}
        representations = {2: {node: node % 2 for node in graph}}

        model_selections = get_possible_model_selection(DegreeCorrectedMicrocanonicalEntropy)
        self.assertEqual(len(model_selections), 6)

        model_selections = get_possible_model_selection(TraditionalMicrocanonicalEntropy)
        self.assertEqual(len(model_selections), 7)

        for model_selection in model_selections:
            self.assertTrue(isinstance(model_selection, ModelSelectionWithExtraFunction))

            model_selection.select_number_of_groups(graph, values, representations)

    def test_AbstractModelSelection(self):
        model_selection = AbstractModelSelection(None)
        with self.assertRaises(NotImplementedError):
            model_selection.create_parameters_for_model_selection_function(None, None)

    def test_model_selections(self):
        function_dict = {"ICL": integrated_complete_likelihood,
                         "AIC": akaike_information_criterion,
                         "MDL": minimum_description_length_peixoto,
                         "BIC sparse": bayesian_information_criterion,
                         "BIC": bayesian_information_criterion,
                         "MDLt": minimum_description_length_traditional,
                         "MDL Peixoto first": minimum_description_length_peixoto_first,
                         }

        parameters_dict = {"ICL": {"partition_representation",
                                   "number_of_nodes",
                                   "is_directed",
                                   "is_degree_corrected"},
                           "AIC": {"partition_representation",
                                   "is_directed",
                                   "is_degree_corrected"},
                           "MDL": {"partition", "is_degree_corrected"},
                           "BIC sparse": {"partition_representation",
                                          "number_of_nodes",
                                          "is_directed",
                                          "is_degree_corrected"},
                           "BIC": {"partition_representation",
                                   "number_of_nodes",
                                   "is_directed",
                                   "is_degree_corrected",
                                   "sparse"},
                           "MDLt": {"partition_representation",
                                    "number_of_edges",
                                    "is_directed",
                                    "is_degree_corrected"},
                           "MDL Peixoto first": {"partition_representation",
                                                 "number_of_nodes",
                                                 "number_of_edges",
                                                 "is_directed",
                                                 "is_degree_corrected"},
                           }

        def call_function_with_parameters(call_function, parameters_description, partition, is_degree_corrected):

            call_parameters = {"is_degree_corrected": is_degree_corrected}

            if "partition_representation" in parameters_description:
                call_parameters["partition_representation"] = partition.get_representation()

            if "number_of_nodes" in parameters_description:
                call_parameters["number_of_nodes"] = partition.get_number_of_nodes()

            if "number_of_edges" in parameters_description:
                call_parameters["number_of_edges"] = partition.get_number_of_edges()

            if "is_directed" in parameters_description:
                call_parameters["is_directed"] = partition.is_graph_directed()

            if "partition" in parameters_description:
                call_parameters["partition"] = partition

            if "sparse" in parameters_description:
                call_parameters["sparse"] = False

            return call_function(**call_parameters)

        graph = nx.karate_club_graph()
        partitions = {number_of_blocks: NxPartition(graph, number_of_blocks=number_of_blocks) for number_of_blocks in
                      range(1, 11)}
        representations = {number_of_blocks: partitions[number_of_blocks].get_representation() for number_of_blocks in
                           partitions}

        objective_function = TraditionalMicrocanonicalEntropy(is_directed=False)
        values = {number_of_blocks: objective_function.calculate(partitions[number_of_blocks]) for number_of_blocks in
                  partitions}

        model_selections = get_possible_model_selection(TraditionalMicrocanonicalEntropy)

        for model_selection in model_selections:
            model_selection_function = function_dict[model_selection.title]
            parameters = parameters_dict[model_selection.title]
            true_values = {
                number_of_blocks:
                    values[number_of_blocks] - call_function_with_parameters(model_selection_function, parameters,
                                                                             partitions[number_of_blocks],
                                                                             False) for number_of_blocks in values
            }

            true_max = max(true_values, key=lambda x: true_values[x])
            true_max_value = true_values[true_max]
            self.assertEqual((true_max, true_max_value, true_values),
                             model_selection.select_number_of_groups(graph, values, representations, True),
                             msg="\nError in " + str(model_selection.title) + " " + str(model_selection_function))

        # now degree corrected
        objective_function = DegreeCorrectedMicrocanonicalEntropy(is_directed=False)
        values = {number_of_blocks: objective_function.calculate(partitions[number_of_blocks]) for number_of_blocks in
                  partitions}

        model_selections = get_possible_model_selection(DegreeCorrectedMicrocanonicalEntropy)

        for model_selection in model_selections:
            model_selection_function = function_dict[model_selection.title]
            parameters = parameters_dict[model_selection.title]
            true_values = {
                number_of_blocks:
                    values[number_of_blocks] - call_function_with_parameters(model_selection_function,
                                                                             parameters,
                                                                             partitions[number_of_blocks],
                                                                             True) for number_of_blocks in values
            }

            true_max = max(true_values, key=lambda x: true_values[x])
            true_max_value = true_values[true_max]
            self.assertEqual((true_max, true_max_value, true_values),
                             model_selection.select_number_of_groups(graph, values, representations, True),
                             msg="\nError in " + str(model_selection.title) + " " + str(model_selection_function))


class TestModelSelectionFunctions(ut.TestCase):

    def setUp(self):
        self.graphs = []

        graph = nx.Graph()
        nx.add_path(graph, range(4))
        self.graphs.append(graph)

        graph = nx.Graph()
        nx.add_path(graph, range(10))
        self.graphs.append(graph)

        graph = nx.DiGraph()
        nx.add_star(graph, range(4))
        self.graphs.append(graph)

        graph = nx.DiGraph()
        nx.add_star(graph, range(10))
        self.graphs.append(graph)

        graph = nx.karate_club_graph()
        self.graphs.append(graph)

        self.block_sizes = [2, 4, 2, 4, 3]

        self.partitions = [NxPartition(graph, representation={node: node % self.block_sizes[i] for node in graph}) for
                           i, graph in enumerate(self.graphs)]

        self.representations = [{node: node % self.block_sizes[i] for node in graph} for i, graph in
                                enumerate(self.graphs)]
        self.is_directed = [graph.is_directed() for graph in self.graphs]
        self.nodes = [len(graph.nodes) for graph in self.graphs]
        self.edges = [len(graph.edges) for graph in self.graphs]

    def test_minimum_description_length_peixoto_first(self):
        correct_results = [10 * log(2),
                           9 * ((1 + 10 / 9) * log(1 + 10 / 9) - 10 / 9 * log(10 / 9)) + 10 * log(4),
                           3 * ((1 + 4 / 3) * log(1 + 4 / 3) - 4 / 3 * log(4 / 3)) + 4 * log(2),
                           9 * ((1 + 16 / 9) * log(1 + 16 / 9) - 16 / 9 * log(16 / 9)) + 10 * log(4),
                           78 * ((1 + 1 / 13) * log(1 + 1 / 13) - 1 / 13 * log(1 / 13)) + 34 * log(3),
                           ]

        for i in range(len(self.graphs)):
            self.assertAlmostEqual(
                minimum_description_length_peixoto_first(self.representations[i], self.nodes[i], self.edges[i],
                                                         self.is_directed[i], is_degree_corrected=False),
                correct_results[i],
                msg="\nEdges= " + str(self.edges[i]) + "\nnodes= " + str(self.nodes[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

            self.assertAlmostEqual(
                minimum_description_length_peixoto_first(self.representations[i], self.nodes[i], self.edges[i],
                                                         self.is_directed[i], is_degree_corrected=True),
                correct_results[i],
                msg="\nEdges= " + str(self.edges[i]) + "\nnodes= " + str(self.nodes[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

    def test_minimum_description_length_traditional(self):
        correct_results = [3 * log(2) + 2 * 3 / 2 * log(3),
                           9 * log(4) + 4 * 5 / 2 * log(9),
                           3 * log(2) + 2 * 2 * log(3),
                           9 * log(4) + 4 * 4 * log(9),
                           78 * log(3) + 3 * 4 / 2 * log(78),
                           ]

        for i in range(len(self.graphs)):
            self.assertAlmostEqual(
                minimum_description_length_traditional(self.representations[i], self.edges[i],
                                                       self.is_directed[i], is_degree_corrected=False),
                correct_results[i],
                msg="\nEdges= " + str(self.edges[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

            self.assertAlmostEqual(
                minimum_description_length_traditional(self.representations[i], self.edges[i],
                                                       self.is_directed[i], is_degree_corrected=True),
                correct_results[i],
                msg="\nEdges= " + str(self.edges[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

    def test_minimum_description_length_peixoto(self):
        correct_results = [
            # log(binom(2+4-1,4))-log(2!)-log(2!)+log(binom(2*3/2+3-1,3))
            -2 * log(2) + log(5) + log(10),
            # log(binom(4+10-1,10))-log(2!)-log(2!)-log(3!)-log(3!)+log(binom(4*5/2+9-1,9))
            -2 * log(2) - 2 * log(6) + log(286) + log(48620),
            # log(binom(2+4-1,4))-log(2!)-log(2!)+log(binom(2*2+3-1,3))
            -2 * log(2) + log(5) + log(20),
            # log(binom(4+10-1,10))-log(2!)-log(2!)-log(3!)-log(3!)+log(binom(4*4+9-1,9))
            -2 * log(2) - 2 * log(6) + log(286) + log(1307504),
            # log(binom(3+34-1,34))-log(12!)-log(11!)-log(11!)+log(binom(3*4/2+78-1,78))
            log(630) + log(29034396) - 2 * log(39916800) - log(479001600),
        ]

        degree_corrected_correction = [-4 * log(.5),
                                       -2 * 3 * 1 / 3 * log(1 / 3) - 2 * 3 * 2 / 3 * log(2 / 3),
                                       -4 * log(.5),
                                       -2 * 3 * 1 / 3 * log(1 / 3) - 2 * 3 * 2 / 3 * log(2 / 3),
                                       53.4782992808356
                                       ]

        for i in range(len(self.graphs)):
            self.assertAlmostEqual(
                minimum_description_length_peixoto(self.partitions[i], is_degree_corrected=False),
                correct_results[i],
                msg="\nEdges= " + str(self.edges[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

            self.assertAlmostEqual(
                minimum_description_length_peixoto(self.partitions[i], is_degree_corrected=True),
                correct_results[i] + degree_corrected_correction[i],
                msg="\nDegree corrected\nEdges= " + str(self.edges[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

    def test_akaike_information_criterion(self):
        correct_results = [2 * 3 / 2 + 2,
                           4 * 5 / 2 + 4,
                           2 * 2 + 2,
                           4 * 4 + 4,
                           3 * 4 / 2 + 3,
                           ]

        degree_corrected_correction = [2,
                                       4,
                                       2,
                                       4,
                                       3
                                       ]

        for i in range(len(self.graphs)):
            self.assertAlmostEqual(
                akaike_information_criterion(self.representations[i],
                                             self.is_directed[i], is_degree_corrected=False),
                correct_results[i],
                msg="\nEdges= " + str(self.edges[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

            self.assertAlmostEqual(
                akaike_information_criterion(self.representations[i],
                                             self.is_directed[i], is_degree_corrected=True),
                correct_results[i] + degree_corrected_correction[i],
                msg="\nDegree Corrected\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

    def test_bayesian_information_criterion(self):
        correct_results = [2 * 3 / 2 * log(4 * 4 * 4),
                           4 * 5 / 2 * log(10 * 10 * 10),
                           2 * 2 * log(4 * 4 * 4),
                           4 * 4 * log(10 * 10 * 10),
                           3 * 4 / 2 * log(34 * 34 * 34),
                           ]

        degree_corrected_correction = [0,
                                       0,
                                       0,
                                       0,
                                       0
                                       ]

        for i in range(len(self.graphs)):
            self.assertAlmostEqual(
                bayesian_information_criterion(self.representations[i], self.nodes[i],
                                               self.is_directed[i], is_degree_corrected=False),
                .5 * correct_results[i],
                msg="\nnodes= " + str(self.nodes[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

            self.assertAlmostEqual(
                bayesian_information_criterion(self.representations[i], self.nodes[i],
                                               self.is_directed[i], is_degree_corrected=True),
                .5 * (correct_results[i] + degree_corrected_correction[i]),
                msg="\nNodes= " + str(self.nodes[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

    def test_bayesian_information_criterion_not_sparse(self):
        correct_results = [2 * 3 / 2 * log(4 * 4),
                           4 * 5 / 2 * log(10 * 10),
                           2 * 2 * log(4 * 4),
                           4 * 4 * log(10 * 10),
                           3 * 4 / 2 * log(34 * 34),
                           ]

        degree_corrected_correction = [0,
                                       0,
                                       0,
                                       0,
                                       0
                                       ]

        for i in range(len(self.graphs)):
            self.assertAlmostEqual(
                bayesian_information_criterion(self.representations[i], self.nodes[i],
                                               self.is_directed[i], is_degree_corrected=False, sparse=False),
                .5 * correct_results[i],
                msg="\nNodes= " + str(self.nodes[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

            self.assertAlmostEqual(
                bayesian_information_criterion(self.representations[i], self.nodes[i],
                                               self.is_directed[i], is_degree_corrected=True, sparse=False),
                .5 * (correct_results[i] + degree_corrected_correction[i]),
                msg="\nNodes= " + str(self.nodes[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

    def test_integrated_complete_likelihood(self):
        correct_results = [(2 - 1) / 2 * log(4) + 2 * 3 / 4 * log(4 * 3 / 2),
                           (4 - 1) / 2 * log(10) + 4 * 5 / 4 * log(10 * 9 / 2),
                           (2 - 1) / 2 * log(4) + 2 * 2 / 2 * log(4 * 4),
                           (4 - 1) / 2 * log(10) + 4 * 4 / 2 * log(10 * 10),
                           (3 - 1) / 2 * log(34) + 3 * 4 / 4 * log(34 * 33 / 2),
                           ]

        for i in range(len(self.graphs)):
            self.assertAlmostEqual(
                integrated_complete_likelihood(self.representations[i], self.nodes[i],
                                               self.is_directed[i], is_degree_corrected=False),
                correct_results[i],
                msg="\nNodes= " + str(self.nodes[i]) + "\ndirected= " + str(
                    self.is_directed[i]) + "\npartition= " + str(
                    self.representations[i]) + "\ncounter= " + str(i))

        with self.assertRaises(NotImplementedError):
            integrated_complete_likelihood(self.representations[0], self.nodes[0],
                                           self.is_directed[0], is_degree_corrected=True)

    def test_ModelSelectionByFunction(self):

        for i in range(len(self.graphs)):
            model_selection = ModelSelectionByFunction(None, self.edges[i], self.nodes[i])

            self.assertEqual(
                model_selection.minimum_description_length_peixoto_first(self.partitions[i], is_degree_corrected=False),
                minimum_description_length_peixoto_first(self.representations[i], self.nodes[i], self.edges[i],
                                                         self.is_directed[i], is_degree_corrected=False)
            )

            self.assertEqual(
                model_selection.minimum_description_length_peixoto_first(self.partitions[i], is_degree_corrected=True),
                minimum_description_length_peixoto_first(self.representations[i], self.nodes[i], self.edges[i],
                                                         self.is_directed[i], is_degree_corrected=True)
            )

            self.assertEqual(
                model_selection.minimum_description_length_traditional(self.partitions[i], is_degree_corrected=False),
                minimum_description_length_traditional(self.representations[i], self.edges[i],
                                                       self.is_directed[i], is_degree_corrected=False)
            )

            self.assertEqual(
                model_selection.minimum_description_length_traditional(self.partitions[i], is_degree_corrected=True),
                minimum_description_length_traditional(self.representations[i], self.edges[i],
                                                       self.is_directed[i], is_degree_corrected=True)
            )

            self.assertEqual(
                model_selection.minimum_description_length_peixoto(self.partitions[i], is_degree_corrected=False),
                minimum_description_length_peixoto(self.partitions[i], is_degree_corrected=False)
            )

            self.assertEqual(
                model_selection.minimum_description_length_peixoto(self.partitions[i], is_degree_corrected=True),
                minimum_description_length_peixoto(self.partitions[i], is_degree_corrected=True)
            )

            self.assertEqual(
                model_selection.akaike_information_criterion(self.partitions[i], is_degree_corrected=False),
                akaike_information_criterion(self.representations[i],
                                             self.is_directed[i], is_degree_corrected=False)
            )

            self.assertEqual(
                model_selection.akaike_information_criterion(self.partitions[i], is_degree_corrected=True),
                akaike_information_criterion(self.representations[i],
                                             self.is_directed[i], is_degree_corrected=True)
            )

            self.assertEqual(
                model_selection.bayesian_information_criterion(self.partitions[i], is_degree_corrected=False),
                bayesian_information_criterion(self.representations[i], self.nodes[i],
                                               self.is_directed[i], is_degree_corrected=False)
            )

            self.assertEqual(
                model_selection.bayesian_information_criterion(self.partitions[i], is_degree_corrected=True),
                bayesian_information_criterion(self.representations[i], self.nodes[i],
                                               self.is_directed[i], is_degree_corrected=True)
            )

            self.assertEqual(
                model_selection.integrated_complete_likelihood(self.partitions[i], is_degree_corrected=False),
                integrated_complete_likelihood(self.representations[i], self.nodes[i],
                                               self.is_directed[i], is_degree_corrected=False)
            )

        model_selection = ModelSelectionByFunction(None, self.edges[0], self.nodes[0])
        with self.assertRaises(NotImplementedError):
            model_selection.integrated_complete_likelihood(self.partitions[0], is_degree_corrected=True)
