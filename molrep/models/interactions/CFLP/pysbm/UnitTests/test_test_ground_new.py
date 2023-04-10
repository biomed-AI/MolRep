import unittest as ut

import six

from pysbm.test_ground_new import *
import os

import contextlib
import sys


class DummyOutput:
    """Replacement of stdout"""

    def __init__(self):
        self.lines = []

    def write(self, x):
        self.lines.append(x)

    def __str__(self):
        return ''.join(self.lines)

    def getvalue(self):
        # method needed for errors inside contextmanager
        return str(self)


@contextlib.contextmanager
def suppress_output():
    """Exchange stdout as context manager including return of output"""
    save_stdout = sys.stdout
    output_replacement = DummyOutput()
    sys.stdout = output_replacement
    yield output_replacement
    sys.stdout = save_stdout


class TestSingleNetworkTest(ut.TestCase):
    def setUp(self):
        self.karate_club = nx.karate_club_graph()
        self.true_partition = {}
        for node in self.karate_club:
            if self.karate_club.node[node]['club'] == 'Mr. Hi':
                self.true_partition[node] = 1
            else:
                self.true_partition[node] = 0

        self.single_network_test = SingleNetworkTest(self.karate_club, self.true_partition)

        # simple graph with two clusters
        self.two_cluster_graph = nx.Graph()
        nx.add_path(self.two_cluster_graph, [0, 1, 2, 0])
        nx.add_path(self.two_cluster_graph, [3, 4, 5, 3])

        self.two_cluster_test = SingleNetworkTest(self.two_cluster_graph, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1})

    def test_apply_inference_algorithm_with_true_number_of_blocks(self):

        optimized = False
        for _ in range(5):
            result = self.single_network_test.apply_inference_algorithm(
                sbm.MetropolisInference,
                sbm.TraditionalUnnormalizedLogLikelyhood)
            partition_representation = result[0][0]
            self.assertTrue(isinstance(partition_representation, dict))
            self.assertTrue(partition_representation is not None)
            self.assertEqual(len(partition_representation), len(self.karate_club))
            self.assertEqual(max(partition_representation.values()) + 1, 2)

            objective_function = sbm.TraditionalUnnormalizedLogLikelyhood(is_directed=False)
            random_partition = sbm.NxPartition(graph=self.karate_club, number_of_blocks=2)
            inferred_partition = sbm.NxPartition(graph=self.karate_club, representation=partition_representation)
            if objective_function.calculate(inferred_partition) > objective_function.calculate(random_partition):
                optimized = True

        self.assertTrue(optimized)

    def test_apply_inference_with_dummy_class(self):
        result = self.single_network_test.apply_inference_algorithm(
            DummyInference,
            sbm.TraditionalUnnormalizedLogLikelyhood)

        self.assertEqual(result[0][0], self.true_partition)

    def test_compare_partition_with_ground_truth(self):

        self.assertEqual(self.single_network_test.compare_partition_with_ground_truth_nmi(
            sbm.NxPartition(graph=self.karate_club, representation=self.true_partition)), 1)

        new_partition = self.true_partition.copy()
        new_partition[5] = 1
        new_partition[10] = 0
        self.assertLess(self.single_network_test.compare_partition_with_ground_truth_nmi(
            sbm.NxPartition(graph=self.karate_club, representation=new_partition)), 1)

    def test_execution_with_execution_parameters(self):
        result = self.single_network_test.apply_inference_algorithm(
            DummyInference,
            sbm.TraditionalUnnormalizedLogLikelyhood,
            inference_execution_parameters=("First Parameter", "Second Parameter")
        )

        self.assertEqual(result[0][0], {node: 0 for node in self.karate_club},
                         msg="Parameters were not passed to execution of inference algorithm")

    def test_execution_with_inference_class_parameters(self):
        result = self.two_cluster_test.apply_inference_algorithm(
            DummyInference,
            sbm.TraditionalUnnormalizedLogLikelyhood,
            inference_creation_parameters=(True,)
        )

        self.assertEqual(result[0][0], {node: 0 for node in self.two_cluster_graph},
                         msg="Parameters were not passed to creation of inference algorithm")

    def test_performance_measures(self):
        start = time.process_time()
        result = self.single_network_test.apply_inference_algorithm(
            DummyTimingInference,
            sbm.TraditionalUnnormalizedLogLikelyhood
        )
        end = time.process_time()

        self.assertTrue(len(result[0]) > 1)
        self.assertTrue(isinstance(result[0][0], dict))

        measured_time, node_moves, calculated_delta, _ = result[0][1:]
        self.assertEqual(node_moves, 20)
        self.assertEqual(calculated_delta, 10)
        # check time
        self.assertGreater(measured_time, 0, msg="No performance time measured or negative!")
        self.assertGreater(measured_time, (end - start) / 2, msg="Performance time to low")
        self.assertLessEqual(measured_time, end - start, msg="Performance time to high")

    def test_calculate_random_partition_nmi(self):
        values = self.single_network_test.calculate_random_partition_nmi(10, 1)

        # test one block -> 10 results all 0
        self.assertEqual(len(values), 10)
        self.assertEqual(sum(values), 0)

        # test error on wrong input
        with self.assertRaises(ValueError):
            self.single_network_test.calculate_random_partition_nmi(10, 0)

        # test different length
        values = self.single_network_test.calculate_random_partition_nmi(20, 4)
        self.assertEqual(len(values), 20)

        # test general
        for nmi in values:
            self.assertLessEqual(nmi, 1)
            self.assertGreaterEqual(nmi, 0)

        # test assumed result
        mean_result = sum(values) * 1.0 / len(values)
        self.assertLessEqual(mean_result, .2)
        self.assertGreaterEqual(mean_result, .005)


class TestTestGround(ut.TestCase):

    @classmethod
    def setUpClass(cls):
        # if not already there create temp directory
        if not os.path.exists(TEMP_DIRECTORY):
            os.makedirs(TEMP_DIRECTORY)

    def setUp(self):
        self.tests = []
        karate_club = nx.karate_club_graph()
        true_partition = {}
        for node in karate_club:
            if karate_club.node[node]['club'] == 'Mr. Hi':
                true_partition[node] = 1
            else:
                true_partition[node] = 0

        self.tests.append(SingleNetworkTest(karate_club, true_partition))

        # simple graph with two clusters
        graph = nx.Graph()
        nx.add_path(graph, [0, 1, 2, 0])
        nx.add_path(graph, [3, 4, 5, 3])

        self.tests.append(SingleNetworkTest(graph, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}, "test"))

        self.inference_classes = [sbm.EMInference, sbm.MetropolisInference, DummyInference]

        self.objective_function_classes = [sbm.TraditionalUnnormalizedLogLikelyhood,
                                           sbm.DegreeCorrectedUnnormalizedLogLikelyhood]

    def test_create_test_ground(self):

        test_ground = TestGround(self.tests, self.inference_classes, self.objective_function_classes)

        self.assertEqual(test_ground.tests, self.tests)
        self.assertEqual(test_ground.inference_classes, self.inference_classes)
        self.assertEqual(test_ground.objective_function_classes, self.objective_function_classes)

    def test_execute_tests(self):

        # create test ground
        test_ground_single = TestGround([self.tests[0]], [self.inference_classes[0]],
                                        [self.objective_function_classes[0]])
        test_ground = TestGround(self.tests, self.inference_classes, self.objective_function_classes)

        result = test_ground_single.execute_tests()
        # check if anything is returned
        self.assertNotEqual(result, None, msg="None return value of execute tests")
        # check length of return
        self.assertEqual(len(result), 1,
                         msg="""Return value of execute tests of test ground with 
                         one test case, one inference class and one objective function has not length 1""")

        result = test_ground.execute_tests()
        self.assertEqual(len(result),
                         len(self.tests) * len(self.inference_classes) * len(self.objective_function_classes),
                         msg="Wrong length of result")

    def test_result_of_execute_tests(self):
        # create test ground
        test_ground_single = TestGround([self.tests[0]], [self.inference_classes[0]],
                                        [self.objective_function_classes[0]])
        test_ground = TestGround(self.tests, self.inference_classes, self.objective_function_classes)

        result = test_ground_single.execute_tests()
        # check if right keys are assigned
        key = (self.tests[0], self.inference_classes[0], None, None, self.objective_function_classes[0], None, 2, False)
        self.assertTrue(key in result,
                        msg="Wrong keys")

        resulting_partition = result[key][0][0]
        # check type of result
        self.assertTrue(isinstance(resulting_partition, dict), msg="Result is not a partition representation")
        # # check if result belongs to the correct graph
        # self.assertEqual(resulting_partition.graph, self.tests[0].network,
        #                  msg="Resulting partition is not of correct netowrk")

        # ---- more complex test ground
        result = test_ground.execute_tests()
        self.assertEqual(len(result),
                         len(self.tests) * len(self.inference_classes) * len(self.objective_function_classes),
                         msg="Wrong length of result")

        test_found = {test_case: 0 for test_case in self.tests}
        inference_found = {inference_class: 0 for inference_class in self.inference_classes}
        objective_function_found = {objective_function: 0 for objective_function in self.objective_function_classes}

        # check right number per test case/inference/objective function
        for key in result:
            test_case, inference_class, _, _, objective_function, _, _, _ = key
            # check if all elements are correct
            self.assertTrue(test_case in test_found, msg="Test case missing")
            self.assertTrue(inference_class in inference_found, msg="Inference class missing")
            self.assertTrue(objective_function in objective_function_found, msg="Objective function missing")

            # increase counters
            test_found[test_case] += 1
            inference_found[inference_class] += 1
            objective_function_found[objective_function] += 1

            # simple check of resulting partition
            resulting_partition = result[key][0][0]
            self.assertTrue(isinstance(resulting_partition, dict), msg="Result is not a partition representation")
            # self.assertEqual(resulting_partition.graph, test_case.network,
            #                  msg="Resulting partition is not of correct network")

            if inference_class == DummyInference:
                if len(test_case.network) == len(nx.karate_club_graph()) or len(test_case.network) == 6:
                    # check if infer stochastic_block_model was called
                    self.assertEqual(
                        test_case.compare_partition_with_ground_truth_nmi(SimplePartition(resulting_partition)), 1,
                        msg="No call of infer_stochastic_block_model")

        for test_case in test_found:
            self.assertEqual(test_found[test_case], len(self.inference_classes) * len(self.objective_function_classes),
                             msg="Wrong number of results for test case" + str(test_case))

        for inference_class in inference_found:
            self.assertEqual(inference_found[inference_class], len(self.tests) * len(self.objective_function_classes),
                             msg="Wrong number of results for inference class" + str(inference_class))

        for objective_function in objective_function_found:
            self.assertEqual(objective_function_found[objective_function],
                             len(self.tests) * len(self.inference_classes),
                             msg="Wrong number of results for objective function" + str(objective_function))

    def test_new_attribute_for_saving(self):
        test_ground_single = TestGround([self.tests[0]], [self.inference_classes[0]],
                                        [self.objective_function_classes[0]])

        self.assertTrue(hasattr(test_ground_single, 'results_of_all_executions'),
                        msg="Missing attribute for saving")

        key_tuple = (self.tests[0], self.inference_classes[0], None, None,
                     self.objective_function_classes[0], None, 2, False)
        results = [test_ground_single.execute_tests()[key_tuple]]
        self.assertTrue(key_tuple in test_ground_single.results_of_all_executions,
                        msg="Missing save after execution")

        self.assertEqual(len(test_ground_single.results_of_all_executions[key_tuple]), 1,
                         msg="Saved more then one entry per execution")

        self.assertEqual(len(test_ground_single.results_of_all_executions), 1,
                         msg="Too many keys in saving structure")

        for i in range(5):
            results.append(test_ground_single.execute_tests()[key_tuple])
            self.assertEqual(len(test_ground_single.results_of_all_executions[key_tuple]), i + 2,
                             msg="Did not append new results of i-th" + str(i + 1) + " execution")

            # self.assertEqual(test_ground_single.results_of_all_executions[key_tuple], results,
            #                  msg="Did not saved correct partitions")

            self.assertEqual(len(test_ground_single.results_of_all_executions), 1,
                             msg="Too many keys in saving structure")

    def test_compare_results_with_ground_truth(self):
        test_ground_single = TestGround([self.tests[0]], [self.inference_classes[0]],
                                        [self.objective_function_classes[0]])

        # check existence of new method
        self.assertTrue(hasattr(test_ground_single, "compare_results_with_ground_truth"),
                        msg="Missing method compare_results_with_ground_truth")

        # check existence of new attribute
        self.assertTrue(hasattr(test_ground_single, "evaluated_results"),
                        msg="Missing attribute evaluated_results")
        key_tuple = (self.tests[0], self.inference_classes[0], None, None,
                     self.objective_function_classes[0], None, 2, False)
        results = [test_ground_single.execute_tests()]

        # NMI

        true_evaluated_results = [
            self.tests[0].compare_partition_with_ground_truth_nmi(SimplePartition(single_result[key_tuple][0][0])) for
            single_result in results]

        evaluated_results = test_ground_single.compare_results_with_ground_truth(
            evaluate_function=TestGround.NORMALIZED_MUTUAL_INFORMATION)

        self.assertTrue(key_tuple in evaluated_results, msg="Missing key in result structure")
        self.assertEqual(evaluated_results[key_tuple], true_evaluated_results, msg="Differ from true resuls")

        for i in range(5):
            result = test_ground_single.execute_tests()
            results.append(result)

            true_evaluated_results.append(
                self.tests[0].compare_partition_with_ground_truth_nmi(SimplePartition(result[key_tuple][0][0])))

            evaluated_results = test_ground_single.compare_results_with_ground_truth(
                evaluate_function=TestGround.NORMALIZED_MUTUAL_INFORMATION)
            self.assertEqual(len(evaluated_results[key_tuple]), i + 2)
            self.assertEqual(evaluated_results[key_tuple], true_evaluated_results, msg="Differ from true resuls")

        # AMI
        true_evaluated_results = [
            self.tests[0].compare_partition_with_ground_truth_ami(single_result[key_tuple][0][0]) for
            single_result in results]

        evaluated_results = test_ground_single.compare_results_with_ground_truth(
            evaluate_function=TestGround.ADJUSTED_MUTUAL_INFORMATION)

        self.assertTrue(key_tuple in evaluated_results, msg="Missing key in result structure")
        self.assertEqual(evaluated_results[key_tuple], true_evaluated_results, msg="Differ from true resuls")

        for i in range(5):
            result = test_ground_single.execute_tests()
            results.append(result)

            true_evaluated_results.append(
                self.tests[0].compare_partition_with_ground_truth_ami(result[key_tuple][0][0]))

            evaluated_results = test_ground_single.compare_results_with_ground_truth()
            self.assertEqual(len(evaluated_results[key_tuple]), i + 7)
            self.assertEqual(evaluated_results[key_tuple], true_evaluated_results, msg="Differ from true resuls")

    def test_passing_of_parameters(self):
        test_ground_single = TestGround([self.tests[0]], [sbm.MetropolisInference],
                                        [self.objective_function_classes[0]],
                                        inference_execution_parameters={sbm.MetropolisInference: [(10,)]})
        result = test_ground_single.execute_tests()
        values = list(result.values())[0][0]
        self.assertEqual(values[3], 10)

        test_ground_single = TestGround([self.tests[0]], [sbm.MetropolisInference],
                                        [self.objective_function_classes[0]],
                                        inference_execution_parameters={sbm.MetropolisInference: [(20,)]})
        result = test_ground_single.execute_tests()
        values = list(result.values())[0][0]
        self.assertEqual(values[3], 20)

        test_ground_single = TestGround([self.tests[0]], [sbm.MetropolisInference],
                                        [self.objective_function_classes[0]],
                                        inference_execution_parameters={sbm.MetropolisInference: [(10,), (20,)]})
        result = test_ground_single.execute_tests()
        values = [value[0][3] for value in result.values()]
        six.assertCountEqual(self, values, [10, 20])

    def test_create_karate_club_test(self):
        original_partition = SingleNetworkSupplier.create_karate_club_test().true_partition
        real_partition = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 0, 10: 1, 11: 1, 12: 1, 13: 1, 14: 0,
                          15: 0, 16: 1, 17: 1, 18: 0, 19: 1, 20: 0, 21: 1, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0,
                          28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0}
        self.assertDictEqual(original_partition, real_partition)

    def test_create_test_with_the_planted_partition_model(self):
        single_network_test = SingleNetworkSupplier.create_test_with_the_planted_partition_model(2, 5, 1, 0)
        number_of_nodes = len(single_network_test.true_partition.values())
        self.assertEqual(number_of_nodes, 2 * 5)
        number_of_edges = len(single_network_test.network.edges())
        if single_network_test.network.is_directed():
            self.assertEqual(number_of_edges, 2 * 5 * 5)
        else:
            self.assertEqual(number_of_edges, 2 * 5 * (5 + 1) * 0.5)

        single_network_test = SingleNetworkSupplier.create_test_with_the_planted_partition_model(4, 3, 1, 0)
        number_of_nodes = len(single_network_test.true_partition.values())
        self.assertEqual(number_of_nodes, 4 * 3)
        number_of_edges = len(single_network_test.network.edges())
        if single_network_test.network.is_directed():
            self.assertEqual(number_of_edges, 4 * 3 * 3)
        else:
            self.assertEqual(number_of_edges, 4 * 3 * (3 + 1) * 0.5)

    def test_create_test_with_the_stochastic_block_model(self):
        single_network_test = SingleNetworkSupplier.create_test_with_the_stochastic_block_model(
            2, [5, 5],
            [[1, 0], [0, 1]],
            SBMGenerator.PROBABILISTIC_EDGES,
            True)
        number_of_nodes = len(single_network_test.true_partition.values())
        self.assertEqual(number_of_nodes, 2 * 5)
        single_network_test = SingleNetworkSupplier.create_test_with_the_stochastic_block_model(
            4, [3, 3, 3, 3],
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            SBMGenerator.PROBABILISTIC_EDGES,
            True)
        number_of_nodes = len(single_network_test.true_partition.values())
        self.assertEqual(number_of_nodes, 4 * 3)

    def test_create_tests_for_girvan_newman_benchmark(self):
        number_of_networks = 10
        list_of_lists_of_graphs = SingleNetworkSupplier.create_tests_for_girvan_newman_benchmark(
            number_of_networks=number_of_networks, directed=False)
        numerator = 0
        divisor = 0
        in_edges = {}
        out_edges = {}
        total_edges = {}
        for test_case in list_of_lists_of_graphs:
            p_out = test_case.information[-1]
            if p_out not in in_edges:
                in_edges[p_out] = 0
                out_edges[p_out] = 0
                total_edges[p_out] = 0
            numerator += 2 * len(test_case.network.edges())
            divisor += 128
            total_edges[p_out] += 2 * len(test_case.network.edges())
            for from_node, to_node in test_case.network.edges():
                if test_case.true_partition[from_node] == test_case.true_partition[to_node]:
                    in_edges[p_out] += 2
                else:
                    out_edges[p_out] += 2
        average_edges = numerator / divisor
        self.assertLess(15, average_edges)
        self.assertGreater(17, average_edges)

        for p_out in total_edges:
            total_edges[p_out] /= 128 * number_of_networks
            in_edges[p_out] /= 128 * number_of_networks
            out_edges[p_out] /= 128 * number_of_networks

        for p_out in total_edges:
            self.assertLess(15, total_edges[p_out])
            self.assertGreater(17, total_edges[p_out])

            self.assertAlmostEqual(total_edges[p_out] - 16, 0, delta=.4,
                                   msg="p_out" + str(p_out) + "\nedges " + str(total_edges[p_out]))

        true_out_edges = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
        for i, p_out in enumerate(out_edges):
            if p_out != 0.0:
                self.assertAlmostEqual(out_edges[p_out] / true_out_edges[i], 1, delta=.1)
            self.assertAlmostEqual(in_edges[p_out] / (16 - true_out_edges[i]), 1, delta=.1)

    def test_create_tests_for_girvan_newman_benchmark_extended(self):
        number_of_networks = 10
        list_of_lists_of_graphs = SingleNetworkSupplier.create_tests_for_girvan_newman_benchmark_extended(
            number_of_networks=number_of_networks, directed=False)
        numerator = 0
        divisor = 0
        in_edges = {}
        out_edges = {}
        total_edges = {}
        for test_case in list_of_lists_of_graphs:
            p_out = test_case.information[-1]
            if p_out not in in_edges:
                in_edges[p_out] = 0
                out_edges[p_out] = 0
                total_edges[p_out] = 0
            numerator += 2 * len(test_case.network.edges())
            divisor += 128
            total_edges[p_out] += 2 * len(test_case.network.edges())
            for from_node, to_node in test_case.network.edges():
                if test_case.true_partition[from_node] == test_case.true_partition[to_node]:
                    in_edges[p_out] += 2
                else:
                    out_edges[p_out] += 2
        average_edges = numerator / divisor
        self.assertLess(15, average_edges)
        self.assertGreater(17, average_edges)

        for p_out in total_edges:
            total_edges[p_out] /= 128 * number_of_networks
            in_edges[p_out] /= 128 * number_of_networks
            out_edges[p_out] /= 128 * number_of_networks

        for p_out in total_edges:
            self.assertLess(15, total_edges[p_out])
            self.assertGreater(17, total_edges[p_out])

            self.assertAlmostEqual(total_edges[p_out] - 16, 0, delta=.4,
                                   msg="p_out" + str(p_out) + "\nedges " + str(total_edges[p_out]))

        true_out_edges = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9.0, 9.5, 10, 10.5,
                          11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
        for i, p_out in enumerate(out_edges):
            if p_out != 0.0:
                self.assertAlmostEqual(out_edges[p_out] / true_out_edges[i], 1, delta=.1,
                                       msg="p_out" + str(p_out) + "\nout edges " + str(out_edges[p_out])
                                           + "\ntrue out edges " + str(true_out_edges[i]))
            if true_out_edges[i] != 16:
                self.assertAlmostEqual(in_edges[p_out] / (16 - true_out_edges[i]), 1, delta=.1)
            else:
                self.assertAlmostEqual(in_edges[p_out], 0, delta=.1)

    def test_create_real_network_tests(self):

        with suppress_output():
            with self.assertRaises(FileNotFoundError):
                SingleNetworkSupplier.create_real_network_tests(
                    include_football=True,
                    include_political_blogs=False,
                    include_political_books=False,
                )

            with self.assertRaises(FileNotFoundError):
                SingleNetworkSupplier.create_real_network_tests(
                    include_football=False,
                    include_political_blogs=True,
                    include_political_books=False,
                )

            with self.assertRaises(FileNotFoundError):
                SingleNetworkSupplier.create_real_network_tests(
                    include_football=False,
                    include_political_blogs=False,
                    include_political_books=True,
                )

        # correct relative path
        SingleNetworkSupplier.REAL_NETWORK_DATA_FOLDER = "../" + SingleNetworkSupplier.REAL_NETWORK_DATA_FOLDER

        # first football network
        tests = SingleNetworkSupplier.create_real_network_tests(
            include_football=True,
            include_corrected_football=False,
            include_political_blogs=False,
            include_political_books=False,
        )

        self.assertEqual(len(tests), 1)
        self.assertEqual(len(tests[0].network), 115)
        self.assertEqual(len(tests[0].network.edges()), 613)
        self.assertEqual(tests[0].network.is_directed(), False)
        self.assertEqual(tests[0].network.is_multigraph(), False)
        self.assertEqual(set(tests[0].network.nodes()), set(tests[0].true_partition.keys()))

        # corrected football network
        tests = SingleNetworkSupplier.create_real_network_tests(
            include_football=False,
            include_corrected_football=True,
            include_political_blogs=False,
            include_political_books=False,
        )

        self.assertEqual(len(tests), 1)
        self.assertEqual(len(tests[0].network), 115)
        self.assertEqual(len(tests[0].network.edges()), 613)
        self.assertEqual(tests[0].network.is_directed(), False)
        self.assertEqual(tests[0].network.is_multigraph(), False)
        self.assertEqual(set(tests[0].network.nodes()), set(tests[0].true_partition.keys()))

        # political blogs
        tests = SingleNetworkSupplier.create_real_network_tests(
            include_football=False,
            include_corrected_football=False,
            include_political_blogs=True,
            include_political_books=False,
            return_largest_weakly_connected_component=False,
        )

        self.assertEqual(len(tests), 2)
        # directed and undirected
        self.assertEqual(len(tests[0].network), 1490)
        self.assertEqual(len(tests[0].network.edges()), 16718)
        self.assertEqual(tests[0].network.is_directed(), False)
        self.assertEqual(tests[0].network.is_multigraph(), False)
        self.assertEqual(set(tests[0].network.nodes()), set(tests[0].true_partition.keys()))

        self.assertEqual(len(tests[1].network), 1490)
        self.assertEqual(len(tests[1].network.edges()), 19025)
        self.assertEqual(tests[1].network.is_directed(), True)
        self.assertEqual(tests[1].network.is_multigraph(), False)
        self.assertEqual(set(tests[1].network.nodes()), set(tests[1].true_partition.keys()))

        # political blogs
        # now with only largest (weakly) connected component
        tests = SingleNetworkSupplier.create_real_network_tests(
            include_football=False,
            include_corrected_football=False,
            include_political_blogs=True,
            include_political_books=False,
            return_largest_weakly_connected_component=True,
        )

        self.assertEqual(len(tests), 2)
        # directed and undirected
        self.assertEqual(len(tests[0].network), 1222)
        self.assertEqual(len(tests[0].network.edges()), 16717)
        self.assertEqual(tests[0].network.is_directed(), False)
        self.assertEqual(tests[0].network.is_multigraph(), False)
        self.assertEqual(set(tests[0].network.nodes()), set(tests[0].true_partition.keys()))

        self.assertEqual(len(tests[1].network), 1222)
        self.assertEqual(len(tests[1].network.edges()), 19024)
        self.assertEqual(tests[1].network.is_directed(), True)
        self.assertEqual(tests[1].network.is_multigraph(), False)
        self.assertEqual(set(tests[1].network.nodes()), set(tests[1].true_partition.keys()))

        # political books
        tests = SingleNetworkSupplier.create_real_network_tests(
            include_football=False,
            include_corrected_football=False,
            include_political_blogs=False,
            include_political_books=True,
        )

        self.assertEqual(len(tests), 1)
        self.assertEqual(len(tests[0].network), 105)
        self.assertEqual(len(tests[0].network.edges()), 441)
        self.assertEqual(tests[0].network.is_directed(), False)
        self.assertEqual(tests[0].network.is_multigraph(), False)
        self.assertEqual(set(tests[0].network.nodes()), set(tests[0].true_partition.keys()))

        # all together
        tests = SingleNetworkSupplier.create_real_network_tests()
        self.assertEqual(len(tests), 5)

    def test_get_search_in_range(self):
        """Test static search in range method of TestGround"""
        self.assertEqual(TestGround._get_search_range(None, range(2)), range(2))
        self.assertEqual(TestGround._get_search_range(None, range(3)), range(3))
        self.assertNotEqual(TestGround._get_search_range(None, range(2)), range(3))

        for network_test in self.tests:
            self.assertEqual(TestGround._get_search_range(network_test, None), [network_test.true_number_of_blocks])

    def test_check_and_read_file(self):
        test_ground = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes)

        all_results = test_ground.execute_tests()
        self.assertGreater(len(all_results), 1)

        # get a part of the result
        key = list(all_results.keys())[0]
        one_result = {key: all_results[key]}

        # test reading and checking of old file
        old_save_file = TEMP_DIRECTORY + "/_" + TEMP_FILE_NAMES + "0"

        def save_results_pickle(results):
            with open(old_save_file, "wb") as file:
                pickle.dump(results, file)

        def save_results_flat(results):
            test_ground.save_results_flat(results, old_save_file, unpack_results_first=False)

        # general test of pickle and flat version
        for save_results, use_flat_saving in [(save_results_pickle, False), (save_results_flat, True)]:
            test_ground.use_flat_saving = use_flat_saving
            save_results(one_result)
            self.assertEqual(test_ground._check_and_read_file(old_save_file, None, 1, already_extended=True),
                             one_result)

            # wrong network
            error_key = (self.tests[0],) + key[1:]
            error_result = {error_key: all_results[key]}
            save_results(error_result)
            with self.assertRaises(ValueError):
                test_ground._check_and_read_file(old_save_file, None, 1, already_extended=True)

            # wrong inference
            error_key = key[:1] + (self.inference_classes[1],) + key[2:]
            error_result = {error_key: all_results[key]}
            save_results(error_result)
            with self.assertRaises(ValueError):
                test_ground._check_and_read_file(old_save_file, None, 1, already_extended=True)

            # wrong objective
            error_key = key[:4] + (sbm.TraditionalMicrocanonicalEntropyDense,) + key[5:]
            error_result = {error_key: all_results[key]}
            save_results(error_result)
            with self.assertRaises(ValueError):
                test_ground._check_and_read_file(old_save_file, None, 1, already_extended=True)

            # wrong number_of_blocks
            error_key = key[:-2] + (1,) + key[-1:]
            error_result = {error_key: all_results[key]}
            save_results(error_result)
            with self.assertRaises(ValueError):
                test_ground._check_and_read_file(old_save_file, None, 1, already_extended=True)

            # wrong number of times
            error_result = {key: [all_results[key][0], all_results[key][0]]}
            save_results(error_result)
            # too less
            with self.assertRaises(ValueError):
                test_ground._check_and_read_file(old_save_file, None, 1, already_extended=True)
            # too much
            with self.assertRaises(ValueError):
                test_ground._check_and_read_file(old_save_file, None, 3, already_extended=True)

        # pickle version allow unpacked saving
        test_ground.use_flat_saving = False
        packed_result = {key[:-2] + key[-1:]: {key[-2]: all_results[key]}}
        save_results_pickle(packed_result)
        self.assertEqual(test_ground._check_and_read_file(old_save_file, None, 1, already_extended=False),
                         one_result)

        # no unpacked version in flat case
        test_ground.use_flat_saving = True
        with self.assertRaises(ValueError):
            test_ground._check_and_read_file(old_save_file, None, 1, already_extended=False)

        os.remove(old_save_file)
        os.remove(old_save_file + ".csv")

    def test_clean_temp_files(self):
        test_ground = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes)

        test_ground.use_flat_saving = False
        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY) if os.path.isfile(name)]), 0)
        # create some files...
        for i in range(0, 20):
            file_path = TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + str(SUCCESS_MESSAGE_AND_SAVE_EVERY_X_PERCENT * i)
            with open(file_path, "w") as file:
                file.write("Test")

        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 21)
        with suppress_output() as output:
            test_ground._clean_temp_files(start=3)
        self.assertEqual(str(output), 'Deleted 17 temporary files\n')
        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 4)

        with suppress_output() as output:
            test_ground._clean_temp_files()
        self.assertEqual(str(output), 'Deleted 3 temporary files\n')
        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 1)

        # with save of csv files
        test_ground.use_flat_saving = True
        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY) if os.path.isfile(name)]), 0)
        # create some files...
        for i in range(0, 20):
            file_path = TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + str(SUCCESS_MESSAGE_AND_SAVE_EVERY_X_PERCENT * i) \
                        + ".csv"
            with open(file_path, "w") as file:
                file.write("Test")

        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 21)
        with suppress_output() as output:
            test_ground._clean_temp_files(start=3)
        self.assertEqual(str(output), 'Deleted 17 temporary files\n')
        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 4)

        with suppress_output() as output:
            test_ground._clean_temp_files()
        self.assertEqual(str(output), 'Deleted 3 temporary files\n')
        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 1)

    def test_execute_parallel(self):
        test_ground_sync = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes)
        test_ground_par = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes)

        test_ground_par.save_memory = False
        with suppress_output():
            sync_results = test_ground_sync.execute_tests()
            par_results, errors = test_ground_par.execute_tests_parallel()

            self.assertEqual(len(sync_results), len(par_results))
            for key in sync_results:
                self.assertTrue(key in par_results)
                self.assertEqual(len(sync_results), len(par_results))

            # save results of first as result_0 and see if it works
            test_ground_par = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes)

            test_ground_sync.save_results_flat(sync_results, TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + "0",
                                               unpack_results_first=False)
            par_results, errors = test_ground_par.execute_tests_parallel()
            self.assertEqual(par_results, sync_results)

            # save results of first as result_5 and see if it works
            test_ground_par = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes)

            test_ground_sync.save_results_flat(sync_results, TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + "5",
                                               unpack_results_first=False)
            par_results, errors = test_ground_par.execute_tests_parallel()
            self.assertEqual(par_results, sync_results)

    def test_execute_parallel_memory_saving(self):
        test_ground_sync = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes)
        test_ground_par = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes)
        test_ground_par.save_memory = True

        with suppress_output():
            sync_results = test_ground_sync.execute_tests()
            par_results, errors = test_ground_par.execute_tests_parallel()

            self.assertEqual(len(sync_results), len(par_results))
            for key in sync_results:
                self.assertTrue(key in par_results)
                self.assertEqual(len(sync_results), len(par_results))

            # save results of first as result_0 and see if it works
            test_ground_par = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes)

            test_ground_sync.save_results_flat(sync_results, TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + "0",
                                               unpack_results_first=False)
            par_results, errors = test_ground_par.execute_tests_parallel()
            self.assertEqual(par_results, sync_results)

            # save results of first as result_5 and see if it works
            test_ground_par = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes)

            test_ground_sync.save_results_flat(sync_results, TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + "5",
                                               unpack_results_first=False)
            par_results, errors = test_ground_par.execute_tests_parallel()
            self.assertEqual(par_results, sync_results)

    def test_read_temp_files(self):
        test_ground = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes[:1])

        # pre check
        # no files
        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 1)

        # check read without any existing file
        with suppress_output():
            self.assertEqual(test_ground._read_temp_files(None, 1), {})
        # still no files
        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 1)

        # generate some results
        results = test_ground.execute_tests()

        # save results as _0 file
        base_save_file_path = TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + "0.csv"
        test_ground.save_results_flat(results, base_save_file_path, unpack_results_first=False)
        with suppress_output():
            self.assertEqual(test_ground._read_temp_files(None, 1), results)

        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 2)
        self.assertTrue(os.path.exists(base_save_file_path))
        self.assertEqual(test_ground.read_results_flat(base_save_file_path), results)
        os.remove(base_save_file_path)

        # save results as _5 file
        file_path = TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + str(SUCCESS_MESSAGE_AND_SAVE_EVERY_X_PERCENT) + ".csv"
        test_ground.save_results_flat(results, file_path, unpack_results_first=False)

        # first with only old data -> return should be empty
        with suppress_output():
            self.assertEqual(test_ground._read_temp_files(None, 1, only_old_data=True), {})

        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 2)

        with suppress_output():
            self.assertEqual(test_ground._read_temp_files(None, 1), results)

        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 2)
        self.assertTrue(os.path.exists(base_save_file_path))

        with suppress_output():
            only_keys = test_ground._read_temp_files(None, 1, return_only_keys=True)

        true_keys = set()
        for key in results:
            # key without number of blocks, which is second from behind
            true_keys.add(key[:-2] + key[-1:])

        self.assertEqual(true_keys, only_keys)
        os.remove(base_save_file_path)

        packed_results = {}
        for key in results:
            short_key = key[:-2] + key[-1:]
            number_of_blocks = key[-2]

            if short_key not in packed_results:
                packed_results[short_key] = {}

            packed_results[short_key][number_of_blocks] = results[key]

        # test old save of results with pickle
        with open(file_path[:-4], "wb") as file:
            pickle.dump(packed_results, file)

        test_ground.use_flat_saving = False
        with suppress_output():
            self.assertEqual(test_ground._read_temp_files(None, 1), results)

        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 2)
        self.assertTrue(os.path.exists(base_save_file_path[:-4]))
        os.remove(base_save_file_path[:-4])

        # _0 file with different key
        test_ground.use_flat_saving = True
        # create a result with wrong key
        key = list(results.keys())[0]
        error_key = (self.tests[0],) + key[1:]
        error_result = {error_key: results[key]}
        test_ground.save_results_flat(error_result, base_save_file_path, unpack_results_first=False)
        with suppress_output():
            self.assertEqual(test_ground._read_temp_files(None, 1), {})
        os.remove(base_save_file_path)

        # _5 file with different key
        test_ground.save_results_flat(error_result, file_path, unpack_results_first=False)
        with suppress_output():
            self.assertEqual(test_ground._read_temp_files(None, 1), {})
        os.remove(file_path)

    def test_read_temp_files_multiple(self):
        # check reading of files _5, _10, ...
        test_ground = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes[:1])

        # pre check
        # no files
        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 1)

        # check read without any existing file
        with suppress_output():
            self.assertEqual(test_ground._read_temp_files(None, 1), {})
        # still no files
        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 1)

        # generate some results
        results = test_ground.execute_tests()

        # save results into different files
        for i, key in enumerate(results):
            file_path = TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + str(int(i * SUCCESS_MESSAGE_AND_SAVE_EVERY_X_PERCENT))
            test_ground.save_results_flat({key: results[key]}, file_path, unpack_results_first=False)

        with suppress_output():
            read_results = test_ground._read_temp_files(None, 1)
        self.assertEqual(read_results, results)

        # clean up at the end
        os.remove(TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + "0.csv")
        self.assertEqual(len([name for name in os.listdir(TEMP_DIRECTORY)]), 1)

    def test_exclusion_file(self):
        test_ground = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes)

        results = test_ground.execute_tests()

        key = list(results.keys())[0]
        one_result = {key: results[key]}
        base_save_file_path = TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + "0.csv"
        test_ground.save_results_flat(one_result, base_save_file_path, unpack_results_first=False)

        with suppress_output():
            test_ground.create_simple_exclusion_file()
            excluded_keys = test_ground.read_simple_exclusion_file()
        os.remove(base_save_file_path)
        os.remove(TEMP_DIRECTORY + "/" + test_ground.EXCLUSION_FILE_NAME)

        self.assertEqual({key[:-2]}, excluded_keys)

    def test_read_and_save_flat_results_hierarchy(self):
        # non hierarchy tested a lot in other
        test_ground = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes[:1])

        # get any result
        results = test_ground.execute_tests()

        # create test ground with hierarchy
        test_ground = TestGround(self.tests[1:2],
                                 [sbm.HierarchicalInference],
                                 [sbm.LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedUniform])

        # transform results
        original_key = list(results.keys())[0]
        network_test = self.tests[1]
        new_key = (network_test, sbm.HierarchicalInference, None, None,
                   sbm.LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedUniform, None, 2, False)

        # 6 nodes
        faked_hierarchy_partition = [{i: i for i in range(6)},
                                     {i: i % 2 for i in range(6)},
                                     {0: 0, 1: 0}
                                     ]
        # create new result with hierarchy partition
        new_results = {new_key: [(faked_hierarchy_partition,) + results[original_key][0][1:]]}
        file_path = TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + "0.csv"
        test_ground.save_results_flat(new_results, file_path, unpack_results_first=False)

        self.assertEqual(test_ground.read_results_flat(file_path), new_results)
        os.remove(file_path)

    def test_random_nmi(self):
        test_ground = TestGround(self.tests[1:2], self.inference_classes[:1], self.objective_function_classes[:1])

        random_nmi_compressed = test_ground.calculate_random_nmi(5)
        self.assertEqual(len(random_nmi_compressed), 5)
        self.assertGreaterEqual(min(random_nmi_compressed), 0)
        self.assertLessEqual(max(random_nmi_compressed), 1)

        random_nmi = test_ground.calculate_random_nmi(10, compressed=False)
        network_test = self.tests[1]
        self.assertTrue(network_test in random_nmi)
        self.assertEqual(len(random_nmi[network_test]), 10)


class TestLFRBenchmark(ut.TestCase):

    def test_read_files(self):
        tests = SingleNetworkSupplier.create_lfr_benchmark_tests("./LFR_benchmark_test_file", [0], 1)
        self.assertEqual(len(tests), 1)

        graph = nx.Graph()
        graph.add_edges_from([(0, 2), (0, 4), (1, 3), (2, 4), (3, 4)])
        true_partition = {0: 1, 1: 0, 2: 3, 3: 3, 4: 2}

        test_case = tests[0]
        self.assertEqual(test_case.network.edges(), graph.edges())
        self.assertEqual(test_case.network.nodes(), graph.nodes())
        self.assertEqual(test_case.true_partition, true_partition)
        self.assertEqual(test_case.true_number_of_blocks, 4)
        self.assertEqual(test_case.information,
                         ['mixing parameter, network instance', 0, 0])


class DummyInference(sbm.Inference):

    def __init__(self, graph, objective_function, partition, new_parameter=False):
        super(DummyInference, self).__init__(graph, objective_function, partition)
        self._instance_parameter = new_parameter

    def infer_stochastic_block_model(self, *args):
        other_partition = False
        if len(args) > 0 and args[0] == "First Parameter" and args[1] == "Second Parameter":
            other_partition = True
        if len(self.partition.graph) == len(nx.karate_club_graph()):
            karate_club = nx.karate_club_graph()
            for node in karate_club:
                if karate_club.node[node]['club'] == 'Mr. Hi':
                    if not other_partition:
                        self.partition.move_node(node, 1)
                    else:
                        self.partition.move_node(node, 0)
                else:
                    self.partition.move_node(node, 0)
        elif len(self.partition.graph) == 6:
            representation = {}
            for node in range(6):
                if node < 3:
                    if not self._instance_parameter:
                        representation[node] = 1
                    else:
                        representation[node] = 0
                else:
                    representation[node] = 0
            self.partition.set_from_representation(representation)
        # in all other cases do nothing

    def infer_stepwise(self):
        raise NotImplementedError()


class DummyTimingInference(sbm.Inference):

    def infer_stochastic_block_model(self, *args):
        # do some nonsense and waste around 2-3 seconds
        test = [list(range(100)) for _ in range(1000)]

        for _ in range(20):
            self.partition.move_node(0, (self.partition.get_block_of_node(0) + 1) % self.partition.B)
        for _ in range(10):
            self.objective_function.calculate_delta(self.partition, 0, 0)

    def infer_stepwise(self):
        pass
