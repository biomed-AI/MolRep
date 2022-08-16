import time

import networkx as nx
import csv

from MolRep.Interactions.link_models.CFLP.pysbm import sbm
from MolRep.Interactions.link_models.CFLP.pysbm.parallel_execution import parallel_execution
from MolRep.Interactions.link_models.CFLP.pysbm.parallel_execution import TEMP_DIRECTORY
from MolRep.Interactions.link_models.CFLP.pysbm.parallel_execution import TEMP_FILE_NAMES
from MolRep.Interactions.link_models.CFLP.pysbm.parallel_execution import SUCCESS_MESSAGE_AND_SAVE_EVERY_X_PERCENT
from MolRep.Interactions.link_models.CFLP.pysbm.test_ground import NormalizedMutualInformation, PlantedPartitionGenerator, SBMGenerator, SimplePartition
import random as rd
import os
import pickle
import math
import typing
from collections import namedtuple

from sklearn.metrics import adjusted_mutual_info_score as adjusted_mutual_information

# @formatter:off
def apply_inference_algorithm(objective_function_class,
                              inference_algorithm_class,
                              graph,
                              search_in_range,
                              number_of_times=1,
                              starting_representation=None,
                              inference_execution_parameters=None,
                              inference_creation_parameters=None,
                              objective_function_creation_parameters=None,
                              ):
    """
    Function to apply inference algorithm with a given objective function and partition
    :param objective_function_class: class reference to an objective function
    :type objective_function_class class sbm.ObjectiveFunction
    :param inference_algorithm_class: class reference to an inference algorithm
    :type inference_algorithm_class class sbm.Inference
    :param graph: graph for clustering
    :type graph nx.Graph
    :param search_in_range: range for number of blocks for partition
    :type search_in_range typing.Iterable(int)
    :param starting_representation: If given the algorithm starts from the given partition instead of a random one
    :type starting_representation dict
    :param number_of_times: number of executions
    :type number_of_times int
    :param inference_execution_parameters: parameters for execution of inference algorithm
    :param inference_creation_parameters: parameters for creation of inference class
    :param objective_function_creation_parameters: parameters for creation of objective function
    :return: resulting partition, needed cpu time, needed node moves, number of calculated deltas
    """
    if len(search_in_range) > 1 and starting_representation is not None:
        raise ValueError("Starting partition and search range for number of blocks with at least 2 members given!")
    results_by_group_size = {}
    for number_of_blocks in search_in_range:
        results = []
        for _ in range(number_of_times):
            if issubclass(inference_algorithm_class, sbm.HierarchicalInference):
                # if hierarchical version of SBM need hierarchical partition
                if starting_representation is None:
                    partition = sbm.NxHierarchicalPartition(graph=graph)
                else:
                    raise ValueError("Starting representation for hierarchy not allowed")
            elif inference_algorithm_class is sbm.KerninghanLinInference:
                # if KL algorithm we need to keep the number of blocks either this
                if starting_representation is None:
                    partition = sbm.NxPartitionWithMoveCounter(graph=graph, number_of_blocks=number_of_blocks)
                    # or
                    # partition = sbm.NxPartitionGraphBasedWithMoveCounter(graph=graph,
                    #                                                      number_of_blocks=number_of_blocks)
                    # partition.with_empty_blocks = True
                else:
                    partition = sbm.NxPartitionWithMoveCounter(graph=graph,
                                                               fill_random=False,
                                                               representation=starting_representation)

            else:
                if starting_representation is None:
                    partition = sbm.NxPartitionGraphBasedWithMoveCounter(graph=graph, number_of_blocks=number_of_blocks)
                else:
                    partition = sbm.NxPartitionGraphBasedWithMoveCounter(graph=graph,
                                                                         fill_random=False,
                                                                         representation=starting_representation)

            used_cpu_time_start = time.process_time()
            if inference_execution_parameters is None:
                inference_execution_parameters = ()
            if inference_creation_parameters is None:
                inference_creation_parameters = ()
            if objective_function_creation_parameters is None:
                objective_function_creation_parameters = ()

            objective_function = objective_function_class(partition.is_graph_directed(),
                                                          *objective_function_creation_parameters)
            calculated_deltas_start = objective_function.number_of_calculated_deltas
            # create inference algorithm instance for this run
            if issubclass(inference_algorithm_class, sbm.HierarchicalInference):
                inference_algorithm = inference_algorithm_class(graph,
                                                                objective_function,
                                                                partition,
                                                                objective_function.calculate_delta_actual_level_removed,
                                                                *inference_creation_parameters)
            else:
                inference_algorithm = inference_algorithm_class(graph, objective_function,
                                                                partition, *inference_creation_parameters)
            # set direction attribute of objective function to correct value
            objective_function.is_directed = partition.is_graph_directed()
            # now run the inference algorithm
            inference_algorithm.infer_stochastic_block_model(*inference_execution_parameters)
            used_cpu_time_end = time.process_time()
            used_cpu_time = used_cpu_time_end - used_cpu_time_start
            calculated_deltas_end = objective_function.number_of_calculated_deltas
            calculated_deltas = calculated_deltas_end - calculated_deltas_start
            try:
                node_moves = partition.node_moves
            except AttributeError:
                node_moves = 0
            results.append((inference_algorithm.partition.get_representation(), used_cpu_time, node_moves,
                            calculated_deltas, objective_function.calculate(inference_algorithm.partition)))
        results_by_group_size[number_of_blocks] = results
    return results_by_group_size


# @formatter:on


class SingleNetworkTest:
    """Object containing a single graph with all known relevant information"""

    def __init__(self, network, true_partition=None, information=None, additional_information=None):
        """

        :param network: Network where the test runs should be executed
        :type network nx.Graph
        :param true_partition: If available the true partition or metadata information
        :type true_partition dict
        :param information: Unique description of the data
        :type information Hashable
        :param additional_information: Further may non unique or non hashable data
        :type additional_information Any
        """
        self.network = network
        self._true_partition = true_partition
        if true_partition is not None:
            self._true_number_of_blocks = max(true_partition.values()) + 1
        else:
            self._true_number_of_blocks = None
        self._normalized_mutual_information = None
        self.information = information
        self.additional_information = additional_information

    @property
    def true_partition(self):
        if self._true_partition is not None:
            return self._true_partition
        raise ValueError("No true partition supplied")

    @property
    def true_number_of_blocks(self):
        if self._true_number_of_blocks is not None:
            return self._true_number_of_blocks
        raise ValueError("No true partition supplied")

    @property
    def normalized_mutual_information(self):
        if self._normalized_mutual_information is None:
            self._normalized_mutual_information = NormalizedMutualInformation(self.true_partition)
        return self._normalized_mutual_information

    def apply_inference_algorithm(self,
                                  inference_algorithm_class,
                                  objective_function_class,
                                  inference_execution_parameters=None,
                                  inference_creation_parameters=None,
                                  objective_function_creation_parameters=None,
                                  number_of_blocks=None,
                                  number_of_times=1,
                                  starting_representation=None,
                                  ):
        """
        Apply inference algorithm to contained network and pass to the algorithm the true number of blocks
        :param inference_algorithm_class: create inference algorithm from this class
        :type inference_algorithm_class class sbm.Inference
        :param objective_function_class: target function to be passed to the inference algorithm
        :type objective_function_class class sbm.ObjectiveFunction
        :param inference_execution_parameters: parameters for execution of inference algorithm
        :param inference_creation_parameters: parameters for creation of inference class
        :param objective_function_creation_parameters: parameters for creation of objective function
        :param number_of_blocks: number of blocks
        :param number_of_times: number of executions
        :return:
        """
        if number_of_blocks is None:
            # create new partition for this run with the true number of blocks
            number_of_blocks = self.true_number_of_blocks
        return apply_inference_algorithm(objective_function_class,
                                         inference_algorithm_class,
                                         self.network,
                                         [number_of_blocks],
                                         number_of_times,
                                         starting_representation,
                                         inference_execution_parameters,
                                         inference_creation_parameters,
                                         objective_function_creation_parameters,
                                         )[number_of_blocks]

    def get_parameters_for_apply_inference_algorithm_on_graph_with_given_number_of_blocks(
            self,
            inference_algorithm_class,
            objective_function_class,
            search_in_range=None,
            number_of_times=1,
            start_from_true_partition=False,
    ):
        if search_in_range is None:
            # create new partition for this run with the true number of blocks
            search_in_range = self.true_number_of_blocks
        if start_from_true_partition:
            starting_representation = self.true_partition
        else:
            starting_representation = None
        return objective_function_class, inference_algorithm_class, self.network, search_in_range, number_of_times, \
               starting_representation,

    def compare_partition_with_ground_truth_nmi(self, partition):
        """
        Compare given partition with the metadata or planted partition
        :param partition: any partition of the same network
        :type partition typing.Union[sbm.Partition, SimplePartition)
        :return:
        """
        return self.normalized_mutual_information.evaluate(partition)

    def compare_partition_with_ground_truth_ami(self, representation):
        """
        Compare given partition with the metadata or planted partition
        :param representation: any partition of the same network
        :type representation dict
        :return:
        """
        return adjusted_mutual_information([self.true_partition[node] for node in representation],
                                           [representation[node] for node in representation])

    def calculate_random_partition_nmi(self, number_of_random_partitions, number_of_groups=None):
        if number_of_groups is None:
            number_of_groups = self.true_number_of_blocks
        if number_of_groups < 1:
            raise ValueError

        values = []
        nodes = list(self.network.nodes)
        for _ in range(number_of_random_partitions):
            # create random partitions
            representation = {}
            # assign each node a random block
            for node in nodes:
                representation[node] = rd.randrange(0, number_of_groups)

            # ensure that in each block is at least one node
            for group in range(number_of_groups):
                representation[rd.choice(nodes)] = group

            # calculate nmi
            values.append(self.normalized_mutual_information.evaluate(SimplePartition(representation)))
        return values


# @formatter:off
class TestGround:
    EXCLUSION_FILE_NAME = "exclude.csv"

    NEW_HEADER_LINE = ["network information",
                       "inference short title",
                       "objective function short title",
                       "number of blocks",
                       "true partition?",
                       "cpu time",
                       "node moves",
                       "# deltas",
                       "objective function value",
                       "node block assignments based on sorted nodes..."]

    NEW_HEADER_LINE_EVALUATED = ["network information",
                                 "inference short title",
                                 "objective function short title",
                                 "number of blocks",
                                 "true partition?",
                                 "Evaluated results"]

    NORMALIZED_MUTUAL_INFORMATION = "NMI"
    ADJUSTED_MUTUAL_INFORMATION = "AMI_max"

    def __init__(self, tests, inference_classes, objective_function_classes, inference_execution_parameters=None,
                 inference_creation_parameters=None, objective_function_creation_parameters=None):
        """
        Container of tests, SBM variances and inference methods with useful methods like (parallel) bulk execution
        :param tests: test cases for the execution
        :type tests typing.Iterable[SingleNetworkTest]
        :param inference_classes: methods for inference
        :type inference_classes typing.Iterable[class sbm.Inference]
        :param objective_function_classes: objective functions of (different) SBM variants
        :type objective_function_classes typing.Iterable[class sbm.objective_function]
        :param inference_execution_parameters: parameters for execution of specific inference classes
        :type inference_execution_parameters typing.Dict[class sbm.Inference, typing.Any]
        :param inference_creation_parameters: parameters for creation of inference classes
        :type inference_creation_parameters typing.Dict[class sbm.Inference, typing.Any]
        :param objective_function_creation_parameters: parameters for creating of objective function
        :type objective_function_creation_parameters typing.Dict[class sbm.objective_function, typing.Any]
        """
        self.tests = tests
        self.inference_classes = inference_classes
        self.objective_function_classes = objective_function_classes
        self.results_of_all_executions = {}
        self.evaluated_results = {}
        if inference_execution_parameters is None:
            inference_execution_parameters = {}
        if inference_creation_parameters is None:
            inference_creation_parameters = {}
        if objective_function_creation_parameters is None:
            objective_function_creation_parameters = {}

        self.inference_execution_parameters = inference_execution_parameters
        self.inference_creation_parameters = inference_creation_parameters
        self.objective_function_creation_parameters = objective_function_creation_parameters
        self._number_of_total_executions = 0

        # avoid usage of pickle
        self.use_flat_saving = True

        # save memory by not keeping results in RAM and save them instead into the files
        self.save_memory = True

    @staticmethod
    def _get_search_range(network, search_in_range=None):
        if search_in_range is None:
            return [network.true_number_of_blocks]
        else:
            return search_in_range

    def execute_tests(self, search_in_range=None, start_from_true_partition=False):
        results = {
            (network_test,
             inference_class,
             inference_creation_parameters,
             inference_execution_parameters,
             objective_function_class,
             objective_function_creation_parameters,
             number_of_blocks,
             start_from_true_partition,
             ): None
            for network_test in self.tests
            for inference_class in self.inference_classes
            for inference_creation_parameters in self.inference_creation_parameters.get(inference_class, [None])
            for inference_execution_parameters in self.inference_execution_parameters.get(inference_class, [None])
            for objective_function_class in self.objective_function_classes
            for objective_function_creation_parameters in
            self.objective_function_creation_parameters.get(objective_function_class, [None])
            for number_of_blocks in self._get_search_range(network_test, search_in_range)
        }
        for key in results:
            network_test, inference_class, inference_creation_parameters, inference_execution_parameters, \
            objective_function_class, objective_function_creation_parameters, number_of_blocks, \
            is_from_starting_partition = key

            if is_from_starting_partition:
                starting_representation = network_test.true_partition
            else:
                starting_representation = None

            results[key] = network_test.apply_inference_algorithm(
                inference_class, objective_function_class,
                inference_creation_parameters=inference_creation_parameters,
                inference_execution_parameters=inference_execution_parameters,
                objective_function_creation_parameters=objective_function_creation_parameters,
                number_of_blocks=number_of_blocks,
                starting_representation=starting_representation,
            )

            if key not in self.results_of_all_executions:
                self.results_of_all_executions[key] = []
            self.results_of_all_executions[key].extend(results[key])

        self._number_of_total_executions += 1
        return results

    def _read_temp_files(self, search_in_range, number_of_times, temp_file_path=None, only_old_data=False,
                         return_only_keys=False):
        old_saved_results_path = TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + "0"
        if self.use_flat_saving:
            old_saved_results_path += ".csv"

        try:
            old_saved_results = self._check_and_read_file(old_saved_results_path, search_in_range, number_of_times,
                                                          already_extended=True)
        except FileNotFoundError:
            print("No file", old_saved_results_path)
            old_saved_results = {}
        except ValueError as exc:
            print(exc)
            old_saved_results = {}

        function_call_results = {}
        if not only_old_data:
            last_file_path = temp_file_path

            if last_file_path is None:
                for i in range(1, round(math.ceil(100 / SUCCESS_MESSAGE_AND_SAVE_EVERY_X_PERCENT) + 1)):
                    file_path = TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES \
                                + str(SUCCESS_MESSAGE_AND_SAVE_EVERY_X_PERCENT * i)
                    if self.use_flat_saving:
                        file_path += ".csv"
                    if os.path.exists(file_path):
                        if self.save_memory:
                            try:
                                intermediate_result = self._check_and_read_file(file_path,
                                                                                search_in_range,
                                                                                number_of_times,
                                                                                already_extended=self.use_flat_saving
                                                                                )
                            except ValueError as exc:
                                print(exc)
                            else:
                                print("Loaded ", len(intermediate_result), "from file", file_path)
                                function_call_results.update(intermediate_result)
                        else:
                            last_file_path = file_path
                            continue
                    else:
                        break

            if last_file_path is not None and not self.save_memory:
                try:
                    function_call_results = self._check_and_read_file(last_file_path,
                                                                      search_in_range,
                                                                      number_of_times,
                                                                      already_extended=self.use_flat_saving
                                                                      )
                except ValueError as exc:
                    print(exc)

        mapped_results = {}
        mapped_results.update(old_saved_results)
        mapped_results.update(function_call_results)

        print("Loaded ", len(mapped_results), "from saved temporary files")
        if mapped_results:
            if self.use_flat_saving:
                self.save_results_flat(mapped_results, old_saved_results_path, unpack_results_first=False)
            else:
                with open(old_saved_results_path, "wb") as file:
                    pickle.dump(mapped_results, file, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved", len(mapped_results), "to", old_saved_results_path)

        if not only_old_data and function_call_results:
            # only delete files if there are consumed, i.e. read and included in above summary
            # and something was read in -> else no correct files or no files at all
            self._clean_temp_files(start=1)

        if return_only_keys:
            keys = set()
            for long_key in mapped_results:
                # add without number of blocks
                keys.add(long_key[:-2] + long_key[-1:])
            return keys

        return mapped_results

    def _check_and_read_file(self, file_path, search_in_range, number_of_times, already_extended=False):

        if self.use_flat_saving:
            if not already_extended:
                raise ValueError("Flat saving and not extended impossible!")
            saved_result = self.read_results_flat(file_path)
        else:
            with open(file_path, "rb") as file:
                saved_result = pickle.load(file)

        # check if the result are from the same data
        network_set = set()
        inference_set = set()
        inference_creation_parameters_set = set()
        inference_execution_parameters_set = set()
        objective_function_classes_set = set()
        objective_function_creation_parameters_set = set()
        # reduce to inputs
        for key in saved_result:
            # flat saving is always extended
            network_test, inference_class, inference_creation_parameters, inference_execution_parameters, \
            objective_function_class, objective_function_creation_parameters = key[:6]

            network_set.add(network_test)
            inference_set.add(inference_class)
            if inference_creation_parameters is not None:
                inference_creation_parameters_set.add((inference_class, inference_creation_parameters))
            if inference_execution_parameters is not None:
                inference_execution_parameters_set.add((inference_class, inference_execution_parameters))
            objective_function_classes_set.add(objective_function_class)
            if objective_function_creation_parameters is not None:
                objective_function_creation_parameters_set.add(
                    (objective_function_class, objective_function_creation_parameters))

        for inference_class in inference_set:
            for check_inference_class in self.inference_classes:
                if inference_class == check_inference_class:
                    break
            else:
                # nothing found => different results
                raise ValueError("File ", file_path, "found, but different inferences")

        for inference_class, inference_creation_parameters in inference_creation_parameters_set:
            if inference_class not in self.inference_creation_parameters:
                raise ValueError("File ", file_path, "found, but different inferences creation parameters")
            for check_parameters in self.inference_creation_parameters[inference_class]:
                if check_parameters == inference_creation_parameters:
                    break
            else:
                raise ValueError("File ", file_path, "found, but different inferences creation parameters")

        for inference_class, inference_execution_parameters in inference_execution_parameters_set:
            if inference_class not in self.inference_execution_parameters:
                raise ValueError("File ", file_path, "found, but different inferences execution parameters")
            for check_parameters in self.inference_execution_parameters[inference_class]:
                if check_parameters == inference_execution_parameters:
                    break
            else:
                raise ValueError("File ", file_path, "found, but different inferences execution parameters")

        for objective_function_class in objective_function_classes_set:
            for check_parameters in self.objective_function_classes:
                if objective_function_class == check_parameters:
                    break
            else:
                raise ValueError("File ", file_path, "found, but different objective_functions")

        for objective_function_class, objective_function_creation_parameters \
                in objective_function_creation_parameters_set:

            if objective_function_class not in self.objective_function_creation_parameters:
                raise ValueError("File ", file_path, "found, but different objective_functions creation parameters")

            for check_parameters in self.objective_function_creation_parameters[objective_function_class]:
                if check_parameters == objective_function_creation_parameters:
                    break
            else:
                raise ValueError("File ", file_path, "found, but different objective_functions creation parameters")

        network_mapping = {}
        for network_test in network_set:
            for check_network_test in self.tests:
                if network_test.network.edges() == check_network_test.network.edges():
                    network_mapping[network_test] = check_network_test
                    break
            else:
                # not identical
                raise ValueError("File ", file_path, "found, but different networks")

        # every check is okay
        # exchange references of network_test
        mapped_results = {}
        included_blocks_by_short_key = {}
        for key in saved_result:
            new_key = (network_mapping[key[0]],) + key[1:]
            if not (already_extended or self.use_flat_saving):
                results_by_number_of_blocks = saved_result[key]
                # check if same region to test
                if list(results_by_number_of_blocks.keys()) != \
                        list(self._get_search_range(network_mapping[key[0]], search_in_range)):
                    raise ValueError("File ", file_path, "found, but different search range")
                # unpack values
                for number_of_blocks in results_by_number_of_blocks:
                    mapped_results[new_key[:-1] + (number_of_blocks,) + new_key[-1:]] = \
                        results_by_number_of_blocks[number_of_blocks]
                    # check number of times
                    if len(results_by_number_of_blocks[number_of_blocks]) != number_of_times:
                        raise ValueError("File ", file_path, "found, but different number of times")
            else:
                if len(saved_result[key]) != number_of_times:
                    raise ValueError("File ", file_path, "found, but different number of times")
                mapped_results[new_key] = saved_result[key]
                # prepare check of search range
                if len(key) == 8:
                    # new format with flag for from start point
                    short_key = key[:-2]
                    number_of_blocks = key[-2]
                else:
                    # old format
                    short_key = key[:-1]
                    number_of_blocks = key[-1]
                if short_key not in included_blocks_by_short_key:
                    included_blocks_by_short_key[short_key] = set()

                included_blocks_by_short_key[short_key].add(number_of_blocks)

        if already_extended or self.use_flat_saving:
            for key in included_blocks_by_short_key:
                if set(self._get_search_range(key[0], search_in_range)) != included_blocks_by_short_key[key]:
                    raise ValueError("File ", file_path, "found, but different search range")

        return mapped_results

    def _clean_temp_files(self, start=0):
        deleted_file_counter = 0
        for i in range(start, round(math.ceil(100 / SUCCESS_MESSAGE_AND_SAVE_EVERY_X_PERCENT) + 1)):
            file_path = TEMP_DIRECTORY + "/" + TEMP_FILE_NAMES + str(SUCCESS_MESSAGE_AND_SAVE_EVERY_X_PERCENT * i)
            if self.use_flat_saving:
                file_path += ".csv"
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
            else:
                deleted_file_counter += 1
        print("Deleted", deleted_file_counter, "temporary files")

    def create_simple_exclusion_file(self, number_of_times=1, search_in_range=None, temp_file_path=None):

        # if wanted check temp files
        if not os.path.exists(TEMP_DIRECTORY):
            raise FileNotFoundError("No directory " + str(TEMP_DIRECTORY) + " found")

        # read keys
        done_keys = self._read_temp_files(search_in_range, number_of_times, temp_file_path, return_only_keys=True)

        # write csv file
        file_path = TEMP_DIRECTORY + "/" + self.EXCLUSION_FILE_NAME
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, dialect=csv.excel)

            for key in done_keys:
                network_test = key[0]
                inference_class = key[1]
                inference_creation_parameters = key[2]
                if inference_creation_parameters is not None:
                    raise NotImplementedError()
                inference_execution_parameters = key[3]
                if inference_execution_parameters is not None:
                    raise NotImplementedError()
                objective_function_class = key[4]
                objective_function_creation_parameters = key[5]
                if objective_function_creation_parameters is not None:
                    raise NotImplementedError()

                writer.writerow([str(network_test.information),
                                 inference_class.short_title,
                                 objective_function_class.short_title])

    def _create_mapping_dicts(self):
        # create simple mapping
        network_mapping = {}
        for network_test in self.tests:
            information = str(network_test.information)
            if information in network_mapping:
                raise ValueError("Two network test with same information: ", information)
            network_mapping[information] = network_test

        inference_mapping = {}
        for inference_class in self.inference_classes:
            title = inference_class.short_title
            if title in inference_mapping:
                raise ValueError("Two inferences with same title: ", title)
            inference_mapping[title] = inference_class

        objective_mapping = {}
        for objective_function_class in self.objective_function_classes:
            title = objective_function_class.short_title
            if title in objective_mapping:
                raise ValueError("Two objectives with same title: ", title)
            objective_mapping[title] = objective_function_class

        return network_mapping, inference_mapping, objective_mapping

    def read_simple_exclusion_file(self):

        network_mapping, inference_mapping, objective_mapping = self._create_mapping_dicts()

        file_path = TEMP_DIRECTORY + "/" + self.EXCLUSION_FILE_NAME
        keys = set()
        try:
            with open(file_path, 'r', newline='') as csv_file:
                reader = csv.reader(csv_file, dialect=csv.excel)

                for row in reader:
                    network_test_information = row[0]
                    inference_class_title = row[1]
                    objective_function_class_title = row[2]

                    network_test = network_mapping[network_test_information]
                    inference_class = inference_mapping[inference_class_title]
                    objective_function_class = objective_mapping[objective_function_class_title]

                    key = network_test, inference_class, None, None, objective_function_class, None

                    keys.add(key)
        except FileNotFoundError:
            pass

        return keys

    @staticmethod
    def save_results_flat(results_dict, file_path, unpack_results_first=True, is_evaluated_results=False):
        if unpack_results_first and not is_evaluated_results:
            # if needed unpack results before writing
            results = {}
            for key, results_by_number_of_block in results_dict.items():
                for number_of_blocks in results_by_number_of_block:
                    results[key[:-1] + (number_of_blocks,) + key[-1:]] = results_by_number_of_block[number_of_blocks]
        else:
            results = results_dict

        # create header row
        if is_evaluated_results:
            lines = [TestGround.NEW_HEADER_LINE_EVALUATED]
        else:
            lines = [TestGround.NEW_HEADER_LINE]

        # loop over results and create one line per entry
        for key, result_per_execution in results.items():
            network_test, inference_class, inference_creation_parameters, inference_execution_parameters, \
                objective_function_class, objective_function_creation_parameters, number_of_blocks, \
                start_from_true_partition = key

            # first basic checks if advanced parameters are none, because saving of them could be more complicated
            if inference_creation_parameters is not None:
                print("Inference creation parameter is not none, line skipped in saving")
                continue
            if inference_execution_parameters is not None:
                print("Inference execution parameter is not none, line skipped in saving")
                continue
            if objective_function_creation_parameters is not None:
                print("Objective function creation parameter is not none, line skipped in saving")
                continue

            if is_evaluated_results:
                # add information with static length and position
                line = [network_test.information,
                        inference_class.short_title,
                        objective_function_class.short_title,
                        number_of_blocks,
                        "T" if start_from_true_partition else "F",
                        ]
                # add evaluated values
                line.extend(result_per_execution)

                lines.append(line)
            else:
                for result in result_per_execution:
                    # add information with static length and position
                    line = [network_test.information,
                            inference_class.short_title,
                            objective_function_class.short_title,
                            number_of_blocks,
                            "T" if start_from_true_partition else "F",
                            result[1],
                            result[2],
                            result[3],
                            result[4]
                            ]

                    # create representation of partition
                    representation = result[0]
                    # distinguish between hierarchy representation and normal flat representation
                    if issubclass(inference_class, sbm.HierarchicalInference):
                        sorted_block_assignments = []
                        for representation_level in representation:
                            sorted_block_assignments.extend(representation_level[node]
                                                            for node in sorted(representation_level))
                    else:
                        sorted_block_assignments = [representation[node] for node in sorted(representation)]

                    line.extend(sorted_block_assignments)

                    lines.append(line)

        # write file
        # mark as csv file if not already done
        if file_path[-4:] != ".csv":
            file_path += ".csv"

        with open(file_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file, dialect=csv.excel)
            writer.writerows(lines)

        return file_path

    def read_results_flat(self, file_path, is_evaluated_results=False):

        # get mapping
        network_mapping, inference_mapping, objective_mapping = self._create_mapping_dicts()

        # mark as csv file if not already done
        if file_path[-4:] != ".csv":
            file_path += ".csv"

        result = {}
        with open(file_path, "r", newline="") as csv_file:
            # skip header row
            if is_evaluated_results:
                formated_header = ",".join(self.NEW_HEADER_LINE_EVALUATED)
            else:
                formated_header = ",".join(self.NEW_HEADER_LINE)
            if csv_file.readline()[:len(formated_header)] != formated_header:
                new_format = False
            else:
                new_format = True
            reader = csv.reader(csv_file, dialect=csv.excel)

            for line in reader:
                network_information = line[0]
                inference_short_title = line[1]
                objective_short_title = line[2]
                number_of_blocks = int(line[3])
                if new_format:
                    offset = 1
                    start_from_true_partition = True if line[4] == 'T' else False
                else:
                    offset = 0
                    start_from_true_partition = False

                try:
                    network_test = network_mapping[network_information]
                except KeyError:
                    raise ValueError("Wrong network information")
                try:
                    inference_class = inference_mapping[inference_short_title]
                except KeyError:
                    raise ValueError("Wrong inference short title")
                try:
                    objective_function_class = objective_mapping[objective_short_title]
                except KeyError:
                    raise ValueError("Wrong objective short title")

                key = network_test, inference_class, None, None, objective_function_class, None, number_of_blocks, \
                    start_from_true_partition

                if is_evaluated_results:
                    evaluated_values = []
                    for raw_value in line[(4+offset):]:
                        evaluated_values.append(float(raw_value))

                    if key not in result:
                        result[key] = evaluated_values
                    else:
                        raise KeyError("Read key " + str(key) + " twice")

                else:
                    cpu_time = float(line[4 + offset])
                    node_moves = int(line[5 + offset])
                    deltas = int(line[6 + offset])
                    objective_function_value = float(line[7 + offset])
                    sorted_block_assignment = line[8 + offset:]

                    # build representation
                    # distinguish between hierarchy representation and normal flat representation
                    if issubclass(inference_class, sbm.HierarchicalInference):
                        representation = []
                        level_representation = {}
                        read_blocks = 0
                        for i, node in enumerate(sorted(network_test.network.nodes)):
                            level_representation[node] = int(sorted_block_assignment[i])

                        representation.append(level_representation)

                        read_blocks += len(level_representation)
                        while len(sorted_block_assignment) > read_blocks:
                            max_level = max(level_representation.values()) + 1
                            level_representation = {}

                            if len(sorted_block_assignment) < max_level + read_blocks:
                                raise ValueError("Representation not long enough")

                            for block in range(max_level):
                                level_representation[block] = int(sorted_block_assignment[read_blocks + block])

                            read_blocks += max_level
                            representation.append(level_representation)
                    else:
                        representation = {}
                        for i, node in enumerate(sorted(network_test.network.nodes)):
                            representation[node] = int(sorted_block_assignment[i])

                    if key not in result:
                        result[key] = []
                    result[key].append((representation, cpu_time, node_moves, deltas, objective_function_value))

        return result

    def determine_parallel_arguments(self, number_of_times=1, search_in_range=None, check_temp_files=False,
                                     check_simple_exclusion_file=True, temp_file_path=None,
                                     start_from_true_partition=False):
        keyed_arguments = {
            (network_test, inference_class, inference_creation_parameters, inference_execution_parameters,
             objective_function_class, objective_function_creation_parameters, start_from_true_partition):
                network_test.get_parameters_for_apply_inference_algorithm_on_graph_with_given_number_of_blocks(
                    inference_class,
                    objective_function_class,
                    self._get_search_range(network_test, search_in_range),
                    number_of_times,
                    start_from_true_partition,
                )
                + (inference_execution_parameters,
                   inference_creation_parameters,
                   objective_function_creation_parameters)

            for network_test in self.tests
            for inference_class in self.inference_classes
            for inference_creation_parameters in self.inference_creation_parameters.get(inference_class, [None])
            for inference_execution_parameters in self.inference_execution_parameters.get(
            inference_class, [None])
            for objective_function_class in self.objective_function_classes
            for objective_function_creation_parameters in
            self.objective_function_creation_parameters.get(objective_function_class, [None])
        }

        # if wanted check temp files
        if check_temp_files and os.path.exists(TEMP_DIRECTORY):
            done_keys = self._read_temp_files(search_in_range, number_of_times, temp_file_path, return_only_keys=True)
            # delete already finished results
            for key in done_keys:
                del keyed_arguments[key]

        if check_simple_exclusion_file:
            done_keys = self.read_simple_exclusion_file()
            # delete already finished results
            for key in done_keys:
                del keyed_arguments[key]

        return keyed_arguments

    def execute_tests_parallel(self, max_workers=1, timeout=10, add_results_to_internal_storage=True, arguments=None,
                               number_of_times=1, search_in_range=None, check_temp_files=True, temp_file_path=None,
                               start_from_true_partition=False,
                               ):
        if arguments is None:
            keyed_arguments = self.determine_parallel_arguments(
                number_of_times=number_of_times,
                search_in_range=search_in_range,
                check_temp_files=check_temp_files,
                temp_file_path=temp_file_path,
                start_from_true_partition=start_from_true_partition,
            )
        else:
            keyed_arguments = arguments

        results = {}

        if search_in_range is None:
            time_multiplier = 1
        else:
            time_multiplier = len(search_in_range)

        if self.use_flat_saving:
            save_method = self.save_results_flat
        else:
            save_method = None

        raw_results, errors_raw = parallel_execution(
            apply_inference_algorithm,
            keyed_arguments,
            max_workers=max_workers,
            maximum_time_per_function_call=timeout * number_of_times * time_multiplier,
            save_method=save_method,
            save_memory=self.save_memory
        )

        # load old data
        if check_temp_files and os.path.exists(TEMP_DIRECTORY):
            # load saved data, if no memory saving is performed only results of previous may aborted run (_0 file)
            # will be included
            # with memory saving all files will be read and included
            results = self._read_temp_files(search_in_range, number_of_times, only_old_data=not self.save_memory)

        # extend packed results
        for short_key in raw_results:
            results_by_number_of_blocks = raw_results[short_key]
            for number_of_blocks in results_by_number_of_blocks:
                results[short_key[:-1] + (number_of_blocks,) + short_key[-1:]] = \
                    results_by_number_of_blocks[number_of_blocks]

        if add_results_to_internal_storage:
            for key in results:
                # key is equal to long key without counter
                if key not in self.results_of_all_executions:
                    self.results_of_all_executions[key] = []
                self.results_of_all_executions[key].extend(results[key])

        # create new dictionary from returned information about keys with errors
        errors = {}
        for key in errors_raw:
            errors[key] = keyed_arguments[key]

        self._number_of_total_executions += number_of_times
        # delete temp files
        self._clean_temp_files()
        return results, errors

    def compare_results_with_ground_truth(self, with_zoom=True, evaluate_function="AMI_max"):

        for key in self.results_of_all_executions:
            network_test = key[0]
            inference_class = key[1]
            evaluated_results = []
            for single_result in self.results_of_all_executions[key]:
                if issubclass(inference_class, sbm.HierarchicalInference):
                    representation = single_result[0][0]
                    if with_zoom:
                        if len(single_result[0]) > 1:
                            raw_partition = single_result[0]
                            representation = {node:
                                                  raw_partition[1][raw_partition[0][node]] for node in raw_partition[0]}
                else:
                    representation = single_result[0]

                if evaluate_function == self.NORMALIZED_MUTUAL_INFORMATION:
                    evaluated_results.append(
                        network_test.compare_partition_with_ground_truth_nmi(SimplePartition(representation)))
                elif evaluate_function == self.ADJUSTED_MUTUAL_INFORMATION:
                    evaluated_results.append(network_test.compare_partition_with_ground_truth_ami(representation))
                else:
                    raise ValueError("No evaluate function with name" + str(evaluate_function))
            self.evaluated_results[key] = evaluated_results
        return self.evaluated_results

    def calculate_random_nmi(self, number_of_random_partitions, compressed=True):
        if compressed:
            random_nmi = []
            for test in self.tests:
                random_nmi.extend(test.calculate_random_partition_nmi(number_of_random_partitions))
        else:
            random_nmi = {}
            for test in self.tests:
                random_nmi[test] = test.calculate_random_partition_nmi(number_of_random_partitions)

        return random_nmi


class ResultHandler:

    def __init__(self, results_of_executions):
        self.Combination = namedtuple('Combination', ['test', 'inference', 'objective'])
        self.FullKey = namedtuple('FullKey',
                                  ['test',
                                   'inference',
                                   'inference_creation_parameter',
                                   'inference_execution_parameter',
                                   'objective',
                                   'objective_parameter',
                                   'number_of_blocks',
                                   'start_from_true_partition'])
        self.ModelSelectionKey = namedtuple('ModelSelectionKey',
                                            ['test', 'inference', 'objective', 'run_counter', 'model_selection_title'])
        self.Results = namedtuple('Results', ['representation',
                                              'cpu_time',
                                              'node_moves',
                                              'calculated_deltas',
                                              'objective_value'])

        self._results_of_executions = {}
        self._tests = []
        self._inference_classes = []
        self._objective_function_classes = []
        self._number_of_groups_per_combination = {}
        self._number_of_executions_per_combination = {}
        self._transformed_results_per_execution = {}
        self._results_per_execution = {}
        self.results_of_executions = results_of_executions

    @property
    def results_of_executions(self):
        return self._results_of_executions

    @property
    def tests(self):
        return self._tests

    @property
    def inference_classes(self):
        return self._inference_classes

    @property
    def objective_function_classes (self):
        return self._objective_function_classes

    @results_of_executions.setter
    def results_of_executions(self, results_of_executions):
        self._transformed_results_per_execution = {}
        self._results_per_execution = {}
        self._number_of_executions_per_combination = {}

        # determine present tests, inferences, objectives
        tests = set()
        inferences = set()
        objectives = set()
        for key in results_of_executions:
            new_key = self.FullKey(*key)
            network = new_key.test
            inference = new_key.inference
            objective = new_key.objective
            number_of_blocks = new_key.number_of_blocks
            start_from_true_partition = new_key.start_from_true_partition

            self._results_of_executions[new_key] = results_of_executions[key]

            tests.add(network)
            inferences.add(inference)
            objectives.add(objective)

            # care about number of groups per combination
            if not start_from_true_partition:
                short_key = self.Combination(network, inference, objective)

                if short_key not in self._number_of_groups_per_combination:
                    self._number_of_groups_per_combination[short_key] = []
                    self._number_of_executions_per_combination[short_key] = len(results_of_executions[key])
                elif self._number_of_executions_per_combination[short_key] != len(results_of_executions[key]):
                    raise ValueError("Different number of executions!")

                self._number_of_groups_per_combination[short_key].append(number_of_blocks)

        # transform to lists
        self._tests = list(tests)
        self._inference_classes = list(inferences)
        self._objective_function_classes = list(objectives)

        # sort number of groups per combination
        for short_key in self._number_of_groups_per_combination:
            self._number_of_groups_per_combination[short_key] = list(
                sorted(self._number_of_groups_per_combination[short_key]))

    def _transform_values_per_execution(self, test, inference, objective):
        short_key = self.Combination(test, inference, objective)

        if short_key in self._transformed_results_per_execution:
            return

        objective_function_values = []
        partition_representations = []

        objective_function_instance = objective(is_directed=test.network.is_directed())

        for number_of_groups in self._number_of_groups_per_combination[short_key]:
            key = self.FullKey(test, inference, None, None, objective, None, number_of_groups, False)

            for execution_counter, single_result in enumerate(self._results_of_executions[key]):
                if execution_counter == len(objective_function_values):
                    objective_function_values.append({})
                    partition_representations.append({})

                single_result = self.Results(*single_result)
                objective_function_values[execution_counter][number_of_groups] = single_result.objective_value
                partition_representations[execution_counter][number_of_groups] = single_result.representation

            # add partition in 1 block
            partition = sbm.NxPartition(test.network, number_of_blocks=1)
            one_block_value = objective_function_instance.calculate(partition)
            one_block_representation = partition.get_representation()
            for values in objective_function_values:
                values[1] = one_block_value
            for representations in partition_representations:
                representations[1] = one_block_representation

        self._transformed_results_per_execution[short_key] = (objective_function_values, partition_representations)

    def perform_model_selection(self, test, inference, objective, model_selection):
        short_key = self.Combination(test, inference, objective)

        self._transform_values_per_execution(test, inference, objective)

        objective_function_values, partition_representations = self._transformed_results_per_execution[short_key]

        # determine per run
        for run_counter in range(len(objective_function_values)):
            single_values = objective_function_values[run_counter]
            single_representations = partition_representations[run_counter]

            best_number_of_groups, value_of_best_group = model_selection.select_number_of_groups(
                test.network, single_values, single_representations)

            long_key = self.ModelSelectionKey(*(short_key + (run_counter, model_selection.title)))
            self._results_per_execution[long_key] = best_number_of_groups, value_of_best_group

    def perform_model_selection_for_all(self, tests=None, inferences=None, objectives=None, model_selections=None,
                                        with_output=False):

        total_number_of_selections = 1
        if tests is None:
            total_number_of_selections *= len(self._tests)
        else:
            total_number_of_selections *= len(tests)

        if inferences is None:
            total_number_of_selections *= len(self._inference_classes)
        else:
            total_number_of_selections *= len(inferences)

        if objectives is None:
            total_number_of_selections *= len(self._objective_function_classes)
        else:
            total_number_of_selections *= len(objectives)

        finished = 0

        for objective in self._objective_function_classes:
            if objectives is not None and objective not in objectives:
                continue

            _model_selections = sbm.get_possible_model_selection(objective)

            for test in self._tests:
                if tests is not None and test not in tests:
                    continue

                for inference in self._inference_classes:
                    if inferences is not None and inference not in inferences:
                        continue

                    for model_selection in _model_selections:
                        if model_selections is not None and model_selection not in model_selections:
                            continue

                        self.perform_model_selection(test, inference, objective, model_selection)

                    finished += 1

                    if with_output and finished%100 == 0:
                        print(round(finished/total_number_of_selections, 2), "completed")

    def get_selected_representations(self, test, inference, objective, model_selection=None, run_counter=None):
        if run_counter is not None:
            if model_selection is None:
                model_selections = sbm.get_possible_model_selection(objective)

                if len(model_selections) > 1:
                    raise ValueError("Model Selection not unique")

                model_selection = model_selections[0]

            long_key = self.ModelSelectionKey(test, inference, objective, run_counter, model_selection.title)
            selected_number_of_groups, model_selection_value = self._results_per_execution[long_key]

            short_key = self.Combination(test, inference, objective)
            return self._transformed_results_per_execution[short_key][1][run_counter][selected_number_of_groups]
        else:
            raise NotImplementedError()


# @formatter:off
class SingleNetworkSupplier:
    KARATE_CLUB_INFORMATION = "karate club graph"
    PLANTED_PARTITION_INFORMATION = "planted partition, nodes, P_in, P_out "
    STOCHASTIC_BLOCKMODEL_INFORMATION = "stochastic block model, nodes, edge matrix "
    LFR_INFORMATION = "mixing parameter, network instance"

    REAL_NETWORK_DATA_FOLDER = "Network Data"
    REAL_NETWORK_BLOCK_INFORMATION = "value"

    FOOTBALL_SUB_PATH = "football/football.gml"
    FOOTBALL_INFORMATION = "football"

    FOOTBALL_CORRECTED_SUB_PATH = "football_corrected/footballTSEinput.gml"
    FOOTBALL_CORRECTED_INFORMATION = "football corrected"

    POLITICAL_BLOGS_SUB_PATH = "polblogs/polblogs.gml"
    POLITICAL_BLOGS_INFORMATION = "polblogs"

    POLITICAL_BO0KS_SUB_PATH = "polbooks/polbooks.gml"
    POLITICAL_BO0KS_INFORMATION = "polbooks"

    @staticmethod
    def create_karate_club_test():
        karate_club = nx.karate_club_graph()
        true_partition = {}
        for node in karate_club:
            if karate_club.node[node]['club'] == 'Mr. Hi':
                true_partition[node] = 1
            else:
                true_partition[node] = 0
        information = SingleNetworkSupplier.KARATE_CLUB_INFORMATION
        return SingleNetworkTest(karate_club, true_partition, information)

    @staticmethod
    def create_test_with_the_planted_partition_model(number_of_groups, number_of_vertices_in_each_group,
                                                     edge_probability_in_group, edge_probability_between_groups):
        planted_partition_generator = PlantedPartitionGenerator(number_of_groups, number_of_vertices_in_each_group,
                                                                edge_probability_in_group,
                                                                edge_probability_between_groups)
        planted_partition_graph, number_of_blocks, returned_ground_truth = planted_partition_generator.generate(
            directed=False, seed=42)
        nodes_in_groups = [number_of_vertices_in_each_group for _ in range(number_of_groups)]
        information = [SingleNetworkSupplier.PLANTED_PARTITION_INFORMATION, nodes_in_groups, edge_probability_in_group,
                       edge_probability_between_groups]
        return SingleNetworkTest(planted_partition_graph, returned_ground_truth, information)

    @staticmethod
    def create_test_with_the_stochastic_block_model(number_of_blocks, nodes_per_block, edge_matrix,
                                                    type_of_edge_matrix, is_directed_edge_matrix):
        sbm_generator = SBMGenerator(number_of_blocks, nodes_per_block, edge_matrix, type_of_edge_matrix,
                                     is_directed_edge_matrix)
        sbm_graph, number_of_blocks, returned_ground_truth = sbm_generator.generate(directed=False, seed=42)
        information = [SingleNetworkSupplier.STOCHASTIC_BLOCKMODEL_INFORMATION, nodes_per_block, edge_matrix]
        return SingleNetworkTest(sbm_graph, returned_ground_truth, information)

    @staticmethod
    def create_tests_for_girvan_newman_benchmark(number_of_networks=10, directed=False):
        p_out = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
        p_in = []
        for i in range(len(p_out)):
            if directed:
                p_out[i] = p_out[i] * 128 / (2 * 6 * 32 * 32)
                p_in.append((128 * 4 - p_out[i] * 2 * 6 * 32 * 32) / (32 * 32))
            else:
                p_out[i] = p_out[i] * 128 / (2 * 6 * 32 * 32)
                p_in.append((128 * 16 - p_out[i] * 2 * 6 * 32 * 32) / (4 * 32 * 33))

        list_of_tests = []
        for i in range(len(p_in)):
            planted_partition_generator = PlantedPartitionGenerator(4, 32, p_in[i], p_out[i])
            for j in range(number_of_networks):
                planted_partition_graph, number_of_blocks, returned_ground_truth = planted_partition_generator.generate(
                    directed=False, seed=j)
                information = [SingleNetworkSupplier.PLANTED_PARTITION_INFORMATION,
                               [32, 32, 32, 32], j, p_in[i], p_out[i]]
                list_of_tests.append(SingleNetworkTest(planted_partition_graph, returned_ground_truth, information))
        return list_of_tests

    @staticmethod
    def create_tests_for_girvan_newman_benchmark_extended(number_of_networks=10, directed=False):
        """"Includes extended P_out range"""
        p_out = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9.0, 9.5, 10, 10.5,
                 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
        p_in = []
        for i in range(len(p_out)):
            if directed:
                p_out[i] = p_out[i] * 128 / (2 * 6 * 32 * 32)
                p_in.append((128 * 4 - p_out[i] * 2 * 6 * 32 * 32) / (32 * 32))
            else:
                p_out[i] = p_out[i] * 128 / (2 * 6 * 32 * 32)
                p_in.append((128 * 16 - p_out[i] * 2 * 6 * 32 * 32) / (4 * 32 * 33))

        list_of_tests = []
        for i in range(len(p_in)):
            planted_partition_generator = PlantedPartitionGenerator(4, 32, p_in[i], p_out[i])
            for j in range(number_of_networks):
                planted_partition_graph, number_of_blocks, returned_ground_truth = planted_partition_generator.generate(
                    directed=False, seed=j)
                information = [SingleNetworkSupplier.PLANTED_PARTITION_INFORMATION,
                               [32, 32, 32, 32], j, p_in[i], p_out[i]]
                list_of_tests.append(SingleNetworkTest(planted_partition_graph, returned_ground_truth, information))
        return list_of_tests

    @staticmethod
    def create_lfr_benchmark_tests(folder_path, mixing_parameters, number_of_networks, read_networks_starting_with=0):
        """
        Read data from files created by C++ program and create SingleNetworkTest from the network
        and community information
        :param folder_path: path to folder with different files
        :param mixing_parameters: iterable with identifier of mixing parameters
        :param number_of_networks: how many different instances per mixing parameter where generated
        :param read_networks_starting_with: add networks instances starting from this number
        :return: list of SingleNetworkTest
        """
        community_file_init = "community"
        network_file_init = "network"
        file_ending = ".dat"

        tests = []
        for mixing_parameter in mixing_parameters:
            for network_counter in range(read_networks_starting_with, read_networks_starting_with + number_of_networks):
                community_file_path = folder_path + "/" + community_file_init \
                                      + "_" + str(mixing_parameter) + "_" + str(network_counter) + file_ending
                network_file_path = folder_path + "/" + network_file_init \
                                    + "_" + str(mixing_parameter) + "_" + str(network_counter) + file_ending

                try:
                    tests.append(SingleNetworkSupplier.create_singe_lfr_test_case(
                        network_file_path, community_file_path,
                        information=[SingleNetworkSupplier.LFR_INFORMATION, mixing_parameter, network_counter]))
                except AssertionError:
                    print("Error at", network_file_path, community_file_path)
                    break

        return tests

    @staticmethod
    def create_singe_lfr_test_case(network_file_path, community_file_path, information=None):
        # read graph
        graph = nx.Graph()
        with open(network_file_path) as network_file:
            for line in csv.reader(network_file, delimiter="\t"):
                assert len(line) == 2
                # cast to int and make node numbers in the range 0 to N-1
                graph.add_edge(int(line[0]) - 1, int(line[1]) - 1)

        # read true community
        true_partition = {}
        with open(community_file_path) as community_file:
            for line in csv.reader(community_file, delimiter="\t"):
                assert len(line) == 2
                true_partition[int(line[0]) - 1] = int(line[1]) - 1

        return SingleNetworkTest(graph, true_partition, information=information)

    @staticmethod
    def create_real_network_tests(include_football=True,
                                  include_corrected_football=True,
                                  include_political_blogs=True,
                                  include_political_books=True,
                                  return_largest_weakly_connected_component=True
                                  ):
        tests = []

        if include_football:
            file_path = SingleNetworkSupplier.REAL_NETWORK_DATA_FOLDER + "/" + SingleNetworkSupplier.FOOTBALL_SUB_PATH
            try:
                read_graph = nx.read_gml(file_path)
            except FileNotFoundError:
                print(""""No football data found. Please download the file from the internet, e.g. ﻿
                          http://www-personal.umich.edu/~mejn/netdata/ """)
                raise

            # already connected graph

            # create true partition
            representation = {}
            for node, block in read_graph.nodes(data=SingleNetworkSupplier.REAL_NETWORK_BLOCK_INFORMATION):
                if block is None:
                    raise ValueError("Read football network miss block information of node" + str(node))
                representation[node] = block

            tests.append(SingleNetworkTest(read_graph,
                                           representation,
                                           information=SingleNetworkSupplier.FOOTBALL_INFORMATION))

        if include_corrected_football:
            file_path = SingleNetworkSupplier.REAL_NETWORK_DATA_FOLDER + "/" \
                        + SingleNetworkSupplier.FOOTBALL_CORRECTED_SUB_PATH
            try:
                read_graph = nx.read_gml(file_path)
            except FileNotFoundError:
                print(""""No corrected football data found. The author of 
                          `Clique Graphs and Overlapping Communities` has the file on his website """)
                raise

            # already connected graph

            # create true partition
            representation = {}
            for node, block in read_graph.nodes(data=SingleNetworkSupplier.REAL_NETWORK_BLOCK_INFORMATION):
                if block is None:
                    raise ValueError("Read football network miss block information of node" + str(node))
                representation[node] = block

            tests.append(SingleNetworkTest(read_graph,
                                           representation,
                                           information=SingleNetworkSupplier.FOOTBALL_CORRECTED_INFORMATION))

        if include_political_blogs:
            file_path = SingleNetworkSupplier.REAL_NETWORK_DATA_FOLDER + "/" \
                        + SingleNetworkSupplier.POLITICAL_BLOGS_SUB_PATH
            try:
                read_graph = nx.read_gml(file_path)
            except nx.NetworkXError:
                print("Insert multigraph 1 in the header! We will cast it to a normal directed graph later.")
                raise
            except FileNotFoundError:
                print(""""No political blocks data found. Please download the file from the internet, e.g. ﻿
                          http://www-personal.umich.edu/~mejn/netdata/ """)
                raise

            graph = nx.Graph(read_graph)

            if return_largest_weakly_connected_component:
                graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

            # create true partition
            representation = {}
            for node, block in graph.nodes(data=SingleNetworkSupplier.REAL_NETWORK_BLOCK_INFORMATION):
                if block is None:
                    raise ValueError("Read political blogs network miss block information of node" + str(node))
                representation[node] = block

            tests.append(SingleNetworkTest(graph,
                                           representation,
                                           information=SingleNetworkSupplier.POLITICAL_BLOGS_INFORMATION))

            graph = nx.DiGraph(read_graph)
            if return_largest_weakly_connected_component:
                graph = graph.subgraph(max(nx.weakly_connected_components(graph), key=len)).copy()

            tests.append(SingleNetworkTest(graph,
                                           representation,
                                           information="directed " + SingleNetworkSupplier.POLITICAL_BLOGS_INFORMATION))

        if include_political_books:
            file_path = SingleNetworkSupplier.REAL_NETWORK_DATA_FOLDER + "/" \
                        + SingleNetworkSupplier.POLITICAL_BO0KS_SUB_PATH
            try:
                read_graph = nx.read_gml(file_path)
            except FileNotFoundError:
                print(""""No political books data found. Please download the file from the internet, e.g. ﻿
                          http://www-personal.umich.edu/~mejn/netdata/ """)
                raise

            # graph is connected...

            # create true partition
            representation = {}
            for node, block in read_graph.nodes(data=SingleNetworkSupplier.REAL_NETWORK_BLOCK_INFORMATION):
                if block is None:
                    raise ValueError("Read political books network miss block information of node" + str(node))
                # correct coding to number
                if block == 'l':
                    # liberal
                    block = 0
                elif block == "n":
                    # neutral
                    block = 1
                elif block == 'c':
                    # conservative
                    block = 2
                representation[node] = block

            tests.append(SingleNetworkTest(read_graph,
                                           representation,
                                           information=SingleNetworkSupplier.POLITICAL_BO0KS_INFORMATION))
        return tests
