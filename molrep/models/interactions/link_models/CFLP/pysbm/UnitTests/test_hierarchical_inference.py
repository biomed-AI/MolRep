import unittest as ut

import networkx as nx

from pysbm.sbm.hierarchical_inference import HierarchicalInference
from pysbm.sbm.hierarchical_inference import PeixotoHierarchicalInference
from pysbm.sbm.nxpartitiongraphbased import NxHierarchicalPartition
from pysbm.sbm.peixotos_hierarchical_sbm import LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmWrapper


class TestHierarchicalInference(ut.TestCase):
    def setUp(self):
        self.karate = nx.karate_club_graph()
        self.karate_partition = NxHierarchicalPartition(graph=self.karate,
                                                        number_of_blocks=1,
                                                        save_degree_distributions=True,
                                                        representation=[{node: 0 for node in self.karate}]
                                                        )

        self.objective_function = LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmWrapper(is_directed=False)
        self.inference = HierarchicalInference(self.karate, self.objective_function, self.karate_partition,
                                               self.objective_function.calculate_delta_actual_level_removed)

    def test_infer_stepwise(self):
        # print(self.objective_function.calculate(self.karate_partition), self.karate_partition.B,
        #       self.karate_partition.max_level, self.karate_partition.get_representation())
        old_value = self.objective_function.calculate(self.karate_partition)
        for _ in range(100):
            try:
                self.inference.infer_stepwise()
                new_value = self.objective_function.calculate(self.karate_partition)
                self.assertGreaterEqual(new_value, old_value)
                old_value = new_value
            except StopIteration:
                break
            # print(self.objective_function.calculate(self.karate_partition), self.karate_partition.B,
            #       self.karate_partition.max_level, self.karate_partition.get_representation())

        new_value = self.objective_function.calculate(self.karate_partition)
        self.assertGreaterEqual(new_value, old_value)

        # print("Final", self.objective_function.calculate(self.karate_partition), self.karate_partition.B,
        #       self.karate_partition.max_level, self.karate_partition.get_representation())

    def test_infer_sbm(self):
        old_value = self.objective_function.calculate(self.karate_partition)
        self.inference.infer_stochastic_block_model()
        new_value = self.objective_function.calculate(self.karate_partition)
        self.assertGreater(new_value, old_value)
        # print(new_value, old_value)


class TestPeixotoHierarchicalInference(ut.TestCase):
    def setUp(self):
        self.karate = nx.karate_club_graph()
        self.karate_partition = NxHierarchicalPartition(graph=self.karate,
                                                        number_of_blocks=1,
                                                        save_degree_distributions=True,
                                                        representation=[{node: 0 for node in self.karate}]
                                                        )

        self.objective_function = LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmWrapper(is_directed=False)
        self.inference = PeixotoHierarchicalInference(self.karate, self.objective_function, self.karate_partition,
                                                      self.objective_function.calculate_delta_actual_level_removed)

    def test_infer_stepwise(self):
        # print(self.objective_function.calculate(self.karate_partition), self.karate_partition.B,
        #       self.karate_partition.max_level, self.karate_partition.get_representation())
        old_value = self.objective_function.calculate(self.karate_partition)
        for _ in range(100):
            try:
                self.inference.infer_stepwise()
                new_value = self.objective_function.calculate(self.karate_partition)
                self.assertGreaterEqual(new_value, old_value)
                old_value = new_value
            except StopIteration:
                break
            # print(self.objective_function.calculate(self.karate_partition), self.karate_partition.B, self.inference.viewed_level, self.inference._level_status,
            #       self.karate_partition.max_level, self.karate_partition.get_representation())

        new_value = self.objective_function.calculate(self.karate_partition)
        # print(_, self.karate_partition.B)
        # print("new", new_value, "old", old_value)
        # self.assertGreaterEqual(new_value, old_value)

        # print("Final", self.objective_function.calculate(self.karate_partition), self.karate_partition.B,
        #       self.karate_partition.max_level, self.karate_partition.get_representation())

    def test_infer_sbm(self):
        old_value = self.objective_function.calculate(self.karate_partition)
        self.inference.infer_stochastic_block_model()
        new_value = self.objective_function.calculate(self.karate_partition)
        self.assertGreater(new_value, old_value)
        # print(new_value, old_value)

    def test_error_on_delete(self):
        # Reproduce some index error
        test_inference = DummyInference()
        test_inference.infer_stochastic_block_model()


class DummyInference(HierarchicalInference):

    @staticmethod
    def dummy_function(*args):
        if len(args) > 0:
            return 0

    def __init__(self, number_of_successful_adds=2, number_of_successful_resize=5, number_of_successful_delete=3):
        graph = nx.karate_club_graph()
        objective_function = LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmWrapper(is_directed=False)
        hierarchical_partition = NxHierarchicalPartition(graph)
        delta_objective_function_for_level_removal = self.dummy_function
        super(DummyInference, self).__init__(graph, objective_function, hierarchical_partition,
                                             delta_objective_function_for_level_removal)

        self.number_of_successful_adds = number_of_successful_adds
        self.number_of_successful_resize = number_of_successful_resize
        self.number_of_successful_delete = number_of_successful_delete

    def add_level(self, level):
        # print("Add", bool(self.number_of_successful_adds))
        if self.number_of_successful_adds > 0:
            self.number_of_successful_adds -= 1
            self.partition.max_level += 1
            return True
        return False

    def resize_level(self, level):
        # print("Resize", bool(self.number_of_successful_resize))
        if self.number_of_successful_resize > 0:
            self.number_of_successful_resize -= 1
            return True
        return False

    def delete_level(self, level):
        # print("Delete", bool(self.number_of_successful_delete))
        if self.number_of_successful_delete > 0:
            self.number_of_successful_delete -= 1
            self.partition.max_level -= 1
            return True
        return False
