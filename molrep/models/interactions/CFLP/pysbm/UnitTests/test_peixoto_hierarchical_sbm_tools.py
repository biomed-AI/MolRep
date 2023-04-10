import math
from unittest import TestCase

import numpy as np

from pysbm.sbm.peixotos_hierarchical_sbm_tools import init_number_of_restricted_partitions, \
    NUMBER_OF_RESTRICTED_PARTITIONS_PRECOMPUTED_LIMIT, \
    get_log_number_of_restricted_partitions

from pysbm.sbm.peixotos_hierarchical_sbm_tools_full import get_number_of_restricted_partitions


class TestBasicFunctions(TestCase):

    # # noinspection PyPep8Naming
    # def __init__(self, methodName='runTest'):
    #     super(TestBasicFunctions, self).__init__(methodName)
    #     sbm.peixotosbm.NUMBER_OF_RESTRICTED_PARTITIONS_PRECOMPUTED_LIMIT = 9
    #     sbm.peixotosbm.NUMBER_OF_RESTRICTED_PARTITIONS_MATRIX = init_number_of_restricted_partitions(9)

    def setUp(self):
        # correct table from the internet https://dlmf.nist.gov/26.9
        self.first_numbers_of_restricted_partitions = np.array([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3,
            0, 1, 3, 4, 5, 5, 5, 5, 5, 5, 5,
            0, 1, 3, 5, 6, 7, 7, 7, 7, 7, 7,
            0, 1, 4, 7, 9, 10, 11, 11, 11, 11, 11,
            0, 1, 4, 8, 11, 13, 14, 15, 15, 15, 15,
            0, 1, 5, 10, 15, 18, 20, 21, 22, 22, 22,
            0, 1, 5, 12, 18, 23, 26, 28, 29, 30, 30,
            0, 1, 6, 14, 23, 30, 35, 38, 40, 41, 42]).reshape(11, 11)

    def test_init_number_of_restricted_partitions(self):

        np.testing.assert_array_equal(self.first_numbers_of_restricted_partitions,
                                      init_number_of_restricted_partitions(11, save_space=False))

        # same with duplicates deleted
        test_results_spare = [
            [1],
            [0, 1],
            [0, 1, 2],
            [0, 1, 2, 3],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 5, 6, 7],
            [0, 1, 4, 7, 9, 10, 11],
            [0, 1, 4, 8, 11, 13, 14, 15],
            [0, 1, 5, 10, 15, 18, 20, 21, 22],
            [0, 1, 5, 12, 18, 23, 26, 28, 29, 30],
            [0, 1, 6, 14, 23, 30, 35, 38, 40, 41, 42]]

        real_spare = init_number_of_restricted_partitions(11, save_space=True)
        for i, values in enumerate(test_results_spare):
            self.assertEqual(test_results_spare[i], real_spare[i])

    def test_get_number_of_restricted_partitions_precomputed(self):
        for m, values in enumerate(self.first_numbers_of_restricted_partitions):
            for n, correct_value in enumerate(values):
                self.assertEqual(get_number_of_restricted_partitions(m, n), correct_value)

    def test_get_number_of_restricted_partitions_not_precomputed(self):
        new_max = NUMBER_OF_RESTRICTED_PARTITIONS_PRECOMPUTED_LIMIT + 100
        real_spare = init_number_of_restricted_partitions(new_max,
                                                          save_space=True)
        # read number above limit and only compare log values
        self.assertAlmostEqual(math.log(real_spare[new_max - 1][new_max - 1]),
                               math.log(get_number_of_restricted_partitions(new_max - 1, new_max - 1)), delta=.1)

    def test_get_log_number_of_restricted_partitions_precomputed(self):
        for m, values in enumerate(self.first_numbers_of_restricted_partitions):
            for n, correct_value in enumerate(values):
                if correct_value != 0:
                    self.assertEqual(get_log_number_of_restricted_partitions(m, n), math.log(correct_value))
                else:
                    self.assertEqual(get_log_number_of_restricted_partitions(m, n), float('-inf'))

    def test_get_log_number_of_restricted_partitions_not_precomputed(self):
        new_max = NUMBER_OF_RESTRICTED_PARTITIONS_PRECOMPUTED_LIMIT + 100
        real_spare = init_number_of_restricted_partitions(new_max,
                                                          save_space=True)
        # read number above limit and only compare log values
        self.assertAlmostEqual(math.log(real_spare[new_max - 1][new_max - 1]),
                               get_log_number_of_restricted_partitions(new_max - 1, new_max - 1), delta=.1)
