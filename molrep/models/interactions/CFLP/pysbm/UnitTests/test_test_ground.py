import math
import unittest as ut

# noinspection SpellCheckingInspection
from pysbm import test_ground as sbmt

from sklearn.metrics.cluster import adjusted_mutual_info_score


class TestSimplePartition(ut.TestCase):
    """Test SimplePartition"""

    def setUp(self):
        """Prepare some simple partitions"""

        self.all_in_one = sbmt.SimplePartition({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
        self.single_groups = sbmt.SimplePartition({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})

    def test_filling(self):
        """Create and assign different Partitions"""
        self.assertEqual(self.all_in_one.node_to_block, {k: v for k, v in enumerate([0, 0, 0, 0, 0])})
        self.assertEqual(self.single_groups.node_to_block, {k: v for k, v in enumerate(range(6))})

        self.all_in_one.node_to_block = self.single_groups.node_to_block

        self.assertEqual(self.all_in_one.node_to_block, {k: v for k, v in enumerate(range(6))})

    def test_getters(self):
        """Test reading the information from simple partitions"""
        # test the partition with all in one block
        self.assertEqual(self.all_in_one.number_of_blocks, 1)
        self.assertEqual(self.all_in_one.get_number_of_nodes_in_block(0), 5)
        self.assertEqual(self.all_in_one.get_nodes_of_block(0), set(range(5)))
        self.assertEqual(self.all_in_one.number_of_nodes, 5)

        # test the partition with all in single blocks
        self.assertEqual(self.single_groups.number_of_blocks, 6)

        for block in range(self.single_groups.number_of_blocks):
            self.assertEqual(self.single_groups.get_number_of_nodes_in_block(block), 1)
            self.assertEqual(self.single_groups.get_nodes_of_block(block), {block})
        self.assertEqual(self.single_groups.number_of_nodes, 6)

    def test_representations(self):
        """Test str and repr casting"""
        self.assertEqual(str(self.all_in_one), str({k: v for k, v in enumerate([0, 0, 0, 0, 0])}))
        self.assertEqual(str(self.single_groups), str({k: v for k, v in enumerate(range(6))}))

        self.assertEqual(repr(self.all_in_one), "SimplePartition({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})")
        self.assertEqual(repr(self.single_groups), "SimplePartition({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})")


class TestNormalizedMutualInformation(ut.TestCase):
    """Test Normalized Mutual Information"""

    def setUp(self):
        self.evaluator = sbmt.NormalizedMutualInformation({0: 0, 1: 0, 2: 1, 3: 1})

        # Create some partitions
        self.four_partitions = []
        self.four_partitions.append([0, 0, 0, 0])
        self.four_partitions.append([0, 0, 0, 1])
        self.four_partitions.append([0, 0, 1, 1])
        self.four_partitions.append([0, 0, 1, 2])
        self.four_partitions.append([0, 1, 1, 1])
        self.four_partitions.append([0, 1, 1, 2])
        self.four_partitions.append([0, 1, 2, 2])
        self.four_partitions.append([0, 1, 2, 3])
        # and transform them into class representation
        for i, partition in enumerate(self.four_partitions):
            self.four_partitions[i] = sbmt.SimplePartition({k: v for k, v in enumerate(partition)})
    def test_base_class(self):
        """ Test base class abstract methods"""
        evaluator = sbmt.EvaluateResult()

        with self.assertRaises(NotImplementedError):
            evaluator.evaluate([0, 1])

    def test_entropy(self):
        """Test some entropy values"""
        #        all in one group
        self.assertEqual(self.evaluator.entropy(self.four_partitions[0]), 0)

        #        one node in another group
        #           half half partition
        self.assertEqual(self.evaluator.entropy(self.four_partitions[1]),
                         -3.0 / 4 * math.log(3.0 / 4) - 1.0 / 4 * math.log(1.0 / 4))

        # half half partition
        self.assertEqual(self.evaluator.entropy(self.four_partitions[2]), -math.log(1.0 / 2))

        # every node in a single group
        self.assertEqual(self.evaluator.entropy(self.four_partitions[7]), -math.log(1.0 / 4))

    def test_mutual_information(self):
        """Test mutual information"""
        #        one own partition it should give the maximum value which is the entropy
        self.assertEqual(self.evaluator.mutual_information(self.four_partitions[2]),
                         self.evaluator.entropy(self.four_partitions[2]))
        #        same on the partition with exchanged block numbers
        self.assertEqual(self.evaluator.mutual_information(sbmt.SimplePartition({0: 1, 1: 1, 2: 0, 3: 0})),
                         self.evaluator.entropy(self.four_partitions[2]))

        for i, partition in enumerate(self.four_partitions):
            self.assertLessEqual(self.evaluator.mutual_information(self.four_partitions[i]),
                                 self.evaluator.entropy(self.four_partitions[2]))

    def test_evaluate(self):
        """Test basic behaviour of normalized mutual information"""

        #        with the same partition return 1
        self.assertEqual(self.evaluator.evaluate(self.four_partitions[2]), 1)
        #        same for shifted block labels
        self.assertEqual(self.evaluator.evaluate(sbmt.SimplePartition({0: 1, 1: 1, 2: 0, 3: 0})), 1)

        #        all others return values smaller than 1
        for i, partition in enumerate(self.four_partitions):
            if i != 2:
                self.assertLess(self.evaluator.evaluate(self.four_partitions[i]),
                                1)


class TestPlantedPartition(ut.TestCase):

    def test_simple_partitions(self):
        # all edges in group
        planted_partition_generator = sbmt.PlantedPartitionGenerator(2, 5, 1, 0)
        graph, number_of_groups, true_partition = planted_partition_generator.generate(directed=False, seed=42)
        self.assertEqual(number_of_groups, 2)
        self.assertEqual(true_partition, {x: 0 if x < 5 else 1 for x in range(10)})
        edges = [(x, y) for x in range(5) for y in range(x, 5)]
        edges.extend([(x, y) for x in range(5, 10) for y in range(x, 10)])
        self.assertEqual(list(graph.edges()), edges)

        graph, number_of_groups, true_partition = planted_partition_generator.generate(directed=True, seed=42)
        self.assertEqual(number_of_groups, 2)
        self.assertEqual(true_partition, {x: 0 if x < 5 else 1 for x in range(10)})
        edges = [(x, y) for x in range(5) for y in range(5)]
        edges.extend([(x, y) for x in range(5, 10) for y in range(5, 10)])
        self.assertEqual(list(graph.edges()), edges)

        # all edges between group
        planted_partition_generator = sbmt.PlantedPartitionGenerator(2, 5, 0, 1)
        graph, number_of_groups, true_partition = planted_partition_generator.generate(directed=False, seed=42)
        self.assertEqual(number_of_groups, 2)
        self.assertEqual(true_partition, {x: 0 if x < 5 else 1 for x in range(10)})
        edges = [(x, y) for x in range(5) for y in range(5, 10)]
        self.assertEqual(list(graph.edges()), edges)

        graph, number_of_groups, true_partition = planted_partition_generator.generate(directed=True, seed=42)
        self.assertEqual(number_of_groups, 2)
        self.assertEqual(true_partition, {x: 0 if x < 5 else 1 for x in range(10)})
        edges = [(x, y) for x in range(5) for y in range(5, 10)]
        edges.extend([(x, y) for x in range(5, 10) for y in range(5)])
        self.assertEqual(list(graph.edges()), edges)

    def test_check_probabilities(self):
        # check if counts match
        true_partition = {x: 0 if x < 100 else 1 for x in range(200)}
        planted_partition_generator = sbmt.PlantedPartitionGenerator(2, 100, .5, 0)
        graph, number_of_groups, partition = planted_partition_generator.generate(directed=False, seed=42)
        self.assertEqual(number_of_groups, 2)
        self.assertEqual(partition, true_partition)
        between_groups = 0
        in_groups = 0
        for from_node, to_node in graph.edges():
            if partition[from_node] == partition[to_node]:
                in_groups += 1
            else:
                between_groups += 1
        # denominator = probability*number_of_groups*number_of_nodes_per_group*(number_of_nodes_per_group)/2
        self.assertAlmostEqual(in_groups / (2 * 100 * 101 / 2), .5, delta=.1)
        self.assertEqual(between_groups, 0)

        graph, number_of_groups, partition = planted_partition_generator.generate(directed=True, seed=42)
        self.assertEqual(number_of_groups, 2)
        self.assertEqual(partition, true_partition)
        between_groups = 0
        in_groups = 0
        for from_node, to_node in graph.edges():
            if partition[from_node] == partition[to_node]:
                in_groups += 1
            else:
                between_groups += 1
        self.assertAlmostEqual(in_groups / (100 * 100 * 2), .5, delta=.1)
        self.assertEqual(between_groups, 0)

        # edges between groups
        planted_partition_generator = sbmt.PlantedPartitionGenerator(2, 100, 0, 0.5)
        graph, number_of_groups, partition = planted_partition_generator.generate(directed=False, seed=42)
        self.assertEqual(number_of_groups, 2)
        self.assertEqual(partition, true_partition)
        between_groups = 0
        in_groups = 0
        for from_node, to_node in graph.edges():
            if partition[from_node] == partition[to_node]:
                in_groups += 1
            else:
                between_groups += 1
        self.assertEqual(in_groups, 0)
        self.assertAlmostEqual(between_groups / (100 * 100), 0.5, delta=.1)

        graph, number_of_groups, partition = planted_partition_generator.generate(directed=True, seed=42)
        self.assertEqual(number_of_groups, 2)
        self.assertEqual(partition, true_partition)
        between_groups = 0
        in_groups = 0
        for from_node, to_node in graph.edges():
            if partition[from_node] == partition[to_node]:
                in_groups += 1
            else:
                between_groups += 1
        self.assertEqual(in_groups, 0)
        self.assertAlmostEqual(between_groups / (2 * 100 * 100), 0.5, delta=.1)

    def test_check_probabilities_with_more_groups(self):
        # check if counts match
        true_partition = {x: 0 if x < 100 else 1 for x in range(200)}
        for x in range(200, 300):
            true_partition[x] = 2
        planted_partition_generator = sbmt.PlantedPartitionGenerator(3, 100, .5, 0)
        graph, number_of_groups, partition = planted_partition_generator.generate(directed=False, seed=42)
        self.assertEqual(number_of_groups, 3)
        self.assertEqual(partition, true_partition)
        between_groups = 0
        in_groups = 0
        for from_node, to_node in graph.edges():
            if partition[from_node] == partition[to_node]:
                in_groups += 1
            else:
                between_groups += 1
        # denominator = probability*number_of_groups*number_of_nodes_per_group*(number_of_nodes_per_group)/2
        self.assertAlmostEqual(in_groups / (3 * 100 * 101 / 2), .5, delta=.1)
        self.assertEqual(between_groups, 0)

        graph, number_of_groups, partition = planted_partition_generator.generate(directed=True, seed=42)
        self.assertEqual(number_of_groups, 3)
        self.assertEqual(partition, true_partition)
        between_groups = 0
        in_groups = 0
        for from_node, to_node in graph.edges():
            if partition[from_node] == partition[to_node]:
                in_groups += 1
            else:
                between_groups += 1
        self.assertAlmostEqual(in_groups / (100 * 100 * 3), .5, delta=.1)
        self.assertEqual(between_groups, 0)

        # edges between groups
        planted_partition_generator = sbmt.PlantedPartitionGenerator(3, 100, 0, 0.5)
        graph, number_of_groups, partition = planted_partition_generator.generate(directed=False, seed=42)
        self.assertEqual(number_of_groups, 3)
        self.assertEqual(partition, true_partition)
        between_groups = 0
        in_groups = 0
        for from_node, to_node in graph.edges():
            if partition[from_node] == partition[to_node]:
                in_groups += 1
            else:
                between_groups += 1
        self.assertEqual(in_groups, 0)
        self.assertAlmostEqual(between_groups / (100 * 100 * 3), 0.5, delta=.1)

        graph, number_of_groups, partition = planted_partition_generator.generate(directed=True, seed=42)
        self.assertEqual(number_of_groups, 3)
        self.assertEqual(partition, true_partition)
        between_groups = 0
        in_groups = 0
        for from_node, to_node in graph.edges():
            if partition[from_node] == partition[to_node]:
                in_groups += 1
            else:
                between_groups += 1
        self.assertEqual(in_groups, 0)
        self.assertAlmostEqual(between_groups / (2 * 3 * 100 * 100), 0.5, delta=.1)
