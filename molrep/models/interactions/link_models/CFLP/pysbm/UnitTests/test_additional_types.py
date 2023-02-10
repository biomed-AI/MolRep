import unittest as ut

import six

from pysbm import additional_types as at


class TestSetDict(ut.TestCase):
    """ Test Class SetDict """

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestSetDict, self).__init__(methodName)

    def setUp(self):
        self.list = at.SetDict()
        self.list.add(1)
        self.list.add(2)
        self.list.add(3)

    def test_remove(self):
        """ Test removing elements"""
        with self.assertRaises(KeyError):
            self.list.remove(4)

        self.list.remove(2)
        six.assertCountEqual(self, [1, 3], self.list.items)
        with self.assertRaises(KeyError):
            self.list.remove(2)

        self.list.remove(1)
        six.assertCountEqual(self, [3], self.list.items)
        with self.assertRaises(KeyError):
            self.list.remove(1)

        self.list.remove(3)
        six.assertCountEqual(self, [], self.list.items)

        with self.assertRaises(KeyError):
            self.list.remove(1)
        with self.assertRaises(KeyError):
            self.list.remove(2)
        with self.assertRaises(KeyError):
            self.list.remove(3)

    def test_choose_random(self):
        """Test random selection"""
        self.assertIn(self.list.choose_random(), [1, 2, 3])
        self.assertIn(self.list.choose_random(), [1, 2, 3])
        self.assertIn(self.list.choose_random(), [1, 2, 3])
        self.assertIn(self.list.choose_random(), [1, 2, 3])

        self.list.add(4)
        self.assertIn(self.list.choose_random(), [1, 2, 3, 4])
        self.assertIn(self.list.choose_random(), [1, 2, 3, 4])
        self.assertIn(self.list.choose_random(), [1, 2, 3, 4])

        self.list.remove(2)
        self.assertIn(self.list.choose_random(), [1, 4, 3])
        self.assertIn(self.list.choose_random(), [1, 4, 3])
        self.assertIn(self.list.choose_random(), [1, 4, 3])
        self.assertIn(self.list.choose_random(), [1, 4, 3])

        #        test that it really gives random values and not e.g. always the first value
        self.list.add(2)
        hits = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        number_of_tries = 10000
        for _ in range(number_of_tries):
            hits[self.list.choose_random()] += 1

        for hit in hits:
            hits[hit] /= number_of_tries
            self.assertAlmostEqual(hits[hit], 1.0 / len(hits), delta=0.1)

    def test_add(self):
        """Test adding elements"""
        self.list.add(4)
        six.assertCountEqual(self, [1, 2, 3, 4], self.list.items)

        self.assertEqual(str([1, 2, 3, 4]), str(self.list))

        with self.assertRaises(KeyError):
            self.list.add(4)

    def test_copy(self):
        """Test copying list"""
        new_list = self.list.copy()

        self.assertNotEqual(id(new_list), id(self.list))

        new_list.add(4)
        six.assertCountEqual(self, [1, 2, 3], self.list.items)
        six.assertCountEqual(self, [1, 2, 3, 4], new_list.items)

        self.assertEqual(3, len(self.list.item_to_position))
        self.assertEqual(4, len(new_list.item_to_position))


class TestListDict(TestSetDict):
    """ Test Class ListDict """

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestListDict, self).__init__(methodName)

    def setUp(self):
        self.list = at.ListDict()
        self.list.add(1)
        self.list.add(2)
        self.list.add(3)

    def test_add(self):
        """Test adding elements"""
        self.list.add(4)
        six.assertCountEqual(self, [1, 2, 3, 4], self.list.items)
        self.assertEqual(str([1, 2, 3, 4]), str(self.list))

        self.list.add(4)
        six.assertCountEqual(self, [1, 2, 3, 4, 4], self.list.items)

    def test_random_extended(self):
        """Test if weighted sampling works"""

        # create list with 1, 2x2, 3x3, 4x4 = 10 entries
        self.list.add(2)
        self.list.add(3)
        self.list.add(3)
        self.list.add(4)
        self.list.add(4)
        self.list.add(4)
        self.list.add(4)

        hits = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        number_of_tries = 10000
        for _ in range(number_of_tries):
            hits[self.list.choose_random()] += 1

        for hit in hits:
            hits[hit] /= number_of_tries
            self.assertAlmostEqual(hits[hit], hit / 10.0, delta=0.1)

    def test_remove_extended(self):
        """Test removing of multiple elements"""
        self.list.add(4)
        self.list.add(4)
        self.list.add(5)
        self.list.add(6)
        self.list.add(7)

        self.list.remove(4)
        self.list.remove(6)
        self.list.remove(4)
        self.list.remove(7)
        self.list.remove(5)
        #        test if same as before
        with self.assertRaises(KeyError):
            self.list.remove(4)
        six.assertCountEqual(self, [1, 2, 3], self.list.items)


class TestWeightedListDict(TestListDict):
    """Test weighted ListDict implementation"""

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestWeightedListDict, self).__init__(methodName)

    def setUp(self):
        self.list = at.WeightedListDict()
        self.list.add(1)
        self.list.add(2)
        self.list.add(3)

    def test_add(self):
        """Test adding elements"""
        self.list.add(4)
        six.assertCountEqual(self, [1, 2, 3, 4], self.list.items)
        self.assertEqual(str([(1, 1), (2, 1), (3, 1), (4, 1)]), str(self.list))

        self.list.add(4)
        six.assertCountEqual(self, [1, 2, 3, 4], self.list.items)
        six.assertCountEqual(self, [1, 1, 1, 2], self.list.weights)

        self.list.add(2, .5)
        six.assertCountEqual(self, [1, 2, 3, 4], self.list.items)
        six.assertCountEqual(self, [1, 1, 1.5, 2], self.list.weights)

    def test_remove_weights(self):
        """Test weights"""
        self.list.add(4, weight=3)

        #        check if weight works with sampling directly
        hits = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        number_of_tries = 10000
        for _ in range(number_of_tries):
            hits[self.list.choose_random()] += 1

        for hit in range(1, 4):
            hits[hit] /= number_of_tries
            self.assertAlmostEqual(hits[hit], 1.0 / 6, delta=0.1)
        hits[4] /= number_of_tries
        self.assertAlmostEqual(hits[4], 0.5, delta=0.1)

        #        test separate remove
        self.list.remove(4, .5)
        self.list.remove(4, 1)
        self.list.remove(4, 1)
        self.list.remove(4, .5)


# noinspection SpellCheckingInspection
class TestnHeap(ut.TestCase):
    """ Test Class nHeap """

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestnHeap, self).__init__(methodName)

    def setUp(self):
        self.heap = at.nHeap(3)

    def test_push_try(self):
        """Test adding Elements"""
        self.heap.push_try((1, 1))
        self.assertEqual([(1, 1)], self.heap.data)

        self.heap.push_try((2, 2))
        six.assertCountEqual(self, [(1, 1), (2, 2)], self.heap.data)

        self.heap.push_try((3, 3))
        six.assertCountEqual(self, [(1, 1), (2, 2), (3, 3)], self.heap.data)

        self.heap.push_try((0, 0))
        six.assertCountEqual(self, [(1, 1), (2, 2), (3, 3)], self.heap.data)

        self.heap.push_try((2.1, "Test"))
        six.assertCountEqual(self, [(2.1, "Test"), (2, 2), (3, 3)], self.heap.data)

    def test_len(self):
        """ Test len method """
        for i in range(0, 5):
            self.assertEqual(min(i, 3), len(self.heap))
            self.heap.push_try((i, i))

    def test_access(self):
        """Test access on data of heap"""
        data = []
        for i in range(0, 2):
            data.append((i, i))
            self.heap.push_try((i, i))

        six.assertCountEqual(self, data, self.heap.data)
        six.assertCountEqual(self, data, list(self.heap))


class TestNode(ut.TestCase):
    """ Test Class for Node """

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestNode, self).__init__(methodName)

    def setUp(self):
        self.node = at.Node()

    def test_node(self):
        """ Test setting and getting next_node and get data """

        new_node = at.Node(data=10, next_node=self.node)
        self.assertEqual(new_node.data, 10)
        self.assertEqual(new_node.next_node, self.node)
        self.assertNotEqual(self.node, new_node)

        self.node.next_node = new_node
        self.assertEqual(self.node.next_node, new_node)


class TestLinkedList(ut.TestCase):
    """ Test Class for Linked List and LinkedListIterator"""

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestLinkedList, self).__init__(methodName)

    def setUp(self):
        self.list = at.LinkedList()

    def test_empty(self):
        """ Test empty check """
        self.assertTrue(self.list.is_empty())
        self.list.append(1)
        self.assertFalse(self.list.is_empty())
        self.list.pop()
        self.assertTrue(self.list.is_empty())

    def test_append_and_pop(self):
        """ Test inserting and reading data"""
        #        check error on popping empty list
        with self.assertRaises(IndexError):
            self.list.pop()
        # insert 10 Elements and read them from the list
        for i in range(10):
            self.list.append(i)

        for i in range(10):
            self.assertEqual(i, self.list.pop())

        for i in range(10):
            self.list.append(i, as_first_element=True)
            self.assertEqual(i, self.list.get_data())

        for i in reversed(range(10)):
            self.assertEqual(i, self.list.pop())

    def test_merge_and_iter(self):
        """Test merging and iteration"""
        second_list = at.LinkedList()
        third_list = at.LinkedList()
        for i in range(10):
            self.list.append(i)
            second_list.append(i + 10)
            third_list.append(i + 20)

        self.assertEqual(str(list(range(10))), str(self.list))
        self.assertEqual(str(list(range(10, 20))), str(second_list))
        self.assertEqual(str(list(range(20, 30))), str(third_list))

        self.assertNotEqual(self.list, second_list)

        self.list.merge(second_list)
        self.assertEqual(second_list, self.list)
        for i, list_value in enumerate(self.list):
            self.assertEqual(i, list_value)

        for i, list_value in enumerate(second_list):
            self.assertEqual(i, list_value)

        third_list.merge(self.list)
        self.assertEqual(self.list, third_list)
        self.assertEqual(self.list, second_list)

        for i, list_value in enumerate(self.list):
            self.assertEqual(i, list_value)

        for i, list_value in enumerate(second_list):
            self.assertEqual(i, list_value)

        for i, list_value in enumerate(third_list):
            self.assertEqual(i, list_value)

        list_iter = iter(self.list)
        for i in range(30):
            self.assertEqual(i, list_iter.next())

        with self.assertRaises(StopIteration):
            list_iter.next()

        self.assertEqual(str(list(range(30))), str(self.list))

        #        check the tail switch
        list_4 = at.LinkedList()
        list_4.append(31)
        second_list.merge(list_4)
        self.assertEqual(list_4.tail, second_list.tail)

    def test_private(self):
        """Test methods like eq, hash,..."""
        #        handling with NotImplemented on other class
        self.assertFalse(self.list == NotImplemented)

        self.list.append(1)
        another_list = at.LinkedList()
        another_list.append(2)
        self.list.merge(another_list)
        self.assertEqual(hash(self.list), hash(another_list))
        self.assertEqual(self.list, another_list)

        iterator = iter(another_list)
        for i, list_value in enumerate(iterator, 1):
            self.assertEqual(i, list_value)


class TestObservedObserver(ut.TestCase):
    """ Test Class for Observed and Observer"""

    # noinspection PyPep8Naming
    def __init__(self, methodName='runTest'):
        super(TestObservedObserver, self).__init__(methodName)

    def setUp(self):
        self.observed = at.Observed(1)
        self.observer = at.Observer(self.observed)

    def test_observed(self):
        """ Test basic functions of observed """
        other_observed = at.Observed(2)

        self.assertTrue(self.observed < other_observed)
        self.assertTrue(self.observed != other_observed)
        self.assertTrue(self.observed <= other_observed)
        self.assertFalse(self.observed > other_observed)
        self.assertFalse(self.observed >= other_observed)

        #        test handling of NotImplemented->top handling -> address comparision
        self.assertFalse(self.observed == NotImplemented)
        # self.assertFalse(self.observed <= NotImplemented)

        self.assertEqual(self.observed.value, 1)
        self.observed.value = 2
        self.assertEqual(self.observed.value, 2)

        self.assertTrue(self.observed == other_observed)

        self.assertEqual(self.observed.observers, [self.observer])

        other_observer = at.Observer(other_observed)

        self.assertEqual(self.observed.observers, [self.observer])
        self.assertEqual(str(self.observed), "2 :-> [2]")
        self.observed.add(other_observer)
        self.assertEqual(str(self.observed), "2 :-> [2, 2]")
        self.assertEqual(self.observed.observers, [self.observer, other_observer])

    def test_observer(self):
        """Test observer functionality"""

        self.assertEqual(self.observer.get_value(), 1)
        self.assertEqual(self.observer.get_observed(), self.observed)

        self.observer.set_value(2)
        self.assertEqual(self.observer.get_value(), 2)
        self.assertEqual(str(self.observer), "2")

        other_observer = at.Observer(self.observed)

        self.assertTrue(self.observer == other_observer)
        #        test handling of NotImplemented->top handling -> address comparision
        self.assertFalse(self.observer == NotImplemented)
        # self.assertFalse(self.observer <= NotImplemented)

        other_observed = at.Observed(3)
        another_observer = at.Observer(other_observed)
        self.assertTrue(self.observer != another_observer)
        # self.assertTrue(self.observer <= another_observer)

        self.observer.update_observed(other_observed)
        self.assertEqual(self.observer.get_value(), 3)
        self.assertEqual(other_observer.get_value(), 3)
        six.assertCountEqual(self, [self.observer, other_observer, another_observer],
                             other_observed.observers)
