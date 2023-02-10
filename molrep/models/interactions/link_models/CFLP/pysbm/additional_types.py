"""
Module for additional needed types
"""

import copy
import functools
import heapq as hp
import random


class SetDict(object):
    """
    Type which implements the following operations with a runtime better than O(n):
    - adding of items (once)
    - removal of items
    - random access
    additional the operation copy() is needed.
    This implementation is based on a post of Amber on stackoverflow:
    http://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice
    Accessed 5.5.2017 10:11
    """

    def __init__(self):
        self.item_to_position = {}
        self.items = []

    def add(self, item):
        """Add a new item"""
        if item in self.item_to_position:
            raise KeyError()
        # return
        self.items.append(item)
        self.item_to_position[item] = len(self.items) - 1

    def remove(self, item):
        """Remove an item"""
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random(self):
        """Return a random element"""
        return random.choice(self.items)

    def copy(self):
        """Create a copy"""
        new = SetDict()
        new.item_to_position = self.item_to_position.copy()
        new.items = self.items[:]
        return new

    def __repr__(self):
        return str(self.items)


class ListDict(object):
    """
    Type which implements the following operations with a runtime better than O(n):
    - adding of items (even multiple times the same)
    - removal of items
    - random access
    additional the operation copy() is needed.
    This implementation is based on a post of Amber on stackoverflow:
    http://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice
    Accessed 5.5.2017 10:11
    """

    def __init__(self):
        self.item_to_position = {}
        self.items = []

    def add(self, item):
        """Add a new item"""
        self.items.append(item)
        if item in self.item_to_position:
            self.item_to_position[item].add(len(self.items) - 1)
        else:
            self.item_to_position[item] = {len(self.items) - 1}

    def remove(self, item):
        """Remove an item"""
        positions = self.item_to_position[item]
        item_position = positions.pop()

        #        if set is empty remove entry from dict
        if not positions:
            del self.item_to_position[item]

        # exchange entry in items list with the last entry of the list and
        #         keep track of this change in the corresponding item_to_position set
        last_item = self.items.pop()
        if item_position != len(self.items):
            self.items[item_position] = last_item
            self.item_to_position[last_item].add(item_position)
            self.item_to_position[last_item].remove(len(self.items))

    def choose_random(self):
        """Return a random element"""
        return random.choice(self.items)

    def copy(self):
        """Create a copy"""
        new = ListDict()
        new.item_to_position = copy.deepcopy(self.item_to_position)
        new.items = self.items[:]
        return new

    def __repr__(self):
        return str(self.items)


class WeightedListDict(object):
    """
    Type which implements the following operations with a runtime better than:
    - adding of items (even multiple times the same -> reduced to weight)
    - removal of items
    - choose random element (with probabilities proportional to the weights)
    additional the operation copy() is needed.

    This implementation saves space by only counting the times an element is added to the list.
    Additionally this allows different weights for the element. Nevertheless it's slow down
    the runtime of random access because concerning the different weights costs time.

    This implementation is based on a post of Amber on stackoverflow:
    http://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice
    Accessed 5.5.2017 10:11
    """

    def __init__(self):
        self.item_to_position = {}
        self.items = []
        self.weights = []
        #        we save the total way for an easy implementation of random access
        self.total_weight = 0

    def add(self, item, weight=1):
        """Add a new item"""
        self.total_weight += weight
        if item in self.item_to_position:
            self.weights[self.item_to_position[item]] += weight
        else:
            self.items.append(item)
            self.weights.append(weight)
            self.item_to_position[item] = len(self.items) - 1

    def remove(self, item, weight=1):
        """Remove an item"""
        position = self.item_to_position[item]
        self.weights[position] -= weight
        self.total_weight -= weight

        #        Element was completely removed from the list
        #         WARNING the actual zero check only works perfect for ints,
        #          -> floats have to be checked with abs < epsilon
        if self.weights[position] == 0:
            del self.item_to_position[item]

            #            exchange entry if last entry was removed, remove only 0 weight entry
            last_item = self.items.pop()
            if position != len(self.items):
                self.items[position] = last_item
                self.weights[position] = self.weights.pop()
                self.item_to_position[last_item] = position
            else:
                self.weights.pop()

    def choose_random(self):
        """Return a random element"""
        #        first get random weight where we will stop
        random_weight = random.random() * self.total_weight
        for position, actual_weight in enumerate(self.weights):
            random_weight -= actual_weight
            if random_weight < 0:
                return self.items[position]

    def copy(self):
        """Create a copy"""
        new = WeightedListDict()
        new.item_to_position = self.item_to_position.copy()
        new.items = self.items[:]
        new.weights = self.weights[:]
        return new

    #
    # def __getitem__(self, item):
    #     return self.weights[self.item_to_position[item]]

    def __repr__(self):
        return str(list(zip(self.items, self.weights)))


# noinspection PyPep8Naming
class nHeap(object):
    """
    Data Structure for saving the max n Elements (tuples  (value, item_data))
    """

    def __init__(self, max_elements):
        #        create min heap
        self._data = []
        hp.heapify(self.data)

        self.max_elements = max_elements

    def push_try(self, data):
        """
        Add element if data contains not more than max_elements else exchange
        the lowest element if new element is greater
        """
        if len(self._data) < self.max_elements:
            hp.heappush(self.data, data)
        elif self._data[0][0] < data[0]:
            hp.heapreplace(self.data, data)

    @property
    def data(self):
        """Returns the contained data"""
        return self._data

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        for value in self._data:
            yield value

    def __len__(self):
        return len(self.data)


class Node(object):
    """
    Node with data and link to next node
    """

    def __init__(self, data=None, next_node=None):
        self._data = data
        self._next_node = next_node

    @property
    def next_node(self):
        """Return reference to next node"""
        return self._next_node

    @next_node.setter
    def next_node(self, new_node):
        """Set reference to next node"""
        self._next_node = new_node

    @property
    def data(self):
        """Return contained value"""
        return self._data


class LinkedList(object):
    """
    Implementation of a simple linked list
    """

    def __init__(self):
        self.head = None
        self.tail = None

    def is_empty(self):
        """Check if List is empty"""
        return self.head is None

    def append(self, data, as_first_element=False):
        """Append new data to linked list"""
        #        if list is Empty...
        if self.head is None:
            node = Node(data)
            self.head = node
            self.tail = node
        elif as_first_element:
            node = Node(data, self.head)
            self.head = node
        else:
            node = Node(data)
            self.tail.next_node = node
            self.tail = node

    def get_data(self):
        """Get first data"""
        return self.head.data

    def merge(self, another_list):
        """Merge two lists"""
        if another_list.get_data() >= self.get_data():
            #            ensure that tail is real tail...
            while self.tail.next_node is not None:
                self.tail = self.tail.next_node
            self.tail.next_node = another_list.head
            self.tail = another_list.tail
            another_list.change_head(self.head)
        else:
            another_list.merge(self)

    def change_head(self, node):
        """Change head - needed for merge"""
        self.head = node

    def pop(self):
        """ Return first data and remove it from the list"""
        if self.head is None:
            raise IndexError
        value = self.head.data
        self.head = self.head.next_node
        return value

    def __iter__(self):
        return LinkedListIterator(self)

    def __repr__(self):
        return str(list(self))

    # implement operations needed for the desired comparision
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.head == other.head
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.head)


class LinkedListIterator(object):
    """Iterator over a LinkedList"""

    # pylint: disable=too-few-public-methods
    # this class is only an iterator and don't need any additional methods
    def __init__(self, linked_list):
        self.next_node = linked_list.head

    def __iter__(self):
        return self

    def next(self):
        """
        Returns data and set pointer to next node or raise StopIteration
        if end of list is reached
        """
        if self.next_node is None:
            raise StopIteration()
        else:
            value = self.next_node.data
            self.next_node = self.next_node.next_node
            return value

    __next__ = next


@functools.total_ordering
class Observed(object):
    """Object containing a value and a list of interested observers"""

    def __init__(self, value):
        self._value = value
        self.observers = []

    def add(self, observer):
        """Add a new observer"""
        self.observers.append(observer)

    def update_ref(self, new_observed):
        """Change observed object for all interested observers"""
        for observer in self.observers:
            observer.update_ref(new_observed)

    @property
    def value(self):
        """Return the contained value"""
        return self._value

    @value.setter
    def value(self, value):
        """Change the contained value (dos not call an update of observers)"""
        self._value = value

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, self.__class__):
            return self.value <= other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return str(self.value) + ' :-> ' + str(self.observers)


@functools.total_ordering
class Observer(object):
    """Class watching at an observed object"""

    def __init__(self, observed):
        self.observed = observed
        self.observed.add(self)

    def update_ref(self, new_observed):
        """Update observed object"""
        self.observed = new_observed
        self.observed.add(self)

    def get_value(self):
        """Return the value of the observed object"""
        return self.observed.value

    def get_observed(self):
        """Return the observed object"""
        return self.observed

    def update_observed(self, new_observed):
        """
        Call the observed object to update all interested objects to change to
        the new observed object
        """
        self.observed.update_ref(new_observed)

    def set_value(self, value):
        """Change value of the observed object"""
        self.observed.value = value

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.observed == other.observed
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, self.__class__):
            return self.observed <= other.observed
        return NotImplemented

    def __hash__(self):
        return hash(self.observed)

    def __repr__(self):
        return str(self.get_value())
