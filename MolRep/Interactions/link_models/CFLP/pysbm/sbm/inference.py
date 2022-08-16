# real non truncating division (same behaviour as in python 3)
from __future__ import division

import random as rd
import networkx as nx
import numpy as np
from numpy import linalg as la
from sklearn import cluster as cl

from MolRep.Interactions.link_models.CFLP.pysbm import additional_types
from .exceptions import NoFreeNodeException
from .objective_function_iclex import IntegratedCompleteLikelihoodExact
from .objective_functions import *
from .partition import Partition
from .peixotos_flat_sbm import LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmWrapper
from .peixotos_hierarchical_sbm import LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmWrapper
from .objective_function_newman_group_size import NewmanReinertDegreeCorrected
from .objective_function_newman_group_size import NewmanReinertNonDegreeCorrected

rd.seed()


class Inference(object):
    """Infer stochastic block model out of a given graph """

    title = "Forgot to replace title"
    short_title = "Forgot to replace short title"

    def __init__(self, graph, objective_function, partition):
        """

        :param graph:
        :param objective_function:
        :type objective_function ObjectiveFunction
        :param partition:
        :type partition Partition
        """
        self.graph = graph
        self._objective_function = objective_function
        self.partition = partition
        self.node_moves = 0

    def infer_stochastic_block_model(self, *args):
        """Get stochastic block model"""
        raise NotImplementedError()

    def infer_stepwise(self):
        """Calculate stochastic block model stepwise"""
        raise NotImplementedError()

    @property
    def objective_function(self):
        """Objective function which should be maximized"""
        return self._objective_function

    @objective_function.setter
    def objective_function(self, new_objective_function):
        self._objective_function = new_objective_function

    def get_moves(self):
        """Return number of node moves"""
        return self.node_moves


class MetropolisInference(Inference):
    """Inference of SBM with Metropolis Algorithm"""
    title = "Metropolis Inference"
    short_title = "MA 5k"

    def __init__(self, graph, objective_function, partition, use_delta=True):
        super(MetropolisInference, self).__init__(graph, objective_function, partition)
        objective_function.old_value = objective_function.calculate(partition)
        self.steps = 0
        self.use_delta = use_delta
        self.min_partition = None
        self._min_dist = 0.0

    def infer_stochastic_block_model(self, *args):
        # default value for save_min = True with saving
        save_min = False
        if len(args) == 2:
            steps = args[0]
            save_min = args[1]
        elif len(args) == 1:
            steps = args[0]
        elif not args:  # equal to len(args)==0
            steps = 5000  # self.partition.get_number_of_nodes()^4
            save_min = True
        else:
            raise ValueError()

        if save_min:
            self.min_partition = self.partition.get_representation()

        for _ in range(steps):
            self._iterate(save_min)

        # if minimum is saved set the partition like the minimum
        if save_min:
            for node in self.min_partition:
                self.partition.move_node(node, self.min_partition[node])

    def infer_stepwise(self):
        self._iterate(False)

    def _iterate(self, save_min):
        """Perform a single step of Metropolis Inference"""
        move_candidate = self.partition.get_random_move()
        node, from_block, to_block = move_candidate

        if self.use_delta:
            parameter = self.partition.precalc_move(move_candidate, self._objective_function)
            delta = self._objective_function.calculate_delta(self.partition,
                                                             from_block,
                                                             to_block,
                                                             *parameter)
            # long version of probability = min(1, math.exp(delta)) and if rd.random() < probability:...
            #  but this version can handle bigger deltas and prevent OverflowError
            if delta >= 0:
                move_node = True
            else:
                move_node = (rd.random() < math.exp(delta))
            if move_node:
                self.partition.move_node(node, to_block)
                self.node_moves += 1
        else:
            self.partition.move_node(node, to_block)
            new_value = self._objective_function.calculate(self.partition)
            delta = new_value - self._objective_function.old_value
            #     same logic as in use delta
            if delta >= 0:
                move_node = True
            else:
                move_node = (rd.random() < math.exp(delta))
            if move_node:
                self._objective_function.old_value = new_value
                self.node_moves += 1
            else:
                self.partition.move_node(node, from_block)

                # if demanded save minimum
        if save_min:
            self._min_dist += delta
            if self._min_dist > 0:
                self.min_partition = self.partition.get_representation()
                self._min_dist = 0.0
        self.steps += 1


class PeixotoInference(Inference):
    """Inference Algorithm described in Peixoto 2014"""
    title = "Peixoto Inference"
    short_title = "PAH"

    _objective_function_classes = (TraditionalMicrocanonicalEntropy,
                                   TraditionalMicrocanonicalEntropyDense,
                                   TraditionalUnnormalizedLogLikelyhood,
                                   LogLikelihoodOfHierarchicalMicrocanonicalDegreeCorrectedSbmWrapper,
                                   LogLikelihoodOfFlatMicrocanonicalDegreeCorrectedSbmWrapper,
                                   IntegratedCompleteLikelihoodExact,
                                   NewmanReinertDegreeCorrected,
                                   NewmanReinertNonDegreeCorrected)

    def __init__(self, graph, objective_function, partition, no_partition_reset=False, limit_possible_blocks=False,
                 add_additional_mergers=True, mcmc_steps=100):
        super(PeixotoInference, self).__init__(graph, objective_function, partition)
        # in the actual setting need to know the final blocksize and therefore take the value from partition
        self.aimed_b = partition.B
        # create a partition with every node in a own block
        if not no_partition_reset:
            self.partition.random_partition(partition.get_number_of_nodes())
        self.mcmc = MetropolisHastingInference(graph, objective_function, self.partition,
                                               limit_possible_blocks=limit_possible_blocks)
        self.mcmc_steps = mcmc_steps
        # free parameter for the selection of block merges in a Metropolis Hasting like fashion
        self.epsilon = 0.1

        # switch edge saving on
        self.partition.set_save_neighbor_edges(True)
        # sigma defines the proportion of blocks which will be merged in an iteration
        #  -> 2 means around the half of the blocks will be merged
        #  keep in mind that due to loops in the merges not exactly half of the blocks will be merged
        self.sigma = 2
        # Parameter of how many merges for each block will be tried
        self.number_of_tries_per_block = 10
        # create additional mergers so that all blocks have block numbers lower then the number of blocks
        #  use this parameter if the partition do not automatically handle such things
        self.add_additional_mergers = add_additional_mergers

        self.limit_possible_blocks = limit_possible_blocks

    def infer_stochastic_block_model(self, *args):
        while self.partition.B > self.aimed_b:
            self.infer_stepwise()

    def _select_new_block_for_block(self, block):
        """
        Select a new block s with probability sum_t p^i_tp(r->s|t) where
        p^i_t is the fraction of neighbors of node i which belongs to block t and
        p(r->s|t)= frac{e_{ts}+epsilon}{degree_of_block(t)+epsilon*B}
        respectively
        p(r->s|t)= frac{e_{ts}+e_{st}+epsilon}
                {in_degree_of_block(t)+out_degree_of_block(t)+epsilon*B}
        (Node i is the node to be moved)
        """

        # Similar to the method select_new_block of Metropolis Hasting
        #  only change is that t is a neighboring block and not a node
        t = self.partition.get_block_of_node(self.partition.get_random_neighboring_edge_of_block(block)[1])
        # 2. Select s randomly from all B choices with equal probability
        if self.limit_possible_blocks:
            s = rd.choice(self.partition.get_possible_blocks(block))
        else:
            s = rd.choice(range(self.partition.B))
        # 3. Accept with probability R_t
        if self.partition.is_graph_directed():
            rt = self.epsilon * self.partition.B \
                 / (self.partition.get_in_degree_of_block(t) + self.partition.get_in_degree_of_block(t)
                    + self.epsilon * self.partition.B)
        else:
            rt = self.epsilon * self.partition.B \
                 / (self.partition.get_degree_of_block(t) + self.epsilon * self.partition.B)
        if rd.random() < rt:
            return s
        # 4. If rejected chose a random edge adjacent to block t and the label s
        #    is taken from the opposite endpoint
        # else:...
        if not self.limit_possible_blocks:
            to_node = self.partition.get_random_neighboring_edge_of_block(t)[1]
            return self.partition.get_block_of_node(to_node)
        # else need additional check if move is allowed
        possible_blocks = self.partition.get_possible_blocks(block, with_in_operation=True)
        for _ in range(25):
            to_node = self.partition.get_random_neighboring_edge_of_block(t)[1]
            new_block = self.partition.get_block_of_node(to_node)
            if new_block in possible_blocks:
                return new_block
        new_block = rd.choice(self.partition.get_possible_blocks(block))
        # else simple return any random block
        return new_block

    def _cleanup_merges(self, delta_with_merges):
        """
        Create final dict of merges without redirecting or loops
        and additional entries for the effect that the result of the merge
        is numbered from 0 to B-1
        """
        # a dictionary with the structure: block: reference to LinkedList containing
        #    all blocks that will be merged and additionally has the smallest block number
        #    as first element
        blocks_to_be_merged = {}
        # number of detected loops: a loop is detected by an entry of delta_with_merge
        #    which does not result in a change of blocks_to_be_merged
        #    This number is needed to correctly add the additional merges for the renumbering
        #    from 0...B-1
        number_of_loops = 0

        for (_, from_block, to_block) in delta_with_merges:
            block_found_from = from_block in blocks_to_be_merged
            block_found_to = to_block in blocks_to_be_merged

            if block_found_from and block_found_to:
                #  if both exists check value
                left_observer = blocks_to_be_merged[from_block]
                right_observer = blocks_to_be_merged[to_block]
                #  if same increase loop counter
                if left_observer == right_observer:
                    number_of_loops += 1
                # else let the higher observer change his value to the lower one
                elif left_observer < right_observer:
                    right_observer.update_observed(left_observer.get_observed())
                else:
                    left_observer.update_observed(right_observer.get_observed())
                continue
            # if only one already exist copy the pointer and later check for change
            elif block_found_from:
                block_observer = blocks_to_be_merged[from_block]
                data = to_block
            elif block_found_to:
                block_observer = blocks_to_be_merged[to_block]
                data = from_block
            else:
                #     if no entry exists create a new Observer and Observed Instance
                if from_block < to_block:
                    new_observed = additional_types.Observed(from_block)
                else:
                    new_observed = additional_types.Observed(to_block)
                new_observer = additional_types.Observer(new_observed)
                blocks_to_be_merged[from_block] = new_observer
                blocks_to_be_merged[to_block] = new_observer
                continue
            # if only one exists check need for update
            if block_observer.get_value() > data:
                #       if lower change value of observed
                block_observer.set_value(data)
            blocks_to_be_merged[data] = block_observer

        merges = {}
        max_block_number = self.partition.B - len(delta_with_merges) + number_of_loops
        free_blocks = additional_types.LinkedList()

        # sort dict by key to first deal with the smallest block
        for block in sorted(blocks_to_be_merged):
            new_block = blocks_to_be_merged[block].get_value()

            #  to ensure that all blocks are merged to a new_block smaller than max_block_number
            #     works due to sorting by key, the first merge met is the x:x entry
            if self.add_additional_mergers and new_block >= max_block_number:
                new_block = free_blocks.pop()
                blocks_to_be_merged[block].update_observed(additional_types.Observed(new_block))
            if block != new_block:
                merges[block] = new_block
                free_blocks.append(block)

        if self.add_additional_mergers:
            # add extra merges for block with a higher block number
            for block in range(max_block_number, self.partition.B):
                if block not in merges:
                    merges[block] = free_blocks.pop()

        return merges, max_block_number

    def infer_stepwise(self):
        number_of_merges = int(self.partition.B / self.sigma)
        # control that not more merges are performed than needed
        if self.partition.B - number_of_merges < self.aimed_b:
            number_of_merges = self.partition.B - self.aimed_b
            if number_of_merges <= 0:
                raise StopIteration
        best_n_merges = additional_types.nHeap(number_of_merges)
        # determine probable moves for each block
        for block in range(self.partition.B):
            # determine parameters:
            best_merge = None
            best_delta = -float('inf')

            parameter, reference = self._precalc_block_merge(block)

            # try some moves
            tries = self.number_of_tries_per_block
            while tries > 0:
                tries -= 1
                #  select new block according to p(r->s|t)
                merge_with_block = self._select_new_block_for_block(block)
                # quick exit if same as before or same block
                if merge_with_block == best_merge or merge_with_block == block:
                    continue
                # adjustment for delta formula
                saved_data = self._adjustment_delta_pre(reference, block, merge_with_block)

                #  insert parameters depending on new block
                delta = self._objective_function.calculate_delta(
                    self.partition, block, merge_with_block, *parameter)

                #      reset neighbor_info kit
                self._adjustment_delta_post(reference, merge_with_block, saved_data)

                #      refresh best move
                if delta > best_delta:
                    best_delta = delta
                    best_merge = merge_with_block

            if best_merge is not None:
                best_n_merges.push_try((best_delta, block, best_merge))
                #    perform the best n moves
                # actual variant may perform slower but do not need to keep an additional
                #    list of nodes for each block which needs to be updated by each step
        merge_info = self._cleanup_merges(best_n_merges)
        self.partition.merge_blocks(*merge_info)

        #  execute Metropolis Hasting algorithm for a few steps
        self.mcmc.beta = 1
        try:
            self.mcmc.infer_stochastic_block_model(self.mcmc_steps)
        except NoFreeNodeException:
            pass

        # abrupt change of beta, now more or less only moves, which improve the partition are performed
        self.mcmc.beta = 1000000
        try:
            self.mcmc.infer_stochastic_block_model(self.mcmc_steps)
        except NoFreeNodeException:
            pass

    def _precalc_block_merge(self, block):
        """ Analog to precalc move of partition calculate the needed parameters for a block merge"""
        include_covariates = self.partition.with_covariate
        kit = {}
        if self.partition.is_graph_directed():
            # in directed case we need to distinguish between incoming and outgoing edges
            #  and need more parameters for the objective function
            kti = {}
            # neighbors include both incoming and outgoing edges therefore check inside loop
            for neighbor in self.partition.get_neighbors_of_block(block):
                if self.partition.get_edge_count(block, neighbor) != 0:
                    kit[neighbor] = self.partition.get_edge_count(block, neighbor)
                if self.partition.get_edge_count(neighbor, block) != 0:
                    kti[neighbor] = self.partition.get_edge_count(neighbor, block)

            kti[block] = 0
            kit[block] = 0

            reference = (kit, kti)
            if isinstance(self._objective_function,
                          self._objective_function_classes):
                parameter = (kit, kti, self.partition.get_edge_count(block, block),
                             self.partition.get_in_degree_of_block(block),
                             self.partition.get_out_degree_of_block(block),
                             self.partition.get_number_of_nodes_in_block(block),
                             False,  # remove block
                             )
            else:
                parameter = (kit, kti, self.partition.get_edge_count(block, block),
                             self.partition.get_in_degree_of_block(block),
                             self.partition.get_out_degree_of_block(block))

            if include_covariates:
                # same as above but read summed covariates
                covariate_kit = {}
                covariate_kti = {}
                reference += reference + (covariate_kit, covariate_kti)
                for neighbor in self.partition.get_neighbors_of_block(block):
                    if self.partition.get_edge_count(block, neighbor) != 0:
                        covariate_kit[neighbor] = self.partition.get_sum_of_covariates(block, neighbor)
                    if self.partition.get_edge_count(neighbor, block) != 0:
                        covariate_kti[neighbor] = self.partition.get_sum_of_covariates(neighbor, block)

                if isinstance(self._objective_function,
                              self._objective_function_classes):
                    parameter = parameter \
                                + (covariate_kit, covariate_kti, self.partition.get_sum_of_covariates(block, block),
                                   sum(covariate_kti.values()),
                                   sum(covariate_kit.values()),
                                   self.partition.get_number_of_nodes_in_block(block),
                                   False,  # remove block
                                   )
                else:
                    parameter = parameter \
                                + (kit, kti, self.partition.get_sum_of_covariates(block, block),
                                   sum(covariate_kti.values()),
                                   sum(covariate_kit.values()))

        else:
            for neighbor in self.partition.get_neighbors_of_block(block):
                kit[neighbor] = self.partition.get_edge_count(block, neighbor)

            kit[block] = 0

            reference = (kit,)
            if isinstance(self._objective_function,
                          self._objective_function_classes):
                parameter = (kit, self.partition.get_edge_count(block, block) / 2,
                             self.partition.get_degree_of_block(block),
                             self.partition.get_number_of_nodes_in_block(block),
                             False,  # remove block
                             )
            else:
                parameter = (kit, self.partition.get_edge_count(block, block) / 2,
                             self.partition.get_degree_of_block(block))

            if include_covariates:
                covariate_kit = {}
                reference = (kit, covariate_kit)

                for neighbor in self.partition.get_neighbors_of_block(block):
                    covariate_kit[neighbor] = self.partition.get_sum_of_covariates(block, neighbor)

                if isinstance(self._objective_function,
                              self._objective_function_classes):
                    parameter = parameter \
                                + (covariate_kit, self.partition.get_sum_of_covariates(block, block),
                                   sum(covariate_kit.values()),
                                   self.partition.get_number_of_nodes_in_block(block),
                                   False,  # remove block
                                   )
                else:
                    parameter = parameter \
                                + (covariate_kit, self.partition.get_sum_of_covariates(block, block),
                                   sum(covariate_kit.values()))
        return parameter, reference

    def _adjustment_delta_pre(self, reference, block, merge_with_block):
        """ Pre delta calculation changes of the parameters """
        if self.partition.is_graph_directed():
            kit, kti = reference[:2]
            old_value_kit = kit.get(merge_with_block, None)
            kit[merge_with_block] = self.partition.get_edge_count(block, merge_with_block)
            old_value_kti = kti.get(merge_with_block, None)
            kti[merge_with_block] = self.partition.get_edge_count(merge_with_block, block)
            # same adjustments for covariates
            if self.partition.with_covariate:
                covariance_kit, covariance_kti = reference[2], reference[3]
                old_value_covariance_kit = covariance_kit.get(merge_with_block, None)
                covariance_kit[merge_with_block] = self.partition.get_sum_of_covariates(block, merge_with_block)
                old_value_covariance_kti = covariance_kti.get(merge_with_block, None)
                covariance_kti[merge_with_block] = self.partition.get_sum_of_covariates(merge_with_block, block)
                return old_value_kit, old_value_kti, old_value_covariance_kit, old_value_covariance_kti
            return old_value_kit, old_value_kti
        # else:...
        kit = reference[0]
        old_value = kit.get(merge_with_block, None)
        kit[merge_with_block] = self.partition.get_edge_count(block, merge_with_block)
        # same adjustments for covariates
        if self.partition.with_covariate:
            covariance_kit, covariance_kti = reference[1]
            old_value_covariance_kit = covariance_kit.get(merge_with_block, None)
            covariance_kit[merge_with_block] = self.partition.get_sum_of_covariates(block, merge_with_block)
            return old_value, old_value_covariance_kit
        return old_value,

    def _adjustment_delta_post(self, reference, merge_with_block, saved_data):
        """ Reset parameters after delta calculations to the old values """
        if self.partition.is_graph_directed():
            kit, kti = reference[:2]
            old_value_kit, old_value_kti = saved_data[:2]
            if old_value_kit is None:
                del kit[merge_with_block]
            else:
                kit[merge_with_block] = old_value_kit
            if old_value_kti is None:
                del kti[merge_with_block]
            else:
                kti[merge_with_block] = old_value_kti
            # same adjustments for covariates
            if self.partition.with_covariate:
                covariance_kit, covariance_kti = reference[2], reference[3]
                old_value_covariance_kit, old_value_covariance_kti = saved_data[2], saved_data[3]
                if old_value_covariance_kit is None:
                    del covariance_kit[merge_with_block]
                else:
                    covariance_kit[merge_with_block] = old_value_covariance_kit
                if old_value_covariance_kti is None:
                    del covariance_kti[merge_with_block]
                else:
                    covariance_kti[merge_with_block] = old_value_covariance_kti
        else:
            kit = reference[0]
            old_value = saved_data[0]
            if old_value is None:
                del kit[merge_with_block]
            else:
                kit[merge_with_block] = old_value
            # same adjustments for covariates
            if self.partition.with_covariate:
                covariance_kit = reference[1]
                old_value_covariance_kit = saved_data[1]
                if old_value_covariance_kit is None:
                    del covariance_kit[merge_with_block]
                else:
                    covariance_kit[merge_with_block] = old_value_covariance_kit

    @property
    def objective_function(self):
        """Objective function which should be maximized"""
        return self._objective_function

    @objective_function.setter
    def objective_function(self, new_objective_function):
        self._objective_function = new_objective_function
        self.mcmc.objective_function = new_objective_function


# @formatter:off
class MetropolisHastingInference(Inference):
    """Metropolis Hasting Inference Algorithm as described in Peixoto 2014"""
    title = "Metropolis Hasting Inference"
    short_title = "MHA 1k"

    def __init__(self, graph, objective_function, partition, limit_possible_blocks=False):
        super(MetropolisHastingInference, self).__init__(graph, objective_function, partition)
        # free parameter of the metropolis hasting markov chain
        self.epsilon = 0.1
        self.beta_zero = None
        self._number_of_random_partitions_for_beta_zero = 20
        self.beta = 1
        # switch edge saving on
        self.partition.set_save_neighbor_edges(True)
        self.limit_possible_blocks = limit_possible_blocks
        self.default_number_of_steps = 1000
        self.performed_steps = 0

    def infer_stochastic_block_model(self, *args):
        self.performed_steps = 0
        if len(args) == 1:
            steps = args[0]
        elif not args:  # equal to len(args) == 0:
            steps = self.default_number_of_steps
        else:
            raise ValueError()
        if self.partition.is_graph_directed():
            while steps > 0:
                steps -= 1
                self.infer_stepwise_directed()
        else:
            while steps > 0:
                steps -= 1
                self.infer_stepwise_undirected()

    def infer_stepwise(self):
        """
        Try a Metropolis-Hasting move
        """
        if self.partition.is_graph_directed():
            self.infer_stepwise_undirected()
        else:
            self.infer_stepwise_directed()

    def update_temperature(self):
        pass

    def _initialize_temperature(self):
        """Initialize beta_zero"""
        # same procedure as in scipy.anneal pick random points and 1.5 times the maximal observed deviation is
        working_partition = self.partition.copy()
        working_partition.random_partition()
        min_value = max_value = self._objective_function.calculate(working_partition)

        for _ in range(self._number_of_random_partitions_for_beta_zero):
            working_partition.random_partition(self.partition.B)
            new_value = self.objective_function.calculate(working_partition)
            if new_value < min_value:
                min_value = new_value
            elif new_value > max_value:
                max_value = new_value

        self.beta_zero = 1.5*(max_value - min_value)

    def infer_stepwise_undirected(self):
        node = self.partition.get_random_node()
        from_block = self.partition.get_block_of_node(node)
        s = self._select_new_block_undirected(node)

        acceptance_probability = self._calculate_acceptance_probability_undirected(node, from_block, s)

        if rd.random() < acceptance_probability and from_block != s:
            self.partition.move_node(node, s)
            self.node_moves += 1

        self.performed_steps += 1
        self.update_temperature()

    def _calculate_acceptance_probability_undirected(self, node, from_block, to_block, debug=False):
        """
        Calculate the acceptance probability, i.e. calculate the following formula:
         acceptance_probability = e^(beta*Delta)
         *(sum_{neighbor t} (fraction of neighbors of node i belonging to block t)*p(s->r|t))
         /(sum_{neighbor t} (fraction of neighbors of node i belonging to block t)*p(r->s|t))
         where top values are calculated after the move, i.e. change of ert and et,
         and for p(s->r|t) see complex or simple probability.
        """
        s = to_block
        # calculate delta and parameter, which are needed for the rest of the formula too
        parameter = self.partition.precalc_move((node, from_block, s), self._objective_function)
        delta = self._objective_function.calculate_delta(self.partition, from_block, s, *parameter)

        top = 0.0
        bottom = 0.0
        if len(parameter) == 3:
            kit, selfloops, degree = parameter
        else:
            kit = parameter[0]
            selfloops = parameter[1]
            degree = parameter[2]
        # for the calculation of the fraction of neighbors belonging to the blocks
        #  selfloops are only counted as one neighbor therefore decrease degree by selfloops
        for neighbor in kit:
            #  special handling for s and from block the same
            if from_block == s and neighbor == s:
                top += (kit[neighbor] + selfloops) / (degree - selfloops) \
                       * self._probability_undirected(from_block, neighbor)
                bottom += (kit[neighbor] + selfloops) / (degree - selfloops) \
                    * self._probability_undirected(s, neighbor)
            # in the case from block != s handle s and from block separate
            elif neighbor == from_block:
                top += kit[neighbor] / (degree - selfloops) \
                       * self._probability_undirected(from_block, neighbor,
                                                      2 * (kit[from_block] + selfloops),
                                                      degree)
                bottom += (kit[neighbor] + selfloops) / (degree - selfloops) \
                    * self._probability_undirected(s, neighbor)
            elif neighbor == s:
                top += (kit[neighbor] + selfloops) / (degree - selfloops) \
                       * self._probability_undirected(from_block, neighbor,
                                                      kit[s] - kit[from_block],
                                                      -degree)
                bottom += kit[neighbor] / (degree - selfloops) * self._probability_undirected(s, neighbor)
            elif from_block != s:
                top += kit[neighbor] / (degree - selfloops) \
                       * self._probability_undirected(from_block, neighbor, kit[neighbor])
                bottom += kit[neighbor] / (degree - selfloops) \
                    * self._probability_undirected(s, neighbor)
            else:
                top += kit[neighbor] / (degree - selfloops) \
                       * self._probability_undirected(from_block, neighbor)
                bottom += kit[neighbor] / (degree - selfloops) \
                    * self._probability_undirected(s, neighbor)
        # for better testing
        if debug:
            return top, bottom
        # bottom = zero means there is no chance that these block can be selected
        #  to ensure no division by zero error catch this case and accept impossible move
        #  (this is the unique continuous extension)
        if bottom == 0:
            return 1
        # else...
        try:
            # catch large values
            # math.exp(710)= inf
            if self.beta * delta > 700:
                value = 1
            else:
                value = min(math.exp(self.beta * delta) / bottom * top, 1)
        except OverflowError:
            value = 1
        return value

    def _select_new_block_undirected(self, node):
        """
        Select a new block s with probability sum_t p^i_tp(r->s|t) where
        p^i_t is the fraction of neighbors of node i which belongs to block t and
        p(r->s|t) is given like stated in _probability.
        (Node i is the node to be moved)
        """
        # see Peixoto possible with the following procedure (at least for undirected case):
        #  idea use that p(r->s|t)=(1-R_t)e_{ts}/degree_of_block(t)+R_t/B
        #   with R_t=\epsilon*B/(degree_of_block(t)+epsilon*B)
        # 1. Select a random neighbor j of the node i and obtain its block membership->t
        t = self.partition.get_block_of_node(
            rd.choice(self.partition.get_neighbors_of_node(node)))
        # 2. Select s randomly from all B choices with equal probability
        if self.limit_possible_blocks:
            block = self.partition.get_block_of_node(node)
            s = rd.choice(self.partition.get_possible_blocks(block))
        else:
            s = rd.choice(range(self.partition.B))
        # 3. Accept with probability R_t
        rt = self.epsilon * self.partition.B \
            / (self.partition.get_degree_of_block(t) + self.epsilon * self.partition.B)
        if rd.random() < rt:
            return s
        # 4. If rejected chose a random edge adjacent to block t and the label s
        #    is taken from the opposite endpoint
        # else:...
        if not self.limit_possible_blocks:
            to_node = self.partition.get_random_neighboring_edge_of_block(t)[1]
            return self.partition.get_block_of_node(to_node)
        # else need additional check if move is allowed
        # noinspection PyUnboundLocalVariable
        possible_blocks = self.partition.get_possible_blocks(block, with_in_operation=True)
        for _ in range(25):
            to_node = self.partition.get_random_neighboring_edge_of_block(t)[1]
            new_block = self.partition.get_block_of_node(to_node)
            if new_block in possible_blocks:
                return new_block
        new_block = rd.choice(self.partition.get_possible_blocks(block))
        # else simple return any random block
        return new_block

    def _probability_undirected(self, to_block, neighbor_block,
                                change_ets=0, change_degree_of_block=0):
        """
        Return the probability p(r-s|t).
        The formula for directed and undirected graphs are different.

        For undirected graphs:
        ---------------------
        Return the following conditional probability:

        p(r->s|t)= frac{e_{ts}+epsilon}{degree_of_block(t)+epsilon*B}
            (r = from_block, s = to_block and t = block of neighbor)

        Parameters
        ----------
        to_block : int
            The block which should be the next block. (Block s of the formula)
        neighbor_block : int
            The neighboring block which information should be used (block t of the formula)

        With the additional optional parameters the values of e_ts and degree of
        block can be changed.

        """
        # TODO format docstrings accordingly to
        # http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy
        #
        return ((self.partition.get_edge_count(neighbor_block, to_block) - change_ets)
                + self.epsilon) \
            / ((self.partition.get_degree_of_block(neighbor_block) - change_degree_of_block)
                + self.epsilon * self.partition.B)

    # ---- directed versions

    def infer_stepwise_directed(self):
        node = self.partition.get_random_node()
        from_block = self.partition.get_block_of_node(node)
        s = self._select_new_block_directed(node)

        acceptance_probability = self._calculate_acceptance_probability_directed(node, from_block, s)

        if rd.random() < acceptance_probability and from_block != s:
            self.partition.move_node(node, s)
            self.node_moves += 1

        self.performed_steps += 1
        self.update_temperature()

    def _calculate_acceptance_probability_directed(self, node, from_block, to_block, debug=False):
        """
        Calculate the acceptance probability, i.e. calculate the following formula:
         acceptance_probability = e^(beta*Delta)
         *(sum_{neighbor t} (fraction of neighbors of node i belonging to block t)*p(s->r|t))
         /(sum_{neighbor t} (fraction of neighbors of node i belonging to block t)*p(r->s|t))
         where top values are calculated after the move, i.e. change of ert and et,
         and for p(s->r|t) see complex or simple probability.
        """
        s = to_block
        # calculate delta and parameter, which are needed for the rest of the formula too
        parameter = self.partition.precalc_move((node, from_block, s), self._objective_function)
        delta = self._objective_function.calculate_delta(self.partition, from_block, s, *parameter)

        top = 0.0
        bottom = 0.0

        kit, kti, selfloops, in_degree, out_degree = parameter
        for neighbor in kit:
            if from_block == s and neighbor == s:
                top += (kit[neighbor] + kti[neighbor] + 2 * selfloops) / (in_degree + out_degree) \
                       * self._probability_directed(from_block, neighbor)
                bottom += (kit[neighbor] + kti[neighbor] + 2 * selfloops) / (in_degree + out_degree) \
                    * self._probability_directed(s, neighbor)
            elif neighbor == from_block:
                kti_value = kti[neighbor]
                top += (kit[neighbor] + kti_value) \
                    / (in_degree + out_degree) \
                    * self._probability_directed(from_block,
                                                 neighbor,
                                                 kit[from_block] + kti_value + selfloops,
                                                 out_degree,
                                                 kit[from_block] + kti_value + selfloops,
                                                 in_degree)
                bottom += (kit[neighbor] + kti_value + 2 * selfloops) \
                    / (in_degree + out_degree) \
                    * self._probability_directed(s, neighbor)
            elif neighbor == s:
                kti_value = kti[neighbor]
                top += (kit[neighbor] + kti_value + 2 * selfloops) \
                    / (in_degree + out_degree) \
                    * self._probability_directed(from_block,
                                                 neighbor,
                                                 -kit[from_block] + kti[neighbor],
                                                 -out_degree,
                                                 -kti[from_block] + kit[neighbor],
                                                 -in_degree)
                bottom += (kit[neighbor] + kti_value) \
                    / (in_degree + out_degree) \
                    * self._probability_directed(s, neighbor)
            elif from_block != s:
                kti_value = kti.pop(neighbor, 0)
                #      change the est and ets values according
                top += (kit[neighbor] + kti_value) \
                    / (in_degree + out_degree) \
                    * self._probability_directed(from_block, neighbor,
                                                 change_est=kit[neighbor], change_ets=kti_value)
                bottom += (kit[neighbor] + kti_value) \
                    / (in_degree + out_degree) \
                    * self._probability_directed(s, neighbor)
            else:
                kti_value = kti.pop(neighbor, 0)
                #      no change else because these e_ts entries aren't influenced by the move
                top += (kit[neighbor] + kti_value) \
                    / (in_degree + out_degree) \
                    * self._probability_directed(from_block, neighbor)
                bottom += (kit[neighbor] + kti_value) \
                    / (in_degree + out_degree) \
                    * self._probability_directed(s, neighbor)

                #  now handle all neighbors in kti not already gotten by kit
                # first remove s and from block because both were already handled in loop before
                #  but wasn't removed because the value was needed in another loop cycle
        kti.pop(from_block)
        if s != from_block:
            kti.pop(s)
        for neighbor in kti:
            # kit value don't exists -> kit=0
            if from_block != s:
                top += kti[neighbor] / (in_degree + out_degree) \
                       * self._probability_directed(from_block, neighbor, change_ets=kti[neighbor])
                bottom += kti[neighbor] / (in_degree + out_degree) \
                    * self._probability_directed(s, neighbor)
            else:
                top += kti[neighbor] / (in_degree + out_degree) \
                       * self._probability_directed(from_block, neighbor)
                bottom += kti[neighbor] / (in_degree + out_degree) \
                    * self._probability_directed(s, neighbor)

        # for better testing
        if debug:
            return top, bottom
        # bottom = zero means there is no chance that these block can be selected
        #  to ensure no division by zero error catch this case and accept impossible move
        #  (this is the unique continuous extension)
        if bottom == 0:
            return 1
        # else...
        try:
            # catch large values
            # math.exp(710)= inf
            if self.beta * delta > 700:
                value = 1
            else:
                value = min(math.exp(self.beta * delta) / bottom * top, 1)
        except OverflowError:
            value = 1
        return value

    def _select_new_block_directed(self, node):
        """
        Select a new block s with probability sum_t p^i_tp(r->s|t) where
        p^i_t is the fraction of neighbors of node i which belongs to block t and
        p(r->s|t) is given like stated in _probability.
        (Node i is the node to be moved)
        """
        # see Peixoto possible with the following procedure (at least for undirected case):
        #  idea use that p(r->s|t)=(1-R_t)e_{ts}/degree_of_block(t)+R_t/B
        #   with R_t=\epsilon*B/(degree_of_block(t)+epsilon*B)
        # 1. Select a random neighbor j of the node i and obtain its block membership->t
        t = self.partition.get_block_of_node(
            rd.choice(self.partition.get_neighbors_of_node(node)))
        # 2. Select s randomly from all B choices with equal probability
        if self.limit_possible_blocks:
            block = self.partition.get_block_of_node(node)
            s = rd.choice(self.partition.get_possible_blocks(block))
        else:
            s = rd.choice(range(self.partition.B))
        # 3. Accept with probability R_t

        #     directed case is an extension of Peixoto's ansatz
        rt = self.epsilon * self.partition.B \
            / (self.partition.get_in_degree_of_block(t) + self.partition.get_out_degree_of_block(t)
                + self.epsilon * self.partition.B)
        if rd.random() < rt:
            return s
        # 4. If rejected chose a random edge adjacent to block t and the label s
        #    is taken from the opposite endpoint
        # else:...
        if not self.limit_possible_blocks:
            to_node = self.partition.get_random_neighboring_edge_of_block(t)[1]
            return self.partition.get_block_of_node(to_node)
        # else need additional check if move is allowed
        # noinspection PyUnboundLocalVariable
        possible_blocks = self.partition.get_possible_blocks(block, with_in_operation=True)
        for _ in range(25):
            to_node = self.partition.get_random_neighboring_edge_of_block(t)[1]
            new_block = self.partition.get_block_of_node(to_node)
            if new_block in possible_blocks:
                return new_block
        new_block = rd.choice(self.partition.get_possible_blocks(block))
        # else simple return any random block
        return new_block

    def _probability_directed(self, to_block, neighbor_block,
                              change_ets=0, change_degree_of_block=0,
                              change_est=0, change_in_degree_of_block=0):
        """
        Return the probability p(r-s|t).
        The formula for directed and undirected graphs are different.

        Parameters
        ----------
        to_block : int
            The block which should be the next block. (Block s of the formula)
        neighbor_block : int
            The neighboring block which information should be used (block t of the formula)

        With the additional optional parameters the values of e_ts and degree of
        block can be changed.

        ------

        For directed graphs:
        p(r->s|t)= frac{e_{ts}+e_{st}+epsilon}{in_degree_of_block(t)+out_degree_of_block(t)+epsilon*B}
        r = from_block
        s = to_block
        t = block of neighbor

        Again with the optional parameters to change the values.
        Change_degree_of_block change the value of out_degree of block.
        """
        return ((self.partition.get_edge_count(neighbor_block, to_block) - change_ets)
                + (self.partition.get_edge_count(to_block, neighbor_block) - change_est)
                + self.epsilon) \
            / ((self.partition.get_out_degree_of_block(neighbor_block) - change_degree_of_block)
                + (self.partition.get_in_degree_of_block(neighbor_block) - change_in_degree_of_block)
                + self.epsilon * self.partition.B)
# @formatter:on


class MetropolisHastingInferenceTenK(MetropolisHastingInference):
    """Fix Number of Steps"""
    title = "Metropolis Hasting Inference 10k"
    short_title = "MHA 10k"

    def __init__(self, graph, objective_function, partition):
        super(MetropolisHastingInferenceTenK, self).__init__(graph, objective_function, partition)
        self.default_number_of_steps = 10000


class MetropolisHastingInferenceFiftyK(MetropolisHastingInference):
    """Fix Number of Steps"""
    title = "Metropolis Hasting Inference 50k"
    short_title = "MHA 50k"

    def __init__(self, graph, objective_function, partition):
        super(MetropolisHastingInferenceFiftyK, self).__init__(graph, objective_function, partition)
        self.default_number_of_steps = 50000


class MetropolisHastingInferenceHundredK(MetropolisHastingInference):
    """Fix Number of Steps"""
    title = "Metropolis Hasting Inference 100k"
    short_title = "MHA 100k"

    def __init__(self, graph, objective_function, partition):
        super(MetropolisHastingInferenceHundredK, self).__init__(graph, objective_function, partition)
        self.default_number_of_steps = 100000


class MetropolisHastingInferenceTwoHundredFiftyK(MetropolisHastingInference):
    """Fix Number of Steps"""
    title = "Metropolis Hasting Inference 250k"
    short_title = "MHA 250k"

    def __init__(self, graph, objective_function, partition):
        super(MetropolisHastingInferenceTwoHundredFiftyK, self).__init__(graph, objective_function, partition)
        self.default_number_of_steps = 250000


class MetropolisHastingInferenceFiveHundredK(MetropolisHastingInference):
    """Fix Number of Steps"""
    title = "Metropolis Hasting Inference 500k"
    short_title = "MHA 500k"

    def __init__(self, graph, objective_function, partition):
        super(MetropolisHastingInferenceFiveHundredK, self).__init__(graph, objective_function, partition)
        self.default_number_of_steps = 500000


class MetropolisHastingInferenceSimulatedAnnealingCauchy(MetropolisHastingInference):
    """Metropolis Hasting Inference Algorithm with Cauchy Simulated Annealing"""
    title = "Metropolis Hasting Inference Simulated Annealing Cauchy"
    short_title = "MHAC 1k"

    def update_temperature(self):
        if self.beta_zero is None:
            self._initialize_temperature()
        self.beta = (1 + self.performed_steps) / self.beta_zero


class MetropolisHastingInferenceSimulatedAnnealingCauchyFiftyK(MetropolisHastingInferenceSimulatedAnnealingCauchy):
    """Fix Number of Steps"""
    title = "Metropolis Hasting Inference Simulated Annealing Cauchy 50k"
    short_title = "MHAC 50k"

    def __init__(self, graph, objective_function, partition):
        super(MetropolisHastingInferenceSimulatedAnnealingCauchy, self).__init__(graph, objective_function, partition)
        self.default_number_of_steps = 50000


class MetropolisHastingInferenceSimulatedAnnealingCauchyTwoHundredFiftyK(
    MetropolisHastingInferenceSimulatedAnnealingCauchy):
    """Fix Number of Steps"""
    title = "Metropolis Hasting Inference Simulated Annealing Cauchy 250k"
    short_title = "MHAC 250k"

    def __init__(self, graph, objective_function, partition):
        super(MetropolisHastingInferenceSimulatedAnnealingCauchy, self).__init__(graph, objective_function, partition)
        self.default_number_of_steps = 250000


class MetropolisHastingInferenceSimulatedAnnealingBoltzman(MetropolisHastingInference):
    """Metropolis Hasting Inference Algorithm with Cauchy Simulated Annealing"""
    title = "Metropolis Hasting Inference Simulated Annealing Boltzman"
    short_title = "MHAB 1k"

    def update_temperature(self):
        if self.beta_zero is None:
            self._initialize_temperature()
        self.beta = math.log(1 + self.performed_steps) / self.beta_zero


class MetropolisHastingInferenceSimulatedAnnealingBoltzmanFiftyK(MetropolisHastingInferenceSimulatedAnnealingBoltzman):
    """Fix Number of Steps"""
    title = "Metropolis Hasting Inference Simulated Annealing Boltzman 50k"
    short_title = "MHAB 50k"

    def __init__(self, graph, objective_function, partition):
        super(MetropolisHastingInferenceSimulatedAnnealingBoltzman, self).__init__(graph, objective_function, partition)
        self.default_number_of_steps = 50000


class MetropolisHastingInferenceSimulatedAnnealingBoltzmanTwoHundredFiftyK(
    MetropolisHastingInferenceSimulatedAnnealingBoltzman):
    """Fix Number of Steps"""
    title = "Metropolis Hasting Inference Simulated Annealing Boltzman 250k"
    short_title = "MHAB 250k"

    def __init__(self, graph, objective_function, partition):
        super(MetropolisHastingInferenceSimulatedAnnealingBoltzman, self).__init__(graph, objective_function, partition)
        self.default_number_of_steps = 250000


class KarrerInference(Inference):
    """
    Heuristic Inference Algorithm described in Karrer and Newman 2011
        slightly enhanced?
    """
    title = "Karrer Inference"
    short_title = "KL-G"

    def __init__(self, graph, objective_function, partition, no_negative_move=False, limit_possible_blocks=False):
        super(KarrerInference, self).__init__(graph, objective_function, partition)
        self.no_negative_move = no_negative_move
        self.limit_possible_blocks = limit_possible_blocks
        self._last_objective_value = float('-inf')

    def infer_stochastic_block_model(self):
        try:
            for _ in range(100):
                self.infer_stepwise()
            else:
                raise Exception("Could not find minimum in 100 steps" + str(self.partition.get_representation()) + str(
                    self.partition.graph.edges()))
        except StopIteration:
            pass

    def infer_stepwise(self):
        saved_representation = self.partition.get_representation()
        improve = 0
        overall_improve = 0
        best_partition_representation = None
        iteration_moves = 0
        moves = 0
        for node in self.partition.get_nodes_iter():
            from_block = self.partition.get_block_of_node(node)
            next_block = from_block
            move_delta = -float("inf")
            if self.limit_possible_blocks:
                possible_blocks = list(self.partition.get_possible_blocks(from_block))
            else:
                possible_blocks = list(range(self.partition.B))
            if self.no_negative_move:
                possible_blocks.remove(from_block)
            for block in possible_blocks:
                if block != from_block:
                    parameter = self.partition.precalc_move((node, from_block, block),
                                                            self._objective_function)
                    delta = self._objective_function.calculate_delta(
                        self.partition, from_block, block, *parameter)
                else:
                    delta = 0.0
                if delta > move_delta:
                    move_delta = delta
                    next_block = block

            if not self.no_negative_move or move_delta > 0:
                self.partition.move_node(node, next_block)
                moves += 1

            improve += move_delta
            if improve > 0:
                overall_improve += improve
                improve = 0
                best_partition_representation = self.partition.get_representation()
                iteration_moves += moves
                moves = 0

        # overall improve real positive to ignore rounding errors
        if overall_improve > 0.001:
            self.partition.set_from_representation(best_partition_representation)
            self.node_moves += iteration_moves

            actual_value = self.objective_function.calculate(self.partition)
            if actual_value < self._last_objective_value + 0.01:
                if actual_value < self._last_objective_value - .1:
                    # if new one is worse then retrieve old state
                    self.partition.set_from_representation(saved_representation)
                raise StopIteration()
            else:
                self._last_objective_value = actual_value
        else:
            # if no improvement set back to old partition
            self.partition.set_from_representation(saved_representation)
            raise StopIteration()


class KarrerInferenceNoNegativeMove(KarrerInference):
    """ Karrer with No Negative Move"""
    title = "Karrer Inference with no negative move"
    short_title = "KL-G nn"

    def __init__(self, graph, objective_function, partition, limit_possible_blocks=False):
        super(KarrerInferenceNoNegativeMove, self).__init__(graph, objective_function, partition, no_negative_move=True,
                                                            limit_possible_blocks=limit_possible_blocks)


class EMInference(Inference):
    """Expectation-Maximization Algorithm for SBM inference"""
    title = "Expectation Maximization Inference"
    short_title = "KL-EM"

    def __init__(self, graph, objective_function, partition, with_toggle_detection=True, limit_possible_blocks=False):
        super(EMInference, self).__init__(graph, objective_function, partition)
        self.with_toggle_detection = with_toggle_detection
        self._old_value = self._objective_function.calculate(partition)
        self.limit_possible_blocks = limit_possible_blocks

    def infer_stochastic_block_model(self):
        if self.partition.is_graph_directed():
            try:
                for _ in range(2 * len(self.graph)):
                    self.infer_stepwise_directed()
                else:
                    print("EMInference: could not find an optimal partition in", 2 * len(self.graph), "steps",
                          self.partition.get_representation(), self.graph.edges())
            except StopIteration:
                pass
        else:
            try:
                for _ in range(2 * len(self.graph)):
                    self.infer_stepwise_undirected()
                else:
                    print("EMInference: could not find an optimal partition in", 2 * len(self.graph), "steps",
                          self.partition.get_representation(), self.graph.edges())
            except StopIteration:
                pass

    def infer_stepwise(self):
        if self.partition.is_graph_directed():
            self.infer_stepwise_directed()
        else:
            self.infer_stepwise_undirected()

    def infer_stepwise_undirected(self):
        """
        For each node retrieve the best block. Then move all nodes to the new best block.

        Easy ansatz tend/allow to toggle between two states in the end.
        Therefore here is a simple approach to detect this status and resolve it included.
        """
        # save representation of partition in case for overall decrease
        saved_representation = self.partition.get_representation()
        # set flag which checks if we find any improve
        improve = False
        # count number of moves, in case everything is fine
        iteration_moves = 0
        # keep list of moves, which will be performed after all calculations
        moves = []
        possible_blocks = list(range(self.partition.B))
        nodes_moved = {block: 0 for block in range(self.partition.B)}

        for node in self.partition.get_nodes_iter():
            from_block = self.partition.get_block_of_node(node)
            #     ensure that one don't move the last node out of the block
            if self.partition.get_number_of_nodes_in_block(from_block) - nodes_moved[from_block] == 1:
                continue
            next_block = from_block
            move_delta = 0

            if self.limit_possible_blocks:
                possible_blocks = self.partition.get_possible_blocks(from_block)

            possible_blocks.remove(from_block)
            parameter = self.partition.precalc_move((node, from_block, from_block),
                                                    self._objective_function)
            for block in possible_blocks:
                if block not in parameter[0]:
                    parameter[0][block] = 0
                delta = self._objective_function.calculate_delta(
                    self.partition, from_block, block, *parameter)

                if parameter[0][block] == 0:
                    del parameter[0][block]
                if delta > move_delta:
                    move_delta = delta
                    next_block = block

            possible_blocks.append(from_block)
            if move_delta > 0.001:
                moves.append((node, next_block))
                nodes_moved[from_block] += 1
                iteration_moves += 1
                improve = True

        # perform moves
        for move in moves:
            self.partition.move_node(*move)

        if improve:
            if self.with_toggle_detection:
                new_value = self._objective_function.calculate(self.partition)
                # detect decreasing objective function value and prevent decrease by only doing last move
                if new_value <= self._old_value:
                    self.partition.set_from_representation(saved_representation)
                    # carefully move all nodes which increase the value
                    for node, to_block in moves:
                        from_block = self.partition.get_block_of_node(node)
                        parameter = self.partition.precalc_move((node, from_block, to_block),
                                                                self._objective_function)
                        delta = self._objective_function.calculate_delta(
                            self.partition, from_block, to_block, *parameter)
                        if delta > 0.001:
                            self.partition.move_node(node, to_block)
                            self.node_moves += 1
                else:
                    self._old_value = new_value
                    self.node_moves += iteration_moves
        else:
            # No move -> no change -> stop algorithm
            raise StopIteration()

    def infer_stepwise_directed(self):
        """
        For each node retrieve the best block. Then move all nodes to the new best block.

        Easy ansatz tend/allow to toggle between two states in the end.
        Therefore here is a simple approach to detect this status and resolve it included.
        """
        # save representation of partition in case for overall decrease
        saved_representation = self.partition.get_representation()
        # set flag which checks if we find any improve
        improve = False
        # count number of moves, in case everything is fine
        iteration_moves = 0
        # keep list of moves, which will be performed after all calculations
        moves = []
        possible_blocks = list(range(self.partition.B))
        nodes_moved = {block: 0 for block in range(self.partition.B)}

        for node in self.partition.get_nodes_iter():
            from_block = self.partition.get_block_of_node(node)
            #     ensure that one don't move the last node out of the block
            if self.partition.get_number_of_nodes_in_block(from_block) - nodes_moved[from_block] == 1:
                continue
            next_block = from_block
            move_delta = 0

            if self.limit_possible_blocks:
                possible_blocks = self.partition.get_possible_blocks(from_block)

            possible_blocks.remove(from_block)
            parameter = self.partition.precalc_move((node, from_block, from_block),
                                                    self._objective_function)
            for block in possible_blocks:
                if block not in parameter[0]:
                    parameter[0][block] = 0
                if block not in parameter[1]:
                    parameter[1][block] = 0
                delta = self._objective_function.calculate_delta(
                    self.partition, from_block, block, *parameter)

                if parameter[0][block] == 0:
                    del parameter[0][block]
                if parameter[1][block] == 0:
                    del parameter[1][block]
                if delta > move_delta:
                    move_delta = delta
                    next_block = block

            possible_blocks.append(from_block)
            if move_delta > 0.001:
                moves.append((node, next_block))
                nodes_moved[from_block] += 1
                iteration_moves += 1
                improve = True

        # perform moves
        for move in moves:
            self.partition.move_node(*move)

        if improve:
            if self.with_toggle_detection:
                new_value = self._objective_function.calculate(self.partition)
                # detect decreasing objective function value and prevent decrease by only doing last move
                if new_value <= self._old_value:
                    self.partition.set_from_representation(saved_representation)
                    # carefully move all nodes which increase the value
                    for node, to_block in moves:
                        from_block = self.partition.get_block_of_node(node)
                        parameter = self.partition.precalc_move((node, from_block, to_block),
                                                                self._objective_function)
                        delta = self._objective_function.calculate_delta(
                            self.partition, from_block, to_block, *parameter)
                        if delta > 0.001:
                            self.partition.move_node(node, to_block)
                            self.node_moves += 1
                else:
                    self._old_value = new_value
                    self.node_moves += iteration_moves
        else:
            # No move -> no change -> stop algorithm
            raise StopIteration()


class KerninghanLinInference(Inference):
    title = "Kerninghan-Lin"
    short_title = "KL"

    def __init__(self, graph, objective_function, partition, no_negative_move=False):
        super(KerninghanLinInference, self).__init__(graph, objective_function, partition)
        self.no_negative_move = no_negative_move
        # always check for improvement as additional condition to break the loop
        self._last_objective_value = objective_function.calculate(partition)

    def infer_stochastic_block_model(self, *args):
        if self.partition.is_graph_directed():
            for _ in range(100):
                try:
                    self.infer_stepwise_directed()
                except StopIteration:
                    break
            else:
                raise Exception("Could not find minimum in 100 steps" + str(self.partition.get_representation()) + str(
                    self.partition.graph.edges()))
        else:
            for _ in range(100):
                try:
                    self.infer_stepwise_undirected()
                except StopIteration:
                    break
            else:
                raise Exception("Could not find minimum in 100 steps" + str(self.partition.get_representation()) + str(
                    self.partition.graph.edges()))

    def infer_stepwise_undirected(self):
        # calculate optimal values
        optimal_values = {}
        optimal_blocks = {}
        overall_optimal_move = ()
        overall_optimal_value = float('-inf')
        negative_move_counter = 3
        start_representation = self.partition.get_representation()
        # init values
        for node in self.partition.get_nodes_iter():
            best_found = float('-inf')
            best_block = None
            actual_block = self.partition.get_block_of_node(node)
            precalc_info = self.partition.precalc_move((node, actual_block, actual_block), self.objective_function)
            for other_block in self.partition.get_possible_blocks(actual_block):
                if other_block == actual_block:
                    continue
                if other_block not in precalc_info[0]:
                    precalc_info[0][other_block] = 0

                delta = self.objective_function.calculate_delta(self.partition, actual_block, other_block,
                                                                *precalc_info)
                if delta > best_found:
                    best_found = delta
                    best_block = other_block

                if precalc_info[0][other_block] == 0:
                    del precalc_info[0][other_block]

            if best_block is not None:
                optimal_values[node] = best_found
                optimal_blocks[node] = best_block

                if best_found > overall_optimal_value:
                    overall_optimal_value = best_found
                    overall_optimal_move = (node, actual_block, best_block)

        history_values = []
        history_moves = []
        while len(optimal_values) > 0:
            if self.no_negative_move and overall_optimal_value < 0:
                negative_move_counter -= 1
                if negative_move_counter == 0:
                    break
            history_values.append(overall_optimal_value)
            history_moves.append(overall_optimal_move)

            # remove from candidates
            del optimal_values[overall_optimal_move[0]]
            del optimal_blocks[overall_optimal_move[0]]

            # perform move
            self.partition.move_node(overall_optimal_move[0], overall_optimal_move[2])

            overall_optimal_move = ()
            overall_optimal_value = float('-inf')

            for node in list(optimal_values.keys()):
                best_found = float('-inf')
                best_block = None
                actual_block = self.partition.get_block_of_node(node)
                precalc_info = self.partition.precalc_move((node, actual_block, actual_block), self.objective_function)
                for other_block in self.partition.get_possible_blocks(actual_block):
                    if other_block == actual_block:
                        continue
                    if other_block not in precalc_info[0]:
                        precalc_info[0][other_block] = 0
                    delta = self.objective_function.calculate_delta(self.partition, actual_block, other_block,
                                                                    *precalc_info)
                    if delta > best_found:
                        best_found = delta
                        best_block = other_block

                    if precalc_info[0][other_block] == 0:
                        del precalc_info[0][other_block]

                if best_block is not None:
                    optimal_values[node] = best_found
                    optimal_blocks[node] = best_block

                    if best_found > overall_optimal_value:
                        overall_optimal_value = best_found
                        overall_optimal_move = (node, actual_block, best_block)
                else:
                    del optimal_values[node]
                    del optimal_blocks[node]

        # all nodes moved
        # determine optimal partial sum
        optimal_sum = 0
        optimal_index = -1
        actual_sum = 0
        for i, value in enumerate(history_values):
            actual_sum += value
            if actual_sum > optimal_sum:
                optimal_sum = actual_sum
                optimal_index = i

        # check if any real improvement is possible
        # difference to total sum (last value in actual sum) to deal with cases where the result is not 0
        if len(history_values) > optimal_index >= 0 and optimal_sum > 0.01 and (
                actual_sum < 0.01 or optimal_sum - actual_sum > 0.01):
            # retrieve state of this case
            for node, old_block, _ in history_moves[optimal_index + 1:]:
                self.partition.move_node(node, old_block)

            # extra checks to stop endless loops
            actual_value = self.objective_function.calculate(self.partition)
            if actual_value < self._last_objective_value + 0.01:
                # only minor improvement or no improvement -> stop
                if actual_value < self._last_objective_value - .1:
                    # if new one is worse then retrieve old state
                    self.partition.set_from_representation(start_representation)
                raise StopIteration()
            self._last_objective_value = actual_value
        else:
            # no improve possible go back to start
            self.partition.set_from_representation(start_representation)
            raise StopIteration()

    def infer_stepwise_directed(self):
        # calculate optimal values
        optimal_values = {}
        optimal_blocks = {}
        overall_optimal_move = ()
        overall_optimal_value = float('-inf')
        negative_move_counter = 3
        start_representation = self.partition.get_representation()
        # init values
        for node in self.partition.get_nodes_iter():
            best_found = float('-inf')
            best_block = None
            actual_block = self.partition.get_block_of_node(node)
            precalc_info = self.partition.precalc_move((node, actual_block, actual_block), self.objective_function)
            for other_block in self.partition.get_possible_blocks(actual_block):
                if other_block == actual_block:
                    continue
                if other_block not in precalc_info[0]:
                    precalc_info[0][other_block] = 0
                if other_block not in precalc_info[1]:
                    precalc_info[1][other_block] = 0
                delta = self.objective_function.calculate_delta(self.partition, actual_block, other_block,
                                                                *precalc_info)
                if delta > best_found:
                    best_found = delta
                    best_block = other_block

                if precalc_info[0][other_block] == 0:
                    del precalc_info[0][other_block]
                if precalc_info[1][other_block] == 0:
                    del precalc_info[1][other_block]

            if best_block is not None:
                optimal_values[node] = best_found
                optimal_blocks[node] = best_block

                if best_found > overall_optimal_value:
                    overall_optimal_value = best_found
                    overall_optimal_move = (node, actual_block, best_block)

        history_values = []
        history_moves = []
        while len(optimal_values) > 0:
            if self.no_negative_move and overall_optimal_value < 0:
                negative_move_counter -= 1
                if negative_move_counter == 0:
                    break
            history_values.append(overall_optimal_value)
            history_moves.append(overall_optimal_move)

            # remove from candidates
            del optimal_values[overall_optimal_move[0]]
            del optimal_blocks[overall_optimal_move[0]]

            # perform move
            self.partition.move_node(overall_optimal_move[0], overall_optimal_move[2])

            overall_optimal_move = ()
            overall_optimal_value = float('-inf')

            for node in list(optimal_values.keys()):
                best_found = float('-inf')
                best_block = None
                actual_block = self.partition.get_block_of_node(node)
                precalc_info = self.partition.precalc_move((node, actual_block, actual_block), self.objective_function)
                for other_block in self.partition.get_possible_blocks(actual_block):
                    if other_block == actual_block:
                        continue
                    if other_block not in precalc_info[0]:
                        precalc_info[0][other_block] = 0
                    if other_block not in precalc_info[1]:
                        precalc_info[1][other_block] = 0
                    delta = self.objective_function.calculate_delta(self.partition, actual_block, other_block,
                                                                    *precalc_info)
                    if delta > best_found:
                        best_found = delta
                        best_block = other_block

                    if precalc_info[0][other_block] == 0:
                        del precalc_info[0][other_block]
                    if precalc_info[1][other_block] == 0:
                        del precalc_info[1][other_block]

                if best_block is not None:
                    optimal_values[node] = best_found
                    optimal_blocks[node] = best_block

                    if best_found > overall_optimal_value:
                        overall_optimal_value = best_found
                        overall_optimal_move = (node, actual_block, best_block)
                else:
                    del optimal_values[node]
                    del optimal_blocks[node]

        # all nodes moved
        # determine optimal partial sum
        optimal_sum = 0
        optimal_index = -1
        actual_sum = 0
        for i, value in enumerate(history_values):
            actual_sum += value
            if actual_sum > optimal_sum:
                optimal_sum = actual_sum
                optimal_index = i

        # check if any real improvement is possible
        # difference to total sum (last value in actual sum) to deal with cases where the result is not 0
        if len(history_values) > optimal_index >= 0 and optimal_sum > 0.01 and (
                actual_sum < 0.01 or optimal_sum - actual_sum > 0.01):
            # retrieve state of this case
            for node, old_block, _ in history_moves[optimal_index + 1:]:
                self.partition.move_node(node, old_block)

            # extra checks to stop endless loops
            actual_value = self.objective_function.calculate(self.partition)
            if actual_value < self._last_objective_value + 0.01:
                # only minor improvement or no improvement -> stop
                if actual_value < self._last_objective_value - .1:
                    # if new one is worse then retrieve old state
                    self.partition.set_from_representation(start_representation)
                raise StopIteration()
            self._last_objective_value = actual_value
        else:
            # no improve possible go back to start
            self.partition.set_from_representation(start_representation)
            raise StopIteration()

    def infer_stepwise(self):
        if self.partition.is_graph_directed():
            self.infer_stepwise_directed()
        else:
            self.infer_stepwise_undirected()


class KerninghanLinInferenceNoNegative(KerninghanLinInference):
    title = "Kerninghan-Lin No Negative Moves"
    short_title = "KL nn"

    def __init__(self, graph, objective_function, partition):
        super(KerninghanLinInferenceNoNegative, self).__init__(graph, objective_function, partition,
                                                               no_negative_move=True)


class SpectralInference(Inference):
    title = "Spectral Inference"
    short_title = "SI"

    # based on Krzakala 2013: "Spectral redemption in clustering sparse networks"

    def __init__(self, graph, objective_function, partition):
        super(SpectralInference, self).__init__(graph, objective_function, partition)
        self.A = nx.to_numpy_matrix(self.graph)
        self.partition = partition

    def infer_stochastic_block_model(self):
        # get B' from the paper (named B here)
        size = len(self.A)
        degree = nx.degree(self.graph, weight='weight')
        degree = list(degree)
        degree_list = []
        # get matrix with entrys of -1 for B' [0][1]
        minus_ones = np.zeros((size, size))
        for i in range(len(degree)):
            degree_list.append(degree[i][1] - 1)  # -1 because of D-1
            minus_ones[i][i] = -1
        D = np.diag(degree_list)
        B = np.zeros((2 * size, 2 * size))
        for i in range(2 * size):
            for j in range(2 * size):
                if i < size and j < size:  # B'[0][0] = 0 (see paper)
                    continue
                if i >= size and j < size:  # B'[0][1] = -1 (see paper)
                    B[i][j] = minus_ones[i - size][j]
                if i < size and j >= size:  # B'[1][0] = D-1 (see paper) but our D is already -1
                    B[i][j] = D[i][j - size]
                if i >= size and j >= size:  # B'[1][1] = A (see paper)
                    B[i][j] = self.A[i - size, j - size]

        # get eigenvalues (values) and eigenvectors (vectors)
        values, vectors = la.eig(B)
        # make sure, that vectors is not empty, even if all eigenvalues are partly imaginary,
        # if possible get rid of vectors of imaginary eigenvalues
        number_of_clusters = max(list(list(self.partition.get_representation().values()))) + 1
        test = vectors[:, abs(values.imag) > .0001]
        vectors_2 = vectors

        # if there are enough eigenvectors without imaginary part:
        if len(test[0]) <= len(vectors[0]) - number_of_clusters:
            vectors = vectors[:, abs(values.imag) < .0001]
            size_i = len(vectors[0])
            size_j = len(vectors)
            vectors_real = np.zeros((size_j, size_i))
            for i in range(size_j):
                for j in range(size_i):
                    vectors_real[i][j] = vectors[i][j].real
            vectors_real = vectors_real[len(self.A):2 * len(self.A), :]
            vectors = vectors_real

        # if there are not enough eigenvectors without imaginary part:
        else:
            vectors_2 = vectors_2[:, abs(values.imag) >= .0001]
            size_i = len(vectors_2[0])
            size_j = len(vectors_2)
            vectors_real = np.zeros((size_j, size_i))
            vectors_imag = np.zeros((size_j, size_i))
            # split real and imaginary part
            for i in range(size_j):
                for j in range(size_i):
                    vectors_real[i][j] = vectors_2[i][j].real
                    vectors_imag[i][j] = vectors_2[i][j].imag

            # use both parts as own vector
            separated_vectors = np.concatenate((vectors_real, vectors_imag), axis=1)
            separated_vectors = separated_vectors[len(self.A):2 * len(self.A), :]
            vectors = separated_vectors
        kmeans = cl.KMeans(n_clusters=number_of_clusters, random_state=None).fit_predict(vectors)
        # update partition
        node_list = list(self.graph.nodes())
        for i in range(len(node_list)):
            self.partition.move_node(node_list[i], kmeans[i])
        return self.partition
