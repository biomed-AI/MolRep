from .inference import Inference
from .inference import KarrerInferenceNoNegativeMove
from .inference import PeixotoInference
from .peixotos_hierarchical_sbm import NxHierarchicalPartition


class HierarchicalInference(Inference):
    """
     Hierarchical Inference based on Peixotos agglomerate algorithm and
     a search of the complete range of possible number of blocks via an adaption of the golden section search
     """
    short_title = "HINF"
    title = "Hierarchical Inference"

    def __init__(self, graph, objective_function, hierarchical_partition, delta_objective_function_for_level_removal,
                 inference_class=KarrerInferenceNoNegativeMove,
                 inference_class_instance_parameters=None):
        """
        :param graph:
        :type graph nx.Digraph
        :param objective_function:
        :param hierarchical_partition:
        :type hierarchical_partition NxHierarchicalPartition
        :param inference_class:
        :type inference_class class SBMInference
        """
        super(HierarchicalInference, self).__init__(graph, objective_function, hierarchical_partition)
        self.inference_class = inference_class
        if inference_class == KarrerInferenceNoNegativeMove and inference_class_instance_parameters is None:
            self.inference_class_instance_parameters = {"limit_possible_blocks": True}
        else:
            self.inference_class_instance_parameters = inference_class_instance_parameters
        self._level_status = [False for _ in range(hierarchical_partition.max_level + 1)]
        self.delta_objective_function_for_level_removal = delta_objective_function_for_level_removal

        self._inference_algorithm_parameters = []
        self.actual_likelihood = self.objective_function.calculate(self.partition)

        self._epsilon = 0.1
        self.viewed_level = hierarchical_partition.max_level
        self._merging_inference = self.inference_class == PeixotoInference

    def infer_stochastic_block_model(self, *args):
        self.viewed_level = self.partition.max_level

        if args:
            self._inference_algorithm_parameters = args

        try:
            for _ in range(100):
                self.infer_stepwise()
            else:
                raise Exception("Could not find minimum in 100 steps" + str(self.partition.get_representation()) + str(
                    self.partition.graph.edges()))
        except StopIteration:
            pass

    def infer_stepwise(self, *args):
        # same as above in inner loop
        if args:
            self._inference_algorithm_parameters = args

        self.actual_likelihood = self.objective_function.calculate(self.partition)

        # care about one level
        if self._level_status[self.viewed_level]:
            # if its already done skip level
            if self.viewed_level > 0:
                self.viewed_level -= 1
                return
            else:
                # minimum found
                raise StopIteration()

        any_move_successful = False
        move_levels_up = 1
        # first try resize
        if self.resize_level(self.viewed_level):
            any_move_successful = True
            self._level_status[self.viewed_level] = True

            # update neighboring information
            if self.viewed_level > 0:
                self._level_status[self.viewed_level - 1] = False
            if self.viewed_level < self.partition.max_level:
                self._level_status[self.viewed_level + 1] = False

        if self.add_level(self.viewed_level):
            any_move_successful = True
            self._level_status[self.viewed_level] = False

            # update neighboring information
            if self.viewed_level > 0:
                self._level_status[self.viewed_level - 1] = False

            if self.viewed_level < self.partition.max_level - 1:
                # if not old max level update the level above
                self._level_status[self.viewed_level + 1] = False
                move_levels_up += 1

            # now add new level (after actual level
            self._level_status.insert(self.viewed_level + 1, False)

        # now try delete if not lowest level!
        if self.viewed_level > 0:
            if self.delete_level(self.viewed_level):
                any_move_successful = True
                # delete entry of deleted level
                del self._level_status[self.viewed_level]
                # now self.viewed_level point to level above

                # take this into account when moving the pointer
                move_levels_up -= 1
                # test on >0 already done -> mark level below as not done
                self._level_status[self.viewed_level - 1] = False
                # check if above level exists and mark as not done
                if self.viewed_level < self.partition.max_level:
                    self._level_status[self.viewed_level] = False

        if any_move_successful and self.viewed_level + move_levels_up <= self.partition.max_level:
            self.viewed_level += move_levels_up
        elif self.viewed_level > 0:
            if self.viewed_level < len(self._level_status):
                self._level_status[self.viewed_level] = True
            self.viewed_level -= 1
        else:
            raise StopIteration()

    def resize_level(self, level):
        # here we don't do a bisection we simply test some group sizes around the actual one
        self.partition.actual_level = level

        successful = False

        # todo thing about performance improvement by finding a way to work with level partition only

        new_likelihood, optimal_representation, optimal_is_actual_partition = \
            self.optimize_single_level(level, self.actual_likelihood)

        # if partition has changed, be on the save site and set from complete representation
        # may allow some worsening?
        if new_likelihood - self.actual_likelihood > self._epsilon:
            successful = True
            self.actual_likelihood = new_likelihood

        if not optimal_is_actual_partition:
            self.partition.set_from_representation(optimal_representation)

        return successful

    def optimize_single_level(self, level, actual_likelihood, number_of_tries=1):

        max_representation = self.partition.get_representation()
        max_likelihood = actual_likelihood

        inference = self.inference_class(self.graph, self.objective_function, self.partition,
                                         **self.inference_class_instance_parameters)

        # try some group sizes and test
        upper_bound = self.partition.get_number_of_blocks_in_level(level - 1)
        lower_bound = self.partition.get_number_of_blocks_in_level(level + 1)

        optimal_is_actual_partition = False

        for _ in range(number_of_tries):
            # clear values for each independent try
            likelihoods = {}
            partitions = {}

            # start with greatest value
            self.partition.random_partition(lower_bound, level=level)
            # no possibilities in that stage -> calculate value
            likelihoods[lower_bound] = self.objective_function.calculate(self.partition)
            partitions[lower_bound] = self.partition.get_representation()

            # do the same with upper bound
            self.partition.random_partition(upper_bound, level=level)
            # no possibilities in that stage -> calculate value
            likelihoods[upper_bound] = self.objective_function.calculate(self.partition)
            partitions[upper_bound] = self.partition.get_representation()

            middle_of_interval = int((upper_bound - lower_bound) / 2) + lower_bound

            # do until no block number is between upper and lower bound
            while upper_bound - lower_bound > 1:
                top_third_quarter = int((upper_bound - middle_of_interval) / 2) + middle_of_interval

                if top_third_quarter not in likelihoods:
                    if self._merging_inference:
                        inference.aimed_b = top_third_quarter
                    else:
                        inference.partition.random_partition(top_third_quarter, level)
                    inference.infer_stochastic_block_model()

                    likelihoods[top_third_quarter] = self.objective_function.calculate(self.partition)
                    partitions[top_third_quarter] = self.partition.get_representation()

                if middle_of_interval not in likelihoods:
                    if self._merging_inference:
                        inference.aimed_b = middle_of_interval
                    else:
                        inference.partition.random_partition(middle_of_interval, level)
                    inference.infer_stochastic_block_model()

                    likelihoods[middle_of_interval] = self.objective_function.calculate(self.partition)
                    partitions[middle_of_interval] = self.partition.get_representation()

                if likelihoods[middle_of_interval] > likelihoods[top_third_quarter]:
                    upper_bound = middle_of_interval
                    middle_of_interval = int((upper_bound - lower_bound) / 2) + lower_bound
                else:
                    lower_bound = middle_of_interval
                    if self._merging_inference:
                        self.partition.set_from_representation(partitions[upper_bound])
                    self.partition.actual_level = level
                    # to be sure that the middle is not recomputed
                    middle_of_interval = top_third_quarter

            # check which bound is optimal
            if likelihoods[upper_bound] > likelihoods[lower_bound]:
                optimal_index = upper_bound
            else:
                optimal_index = lower_bound

            # check if new optimum was found
            if likelihoods[optimal_index] > max_likelihood:
                max_representation = partitions[optimal_index]
                max_likelihood = likelihoods[optimal_index]

        return max_likelihood, max_representation, optimal_is_actual_partition

    def delete_level(self, level):
        # check if delete improves partition
        self.partition.actual_level = level
        delta = self.delta_objective_function_for_level_removal(self.partition)
        if delta > self._epsilon:
            # if this is the case then delete this level
            self.partition.delete_actual_level()
            # and update likelihood
            self.actual_likelihood += delta
            return True
        return False

    def add_level(self, level):
        self.partition.actual_level = level
        successful = False

        # add new level and set focus to that level
        self.partition.add_level_above_actual_level()
        self.partition.actual_level += 1

        # find optimal partition with new level
        new_likelihood, optimal_representation, optimal_is_actual_partition = \
            self.optimize_single_level(
                level + 1,
                -float('inf'),
                # number_of_tries=2,
            )

        # if partition has changed, be on the save site and set from complete representation
        # may allow some worsening?
        if new_likelihood - self.actual_likelihood > self._epsilon:
            successful = True
            self.actual_likelihood = new_likelihood
            if not optimal_is_actual_partition:
                self.partition.set_from_representation(optimal_representation)
        else:
            self.partition.actual_level = level + 1
            # remove added level if no improve
            self.partition.delete_actual_level()

        return successful


class PeixotoHierarchicalInference(HierarchicalInference):
    """Same as above only with default usage of Peixotos inference algorithm"""

    title = "Hierarchical Inference with Peixotos Agglomorative Heuristic"
    short_title = "HPAH"

    def __init__(self, graph, objective_function, hierarchical_partition, delta_objective_function_for_level_removal,
                 inference_class_instance_parameters=None):
        inference_class = PeixotoInference
        if inference_class_instance_parameters is None:
            inference_class_instance_parameters = {"no_partition_reset": True,
                                                   "limit_possible_blocks": True,
                                                   "add_additional_mergers": False,
                                                   "mcmc_steps": 100}
        else:
            inference_class_instance_parameters = inference_class_instance_parameters
        super(PeixotoHierarchicalInference, self).__init__(graph, objective_function, hierarchical_partition,
                                                           delta_objective_function_for_level_removal,
                                                           inference_class, inference_class_instance_parameters)


class PeixotoHierarchicalInference1k(PeixotoHierarchicalInference):
    """Same as above only with default usage of Peixotos inference algorithm with 1k MCMC steps"""

    title = "Hierarchical Inference with Peixotos Agglomorative Heuristic 1k steps"
    short_title = "HPAH1k"

    def __init__(self, graph, objective_function, hierarchical_partition, delta_objective_function_for_level_removal):
        inference_class_instance_parameters = {"no_partition_reset": True,
                                               "limit_possible_blocks": True,
                                               "add_additional_mergers": False,
                                               "mcmc_steps": 1000}
        super(PeixotoHierarchicalInference1k, self).__init__(graph, objective_function, hierarchical_partition,
                                                             delta_objective_function_for_level_removal,
                                                             inference_class_instance_parameters)
