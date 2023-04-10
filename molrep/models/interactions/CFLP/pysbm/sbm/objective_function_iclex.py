"""
Likelihood function ICLex like described in
Model selection and clustering in stochastic block models with the exact integrated complete data likelihood
from Etienne Come  and Pierre Latouche (2014)
"""

import math

from .objective_functions import ObjectiveFunction


# @formatter:off
def calculate_icl_ex_jeffrey_hyperprior_directed(partition):
    """
    Calculate the integrated complete likelihood (exact) from Etienne Come and Pierre Latouche
    Returns the result of formula (2.5):
    ICL_ex (Z,K) = log p(X,Z|K)
        = \sum_{k,l}^B\log \frac{\Gamma(\eta_{kl}^0+\zeta_{kl}^0)\Gamma(\eta_{kl})\Gamma(\zeta_{kl})}
        {\Gamma(\eta_{kl}+\zeta_{kl})\Gamma(\eta_{kl}^0)\Gamma(\zeta_{kl}^0)}
    + \log\frac{\Gamma(\sum_{r=1}^{B}n_k^0)\prod_{r=1}^B\Gamma(n_k)}
    {\Gamma(\sum_{r=1}^Bn_k)\prod_{r=1}^{B}\Gamma(n_k^0)}
    where
    n_k = n_k^0 + |b_k|,\\
    \eta_{kl} = \eta_{kl}^0 +e_{kl},\\
    \zeta_{kl} = \zeta_{kl}^0 + |b_k||b_l|-e_{kl}
    and n_k^0,\ \eta_{kl}^0 and \zeta_{kl}^0 are the hyperparameters of the prior distributions,
    which are set to 1/2 for a Jeffrey distribution.
    :param partition:
    :return:
    """
    likelihood = 0

    eta_kl_zero = .5
    zeta_kl_zero = .5
    n_k_zero = .5

    for from_block in range(partition.B):
        number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(
            from_block)
        for to_block in range(partition.B):
            # first term
            eta_kl = eta_kl_zero + partition.get_edge_count(from_block, to_block)
            zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * partition.get_number_of_nodes_in_block(to_block) \
                - partition.get_edge_count(from_block, to_block)

            # math.lgamma(eta_kl_zero + zeta_kl_zero) = math.lgamma(1) = 0
            likelihood += math.lgamma(eta_kl) + math.lgamma(zeta_kl) \
                - math.lgamma(eta_kl + zeta_kl) - math.lgamma(eta_kl_zero) - math.lgamma(zeta_kl_zero)

        # calculate second term
        likelihood += math.lgamma(n_k_zero + number_of_nodes_in_from_block) - math.lgamma(n_k_zero)

    # add "global" terms of second term
    likelihood += math.lgamma(n_k_zero * partition.B) - math.lgamma(n_k_zero * partition.B + partition.get_number_of_nodes())

    return likelihood


def calculate_icl_ex_uniform_hyperprior_directed(partition):
    """
    Calculate the integrated complete likelihood (exact) from Etienne Come and Pierre Latouche
    Returns the result of formula (2.5):
    ICL_ex (Z,K) = log p(X,Z|K)
        = \sum_{k,l}^B\log \frac{\Gamma(\eta_{kl}^0+\zeta_{kl}^0)\Gamma(\eta_{kl})\Gamma(\zeta_{kl})}
        {\Gamma(\eta_{kl}+\zeta_{kl})\Gamma(\eta_{kl}^0)\Gamma(\zeta_{kl}^0)}
    + \log\frac{\Gamma(\sum_{r=1}^{B}n_k^0)\prod_{r=1}^B\Gamma(n_k)}
    {\Gamma(\sum_{r=1}^Bn_k)\prod_{r=1}^{B}\Gamma(n_k^0)}
    where
    n_k = n_k^0 + |b_k|,\\
    \eta_{kl} = \eta_{kl}^0 +e_{kl},\\
    \zeta_{kl} = \zeta_{kl}^0 + |b_k||b_l|-e_{kl}
    and n_k^0,\ \eta_{kl}^0 and \zeta_{kl}^0 are the hyperparameters of the prior distributions,
    which are set to 1 for a uniform distribution.
    :param partition:
    :return:
    """
    likelihood = 0

    eta_kl_zero = 1
    zeta_kl_zero = 1

    for from_block in range(partition.B):
        number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(
            from_block)
        for to_block in range(partition.B):
            # first term
            eta_kl = eta_kl_zero + partition.get_edge_count(from_block, to_block)
            zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * partition.get_number_of_nodes_in_block(to_block) \
                - partition.get_edge_count(from_block, to_block)

            # math.lgamma(eta_kl_zero + zeta_kl_zero) = math.lgamma(2) = 0
            # math.lgamma(eta_kl_zero) = math.lgamma(zeta_kl_zero) = math.lgamma(1) = 0
            likelihood += + math.lgamma(eta_kl) + math.lgamma(zeta_kl) \
                - math.lgamma(eta_kl + zeta_kl)

        # calculate second term
        likelihood += math.lgamma(1 + number_of_nodes_in_from_block)

    # add "global" terms of second term
    likelihood += math.lgamma(partition.B) - math.lgamma(partition.B + partition.get_number_of_nodes())

    return likelihood


def calculate_icl_ex_jeffrey_hyperprior_undirected(partition):
    """
    Calculate the integrated complete likelihood (exact) from Etienne Come and Pierre Latouche
    Returns the result of formula (2.5):
    ICL_ex (Z,K) = log p(X,Z|K)
        = \sum_{k,l}^B\log \frac{\Gamma(\eta_{kl}^0+\zeta_{kl}^0)\Gamma(\eta_{kl})\Gamma(\zeta_{kl})}
        {\Gamma(\eta_{kl}+\zeta_{kl})\Gamma(\eta_{kl}^0)\Gamma(\zeta_{kl}^0)}
    + \log\frac{\Gamma(\sum_{r=1}^{B}n_k^0)\prod_{r=1}^B\Gamma(n_k)}
    {\Gamma(\sum_{r=1}^Bn_k)\prod_{r=1}^{B}\Gamma(n_k^0)}
    where
    n_k = n_k^0 + |b_k|,\\
    \eta_{kl} = \eta_{kl}^0 +e_{kl},\\
    \zeta_{kl} = \zeta_{kl}^0 + |b_k||b_l|-e_{kl}
    and n_k^0,\ \eta_{kl}^0 and \zeta_{kl}^0 are the hyperparameters of the prior distributions,
    which are set to 1/2 for a Jeffrey distribution.
    :param partition:
    :return:
    """
    likelihood = 0

    eta_kl_zero = .5
    zeta_kl_zero = .5
    n_k_zero = .5

    for from_block in range(partition.B):
        number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(
            from_block)
        for to_block in range(from_block + 1, partition.B):
            # first term
            eta_kl = eta_kl_zero + partition.get_edge_count(from_block, to_block)
            zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * partition.get_number_of_nodes_in_block(to_block) \
                - partition.get_edge_count(from_block, to_block)

            # math.lgamma(eta_kl_zero + zeta_kl_zero) = math.lgamma(1) = 0
            likelihood += math.lgamma(eta_kl) + math.lgamma(zeta_kl) \
                - math.lgamma(eta_kl + zeta_kl) - math.lgamma(eta_kl_zero) - math.lgamma(zeta_kl_zero)

        # add term for from_block->from_block
        # in undirected case those edges are counted twice cover this
        eta_kl = eta_kl_zero + partition.get_edge_count(from_block, from_block) / 2
        # and reduce the number of possibilities accordingly
        zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * (number_of_nodes_in_from_block + 1) / 2 \
            - partition.get_edge_count(from_block, from_block) / 2

        likelihood += math.lgamma(eta_kl) + math.lgamma(zeta_kl) \
            - math.lgamma(eta_kl + zeta_kl) - math.lgamma(eta_kl_zero) - math.lgamma(zeta_kl_zero)

        # calculate second term
        likelihood += math.lgamma(n_k_zero + number_of_nodes_in_from_block) - math.lgamma(n_k_zero)

    # add "global" terms of second term
    likelihood += math.lgamma(n_k_zero * partition.B) - math.lgamma(n_k_zero * partition.B + partition.get_number_of_nodes())

    return likelihood


def calculate_icl_ex_uniform_hyperprior_undirected(partition):
    """
    Calculate the integrated complete likelihood (exact) from Etienne Come and Pierre Latouche
    Returns the result of formula (2.5):
    ICL_ex (Z,K) = log p(X,Z|K)
        = \sum_{k,l}^B\log \frac{\Gamma(\eta_{kl}^0+\zeta_{kl}^0)\Gamma(\eta_{kl})\Gamma(\zeta_{kl})}
        {\Gamma(\eta_{kl}+\zeta_{kl})\Gamma(\eta_{kl}^0)\Gamma(\zeta_{kl}^0)}
    + \log\frac{\Gamma(\sum_{r=1}^{B}n_k^0)\prod_{r=1}^B\Gamma(n_k)}
    {\Gamma(\sum_{r=1}^Bn_k)\prod_{r=1}^{B}\Gamma(n_k^0)}
    where
    n_k = n_k^0 + |b_k|,\\
    \eta_{kl} = \eta_{kl}^0 +e_{kl},\\
    \zeta_{kl} = \zeta_{kl}^0 + |b_k||b_l|-e_{kl}
    and n_k^0,\ \eta_{kl}^0 and \zeta_{kl}^0 are the hyperparameters of the prior distributions,
    which are set to 1 for a uniform distribution.
    :param partition:
    :return:
    """
    likelihood = 0

    eta_kl_zero = 1
    zeta_kl_zero = 1

    for from_block in range(partition.B):
        number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(
            from_block)
        for to_block in range(from_block + 1, partition.B):
            # first term
            eta_kl = eta_kl_zero + partition.get_edge_count(from_block, to_block)
            zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * partition.get_number_of_nodes_in_block(to_block) \
                - partition.get_edge_count(from_block, to_block)

            # math.lgamma(eta_kl_zero + zeta_kl_zero) = math.lgamma(2) = 0
            # math.lgamma(eta_kl_zero) = math.lgamma(zeta_kl_zero) = math.lgamma(1) = 0
            likelihood += + math.lgamma(eta_kl) + math.lgamma(zeta_kl) \
                - math.lgamma(eta_kl + zeta_kl)

        # add term for from_block->from_block
        # in undirected case those edges are counted twice cover this
        eta_kl = eta_kl_zero + partition.get_edge_count(from_block, from_block) / 2
        # and reduce the number of possibilities accordingly
        zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * (number_of_nodes_in_from_block + 1) / 2 \
            - partition.get_edge_count(from_block, from_block) / 2

        likelihood += + math.lgamma(eta_kl) + math.lgamma(zeta_kl) \
            - math.lgamma(eta_kl + zeta_kl)

        # calculate second term
        likelihood += math.lgamma(1 + number_of_nodes_in_from_block)

    # add "global" terms of second term
    likelihood += math.lgamma(partition.B) - math.lgamma(partition.B + partition.get_number_of_nodes())

    return likelihood


def delta_calculate_icl_ex_jeffrey_hyperprior_undirected(partition, from_block, to_block, *args):
    #     distinguish by given parameters
    if len(args) == 3:
        kit, selfloops, _ = args
        nodes_moved = 1
        nodes_remaining = not (partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
    elif len(args) == 5:
        kit, selfloops, _, nodes_moved, nodes_remaining = args
    else:
        raise ValueError("Wrong number of parameters " + str(args))

    delta = 0

    to_block_exists = to_block != partition.B

    # fast exit for "zero move"
    if not to_block_exists and not nodes_remaining:
        return delta

    eta_kl_zero = .5
    zeta_kl_zero = .5
    n_k_zero = .5

    number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
    if to_block_exists:
        number_of_nodes_in_to_block = partition.get_number_of_nodes_in_block(to_block)
    else:
        number_of_nodes_in_to_block = 0

    # loop over all blocks, because the node count in both block will change
    for block in range(partition.B):
        if block != from_block and block != to_block:
            number_of_nodes_in_block = partition.get_number_of_nodes_in_block(block)
            # from_block -> block
            # old value
            eta_kl = eta_kl_zero + partition.get_edge_count(from_block, block)
            zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_block \
                - partition.get_edge_count(from_block, block)
            delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

            if nodes_remaining:
                # new value
                eta_kl -= kit.get(block, 0)
                zeta_kl -= nodes_moved * number_of_nodes_in_block - kit.get(block, 0)
                delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            else:
                # add missing terms from old value, which else cancel out
                delta += math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

            # to_block -> block
            if to_block_exists:
                # old value
                eta_kl = eta_kl_zero + partition.get_edge_count(to_block, block)
                zeta_kl = zeta_kl_zero + number_of_nodes_in_to_block * number_of_nodes_in_block \
                    - partition.get_edge_count(to_block, block)
                delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            else:
                eta_kl = eta_kl_zero
                zeta_kl = zeta_kl_zero
                # subtract missing terms of new term, which else would cancel out
                delta -= math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

            # new value
            eta_kl += kit.get(block, 0)
            zeta_kl += nodes_moved * number_of_nodes_in_block - kit.get(block, 0)
            delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # add terms for from_block -> to_block
    if to_block_exists:
        # old value
        eta_kl = eta_kl_zero + partition.get_edge_count(from_block, to_block)
        zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - partition.get_edge_count(from_block, to_block)
        delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        eta_kl = eta_kl_zero
        zeta_kl = zeta_kl_zero
        # subtract missing terms of new term, which else would cancel out
        delta -= math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

    if nodes_remaining:
        # new value
        eta_kl += kit[from_block] - kit[to_block]
        zeta_kl -= number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - (number_of_nodes_in_from_block - nodes_moved) * (number_of_nodes_in_to_block + nodes_moved) \
            + kit[from_block] - kit[to_block]
        delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        # add missing terms from old value, which else cancel out
        delta += math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

    # to_block -> to_block
    if to_block_exists:
        # old value
        eta_kl = eta_kl_zero + partition.get_edge_count(to_block, to_block) / 2
        zeta_kl = zeta_kl_zero + number_of_nodes_in_to_block * (number_of_nodes_in_to_block + 1) / 2\
            - partition.get_edge_count(to_block, to_block) / 2
        delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        eta_kl = eta_kl_zero
        zeta_kl = zeta_kl_zero
        # subtract missing terms of new term, which else would cancel out
        delta -= math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

    # new value
    eta_kl += kit[to_block] + selfloops
    zeta_kl -= number_of_nodes_in_to_block * (number_of_nodes_in_to_block + 1) / 2 \
        - (number_of_nodes_in_to_block + nodes_moved) * (number_of_nodes_in_to_block + nodes_moved + 1) / 2 \
        + kit[to_block] + selfloops
    delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # from_block -> from_block
    # old value
    eta_kl = eta_kl_zero + partition.get_edge_count(from_block, from_block) / 2
    zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * (number_of_nodes_in_from_block + 1) / 2\
        - partition.get_edge_count(from_block, from_block) / 2
    delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # new value
    if nodes_remaining:
        eta_kl -= kit[from_block] + selfloops
        zeta_kl -= number_of_nodes_in_from_block * (number_of_nodes_in_from_block + 1) / 2 \
            - (number_of_nodes_in_from_block - nodes_moved) * (number_of_nodes_in_from_block - nodes_moved + 1) / 2 \
            - kit[from_block] - selfloops
        delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        # add missing terms from old value, which else cancel out
        delta += math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)
    # ---- end first term

    # second term
    if nodes_remaining:
        # add term of \prod \gamma(n_k)
        delta += math.lgamma(n_k_zero + number_of_nodes_in_from_block - nodes_moved)
    else:
        # decrease of block count
        # handle term \Gamma(\sum n_k^0) and add one \gamma(n_k^0) for the reduced count in the denominator
        delta += math.lgamma((partition.B - 1) * n_k_zero) - math.lgamma(partition.B * n_k_zero) + math.lgamma(n_k_zero) \
                 - math.lgamma(n_k_zero * (partition.B - 1) + partition.get_number_of_nodes()) \
                 + math.lgamma(n_k_zero * partition.B + partition.get_number_of_nodes())

    if to_block_exists:
        # subtract term of \prod \gamma(n_k)
        delta -= math.lgamma(n_k_zero + number_of_nodes_in_to_block)
    else:
        # increase of block count
        # handle term \Gamma(\sum n_k^0) and subtract one \gamma(n_k^0) for the increased count in the denominator
        delta += math.lgamma((partition.B + 1) * n_k_zero) - math.lgamma(partition.B * n_k_zero) - math.lgamma(n_k_zero) \
            - math.lgamma(n_k_zero * (partition.B + 1) + partition.get_number_of_nodes()) \
            + math.lgamma(n_k_zero * partition.B + partition.get_number_of_nodes())

    # handle term \prod \gamma(n_k)
    delta += math.lgamma(n_k_zero + number_of_nodes_in_to_block + nodes_moved) \
        - math.lgamma(n_k_zero + number_of_nodes_in_from_block)

    return delta


def delta_calculate_icl_ex_uniform_hyperprior_undirected(partition, from_block, to_block, *args):
    #     distinguish by given parameters
    if len(args) == 3:
        kit, selfloops, _ = args
        nodes_moved = 1
        nodes_remaining = not (partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
    elif len(args) == 5:
        kit, selfloops, _, nodes_moved, nodes_remaining = args
    else:
        raise ValueError("Wrong number of parameters " + str(args))

    delta = 0

    to_block_exists = to_block != partition.B

    # fast exit for "zero move"
    if not to_block_exists and not nodes_remaining:
        return delta

    eta_kl_zero = 1
    zeta_kl_zero = 1
    # n_k_zero = 1

    number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
    if to_block_exists:
        number_of_nodes_in_to_block = partition.get_number_of_nodes_in_block(to_block)
    else:
        number_of_nodes_in_to_block = 0

    # loop over all blocks, because the node count in both block will change
    for block in range(partition.B):
        if block != from_block and block != to_block:
            number_of_nodes_in_block = partition.get_number_of_nodes_in_block(block)
            # from_block -> block
            # old value
            eta_kl = eta_kl_zero + partition.get_edge_count(from_block, block)
            zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_block \
                - partition.get_edge_count(from_block, block)
            delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

            if nodes_remaining:
                # new value
                eta_kl -= kit.get(block, 0)
                zeta_kl -= nodes_moved * number_of_nodes_in_block - kit.get(block, 0)
                delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            # no extra handling needed because both terms are zero

            # to_block -> block
            if to_block_exists:
                # old value
                eta_kl = eta_kl_zero + partition.get_edge_count(to_block, block)
                zeta_kl = zeta_kl_zero + number_of_nodes_in_to_block * number_of_nodes_in_block \
                    - partition.get_edge_count(to_block, block)
                delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            else:
                eta_kl = eta_kl_zero
                zeta_kl = zeta_kl_zero
                # no extra handling needed because both terms are zero

            # new value
            eta_kl += kit.get(block, 0)
            zeta_kl += nodes_moved * number_of_nodes_in_block - kit.get(block, 0)
            delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # add terms for from_block -> to_block
    if to_block_exists:
        # old value
        eta_kl = eta_kl_zero + partition.get_edge_count(from_block, to_block)
        zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - partition.get_edge_count(from_block, to_block)
        delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        eta_kl = eta_kl_zero
        zeta_kl = zeta_kl_zero
        # add missing terms of new term, which else would cancel out
        delta += math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

    if nodes_remaining:
        # new value
        eta_kl += kit[from_block] - kit[to_block]
        zeta_kl -= number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - (number_of_nodes_in_from_block - nodes_moved) * (number_of_nodes_in_to_block + nodes_moved) \
            + kit[from_block] - kit[to_block]
        delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    # no extra handling needed because both terms are zero

    # to_block -> to_block
    if to_block_exists:
        # old value
        eta_kl = eta_kl_zero + partition.get_edge_count(to_block, to_block) / 2
        zeta_kl = zeta_kl_zero + number_of_nodes_in_to_block * (number_of_nodes_in_to_block + 1) / 2\
            - partition.get_edge_count(to_block, to_block) / 2
        delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        eta_kl = eta_kl_zero
        zeta_kl = zeta_kl_zero
        # no extra handling needed because both terms are zero

    # new value
    eta_kl += kit[to_block] + selfloops
    zeta_kl -= number_of_nodes_in_to_block * (number_of_nodes_in_to_block + 1) / 2 \
        - (number_of_nodes_in_to_block + nodes_moved) * (number_of_nodes_in_to_block + nodes_moved + 1) / 2 \
        + kit[to_block] + selfloops
    delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # from_block -> from_block
    # old value
    eta_kl = eta_kl_zero + partition.get_edge_count(from_block, from_block) / 2
    zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * (number_of_nodes_in_from_block + 1) / 2\
        - partition.get_edge_count(from_block, from_block) / 2
    delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # new value
    if nodes_remaining:
        eta_kl -= kit[from_block] + selfloops
        zeta_kl -= number_of_nodes_in_from_block * (number_of_nodes_in_from_block + 1) / 2 \
            - (number_of_nodes_in_from_block - nodes_moved) * (number_of_nodes_in_from_block - nodes_moved + 1) / 2 \
            - kit[from_block] - selfloops
        delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    # no extra handling needed because both terms are zero
    # ---- end first term

    # second term
    if nodes_remaining:
        # add term of \prod \gamma(n_k)
        delta += math.lgamma(1 + number_of_nodes_in_from_block - nodes_moved)
    else:
        # decrease of block count
        # handle term \Gamma(\sum n_k^0) and add one \gamma(n_k^0) for the reduced count in the denominator
        # gamma(n_k^0) = gamma(1) = 0
        delta += math.lgamma(partition.B - 1) - math.lgamma(partition.B) \
                 - math.lgamma(partition.B - 1 + partition.get_number_of_nodes()) \
                 + math.lgamma(partition.B + partition.get_number_of_nodes())

    if to_block_exists:
        # subtract term of \prod \gamma(n_k)
        delta -= math.lgamma(1 + number_of_nodes_in_to_block)
    else:
        # increase of block count
        # handle term \Gamma(\sum n_k^0) and subtract one \gamma(n_k^0) for the increased count in the denominator
        # gamma(n_k^0) = gamma(1) = 0
        delta += math.lgamma(partition.B + 1) - math.lgamma(partition.B) \
            - math.lgamma(partition.B + 1 + partition.get_number_of_nodes()) \
            + math.lgamma(partition.B + partition.get_number_of_nodes())

    # handle term \prod \gamma(n_k)
    delta += math.lgamma(1 + number_of_nodes_in_to_block + nodes_moved) \
        - math.lgamma(1 + number_of_nodes_in_from_block)

    return delta


def delta_calculate_icl_ex_jeffrey_hyperprior_directed(partition, from_block, to_block, *args):
    #     distinguish by given parameters
    if len(args) == 5:
        kit, kti, selfloops, _, _ = args
        nodes_moved = 1
        nodes_remaining = not (partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
    elif len(args) == 7:
        kit, kti, selfloops, _, _, nodes_moved, nodes_remaining = args
    else:
        raise ValueError("Wrong number of parameters " + str(args))

    delta = 0

    to_block_exists = to_block != partition.B

    # fast exit for "zero move"
    if not to_block_exists and not nodes_remaining:
        return delta

    eta_kl_zero = .5
    zeta_kl_zero = .5
    n_k_zero = .5

    number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
    if to_block_exists:
        number_of_nodes_in_to_block = partition.get_number_of_nodes_in_block(to_block)
    else:
        number_of_nodes_in_to_block = 0

    # loop over all blocks, because the node count in both block will change
    for block in range(partition.B):
        if block != from_block and block != to_block:
            number_of_nodes_in_block = partition.get_number_of_nodes_in_block(block)
            # from_block -> block
            # old value
            eta_kl = eta_kl_zero + partition.get_edge_count(from_block, block)
            zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_block \
                - partition.get_edge_count(from_block, block)
            delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

            if nodes_remaining:
                # new value
                eta_kl -= kit.get(block, 0)
                zeta_kl -= nodes_moved * number_of_nodes_in_block - kit.get(block, 0)
                delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            else:
                # add missing terms from old value, which else cancel out
                delta += math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

            # to_block -> block
            if to_block_exists:
                # old value
                eta_kl = eta_kl_zero + partition.get_edge_count(to_block, block)
                zeta_kl = zeta_kl_zero + number_of_nodes_in_to_block * number_of_nodes_in_block \
                    - partition.get_edge_count(to_block, block)
                delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            else:
                eta_kl = eta_kl_zero
                zeta_kl = zeta_kl_zero
                # subtract missing terms of new term, which else would cancel out
                delta -= math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

            # new value
            eta_kl += kit.get(block, 0)
            zeta_kl += nodes_moved * number_of_nodes_in_block - kit.get(block, 0)
            delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

            # ----- other directions
            # block -> from_block
            # old value
            eta_kl = eta_kl_zero + partition.get_edge_count(block, from_block)
            zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_block \
                - partition.get_edge_count(block, from_block)
            delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

            if nodes_remaining:
                # new value
                eta_kl -= kti.get(block, 0)
                zeta_kl -= nodes_moved * number_of_nodes_in_block - kti.get(block, 0)
                delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            else:
                # add missing terms from old value, which else cancel out
                delta += math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

            # to_block -> block
            if to_block_exists:
                # old value
                eta_kl = eta_kl_zero + partition.get_edge_count(block, to_block)
                zeta_kl = zeta_kl_zero + number_of_nodes_in_to_block * number_of_nodes_in_block \
                    - partition.get_edge_count(block, to_block)
                delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            else:
                eta_kl = eta_kl_zero
                zeta_kl = zeta_kl_zero
                # subtract missing terms of new term, which else would cancel out
                delta -= math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

            # new value
            eta_kl += kti.get(block, 0)
            zeta_kl += nodes_moved * number_of_nodes_in_block - kti.get(block, 0)
            delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # add terms for from_block -> to_block
    if to_block_exists:
        # old value
        eta_kl = eta_kl_zero + partition.get_edge_count(from_block, to_block)
        zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - partition.get_edge_count(from_block, to_block)
        delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        eta_kl = eta_kl_zero
        zeta_kl = zeta_kl_zero
        # subtract missing terms of new term, which else would cancel out
        delta -= math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

    if nodes_remaining:
        # new value
        eta_kl += kti[from_block] - kit[to_block]
        zeta_kl -= number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - (number_of_nodes_in_from_block - nodes_moved) * (number_of_nodes_in_to_block + nodes_moved) \
            + kti[from_block] - kit[to_block]
        delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        # add missing terms from old value, which else cancel out
        delta += math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

    # add terms for to_block -> from_block
    if to_block_exists:
        # old value
        eta_kl = eta_kl_zero + partition.get_edge_count(to_block, from_block)
        zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - partition.get_edge_count(to_block, from_block)
        delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        eta_kl = eta_kl_zero
        zeta_kl = zeta_kl_zero
        # subtract missing terms of new term, which else would cancel out
        delta -= math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

    if nodes_remaining:
        # new value
        eta_kl += kit[from_block] - kti[to_block]
        zeta_kl -= number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - (number_of_nodes_in_from_block - nodes_moved) * (number_of_nodes_in_to_block + nodes_moved) \
            + kit[from_block] - kti[to_block]
        delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        # add missing terms from old value, which else cancel out
        delta += math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

    # to_block -> to_block
    if to_block_exists:
        # old value
        eta_kl = eta_kl_zero + partition.get_edge_count(to_block, to_block)
        zeta_kl = zeta_kl_zero + number_of_nodes_in_to_block * number_of_nodes_in_to_block \
            - partition.get_edge_count(to_block, to_block)
        delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        eta_kl = eta_kl_zero
        zeta_kl = zeta_kl_zero
        # subtract missing terms of new term, which else would cancel out
        delta -= math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

    # new value
    eta_kl += kit[to_block] + kti[to_block] + selfloops
    zeta_kl -= number_of_nodes_in_to_block * number_of_nodes_in_to_block \
        - (number_of_nodes_in_to_block + nodes_moved) * (number_of_nodes_in_to_block + nodes_moved) \
        + kit[to_block] + kti[to_block] + selfloops
    delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # from_block -> from_block
    # old value
    eta_kl = eta_kl_zero + partition.get_edge_count(from_block, from_block)
    zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_from_block \
        - partition.get_edge_count(from_block, from_block)
    delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # new value
    if nodes_remaining:
        eta_kl -= kit[from_block] + kti[from_block] + selfloops
        zeta_kl -= number_of_nodes_in_from_block * number_of_nodes_in_from_block \
            - (number_of_nodes_in_from_block - nodes_moved) * (number_of_nodes_in_from_block - nodes_moved) \
            - kit[from_block] - kti[from_block] - selfloops
        delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        # add missing terms from old value, which else cancel out
        delta += math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)
    # ---- end first term

    # second term
    if nodes_remaining:
        # add term of \prod \gamma(n_k)
        delta += math.lgamma(n_k_zero + number_of_nodes_in_from_block - nodes_moved)
    else:
        # decrease of block count
        # handle term \Gamma(\sum n_k^0) and add one \gamma(n_k^0) for the reduced count in the denominator
        delta += math.lgamma((partition.B - 1) * n_k_zero) - math.lgamma(partition.B * n_k_zero) + math.lgamma(n_k_zero) \
                 - math.lgamma(n_k_zero * (partition.B - 1) + partition.get_number_of_nodes()) \
                 + math.lgamma(n_k_zero * partition.B + partition.get_number_of_nodes())

    if to_block_exists:
        # subtract term of \prod \gamma(n_k)
        delta -= math.lgamma(n_k_zero + number_of_nodes_in_to_block)
    else:
        # increase of block count
        # handle term \Gamma(\sum n_k^0) and subtract one \gamma(n_k^0) for the increased count in the denominator
        delta += math.lgamma((partition.B + 1) * n_k_zero) - math.lgamma(partition.B * n_k_zero) - math.lgamma(n_k_zero) \
            - math.lgamma(n_k_zero * (partition.B + 1) + partition.get_number_of_nodes()) \
            + math.lgamma(n_k_zero * partition.B + partition.get_number_of_nodes())

    # handle term \prod \gamma(n_k)
    delta += math.lgamma(n_k_zero + number_of_nodes_in_to_block + nodes_moved) \
        - math.lgamma(n_k_zero + number_of_nodes_in_from_block)

    return delta


def delta_calculate_icl_ex_uniform_hyperprior_directed(partition, from_block, to_block, *args):
    #     distinguish by given parameters
    if len(args) == 5:
        kit, kti, selfloops, _, _ = args
        nodes_moved = 1
        nodes_remaining = not (partition.get_number_of_nodes_in_block(from_block) == nodes_moved)
    elif len(args) == 7:
        kit, kti, selfloops, _, _, nodes_moved, nodes_remaining = args
    else:
        raise ValueError("Wrong number of parameters " + str(args))

    delta = 0

    to_block_exists = to_block != partition.B

    # fast exit for "zero move"
    if not to_block_exists and not nodes_remaining:
        return delta

    eta_kl_zero = 1
    zeta_kl_zero = 1
    # n_k_zero = 1

    number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
    if to_block_exists:
        number_of_nodes_in_to_block = partition.get_number_of_nodes_in_block(to_block)
    else:
        number_of_nodes_in_to_block = 0

    # loop over all blocks, because the node count in both block will change
    for block in range(partition.B):
        if block != from_block and block != to_block:
            number_of_nodes_in_block = partition.get_number_of_nodes_in_block(block)
            # from_block -> block
            # old value
            eta_kl = eta_kl_zero + partition.get_edge_count(from_block, block)
            zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_block \
                - partition.get_edge_count(from_block, block)
            delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

            if nodes_remaining:
                # new value
                eta_kl -= kit.get(block, 0)
                zeta_kl -= nodes_moved * number_of_nodes_in_block - kit.get(block, 0)
                delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            # no extra handling needed because both terms are zero

            # to_block -> block
            if to_block_exists:
                # old value
                eta_kl = eta_kl_zero + partition.get_edge_count(to_block, block)
                zeta_kl = zeta_kl_zero + number_of_nodes_in_to_block * number_of_nodes_in_block \
                    - partition.get_edge_count(to_block, block)
                delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            else:
                eta_kl = eta_kl_zero
                zeta_kl = zeta_kl_zero
                # no extra handling needed because both terms are zero

            # new value
            eta_kl += kit.get(block, 0)
            zeta_kl += nodes_moved * number_of_nodes_in_block - kit.get(block, 0)
            delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

            # --- other direction
            # block -> from_block
            # old value
            eta_kl = eta_kl_zero + partition.get_edge_count(block, from_block)
            zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_block \
                - partition.get_edge_count(block, from_block)
            delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

            if nodes_remaining:
                # new value
                eta_kl -= kti.get(block, 0)
                zeta_kl -= nodes_moved * number_of_nodes_in_block - kti.get(block, 0)
                delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            # no extra handling needed because both terms are zero

            # to_block -> block
            if to_block_exists:
                # old value
                eta_kl = eta_kl_zero + partition.get_edge_count(block, to_block)
                zeta_kl = zeta_kl_zero + number_of_nodes_in_to_block * number_of_nodes_in_block \
                    - partition.get_edge_count(block, to_block)
                delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
            else:
                eta_kl = eta_kl_zero
                zeta_kl = zeta_kl_zero
                # no extra handling needed because both terms are zero

            # new value
            eta_kl += kti.get(block, 0)
            zeta_kl += nodes_moved * number_of_nodes_in_block - kti.get(block, 0)
            delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # add terms for from_block -> to_block
    if to_block_exists:
        # old value
        eta_kl = eta_kl_zero + partition.get_edge_count(from_block, to_block)
        zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - partition.get_edge_count(from_block, to_block)
        delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        eta_kl = eta_kl_zero
        zeta_kl = zeta_kl_zero
        # add missing terms of new term, which else would cancel out
        delta += math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

    if nodes_remaining:
        # new value
        eta_kl += kti[from_block] - kit[to_block]
        zeta_kl -= number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - (number_of_nodes_in_from_block - nodes_moved) * (number_of_nodes_in_to_block + nodes_moved) \
            + kti[from_block] - kit[to_block]
        delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    # no extra handling needed because both terms are zero

    # add terms for to_block -> from_block
    if to_block_exists:
        # old value
        eta_kl = eta_kl_zero + partition.get_edge_count(to_block, from_block)
        zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - partition.get_edge_count(to_block, from_block)
        delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        eta_kl = eta_kl_zero
        zeta_kl = zeta_kl_zero
        # add missing terms of new term, which else would cancel out
        delta += math.lgamma(eta_kl_zero) + math.lgamma(zeta_kl_zero)

    if nodes_remaining:
        # new value
        eta_kl += kit[from_block] - kti[to_block]
        zeta_kl -= number_of_nodes_in_from_block * number_of_nodes_in_to_block \
            - (number_of_nodes_in_from_block - nodes_moved) * (number_of_nodes_in_to_block + nodes_moved) \
            + kit[from_block] - kti[to_block]
        delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    # no extra handling needed because both terms are zero

    # to_block -> to_block
    if to_block_exists:
        # old value
        eta_kl = eta_kl_zero + partition.get_edge_count(to_block, to_block)
        zeta_kl = zeta_kl_zero + number_of_nodes_in_to_block * number_of_nodes_in_to_block\
            - partition.get_edge_count(to_block, to_block)
        delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    else:
        eta_kl = eta_kl_zero
        zeta_kl = zeta_kl_zero
        # no extra handling needed because both terms are zero

    # new value
    eta_kl += kit[to_block] + kti[to_block] + selfloops
    zeta_kl -= number_of_nodes_in_to_block * number_of_nodes_in_to_block \
        - (number_of_nodes_in_to_block + nodes_moved) * (number_of_nodes_in_to_block + nodes_moved) \
        + kit[to_block] + kti[to_block] + selfloops
    delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # from_block -> from_block
    # old value
    eta_kl = eta_kl_zero + partition.get_edge_count(from_block, from_block)
    zeta_kl = zeta_kl_zero + number_of_nodes_in_from_block * number_of_nodes_in_from_block \
        - partition.get_edge_count(from_block, from_block)
    delta -= math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)

    # new value
    if nodes_remaining:
        eta_kl -= kit[from_block] + kti[from_block] + selfloops
        zeta_kl -= number_of_nodes_in_from_block * number_of_nodes_in_from_block \
            - (number_of_nodes_in_from_block - nodes_moved) * (number_of_nodes_in_from_block - nodes_moved) \
            - kit[from_block] - kti[from_block] - selfloops
        delta += math.lgamma(eta_kl) + math.lgamma(zeta_kl) - math.lgamma(eta_kl + zeta_kl)
    # no extra handling needed because both terms are zero
    # ---- end first term

    # second term
    if nodes_remaining:
        # add term of \prod \gamma(n_k)
        delta += math.lgamma(1 + number_of_nodes_in_from_block - nodes_moved)
    else:
        # decrease of block count
        # handle term \Gamma(\sum n_k^0) and add one \gamma(n_k^0) for the reduced count in the denominator
        # gamma(n_k^0) = gamma(1) = 0
        delta += math.lgamma(partition.B - 1) - math.lgamma(partition.B) \
                 - math.lgamma(partition.B - 1 + partition.get_number_of_nodes()) \
                 + math.lgamma(partition.B + partition.get_number_of_nodes())

    if to_block_exists:
        # subtract term of \prod \gamma(n_k)
        delta -= math.lgamma(1 + number_of_nodes_in_to_block)
    else:
        # increase of block count
        # handle term \Gamma(\sum n_k^0) and subtract one \gamma(n_k^0) for the increased count in the denominator
        # gamma(n_k^0) = gamma(1) = 0
        delta += math.lgamma(partition.B + 1) - math.lgamma(partition.B) \
            - math.lgamma(partition.B + 1 + partition.get_number_of_nodes()) \
            + math.lgamma(partition.B + partition.get_number_of_nodes())

    # handle term \prod \gamma(n_k)
    delta += math.lgamma(1 + number_of_nodes_in_to_block + nodes_moved) \
        - math.lgamma(1 + number_of_nodes_in_from_block)

    return delta

# ----------------------------------------------
# As Objective Function Class
# ----------------------------------------------


class IntegratedCompleteLikelihoodExact(ObjectiveFunction):
    
    JEFFREY_HYPERPRIOR = "jeffrey"
    UNIFORM_HYPERPRIOR = "uniform"
    
    def __init__(self, is_directed, hyperprior=None):
        if hyperprior is None:
            hyperprior = self.JEFFREY_HYPERPRIOR
            
        self._hyperprior = hyperprior
        
        if hyperprior == self.JEFFREY_HYPERPRIOR:
            function_calculate_undirected = calculate_icl_ex_jeffrey_hyperprior_undirected
            function_calculate_directed = calculate_icl_ex_jeffrey_hyperprior_directed
            function_calculate_delta_undirected = delta_calculate_icl_ex_jeffrey_hyperprior_undirected
            function_calculate_delta_directed = delta_calculate_icl_ex_jeffrey_hyperprior_directed
        else:
            function_calculate_undirected = calculate_icl_ex_uniform_hyperprior_undirected
            function_calculate_directed = calculate_icl_ex_uniform_hyperprior_directed
            function_calculate_delta_undirected = delta_calculate_icl_ex_uniform_hyperprior_undirected
            function_calculate_delta_directed = delta_calculate_icl_ex_uniform_hyperprior_directed
        
        super(IntegratedCompleteLikelihoodExact, self).__init__(
            is_directed, 
            function_calculate_undirected, 
            function_calculate_directed, 
            function_calculate_delta_undirected, 
            function_calculate_delta_directed
        )


class IntegratedCompleteLikelihoodExactJeffrey(IntegratedCompleteLikelihoodExact):
    title = "ICLex Jeffrey"
    short_title ="ICLexJ"

    def __init__(self, is_directed):
        super(IntegratedCompleteLikelihoodExactJeffrey, self).__init__(is_directed, hyperprior=self.JEFFREY_HYPERPRIOR)


class IntegratedCompleteLikelihoodExactUniform(IntegratedCompleteLikelihoodExact):
    title = "ICLex Uniform"
    short_title = "ICLexU"

    def __init__(self, is_directed):
        super(IntegratedCompleteLikelihoodExactUniform, self).__init__(is_directed, hyperprior=self.UNIFORM_HYPERPRIOR)
