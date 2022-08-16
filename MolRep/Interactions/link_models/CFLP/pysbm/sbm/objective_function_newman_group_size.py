"""
Based on
﻿Estimating the Number of Communities in a Network
﻿Newman, M. E. J. and Reinert, G. 2016
"""

from math import gamma
from math import lgamma
from math import log

from .partition import Partition

from .objective_functions import ObjectiveFunction


# @formatter:off
def newman_reinert_non_degree_corrected_undirected(partition):
    """
    Calculates
    P(B,b|G) = P(B)P(b|B)P(G|b)
    * ignoring the unknown constant P(G)
    with
    P(B) = 1/N
    P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    P(G|b) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    and
    p = 2|E|/N^2
    :param partition: partition with all information including the graph and the node partition
    :type partition Partition
    :return: likelihood
    """
    # P(B) = 1/N
    likelihood = 1.0 / partition.get_number_of_nodes()

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    likelihood *= gamma(partition.B) / gamma(partition.get_number_of_nodes() + partition.B)
    for block in range(partition.B):
        likelihood *= gamma(partition.get_number_of_nodes_in_block(block) + 1)

    # P(G|b) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = 2 * partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())
    for from_block in range(partition.B):
        likelihood *= gamma(partition.get_edge_count(from_block, from_block) + 1) / pow(
            .5 * p * partition.get_number_of_nodes_in_block(from_block)
            * partition.get_number_of_nodes_in_block(from_block) + 1,
            partition.get_edge_count(from_block, from_block) + 1)

        for to_block in range(from_block + 1, partition.B):
            likelihood *= gamma(partition.get_edge_count(from_block, to_block) + 1) / pow(
                p * partition.get_number_of_nodes_in_block(from_block)
                * partition.get_number_of_nodes_in_block(to_block) + 1,
                partition.get_edge_count(from_block, to_block) + 1)

    return likelihood


def log_newman_reinert_non_degree_corrected_undirected(partition):
    """
    Calculates
    log P(B,b|G) = log(P(B)P(b|B)P(G|b))
    * ignoring the unknown constant P(G)
    with
    P(B) = 1/N
    P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    P(G|b) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    and
    p = 2|E|/N^2
    should be the likelihood of observing an edge... in undirected case should be 2*|E|/(N*(N+1)/2)
    we use to the formula given by Newman and Reinert
    :param partition: partition with all information including the graph and the node partition
    :type partition Partition
    :return: log likelihood
    """
    # P(B) = 1/N
    likelihood = - log(partition.get_number_of_nodes())

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    likelihood += lgamma(partition.B) - lgamma(partition.get_number_of_nodes() + partition.B)
    for block in range(partition.B):
        likelihood += lgamma(partition.get_number_of_nodes_in_block(block) + 1)

    # P(G|b) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = 2 * partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())
    for from_block in range(partition.B):
        likelihood += lgamma(partition.get_edge_count(from_block, from_block) + 1) - log(
            .5 * p * partition.get_number_of_nodes_in_block(from_block)
            * partition.get_number_of_nodes_in_block(from_block) + 1) \
                      * (partition.get_edge_count(from_block, from_block) + 1)

        for to_block in range(from_block + 1, partition.B):
            likelihood += lgamma(partition.get_edge_count(from_block, to_block) + 1) - log(
                p * partition.get_number_of_nodes_in_block(from_block)
                * partition.get_number_of_nodes_in_block(to_block) + 1) \
                          * (partition.get_edge_count(from_block, to_block) + 1)

    return likelihood


def delta_log_newman_reinert_non_degree_corrected_undirected(partition, from_block, to_block, *args):
    """
    Calculates
    \Delta P(B,b|G)
    :param partition: partition with all information including the graph and the node partition
    :param from_block: node(s) moved from block
    :param to_block: node(s) move to this block
    :type partition Partition
    :return: delta of log likelihood
    """
    if len(args) == 3:
        kit, selfloops, _ = args
        nodes_moved = 1
        no_merger = True
    elif len(args) == 5:
        kit, selfloops, _, nodes_moved, no_merger = args
    else:
        raise ValueError("Wrong number of parameters " + str(args))

    delta = 0

    nodes_remaining = nodes_moved != partition.get_number_of_nodes_in_block(from_block)
    to_block_exists = to_block != partition.B

    # fast exit for "zero move"
    if not to_block_exists and not nodes_remaining:
        return delta

    # P(B) = 1/N
    # no change always fixed

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    # calculate changes of (B-1)!/(N+B-1)!
    if (not nodes_remaining and (not partition.with_empty_blocks or not no_merger)) or not to_block_exists:
        # old term
        delta -= lgamma(partition.B) - lgamma(partition.get_number_of_nodes() + partition.B)

        new_number_of_blocks = partition.B
        if not nodes_remaining:
            new_number_of_blocks -= 1
        else:
            # to block does not exists (combination of if condition for this block and pre check
            new_number_of_blocks += 1
        delta += lgamma(new_number_of_blocks) - lgamma(partition.get_number_of_nodes() + new_number_of_blocks)

    # changes of \prod_r n_r!
    # add old n_from_block! term
    old_number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
    new_number_of_nodes_in_from_block = old_number_of_nodes_in_from_block - nodes_moved
    delta -= lgamma(old_number_of_nodes_in_from_block + 1)
    # new n_from_block! term
    delta += lgamma(new_number_of_nodes_in_from_block + 1)

    # old n_to_block! term
    if to_block_exists:
        old_number_of_nodes_in_to_block = partition.get_number_of_nodes_in_block(to_block)
        delta -= lgamma(old_number_of_nodes_in_to_block + 1)
    else:
        old_number_of_nodes_in_to_block = 0
    new_number_of_nodes_in_to_block = old_number_of_nodes_in_to_block + nodes_moved
    delta += lgamma(new_number_of_nodes_in_to_block + 1)

    # P(G|b) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = 2 * partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())

    for block in range(partition.B):
        if block != from_block and block != to_block:
            # old value from block
            edge_count = partition.get_edge_count(from_block, block)
            delta -= lgamma(edge_count + 1) - log(
                p * old_number_of_nodes_in_from_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

            if nodes_remaining:
                # new term of from block
                if block in kit:
                    edge_count -= kit[block]

                delta += lgamma(edge_count + 1) - log(
                    p * new_number_of_nodes_in_from_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)

            # old value to block
            if to_block_exists:
                edge_count = partition.get_edge_count(to_block, block)
                delta -= lgamma(edge_count + 1) - log(
                    p * old_number_of_nodes_in_to_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)
            else:
                edge_count = 0

            # new term of to block
            if block in kit:
                edge_count += kit[block]

            delta += lgamma(edge_count + 1) - log(
                p * new_number_of_nodes_in_to_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

    # terms with from and to block only
    # from_block <-> from_block
    # old value
    edge_count = partition.get_edge_count(from_block, from_block)
    delta -= lgamma(edge_count + 1) - log(
        .5 * p * old_number_of_nodes_in_from_block
        * old_number_of_nodes_in_from_block + 1) \
        * (edge_count + 1)

    # new term
    if nodes_remaining:
        edge_count -= 2 * (kit[from_block] + selfloops)
        delta += lgamma(edge_count + 1) - log(
            .5 * p * new_number_of_nodes_in_from_block
            * new_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)

    # to_block <-> to_block
    # old value
    if to_block_exists:
        edge_count = partition.get_edge_count(to_block, to_block)
        delta -= lgamma(edge_count + 1) - log(
            .5 * p * old_number_of_nodes_in_to_block
            * old_number_of_nodes_in_to_block + 1) \
            * (edge_count + 1)
    else:
        edge_count = 0

    # new term
    edge_count += 2 * (kit[to_block] + selfloops)
    delta += lgamma(edge_count + 1) - log(
        .5 * p * new_number_of_nodes_in_to_block
        * new_number_of_nodes_in_to_block + 1) \
        * (edge_count + 1)

    # to_block -> from_block
    # old value
    if to_block_exists:
        edge_count = partition.get_edge_count(to_block, from_block)
        delta -= lgamma(edge_count + 1) - log(
            p * old_number_of_nodes_in_to_block
            * old_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)
    else:
        edge_count = 0

    # new term
    if nodes_remaining:
        edge_count += kit[from_block] - kit[to_block]
        delta += lgamma(edge_count + 1) - log(
            p * new_number_of_nodes_in_to_block
            * new_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)

    return delta


# ------ directed


def newman_reinert_non_degree_corrected_directed(partition):
    """
    Own adaption into a directed version:
    Calculates
    P(B,b|G) = P(B)P(b|B)P(G|b)
    * ignoring the unknown constant P(G)
    with
    P(B) = 1/N
    P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    P(G|b) = \prod_{r,s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    and
    p = |E|/N^2
    :param partition: partition with all information including the graph and the node partition
    :type partition Partition
    :return: likelihood
    """
    # P(B) = 1/N
    likelihood = 1.0 / partition.get_number_of_nodes()

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    likelihood *= gamma(partition.B) / gamma(partition.get_number_of_nodes() + partition.B)
    for block in range(partition.B):
        likelihood *= gamma(partition.get_number_of_nodes_in_block(block) + 1)

    # P(G|b) = \prod_{rs} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())
    for from_block in range(partition.B):
        for to_block in range(partition.B):
            likelihood *= gamma(partition.get_edge_count(from_block, to_block) + 1) / pow(
                p * partition.get_number_of_nodes_in_block(from_block)
                * partition.get_number_of_nodes_in_block(to_block) + 1,
                partition.get_edge_count(from_block, to_block) + 1)

    return likelihood


def log_newman_reinert_non_degree_corrected_directed(partition):
    """
    Calculates
    log P(B,b|G) = log(P(B)P(b|B)P(G|b))
    * ignoring the unknown constant P(G)
    with
    P(B) = 1/N
    P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    P(G|b) = \prod_{r,s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    and
    p = |E|/N^2
    :param partition: partition with all information including the graph and the node partition
    :type partition Partition
    :return: log likelihood
    """
    # P(B) = 1/N
    likelihood = - log(partition.get_number_of_nodes())

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    likelihood += lgamma(partition.B) - lgamma(partition.get_number_of_nodes() + partition.B)
    for block in range(partition.B):
        likelihood += lgamma(partition.get_number_of_nodes_in_block(block) + 1)

    # P(G|b) = \prod_{rs} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())
    for from_block in range(partition.B):
        for to_block in range(partition.B):
            likelihood += lgamma(partition.get_edge_count(from_block, to_block) + 1) - log(
                p * partition.get_number_of_nodes_in_block(from_block)
                * partition.get_number_of_nodes_in_block(to_block) + 1) \
                          * (partition.get_edge_count(from_block, to_block) + 1)

    return likelihood


def delta_log_newman_reinert_non_degree_corrected_directed(partition, from_block, to_block, *args):
    """
    Calculates
    \Delta P(B,b|G)
    :param partition: partition with all information including the graph and the node partition
    :param from_block: node(s) moved from block
    :param to_block: node(s) move to this block
    :type partition Partition
    :return: delta of log likelihood
    """
    #     distinguish by given parameters
    if len(args) == 5:
        kit, kti, selfloops, _, _ = args
        nodes_moved = 1
        no_merger = True
    elif len(args) == 7:
        kit, kti, selfloops, _, _, nodes_moved, no_merger = args
    else:
        raise ValueError("Wrong number of parameters " + str(args))

    delta = 0

    nodes_remaining = nodes_moved != partition.get_number_of_nodes_in_block(from_block)
    to_block_exists = to_block != partition.B

    # fast exit for "zero move"
    if not to_block_exists and not nodes_remaining:
        return delta

    # P(B) = 1/N
    # no change always fixed

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    # calculate changes of (B-1)!/(N+B-1)!
    if (not nodes_remaining and (not partition.with_empty_blocks or not no_merger)) or not to_block_exists:
        # old term
        delta -= lgamma(partition.B) - lgamma(partition.get_number_of_nodes() + partition.B)

        new_number_of_blocks = partition.B
        if not nodes_remaining:
            new_number_of_blocks -= 1
        else:
            # to block does not exists (combination of if condition for this block and pre check
            new_number_of_blocks += 1
        delta += lgamma(new_number_of_blocks) - lgamma(partition.get_number_of_nodes() + new_number_of_blocks)

    # changes of \prod_r n_r!
    # add old n_from_block! term
    old_number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
    new_number_of_nodes_in_from_block = old_number_of_nodes_in_from_block - nodes_moved
    delta -= lgamma(old_number_of_nodes_in_from_block + 1)
    # new n_from_block! term
    delta += lgamma(new_number_of_nodes_in_from_block + 1)

    # old n_to_block! term
    if to_block_exists:
        old_number_of_nodes_in_to_block = partition.get_number_of_nodes_in_block(to_block)
        delta -= lgamma(old_number_of_nodes_in_to_block + 1)
    else:
        old_number_of_nodes_in_to_block = 0
    new_number_of_nodes_in_to_block = old_number_of_nodes_in_to_block + nodes_moved
    delta += lgamma(new_number_of_nodes_in_to_block + 1)

    # P(G|b) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())

    for block in range(partition.B):
        if block != from_block and block != to_block:
            # (from_block, to_block)->block
            # old value from block
            edge_count = partition.get_edge_count(from_block, block)
            delta -= lgamma(edge_count + 1) - log(
                p * old_number_of_nodes_in_from_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

            if nodes_remaining:
                # new term of from block
                if block in kit:
                    edge_count -= kit[block]

                delta += lgamma(edge_count + 1) - log(
                    p * new_number_of_nodes_in_from_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)

            # old value to block
            if to_block_exists:
                edge_count = partition.get_edge_count(to_block, block)
                delta -= lgamma(edge_count + 1) - log(
                    p * old_number_of_nodes_in_to_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)
            else:
                edge_count = 0

            # new term of to block
            if block in kit:
                edge_count += kit[block]

            delta += lgamma(edge_count + 1) - log(
                p * new_number_of_nodes_in_to_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

            # ----> other directions
            # block -> (from_block, to_block)
            # old value from block
            edge_count = partition.get_edge_count(block, from_block)
            delta -= lgamma(edge_count + 1) - log(
                p * old_number_of_nodes_in_from_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

            if nodes_remaining:
                # new term of from block
                if block in kti:
                    edge_count -= kti[block]

                delta += lgamma(edge_count + 1) - log(
                    p * new_number_of_nodes_in_from_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)

            # old value to block
            if to_block_exists:
                edge_count = partition.get_edge_count(block, to_block)
                delta -= lgamma(edge_count + 1) - log(
                    p * old_number_of_nodes_in_to_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)
            else:
                edge_count = 0

            # new term of to block
            if block in kti:
                edge_count += kti[block]

            delta += lgamma(edge_count + 1) - log(
                p * new_number_of_nodes_in_to_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

    # terms with from and to block only
    # from_block <-> from_block
    # old value
    edge_count = partition.get_edge_count(from_block, from_block)
    delta -= lgamma(edge_count + 1) - log(
        p * old_number_of_nodes_in_from_block
        * old_number_of_nodes_in_from_block + 1) \
        * (edge_count + 1)

    # new term
    if nodes_remaining:
        edge_count -= kit[from_block] + kti[from_block] + selfloops
        delta += lgamma(edge_count + 1) - log(
            p * new_number_of_nodes_in_from_block
            * new_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)

    # to_block <-> to_block
    # old value
    if to_block_exists:
        edge_count = partition.get_edge_count(to_block, to_block)
        delta -= lgamma(edge_count + 1) - log(
            p * old_number_of_nodes_in_to_block
            * old_number_of_nodes_in_to_block + 1) \
            * (edge_count + 1)
    else:
        edge_count = 0

    # new term
    edge_count += kit[to_block] + kti[to_block] + selfloops
    delta += lgamma(edge_count + 1) - log(
        p * new_number_of_nodes_in_to_block
        * new_number_of_nodes_in_to_block + 1) \
        * (edge_count + 1)

    # to_block -> from_block
    # old value
    if to_block_exists:
        edge_count = partition.get_edge_count(to_block, from_block)
        delta -= lgamma(edge_count + 1) - log(
            p * old_number_of_nodes_in_to_block
            * old_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)
    else:
        edge_count = 0

    # new term
    if nodes_remaining:
        edge_count += kit[from_block] - kti[to_block]
        delta += lgamma(edge_count + 1) - log(
            p * new_number_of_nodes_in_to_block
            * new_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)

    # from_block -> to_block
    # old value
    if to_block_exists:
        edge_count = partition.get_edge_count(from_block, to_block)
        delta -= lgamma(edge_count + 1) - log(
            p * old_number_of_nodes_in_to_block
            * old_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)
    else:
        edge_count = 0

    # new term
    if nodes_remaining:
        edge_count += kti[from_block] - kit[to_block]
        delta += lgamma(edge_count + 1) - log(
            p * new_number_of_nodes_in_to_block
            * new_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)

    return delta


# -------------------------------------
# Degree Corrected Version
# -------------------------------------


def newman_reinert_degree_corrected_undirected(partition):
    """
    Calculates
    P(B,b|G) = P(B)P(b|B)P(G|b)
    * ignoring the unknown constant P(G)
    with
    P(B) = 1/N
    P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    P_standard SBM(G|B) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    P_dc (G|b) = P_standard SBM(G|b) \prod_{r,n_r\ne0} n_r^e_r(n_r-1)!/(n_r+e_r-1)!
    and
    p = 2|E|/N^2
    :param partition: partition with all information including the graph and the node partition
    :type partition Partition
    :return: likelihood
    """
    # P(B) = 1/N
    likelihood = 1.0 / partition.get_number_of_nodes()

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    likelihood *= gamma(partition.B) / gamma(partition.get_number_of_nodes() + partition.B)
    for block in range(partition.B):
        likelihood *= gamma(partition.get_number_of_nodes_in_block(block) + 1)

    # P(G|b) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = 2 * partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())
    for from_block in range(partition.B):
        number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
        likelihood *= gamma(partition.get_edge_count(from_block, from_block) + 1) / pow(
            .5 * p * number_of_nodes_in_from_block * number_of_nodes_in_from_block + 1,
            partition.get_edge_count(from_block, from_block) + 1)

        # extra factor for degree correction
        if number_of_nodes_in_from_block > 0:
            likelihood *= pow(number_of_nodes_in_from_block, partition.get_degree_of_block(from_block)) \
                          * gamma(number_of_nodes_in_from_block) \
                          / gamma(number_of_nodes_in_from_block + partition.get_degree_of_block(from_block))

        for to_block in range(from_block + 1, partition.B):
            likelihood *= gamma(partition.get_edge_count(from_block, to_block) + 1) / pow(
                p * number_of_nodes_in_from_block
                * partition.get_number_of_nodes_in_block(to_block) + 1,
                partition.get_edge_count(from_block, to_block) + 1)

    return likelihood


def log_newman_reinert_degree_corrected_undirected(partition):
    """
    Calculates
    log P(B,b|G) = log(P(B)P(b|B)P(G|b))
    * ignoring the unknown constant P(G)
    with
    P(B) = 1/N
    P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    P_standard SBM(G|B) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    P_dc (G|b) = P_standard SBM(G|b) \prod_{r,n_r\ne0} n_r^e_r(n_r-1)!/(n_r+e_r-1)!
    and
    p = 2|E|/N^2
    should be the likelihood of observing an edge... in undirected case should be 2*|E|/(N*(N+1)/2)
    we use to the formula given by Newman and Reinert
    :param partition: partition with all information including the graph and the node partition
    :type partition Partition
    :return: log likelihood
    """
    # P(B) = 1/N
    likelihood = - log(partition.get_number_of_nodes())

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    likelihood += lgamma(partition.B) - lgamma(partition.get_number_of_nodes() + partition.B)
    for block in range(partition.B):
        likelihood += lgamma(partition.get_number_of_nodes_in_block(block) + 1)

    # P(G|b) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = 2 * partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())
    for from_block in range(partition.B):
        number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
        likelihood += lgamma(partition.get_edge_count(from_block, from_block) + 1) - log(
            .5 * p * number_of_nodes_in_from_block * number_of_nodes_in_from_block + 1) \
            * (partition.get_edge_count(from_block, from_block) + 1)

        # extra factor for degree correction
        if number_of_nodes_in_from_block > 0:
            likelihood += log(number_of_nodes_in_from_block) * partition.get_degree_of_block(from_block) \
                          + lgamma(number_of_nodes_in_from_block) \
                          - lgamma(number_of_nodes_in_from_block + partition.get_degree_of_block(from_block))

        for to_block in range(from_block + 1, partition.B):
            likelihood += lgamma(partition.get_edge_count(from_block, to_block) + 1) - log(
                p * number_of_nodes_in_from_block
                * partition.get_number_of_nodes_in_block(to_block) + 1) \
                          * (partition.get_edge_count(from_block, to_block) + 1)

    return likelihood


def delta_log_newman_reinert_degree_corrected_undirected(partition, from_block, to_block, *args):
    """
    Calculates
    \Delta P(B,b|G)
    :param partition: partition with all information including the graph and the node partition
    :param from_block: node(s) moved from block
    :param to_block: node(s) move to this block
    :type partition Partition
    :return: delta of log likelihood
    """
    if len(args) == 3:
        kit, selfloops, degree = args
        nodes_moved = 1
        no_merger = True
    elif len(args) == 5:
        kit, selfloops, degree, nodes_moved, no_merger = args
    else:
        raise ValueError("Wrong number of parameters " + str(args))

    delta = 0

    nodes_remaining = nodes_moved != partition.get_number_of_nodes_in_block(from_block)
    to_block_exists = to_block != partition.B

    # fast exit for "zero move"
    if not to_block_exists and not nodes_remaining:
        return delta

    # P(B) = 1/N
    # no change always fixed

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    # calculate changes of (B-1)!/(N+B-1)!
    if (not nodes_remaining and (not partition.with_empty_blocks or not no_merger)) or not to_block_exists:
        # old term
        delta -= lgamma(partition.B) - lgamma(partition.get_number_of_nodes() + partition.B)

        new_number_of_blocks = partition.B
        if not nodes_remaining:
            new_number_of_blocks -= 1
        else:
            # to block does not exists (combination of if condition for this block and pre check
            new_number_of_blocks += 1
        delta += lgamma(new_number_of_blocks) - lgamma(partition.get_number_of_nodes() + new_number_of_blocks)

    # changes of \prod_r n_r!
    # add old n_from_block! term
    old_number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
    new_number_of_nodes_in_from_block = old_number_of_nodes_in_from_block - nodes_moved
    delta -= lgamma(old_number_of_nodes_in_from_block + 1)
    # new n_from_block! term
    delta += lgamma(new_number_of_nodes_in_from_block + 1)

    # old n_to_block! term
    if to_block_exists:
        old_number_of_nodes_in_to_block = partition.get_number_of_nodes_in_block(to_block)
        delta -= lgamma(old_number_of_nodes_in_to_block + 1)
        old_degree_of_to_block = partition.get_degree_of_block(to_block)
    else:
        old_number_of_nodes_in_to_block = 0
        old_degree_of_to_block = 0
    new_number_of_nodes_in_to_block = old_number_of_nodes_in_to_block + nodes_moved
    delta += lgamma(new_number_of_nodes_in_to_block + 1)

    # P(G|b) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = 2 * partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())

    for block in range(partition.B):
        if block != from_block and block != to_block:
            # old value from block
            edge_count = partition.get_edge_count(from_block, block)
            delta -= lgamma(edge_count + 1) - log(
                p * old_number_of_nodes_in_from_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

            if nodes_remaining:
                # new term of from block
                if block in kit:
                    edge_count -= kit[block]

                delta += lgamma(edge_count + 1) - log(
                    p * new_number_of_nodes_in_from_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)

            # old value to block
            if to_block_exists:
                edge_count = partition.get_edge_count(to_block, block)
                delta -= lgamma(edge_count + 1) - log(
                    p * old_number_of_nodes_in_to_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)
            else:
                edge_count = 0

            # new term of to block
            if block in kit:
                edge_count += kit[block]

            delta += lgamma(edge_count + 1) - log(
                p * new_number_of_nodes_in_to_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

    # terms with from and to block only
    # from_block <-> from_block
    # old value
    edge_count = partition.get_edge_count(from_block, from_block)
    delta -= lgamma(edge_count + 1) - log(
        .5 * p * old_number_of_nodes_in_from_block
        * old_number_of_nodes_in_from_block + 1) \
        * (edge_count + 1)

    # new term
    if nodes_remaining:
        edge_count -= 2 * (kit[from_block] + selfloops)
        delta += lgamma(edge_count + 1) - log(
            .5 * p * new_number_of_nodes_in_from_block
            * new_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)

    # to_block <-> to_block
    # old value
    if to_block_exists:
        edge_count = partition.get_edge_count(to_block, to_block)
        delta -= lgamma(edge_count + 1) - log(
            .5 * p * old_number_of_nodes_in_to_block
            * old_number_of_nodes_in_to_block + 1) \
            * (edge_count + 1)
    else:
        edge_count = 0

    # new term
    edge_count += 2 * (kit[to_block] + selfloops)
    delta += lgamma(edge_count + 1) - log(
        .5 * p * new_number_of_nodes_in_to_block
        * new_number_of_nodes_in_to_block + 1) \
        * (edge_count + 1)

    # to_block -> from_block
    # old value
    if to_block_exists:
        edge_count = partition.get_edge_count(to_block, from_block)
        delta -= lgamma(edge_count + 1) - log(
            p * old_number_of_nodes_in_to_block
            * old_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)
    else:
        edge_count = 0

    # new term
    if nodes_remaining:
        edge_count += kit[from_block] - kit[to_block]
        delta += lgamma(edge_count + 1) - log(
            p * new_number_of_nodes_in_to_block
            * new_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)

    # -------------------------------
    # new terms for degree correction
    # term of from block
    # old
    if old_number_of_nodes_in_from_block > 0:
        delta -= log(old_number_of_nodes_in_from_block) * partition.get_degree_of_block(from_block) \
                 + lgamma(old_number_of_nodes_in_from_block) \
                 - lgamma(old_number_of_nodes_in_from_block + partition.get_degree_of_block(from_block))

    # new
    if nodes_remaining:
        delta += log(new_number_of_nodes_in_from_block) * (partition.get_degree_of_block(from_block) - degree) \
                 + lgamma(new_number_of_nodes_in_from_block) \
                 - lgamma(new_number_of_nodes_in_from_block + partition.get_degree_of_block(from_block) - degree)

    # terms of to block
    # this includes the check if the to block exists (because then there would be no node inside)
    if old_number_of_nodes_in_to_block > 0:
        delta -= log(old_number_of_nodes_in_to_block) * old_degree_of_to_block \
                 + lgamma(old_number_of_nodes_in_to_block) \
                 - lgamma(old_number_of_nodes_in_to_block + old_degree_of_to_block)

    delta += log(new_number_of_nodes_in_to_block) * (old_degree_of_to_block + degree) \
        + lgamma(new_number_of_nodes_in_to_block) \
        - lgamma(new_number_of_nodes_in_to_block + old_degree_of_to_block + degree)

    return delta


# ------ directed
# directed degree corrected formula not stated in the paper!

def newman_reinert_degree_corrected_directed(partition):
    """
    Own adaption into a directed version:
    Calculates
    P(B,b|G) = P(B)P(b|B)P(G|b)
    * ignoring the unknown constant P(G)
    with
    P(B) = 1/N
    P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    P_standard SBM(G|B) = \prod_{rs} e_rs!/(pn_rn_s+1)^{e_rs+1}
    P_dc (G|b) = P_standard SBM(G|b) \prod_{r,n_r\ne0} n_r^(e^in_r+e^out_r)(n_r-1)!/(n_r+e^in_r+e^out_r-1)!
    and
    p = |E|/N^2
    :param partition: partition with all information including the graph and the node partition
    :type partition Partition
    :return: likelihood
    """
    # P(B) = 1/N
    likelihood = 1.0 / partition.get_number_of_nodes()

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    likelihood *= gamma(partition.B) / gamma(partition.get_number_of_nodes() + partition.B)
    for block in range(partition.B):
        likelihood *= gamma(partition.get_number_of_nodes_in_block(block) + 1)

    # P(G|b) = \prod_{rs} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())
    for from_block in range(partition.B):
        number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
        # extra factor for degree correction
        if number_of_nodes_in_from_block > 0:
            likelihood *= pow(number_of_nodes_in_from_block,
                              partition.get_in_degree_of_block(from_block)
                              + partition.get_out_degree_of_block(from_block)) \
                          * gamma(number_of_nodes_in_from_block) \
                          / gamma(number_of_nodes_in_from_block
                                  + partition.get_in_degree_of_block(from_block)
                                  + partition.get_out_degree_of_block(from_block))
        for to_block in range(partition.B):
            likelihood *= gamma(partition.get_edge_count(from_block, to_block) + 1) / pow(
                p * number_of_nodes_in_from_block
                * partition.get_number_of_nodes_in_block(to_block) + 1,
                partition.get_edge_count(from_block, to_block) + 1)

    return likelihood


def log_newman_reinert_degree_corrected_directed(partition):
    """
    Calculates
    log P(B,b|G) = log(P(B)P(b|B)P(G|b))
    * ignoring the unknown constant P(G)
    with
    P(B) = 1/N
    P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    P_standard SBM(G|B) = \prod_{rs} e_rs!/(pn_rn_s+1)^{e_rs+1}
    P_dc (G|b) = P_standard SBM(G|b) \prod_{r,n_r\ne0} n_r^(e^in_r+e^out_r)(n_r-1)!/(n_r+e^in_r+e^out_r-1)!
    and
    p = |E|/N^2
    :param partition: partition with all information including the graph and the node partition
    :type partition Partition
    :return: log likelihood
    """
    # P(B) = 1/N
    likelihood = - log(partition.get_number_of_nodes())

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    likelihood += lgamma(partition.B) - lgamma(partition.get_number_of_nodes() + partition.B)
    for block in range(partition.B):
        likelihood += lgamma(partition.get_number_of_nodes_in_block(block) + 1)

    # P(G|b) = \prod_{rs} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())
    for from_block in range(partition.B):
        number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
        # extra factor for degree correction
        if number_of_nodes_in_from_block > 0:
            likelihood += log(number_of_nodes_in_from_block) \
                          * (partition.get_in_degree_of_block(from_block)
                             + partition.get_out_degree_of_block(from_block)) \
                          + lgamma(number_of_nodes_in_from_block) \
                          - lgamma(number_of_nodes_in_from_block
                                   + partition.get_in_degree_of_block(from_block)
                                   + partition.get_out_degree_of_block(from_block))
        for to_block in range(partition.B):
            likelihood += lgamma(partition.get_edge_count(from_block, to_block) + 1) - log(
                p * number_of_nodes_in_from_block
                * partition.get_number_of_nodes_in_block(to_block) + 1) \
                          * (partition.get_edge_count(from_block, to_block) + 1)

    return likelihood


def delta_log_newman_reinert_degree_corrected_directed(partition, from_block, to_block, *args):
    """
    Calculates
    \Delta P(B,b|G)
    :param partition: partition with all information including the graph and the node partition
    :param from_block: node(s) moved from block
    :param to_block: node(s) move to this block
    :type partition Partition
    :return: delta of log likelihood
    """
    #     distinguish by given parameters
    if len(args) == 5:
        kit, kti, selfloops, in_degree, out_degree = args
        nodes_moved = 1
        no_merger = True
    elif len(args) == 7:
        kit, kti, selfloops, in_degree, out_degree, nodes_moved, no_merger = args
    else:
        raise ValueError("Wrong number of parameters " + str(args))

    delta = 0

    nodes_remaining = nodes_moved != partition.get_number_of_nodes_in_block(from_block)
    to_block_exists = to_block != partition.B

    # fast exit for "zero move"
    if not to_block_exists and not nodes_remaining:
        return delta

    # P(B) = 1/N
    # no change always fixed

    # P(b|B) = (B-1)!/(N+B-1)! \prod_r n_r!
    # calculate changes of (B-1)!/(N+B-1)!
    if (not nodes_remaining and (not partition.with_empty_blocks or not no_merger)) or not to_block_exists:
        # old term
        delta -= lgamma(partition.B) - lgamma(partition.get_number_of_nodes() + partition.B)

        new_number_of_blocks = partition.B
        if not nodes_remaining:
            new_number_of_blocks -= 1
        else:
            # to block does not exists (combination of if condition for this block and pre check
            new_number_of_blocks += 1
        delta += lgamma(new_number_of_blocks) - lgamma(partition.get_number_of_nodes() + new_number_of_blocks)

    # changes of \prod_r n_r!
    # add old n_from_block! term
    old_number_of_nodes_in_from_block = partition.get_number_of_nodes_in_block(from_block)
    new_number_of_nodes_in_from_block = old_number_of_nodes_in_from_block - nodes_moved
    delta -= lgamma(old_number_of_nodes_in_from_block + 1)
    # new n_from_block! term
    delta += lgamma(new_number_of_nodes_in_from_block + 1)

    # old n_to_block! term
    if to_block_exists:
        old_number_of_nodes_in_to_block = partition.get_number_of_nodes_in_block(to_block)
        delta -= lgamma(old_number_of_nodes_in_to_block + 1)
        old_in_degree_of_to_block = partition.get_in_degree_of_block(to_block)
        old_out_degree_of_to_block = partition.get_out_degree_of_block(to_block)
    else:
        old_number_of_nodes_in_to_block = 0
        old_in_degree_of_to_block = 0
        old_out_degree_of_to_block = 0
    new_number_of_nodes_in_to_block = old_number_of_nodes_in_to_block + nodes_moved
    delta += lgamma(new_number_of_nodes_in_to_block + 1)

    # P(G|b) = \prod_r e_rr! / (1/2 p n_r^2 + 1)^{e_rr+1} \prod_{r<s} e_rs!/(pn_rn_s+1)^{e_rs+1}
    p = partition.get_number_of_edges() / (partition.get_number_of_nodes() * partition.get_number_of_nodes())

    for block in range(partition.B):
        if block != from_block and block != to_block:
            # (from_block, to_block)->block
            # old value from block
            edge_count = partition.get_edge_count(from_block, block)
            delta -= lgamma(edge_count + 1) - log(
                p * old_number_of_nodes_in_from_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

            if nodes_remaining:
                # new term of from block
                if block in kit:
                    edge_count -= kit[block]

                delta += lgamma(edge_count + 1) - log(
                    p * new_number_of_nodes_in_from_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)

            # old value to block
            if to_block_exists:
                edge_count = partition.get_edge_count(to_block, block)
                delta -= lgamma(edge_count + 1) - log(
                    p * old_number_of_nodes_in_to_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)
            else:
                edge_count = 0

            # new term of to block
            if block in kit:
                edge_count += kit[block]

            delta += lgamma(edge_count + 1) - log(
                p * new_number_of_nodes_in_to_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

            # ----> other directions
            # block -> (from_block, to_block)
            # old value from block
            edge_count = partition.get_edge_count(block, from_block)
            delta -= lgamma(edge_count + 1) - log(
                p * old_number_of_nodes_in_from_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

            if nodes_remaining:
                # new term of from block
                if block in kti:
                    edge_count -= kti[block]

                delta += lgamma(edge_count + 1) - log(
                    p * new_number_of_nodes_in_from_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)

            # old value to block
            if to_block_exists:
                edge_count = partition.get_edge_count(block, to_block)
                delta -= lgamma(edge_count + 1) - log(
                    p * old_number_of_nodes_in_to_block
                    * partition.get_number_of_nodes_in_block(block) + 1) \
                    * (edge_count + 1)
            else:
                edge_count = 0

            # new term of to block
            if block in kti:
                edge_count += kti[block]

            delta += lgamma(edge_count + 1) - log(
                p * new_number_of_nodes_in_to_block
                * partition.get_number_of_nodes_in_block(block) + 1) \
                * (edge_count + 1)

    # terms with from and to block only
    # from_block <-> from_block
    # old value
    edge_count = partition.get_edge_count(from_block, from_block)
    delta -= lgamma(edge_count + 1) - log(
        p * old_number_of_nodes_in_from_block
        * old_number_of_nodes_in_from_block + 1) \
        * (edge_count + 1)

    # new term
    if nodes_remaining:
        edge_count -= kit[from_block] + kti[from_block] + selfloops
        delta += lgamma(edge_count + 1) - log(
            p * new_number_of_nodes_in_from_block
            * new_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)

    # to_block <-> to_block
    # old value
    if to_block_exists:
        edge_count = partition.get_edge_count(to_block, to_block)
        delta -= lgamma(edge_count + 1) - log(
            p * old_number_of_nodes_in_to_block
            * old_number_of_nodes_in_to_block + 1) \
            * (edge_count + 1)
    else:
        edge_count = 0

    # new term
    edge_count += kit[to_block] + kti[to_block] + selfloops
    delta += lgamma(edge_count + 1) - log(
        p * new_number_of_nodes_in_to_block
        * new_number_of_nodes_in_to_block + 1) \
        * (edge_count + 1)

    # to_block -> from_block
    # old value
    if to_block_exists:
        edge_count = partition.get_edge_count(to_block, from_block)
        delta -= lgamma(edge_count + 1) - log(
            p * old_number_of_nodes_in_to_block
            * old_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)
    else:
        edge_count = 0

    # new term
    if nodes_remaining:
        edge_count += kit[from_block] - kti[to_block]
        delta += lgamma(edge_count + 1) - log(
            p * new_number_of_nodes_in_to_block
            * new_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)

    # from_block -> to_block
    # old value
    if to_block_exists:
        edge_count = partition.get_edge_count(from_block, to_block)
        delta -= lgamma(edge_count + 1) - log(
            p * old_number_of_nodes_in_to_block
            * old_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)
    else:
        edge_count = 0

    # new term
    if nodes_remaining:
        edge_count += kti[from_block] - kit[to_block]
        delta += lgamma(edge_count + 1) - log(
            p * new_number_of_nodes_in_to_block
            * new_number_of_nodes_in_from_block + 1) \
            * (edge_count + 1)

    # -------------------------------
    # new terms for degree correction
    # term of from block
    # old
    if old_number_of_nodes_in_from_block > 0:
        delta -= log(old_number_of_nodes_in_from_block) \
                 * (partition.get_in_degree_of_block(from_block) + partition.get_out_degree_of_block(from_block)) \
                 + lgamma(old_number_of_nodes_in_from_block) \
                 - lgamma(old_number_of_nodes_in_from_block
                          + partition.get_in_degree_of_block(from_block)
                          + partition.get_out_degree_of_block(from_block))

    # new
    if nodes_remaining:
        delta += log(new_number_of_nodes_in_from_block) \
                 * (partition.get_in_degree_of_block(from_block) + partition.get_out_degree_of_block(from_block)
                    - in_degree - out_degree) \
                 + lgamma(new_number_of_nodes_in_from_block) \
                 - lgamma(new_number_of_nodes_in_from_block
                          + partition.get_in_degree_of_block(from_block) + partition.get_out_degree_of_block(from_block)
                          - in_degree - out_degree)

    # terms of to block
    # this includes the check if the to block exists (because then there would be no node inside)
    if old_number_of_nodes_in_to_block > 0:
        delta -= log(old_number_of_nodes_in_to_block) * (old_in_degree_of_to_block + old_out_degree_of_to_block) \
                 + lgamma(old_number_of_nodes_in_to_block) \
                 - lgamma(old_number_of_nodes_in_to_block + old_in_degree_of_to_block + old_out_degree_of_to_block)

    delta += log(new_number_of_nodes_in_to_block) \
        * (old_in_degree_of_to_block + old_out_degree_of_to_block + in_degree + out_degree) \
        + lgamma(new_number_of_nodes_in_to_block) \
        - lgamma(new_number_of_nodes_in_to_block
                 + old_in_degree_of_to_block + old_out_degree_of_to_block + in_degree + out_degree)

    return delta
# @formatter:on


class NewmanReinertNonDegreeCorrected(ObjectiveFunction):
    title = "Non Degree Corrected Newman Reinert"
    short_title = "SNR"

    def __init__(self, is_directed):
        function_calculate_undirected = log_newman_reinert_non_degree_corrected_undirected
        function_calculate_directed = log_newman_reinert_non_degree_corrected_directed
        function_calculate_delta_undirected = delta_log_newman_reinert_non_degree_corrected_undirected
        function_calculate_delta_directed = delta_log_newman_reinert_non_degree_corrected_directed

        super(NewmanReinertNonDegreeCorrected, self).__init__(
            is_directed,
            function_calculate_undirected,
            function_calculate_directed,
            function_calculate_delta_undirected,
            function_calculate_delta_directed
        )


class NewmanReinertDegreeCorrected(ObjectiveFunction):
    title = "Degree Corrected Newman Reinert"
    short_title = "DCNR"

    def __init__(self, is_directed):
        function_calculate_undirected = log_newman_reinert_degree_corrected_undirected
        function_calculate_directed = log_newman_reinert_degree_corrected_directed
        function_calculate_delta_undirected = delta_log_newman_reinert_degree_corrected_undirected
        function_calculate_delta_directed = delta_log_newman_reinert_degree_corrected_directed

        super(NewmanReinertDegreeCorrected, self).__init__(
            is_directed,
            function_calculate_undirected,
            function_calculate_directed,
            function_calculate_delta_undirected,
            function_calculate_delta_directed
        )
