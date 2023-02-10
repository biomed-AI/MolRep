# A collection of help functions needed in peixotos_hierarchical_sbm.py


import math

from scipy.special import binom
from .peixotos_hierarchical_sbm_tools import NUMBER_OF_RESTRICTED_PARTITIONS_PRECOMPUTED_LIMIT
from .peixotos_hierarchical_sbm_tools import spence
from .peixotos_hierarchical_sbm_tools import init_number_of_restricted_partitions

BINOMIALS = {}


def get_number_of_restricted_partitions(m, n):
    if n <= 0 or m <= 0:
        if m == 0:
            return 1
        return 0
    elif m < NUMBER_OF_RESTRICTED_PARTITIONS_PRECOMPUTED_LIMIT:
        if m < n:
            return NUMBER_OF_RESTRICTED_PARTITIONS_MATRIX[m][m]

        return NUMBER_OF_RESTRICTED_PARTITIONS_MATRIX[m][n]
    else:
        if n == 1:
            return 1
        elif n < m ** (1.0 / 4):
            binomial_values = (m - 1, n - 1)
            if binomial_values not in BINOMIALS:
                BINOMIALS[binomial_values] = binom(*binomial_values)
            # value is always below zero... think about another option
            return BINOMIALS[binomial_values] / math.factorial(m)
        else:
            u = n / math.sqrt(m)
            v = u
            for _ in range(100):
                # formula from Peixoto's implementation
                # https://git.skewed.de/count0/graph-tool/blob/master/src/graph/inference/support/int_part.cc
                v_new = u * math.sqrt(spence(math.exp(-v)))
                if math.fabs(v - v_new) < .0001:
                    v = v_new
                    break
                v = v_new
            else:
                raise Exception("Fix Point iteration as search for v not converged in 100 steps:", v)

            f_u = v / (2 ** 1.5 * math.pi * u * math.sqrt(1 - (1 + u * u / 2) * math.exp(-v)))
            g_u = 2 * v / u - u * math.log(1 - math.exp(-v))
            return f_u / m * math.exp(math.sqrt(m) * g_u)


NUMBER_OF_RESTRICTED_PARTITIONS_MATRIX = \
    init_number_of_restricted_partitions(NUMBER_OF_RESTRICTED_PARTITIONS_PRECOMPUTED_LIMIT)
