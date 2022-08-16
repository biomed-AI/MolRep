# A collection of help functions needed in peixotos_hierarchical_sbm.py


import math

import numpy as np

NUMBER_OF_RESTRICTED_PARTITIONS_PRECOMPUTED_LIMIT = 1000

spence_constants_a = [
    4.65128586073990045278E-5,
    7.31589045238094711071E-3,
    1.33847639578309018650E-1,
    8.79691311754530315341E-1,
    2.71149851196553469920E0,
    4.25697156008121755724E0,
    3.29771340985225106936E0,
    1.00000000000000000126E0,
]

spence_constants_b = [
    6.90990488912553276999E-4,
    2.54043763932544379113E-2,
    2.82974860602568089943E-1,
    1.41172597751831069617E0,
    3.63800533345137075418E0,
    5.03278880143316990390E0,
    3.54771340985225096217E0,
    9.99999999999999998740E-1,
]


def spence(x):
    """Implementation of spence like in scipy.special.spence"""
    if x < 0.0:
        return np.nan

    if x == 1.0:
        return 0.0

    if x == 0.0:
        return np.pi * np.pi / 6.0

    flag = 0

    if x > 2.0:
        x = 1.0 / x
        flag |= 2

    if x > 1.5:
        w = (1.0 / x) - 1.0
        flag |= 2
    elif x < 0.5:
        w = -x
        flag |= 1
    else:
        w = x - 1.0

    y = -w
    temp = spence_constants_a[0]
    for i in range(1, 8):
        temp = temp * w + spence_constants_a[i]

    y *= temp

    temp = spence_constants_b[0]
    for i in range(1, 8):
        temp = temp * w + spence_constants_b[i]

    y /= temp

    # y = -w * polevl(w, A, 7) / polevl(w, B, 7);

    if flag & 1:
        y = (np.pi * np.pi) / 6.0 - math.log(x) * math.log(1.0 - x) - y

    if flag & 2:
        z = math.log(x)
        y = -0.5 * z * z - y

    return y


def init_number_of_restricted_partitions(limit, save_space=True, return_log=False):
    number_of_restricted_partitions = np.zeros(shape=(limit, limit), dtype=np.uint64)
    number_of_restricted_partitions[:, 1] = np.ones(shape=limit, dtype=np.uint64)
    number_of_restricted_partitions[0, :] = np.ones(shape=limit, dtype=np.uint64)
    number_of_restricted_partitions = list(number_of_restricted_partitions.tolist())
    if save_space:
        del number_of_restricted_partitions[0][1:]
        for m in range(1, limit):
            for n in range(2, m + 1):
                if m - n <= n:
                    number_of_restricted_partitions[m][n] = number_of_restricted_partitions[m][n - 1] \
                                                            + number_of_restricted_partitions[m - n][m - n]
                elif m - n >= 0:
                    number_of_restricted_partitions[m][n] = number_of_restricted_partitions[m][n - 1] \
                                                            + number_of_restricted_partitions[m - n][n]
                else:
                    number_of_restricted_partitions[m][n] = number_of_restricted_partitions[m][n - 1]
            # rest the same
            if save_space:
                del number_of_restricted_partitions[m][(m + 1):]
            else:
                number_of_restricted_partitions[m][(m + 1):] = [number_of_restricted_partitions[m][m]] * (
                        limit - (m + 1))
    else:
        for m in range(1, limit):
            for n in range(2, m + 1):
                if m - n >= 0:
                    number_of_restricted_partitions[m][n] = number_of_restricted_partitions[m][n - 1] \
                                                            + number_of_restricted_partitions[m - n][n]
                else:
                    number_of_restricted_partitions[m][n] = number_of_restricted_partitions[m][n - 1]
            # rest the same
            number_of_restricted_partitions[m][(m + 1):] = [number_of_restricted_partitions[m][m]] * (limit - (m + 1))
    if return_log:
        number_of_restricted_partitions = [
            [math.log(value) if value != 0 else float('-inf') for value in value_m] for value_m in
            number_of_restricted_partitions]

    return number_of_restricted_partitions


def get_log_number_of_restricted_partitions(m, n):
    """Get Number of restricted partitions of m into at most n parts"""
    if n <= 0 or m <= 0:
        if m == 0:
            return 0
        return float('-inf')
    elif m < NUMBER_OF_RESTRICTED_PARTITIONS_PRECOMPUTED_LIMIT:
        if m < n:
            return LOG_NUMBER_OF_RESTRICTED_PARTITIONS_MATRIX[m][m]

        return LOG_NUMBER_OF_RESTRICTED_PARTITIONS_MATRIX[m][n]
    else:
        if n == 1:
            return 1
        elif n < m ** (1.0 / 4):
            # value is always below zero... think about another option
            return log_binom(m - 1, n - 1) - math.lgamma(m + 1)
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

            log_f_u = math.log(v) - 1.5 * math.log(2) - math.log(math.pi) - math.log(u) - .5 * math.log(
                1 - (1 + u * u / 2) * math.exp(-v))
            # log_g_u = math.log(2 * v / u - u * math.log(1 - math.exp(-v)))
            # f_u = v / (2 ** 1.5 * math.pi * u * math.sqrt(1 - (1 + u * u / 2) * math.exp(-v)))
            g_u = 2 * v / u - u * math.log(1 - math.exp(-v))
            return log_f_u - math.log(m) + math.sqrt(m) * g_u


LOG_NUMBER_OF_RESTRICTED_PARTITIONS_MATRIX = \
    init_number_of_restricted_partitions(NUMBER_OF_RESTRICTED_PARTITIONS_PRECOMPUTED_LIMIT, return_log=True)


def log_binom(n, k):
    """Approximation of log(binom(n,k)) for large values"""
    if n < 0 or k < 0:
        raise Exception("Negative values")
    if k >= n:
        return 0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
