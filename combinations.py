import numpy as np
from scipy.special import binom, comb, factorial


def partitions(number, m, lower_bound=1):
    """
        Generates all partitions of the form number = a[1] + ... + a[m]
        with restriction a[1] >= a[2] >= ... >= a[m] >= lower_bound
        see Knuth, Art of Programming, 7.2.1.4
        :param number: int, integer to split into summands
        :param m: int, number of summands
        :param lower_bound: int, lower bound for summands
        :return: sequence of np.array with shape (m,)
    """
    a = lower_bound * np.ones(m + 1, dtype=int)
    a[0] = number - m * lower_bound + lower_bound
    a[m] = lower_bound - 2
    while True:
        yield a[:-1]
        if a[1] < a[0] - 1:
            a[0] -= 1
            a[1] += 1
            continue
        j = 2
        s = a[0] + a[1] - 1
        while a[j] >= a[0] - 1:
            s += a[j]
            j += 1
        if j >= m:
            return
        x = a[j] + 1
        a[j] = x
        j -= 1
        while j > 0:
            a[j] = x
            s -= x
            j -= 1
        a[0] = s


def compositions(number, s):
    """
        Generates all compositions (ordered partitions) of integer number into s non-negative summands;
        see Knuth, Art of Programming, 7.2.1.3, problem 3
        :param number: int, integer to split into summands
        :param s: int, number of summands
        :return sequence of np.array with shape (s,)
    """
    assert s > 0, "Number of summands should be positive!"
    if s == 1:
        yield np.array([number], dtype=int)
        return
    q = np.zeros(s, dtype=int)
    r = None
    q[0] = number
    while True:
        yield q
        if q[0] == 0:
            if r == s - 1:
                break
            else:
                q[0] = q[r] - 1
                q[r] = 0
                r += 1
        else:
            q[0] -= 1
            r = 1
        q[r] += 1


def bounded_compositions(number, s, bound: [list]):
    """
        Generates all compositions (ordered partitions) of the form number = q[1] + ... + q[s]
        with restriction 0 <= q[j] <= bound[j]; see Knuth, Art of Programming, 7.2.1.3, problem 60
        :param number: int, integer to split into summands
        :param s: int, number of summands
        :param bound: list of non-negative integers of size s
        :return sequence of np.array with shape (s,)
    """
    # Q1
    if sum(bound) < number:
        return
    if s == 1:
        yield np.array([number], dtype=int)
        return

    q = np.zeros(s, dtype=int)
    x = number

    while True:
        # Q2
        j = 0
        while x > bound[j]:
            q[j] = bound[j]
            x -= bound[j]
            j += 1
        q[j] = x

        # Q3
        while True:
            yield q

            flag = False
            # Q4
            if j == 0:
                x = q[0] - 1
                j = 1
            else:
                if q[0] == 0:
                    x = q[j] - 1
                    q[j] = 0
                    j += 1
                else:
                    flag = True

            if not flag:
                # Q5
                while j < s and q[j] == bound[j]:
                    x += bound[j]
                    q[j] = 0
                    j += 1
                if j >= s:
                    return

                # Q6
                q[j] += 1
                if x == 0:
                    q[0] = 0
                    continue
                else:
                    break

            # Q7
            while q[j] == bound[j]:
                j += 1
                if j >= s:
                    return
            q[j] += 1
            j -= 1
            q[j] -= 1
            if q[0] == 0:
                j = 1


def contingency_table(row_sums, column_sums: [list]):
    """
        Generates all contingency tables with given sums of rows and columns;
        see Knuth, Art of Programming, 7.2.1.3, problem 62
        :param row_sums: list of positive integers, the contingency table row sums
        :param column_sums: list of positive integers, the contingency table column sums
        :return sequence of np.array with shape (len(r), len(c))
    """
    assert sum(row_sums) == sum(column_sums), "row and column sums must be equal"
    if len(row_sums) == 1:
        yield np.array(column_sums)
    elif len(row_sums) == 2:
        for composition in bounded_compositions(row_sums[0], len(column_sums), column_sums):
            yield np.vstack((composition, column_sums - composition))
    else:
        for composition in bounded_compositions(row_sums[0], len(column_sums), column_sums):
            residuals = column_sums - composition
            non_zero_idx = np.argwhere(residuals).ravel()
            for residual_composition in contingency_table(row_sums[1:], residuals[non_zero_idx]):
                res = np.zeros((len(row_sums) - 1, len(column_sums)), dtype=int)
                res[:, non_zero_idx] = residual_composition
                yield np.vstack((composition, res))


def check_lex_order_for(a1, a2):
    for i, item in enumerate(a1):
        if item < a2[i]:
            return True
        elif item > a2[i]:
            return False
    return True


def enumerate_deals(suit_sizes, hand_sizes, trump=False, reduce_perms=False):
    """
        Enumerates all deals with given suit sizes and hand sizes
        :param suit_sizes: list of non-negative integers, the contingency table row sums;
         must be sorted with respect to non-trump suits
        :param hand_sizes: list of non-negative integers, the contingency table column sums
        :param trump: if True, then 0th suit is trump
        :param reduce_perms: if True then reduce suit permutations
        :return dict {key: value}, key is raveled contingency matrix as a tuple, value is the total number of deals
    """
    result = {}
    suit_array = np.array(suit_sizes)
    non_zero_suits = suit_array[suit_array > 0]
    matrix_fact_prod = factorial(non_zero_suits).prod()
    for d in contingency_table(list(non_zero_suits), hand_sizes):
        if len(d.shape) == 1:
            d = d[None, :]

        if not reduce_perms:
            variants = matrix_fact_prod / factorial(d).prod()
        else:
            # check if equal suits are ordered lexicographically
            lex_flag = False
            for i in range(int(trump), len(non_zero_suits) - 1):
                if non_zero_suits[i] == non_zero_suits[i + 1] and not check_lex_order_for(d[i], d[i + 1]):
                    lex_flag = True
                    break
            if lex_flag:
                continue

            # calculate number of deals for given suit distribution d
            begin_suit = 0
            end_suit = 1
            variants = 1
            while end_suit <= d.shape[0]:
                fact_prod = factorial(d[begin_suit], exact=True).prod()
                current_variants = factorial(non_zero_suits[begin_suit], exact=True) // fact_prod

                if not trump or end_suit != 1:
                    while end_suit < d.shape[0] and non_zero_suits[begin_suit] == non_zero_suits[end_suit] and \
                            np.array_equal(d[begin_suit], d[end_suit]):
                        end_suit += 1
                variants *= comb(current_variants + end_suit - begin_suit - 1, end_suit - begin_suit, exact=True)
                begin_suit, end_suit = end_suit, end_suit + 1

        key = np.zeros((len(suit_sizes), len(hand_sizes)), dtype=np.int32)
        key[suit_array > 0, :] = d
        result[tuple(key.ravel())] = int(variants)
    return result


def generate_suit_size_distributions(hand_sizes, n_suits=4, max_suit_size=8, trump_idx=None):
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    for suit_sizes in bounded_compositions(sum(hand_sizes), n_suits, n_suits * [max_suit_size, ]):
        if trump_idx is None and not is_sorted(suit_sizes):
            continue
        if trump_idx is not None and (suit_sizes[trump_idx] == 0 or
                                      not is_sorted(np.delete(suit_sizes, trump_idx))):
            continue
        yield suit_sizes


def get_card_distributions(hand_size, n_hands=3, max_suit_size=8, trump=False, reduce_perms=True):
    result = {}
    for suit_sizes in generate_suit_size_distributions(n_hands * [hand_size, ], trump_idx=0 if trump else None):
        result[tuple(suit_sizes)] = enumerate_deals(
            suit_sizes, n_hands * [hand_size, ], trump=trump, reduce_perms=reduce_perms
        )
    return result
