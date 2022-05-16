import numpy as np


def multinomial(ns: list[int]):
    """
    Calculates multinomial coefficient (n; ns), n = ns[1] + ns[2] + ... + ns[k]
    :param ns: list of ints; preferred to be sorted in the non-decreasing order
    :return: int, multinomial coefficient
    """
    if len(ns) == 1:
        return 1
    res, i = 1, sum(ns)
    for a in ns[:-1]:
        for j in range(1, a+1):
            res *= int(i)
            res //= int(j)
            i -= 1
    return res


def multiperm2index(multiperm, ns):
    """
        Calculates index of a multiset permutation,
        see Knuth, Art of Programming, 7.2.1.2, ex. 4
        :param ns: list, sizes of elements of the multiset, ns = (n_1, ..,, n_k)
        :param multiperm: list, permutation of the multiset {n_1*0, ..., n_k*(k-1)}
        :return: int, index, 0 <= index < (n; ns) (multinomial coefficient)
    """
    result = 0
    for element in multiperm:
        multi_coef = multinomial(ns)
        n = sum(ns)
        for j in range(element):
            result += multi_coef * ns[j] // n
        ns[element] -= 1
    return result


def index2multiperm(index, ns):
    """
        Calculates multiset permutation by its index,
        see Knuth, Art of Programming, 7.2.1.2, ex. 4
        :param ns: np.array, sizes of elements of the multiset, ns = (n_1, ..,, n_k)
        :param index: int, index, 0 <= index < (n; ns) (multinomial coefficient)
        :return: np.array, permutation of the multiset {n_1*0, ..., n_k*(k-1)}
    """
    if ns.sum() == 1:
        return np.array(np.argmax(ns))
    multi_coef = multinomial(ns)
    assert index < multi_coef, "Index should be less than multinomial coefficient!"
    cum_sum = 0
    for j in range(len(ns)):
        N_j = multi_coef * ns[j] // ns.sum()
        if index < cum_sum + N_j:
            sub_ns = np.copy(ns)
            sub_ns[j] -= 1
            if sub_ns[j] == 0:
                np.delete(sub_ns, j)
            sub_result = index2multiperm(index - cum_sum, sub_ns)
            return np.append(j, sub_result)
        cum_sum += N_j