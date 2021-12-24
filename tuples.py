import numpy as np


def generate_tuples(size, upper_bound, lower_bound=0):
    """
        Generates all tuples of the form (a[0], a[1], ..., a[size-1])
        with restriction lower_bound[i] <= a[i] < upper_bound[i] (bounds may be constant)
        see Knuth, Art of Programming, 7.2.1.1, Algorithm M
        :param size: int, size of tuple
        :param upper_bound: int or list, upper bound for tuple items; if int than the bound is the same for all values
        :param lower_bound: int or list, lower bound for tuple items; if int than the bound is the same for all values
        :return: sequence of np.array with shape (size,)
    """
    if isinstance(upper_bound, int):
        upper_bound = np.ones(size, dtype=np.int32) * upper_bound
    if isinstance(lower_bound, int):
        lower_bound = np.ones(size, dtype=np.int32) * lower_bound
    a = np.array(lower_bound, dtype=np.int32)
    j = size
    while j >= 0:
        yield a
        j = size - 1
        while j >= 0 and a[j] == upper_bound[j] - 1:
            a[j] = lower_bound[j]
            j -= 1
        if j >= 0:
            a[j] += 1


def generate_binary_gray_tuples(size):
    """
        Generates all binary tuples of the form (a[size-1], a[size-2], ..., a[1], a[0])
        starting from (0,...,0) and swapping one pair of bits at a time
        see Knuth, Art of Programming, 7.2.1.1, Algorithm G
        :param size: int, size of tuple
        :return: sequence of np.array with shape (size,)
    """
    a = np.zeros(size, dtype=np.int8)
    a_inf = False
    j = 0
    while j < size:
        yield a
        a_inf = not a_inf
        if a_inf:
            j = 0
        else:
            try:
                j = np.argwhere(a[::-1])[0] + 1
            except Exception as e:
                print(e)
                print(a)
                break
        if j < size:
            a[size - 1 - j] = 1 - a[size - 1 - j]
