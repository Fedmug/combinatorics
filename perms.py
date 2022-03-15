import numpy as np
from scipy.special import factorial


def generate_perms(a):
    """
        Generates all permutations of non-negative array a[1] <= a[2] <= ... <= a[n],
        see Knuth, Art of Programming, 7.2.1.2, Algorithm L
        :param a: non-decreasing np.array
        :return: sequence of np.array like a
    """
    n = len(a)
    if n == 0:
        return
    if n == 1:
        yield a
        return

    arr = np.zeros(n + 1, dtype=np.int8)
    arr[0] = -1
    arr[1:] = a
    while True:
        yield arr[1:]
        j = n - 1
        while arr[j] >= arr[j + 1]:
            j -= 1
        if j == 0:
            break
        m = n
        while arr[j] >= arr[m]:
            m -= 1
        arr[j], arr[m] = arr[m], arr[j]
        k, m = j + 1, n
        while k < m:
            arr[k], arr[m] = arr[m], arr[k]
            k += 1
            m -= 1


def generate_perms_fast(a):
    """
        Generates all permutations of non-negative array a[1] <= a[2] <= ... <= a[n],
        see Knuth, Art of Programming, 7.2.1.2, ex. 1, Algorithm L'
        :param a: non-decreasing np.array
        :return: sequence of np.array like a
    """
    n = len(a)
    if n == 0:
        return
    if n == 1:
        yield a
        return
    if n == 2:
        generate_perms(a)
        return

    arr = np.zeros(n + 1, dtype=np.int8)
    arr[0] = -1
    arr[1:] = a
    while True:
        yield arr[1:]
        y, z = arr[n - 1], arr[n]
        if y < z:
            arr[n - 1], arr[n] = z, y
            continue
        x = arr[n - 2]
        if x < y:
            if x < z:
                arr[-3:] = [z, x, y]
            else:
                arr[-3:] = [y, z, x]
            continue
        j = n - 3
        y = arr[j]
        while y >= x:
            j -= 1
            x, y = y, arr[j]
        if j == 0:
            break

        if y >= z:
            m = n - 1
            while y >= arr[m]:
                m -= 1
            arr[j], arr[m] = arr[m], y
            arr[n], arr[j + 1] = arr[j + 1], z
        else:
            arr[j], arr[j + 1], arr[n] = z, y, x
        k, m = j + 2, n - 1
        while k < m:
            arr[k], arr[m] = arr[m], arr[k]
            k += 1
            m -= 1


def inversion_table2perm(b):
    """
        Calculates permutation (a[1], a[2], ...,a[n]) from its inversion table (b[1], b[2], ..., b[n])
        see Knuth, Art of Programming, 7.2.1.2, ex. 4
        :param b: np.array, inversion table
        :return: np.array, permutation
    """
    x = np.empty(1 + len(b), dtype=np.int32)
    a = np.empty_like(b)
    x[0] = 0
    for k in range(len(b), 0, -1):
        j = 0
        for i in range(b[k - 1]):
            j = x[j]
        x[k], x[j] = x[j], k
    j = 0
    for k in range(len(b)):
        a[k], j = x[j] - 1, x[j]
    return a


def invert_permutation(a):
    """
    Inverts permutation a
    :param a: np.array, permutation of elements (0, 1, ..., n-1)
    :return: np.array, inverse permutation
    """
    inv = np.empty_like(a)
    inv[a] = np.arange(len(inv), dtype=inv.dtype)
    return inv


def inversion_table(a, invert=True):
    """
        Calculates the inversion table b[1], ..., b[n] of the permutation a
        see Knuth, Art of Programming, 5.1.1, ex. 6
        :param invert: bool, invert permutation a before calculation
        :param a: np.array, permutation of 0, 1, ..., n-1
        :return: np.array, inversion table b[1], ..., b[n]
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=np.int32)
    assert (np.bincount(a) != 0).sum() == len(a), "Elements should be different!"
    lower = np.min(a)
    assert np.max(a) == lower + len(a) - 1, "No gaps allowed!"
    if invert:
        a = invert_permutation(a)
    b = np.zeros_like(a, dtype=np.int32)
    for k in range(int(np.floor(np.log2(len(a)))), -1, -1):
        power_2 = 2 ** k
        x = np.zeros(shape=(len(a) // (2 * power_2) + 1,), dtype=np.int32)
        for j in range(len(a)):
            s = a[j] // (2 * power_2)
            if (a[j] // power_2) % 2:
                x[s] += 1
            else:
                b[a[j] - lower] += x[s]
    return b


def factoradic2decimal(digits):
    return np.dot(digits, factorial(np.arange(len(digits)), exact=True))


def decimal2factoradic(n: int):
    result = [0, ]
    digit = 2
    while n:
        result.append(n % digit)
        n //= digit
        digit += 1
    return np.array(result)


def perm2index(perm, invert=True):
    """
    Calculates index of the permutation, using inversion table
    :param perm: np.array, permutation of (0, 1, ..., n-1)
    :param invert: bool, invert permutation before calculating inversion table
    :return: int, index, 0 <= index < n!, n = len(perm)
    """
    it = inversion_table(perm, invert=invert)[::-1]
    return factoradic2decimal(it)


def index2perm(index, size, invert=True):
    """
    Retrieves permutation by its index
    :param size: int, size of permutation
    :param index: int, index of permutation, 0 <= index < size!
    :param invert: bool, invert permutation
    :return: np.array
    """
    assert index < factorial(size, exact=True), "Index must less than n!"
    factoradic = decimal2factoradic(index)
    if len(factoradic) < size:
        factoradic = np.append(factoradic, np.zeros(size - len(factoradic), dtype=np.int32))
    perm = inversion_table2perm(factoradic[::-1])
    if invert:
        perm = invert_permutation(perm)
    return perm


def get_all_perms(size, optimize: bool):
    generator = generate_perms_fast if optimize else generate_perms
    result = np.zeros(shape=(factorial(size, exact=True), size), dtype=np.int8)
    for i, perm in enumerate(generator(np.arange(size))):
        result[i, :] = perm
    print("Mean column values:", result.mean(axis=0))
    print("Column std:", result.std(axis=0))
    return result
