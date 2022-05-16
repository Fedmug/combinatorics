import numpy as np
from combinations import check_lex_order_for
from multiperms import multinomial
from sympy.utilities.iterables import multiset_permutations

SUIT_SYMBOLS = ("\u2660", "\u2663", "\u2666", "\u2665")
# CARD_RANKS = {'2', '3', '4', '5', '6', '7', '8', '9', 'T', '10', 'J', 'Q', 'K', 'A'}
# SUIT_STRING_TO_INDEX = {'s': 0, 'spades': 0, "\u2660": 0,
#                         'c': 1, 'clubs': 1, "\u2663": 1,
#                         'd': 2, 'diamonds': 2, "\u2666": 2,
#                         'h': 3, 'hearts': 3, "\u2665": 3}
# NUMBER_OF_SUITS = 4


def permute_rows(matrix, index):
    suit_sizes = np.sum(matrix, axis=1)
    for i in range(index, suit_sizes.shape[0] - 1):
        if suit_sizes[i] < suit_sizes[i + 1]:
            return
        elif suit_sizes[i] > suit_sizes[i + 1] or not check_lex_order_for(matrix[i], matrix[i + 1]):
            matrix[[i, i + 1]] = matrix[[i + 1, i]]


def perm2suit(perm, n_hands=3):
    result = [[] for i in range(n_hands)]
    for i, hand_index in enumerate(perm):
        result[hand_index].append(i)
    return result


def count_reduced_moves(suit_sizes):
    result = np.zeros(len(suit_sizes))
    hand_list = []
    for i in range(len(suit_sizes)):
        hand_list.extend(suit_sizes[i] * [i,])
    total_perms = 0
    for perm in multiset_permutations(hand_list):
        total_perms += 1
        card_indices_by_hand = perm2suit(perm, len(suit_sizes))
        for i, card_indices in enumerate(card_indices_by_hand):
            for j, card in enumerate(card_indices):
                if j + 1 == len(card_indices) or card + 1 != card_indices[j + 1]:
                    result[i] += 1
    result /= total_perms
    return result


def widow_column2code(col: np.array):
    if np.max(col) == 2:
        code = np.argmax(col)
        if code == 3:
            code += 4
    else:
        code = 0
        for i in np.nonzero(col)[0]:
            code += 1 << i
    return int(code)


class PrefContingencyTable:
    def __init__(self, matrix: np.array):
        """
        :param matrix: contingency table with shape (n_suits, n_hands)
        """
        self.matrix = np.copy(matrix)
        if len(self.matrix.shape) == 1:
            self.matrix = self.matrix[None, :]

    def count_deals(self):
        """
        Calculates the number of deals with given contingency table
        :return: int
        """
        result = 1
        for suit in self.matrix:
            result *= multinomial(suit)
        return result

    def count_moves(self, trump: bool, reduce: bool, suit_moves=None):
        """
        :param trump: True, if 0th suit is trump
        :param reduce: if True, then reduce equivalent (dense) moves
        :param suit_moves: precalculated dict with average reduced moves over given suit sizes distribution
        :return: np.array with shape (n_hands,)
        """
        move_matrix = np.array(self.matrix, dtype=np.float32)
        if reduce:
            for i in range(move_matrix.shape[0]):
                if suit_moves is None:
                    move_matrix[i, :] = count_reduced_moves(self.matrix[i])
                else:
                    move_matrix[i, :] = suit_moves[tuple(self.matrix[i])]

        for j in range(1, move_matrix.shape[1]):
            for i in range(move_matrix.shape[0]):
                if move_matrix[i, j] == 0:
                    if i > 0 and trump and move_matrix[0, j] > 0:
                        move_matrix[i, j] = move_matrix[0, j]
                    else:
                        move_matrix[i, j] = move_matrix[:, j].sum()
        result = np.zeros(move_matrix.shape[1])
        move_cum_prod = move_matrix.cumprod(axis=1)
        result[0] = move_matrix[:, 0].sum()
        for i in range(1, move_matrix.shape[1]):
            result[i] = move_cum_prod[:, i].sum() / move_cum_prod[:, i - 1].sum()
        return result

    def __eq__(self, other):
        return np.array_equal(self.matrix, other.matrix)

    def __repr__(self):
        result = ""
        for i in range(self.matrix.shape[0]):
            result += SUIT_SYMBOLS[i] + ' '
            for item in self.matrix[i]:
                result += str(item)
            result += '\n'
        return result

    def __hash__(self):
        """
        :return: 32-bit int
        """
        result = 0
        suit_sizes = self.matrix.sum(axis=1)
        n_cards = suit_sizes.sum()
        if n_cards == 30:
            result = 1
        elif n_cards < 30:
            if suit_sizes.shape[0] == 4:
                result = 2
            else:
                result = 3

        offset = 2
        if n_cards > 30:
            widow_code = widow_column2code(self.matrix[:, -1])
            result += widow_code << 2
            offset = 6
        elif n_cards == 30:
            widow_code = widow_column2code(8 * np.ones_like(suit_sizes) - suit_sizes)
            result += widow_code << 2
            offset = 6
        elif suit_sizes.shape[0] < 4:
            result += suit_sizes.shape[0] << 2
            offset = 4

        if suit_sizes.sum() < 30:
            for suit_size in suit_sizes:
                result += int(suit_size) - 1 << offset
                offset += 3

        for i in range(self.matrix.shape[0] - 1):
            first_hand = self.matrix[i][0]
            second_hand = self.matrix[i][1]
            if first_hand == 8:
                first_hand = 7
                second_hand = 2
            elif second_hand == 8:
                first_hand = 2
                second_hand = 7

            result += int(first_hand) << offset
            offset += 3

            result += int(second_hand) << offset
            offset += 3
        return result

    @classmethod
    def from_hash(cls, index):
        code = index & 0b11
        n_hands = 3 if code else 4
        n_suits = 4
        offset = 2
        if code == 3:
            n_suits = (index >> 2) & 0b11
            offset = 4
        matrix = np.zeros((n_suits, n_hands), dtype=np.int8)
        suit_sizes = 8 * np.ones(n_suits, dtype=np.int8)

        if code < 2:
            widow_code = (index >> offset) & 0b1111
            widow_column = np.zeros(n_suits, dtype=np.int8)
            if widow_code < 3 or widow_code == 7:
                widow_column[widow_code & 0b11] = 2
            else:
                for i in range(4):
                    if widow_code & (1 << i):
                        widow_column[i] = 1
            if code == 0:
                matrix[:, -1] = widow_column
            else:
                suit_sizes -= widow_column
            offset += 4
        else:
            for i in range(n_suits):
                suit_sizes[i] = ((index >> offset) & 0b111) + 1
                offset += 3

        for i in range(matrix.shape[0] - 1):
            first_hand = (index >> offset) & 0b111
            offset += 3
            second_hand = (index >> offset) & 0b111
            offset += 3
            if first_hand == 7 and second_hand == 2:
                first_hand, second_hand = 8, 0
            elif first_hand == 2 and second_hand == 7:
                first_hand, second_hand = 0, 8
            matrix[i, 0] = first_hand
            matrix[i, 1] = second_hand
            if code:
                matrix[i, 2] = suit_sizes[i] - first_hand - second_hand
            else:
                matrix[i, 2] = suit_sizes[i] - first_hand - second_hand - matrix[i, 3]

        if code == 0:
            hands = np.array([10, 10, 10, 2], dtype=np.int8)
        else:
            hands = (np.sum(suit_sizes) // 3) * np.ones(3, dtype=np.int8)
        matrix[-1, :] = hands - np.sum(matrix[:-1, :], axis=0)
        return cls(matrix)
