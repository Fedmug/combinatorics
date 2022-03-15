import numpy as np

SUIT_SYMBOLS = ("\u2660", "\u2663", "\u2666", "\u2665")
# CARD_RANKS = {'2', '3', '4', '5', '6', '7', '8', '9', 'T', '10', 'J', 'Q', 'K', 'A'}
# SUIT_STRING_TO_INDEX = {'s': 0, 'spades': 0, "\u2660": 0,
#                         'c': 1, 'clubs': 1, "\u2663": 1,
#                         'd': 2, 'diamonds': 2, "\u2666": 2,
#                         'h': 3, 'hearts': 3, "\u2665": 3}
# NUMBER_OF_SUITS = 4


class PrefContingencyTable:
    def __init__(self, matrix: np.array, trump: bool):
        """
        param matrix: contingency table with shape (n_suits, n_hands=3)
        :param trump: True, if 0th suit is trump
        """
        self.matrix = np.copy(matrix)
        if len(self.matrix.shape) == 1:
            self.matrix = self.matrix[None, :]
        assert self.matrix.shape[1] == 3, "n_hands must be equal to 3"
        self.trump = trump

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
        result = 1 if self.trump else 0
        suit_sizes = self.matrix.sum(axis=1)
        offset = 2
        if suit_sizes.shape[0] < 4:
            result += suit_sizes.shape[0] << 2
            result += 2
            offset = 4
        if suit_sizes.sum() < 30:
            for suit_size in suit_sizes:
                result += int(suit_size) - 1 << offset
                offset += 3
        else:
            result += 2
            if suit_sizes[2] < 8:
                result += 3 << 4
            else:
                result += 8 - int(suit_sizes[1]) << 4
            offset = 6
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
        trump = index & 1 == 1
        flag30 = False
        if index & 2 == 0:
            n_suits = 4
            offset = 2
        else:
            n_suits = (index >> 2) & 0b11
            if n_suits == 0:
                n_suits = 4
                flag30 = True
            offset = 4
        matrix = np.zeros((n_suits, 3), dtype=np.int8)
        suit_sizes = 8 * np.ones(n_suits, dtype=np.int8)
        if flag30:
            suit_code = (index >> offset) & 0b11
            offset += 2
            if suit_code == 0:
                suit_sizes[0] -= 2
            elif suit_code == 1:
                suit_sizes[:2] -= 1
            elif suit_code == 2:
                suit_sizes[1] -= 2
            else:
                suit_sizes[1:3] -= 1
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
            matrix[i, 2] = suit_sizes[i] - first_hand - second_hand

        hands = (np.sum(suit_sizes) // 3) * np.ones(3, dtype=np.int8)
        matrix[-1, :] = hands - np.sum(matrix[:-1, :], axis=0)
        return cls(matrix, trump)
