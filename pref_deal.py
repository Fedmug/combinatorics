import numpy as np
import re
from multiperms import multinomial, multiperm2index, index2multiperm
from pref_contingency_tables import PrefContingencyTable

NUMBER_OF_SUITS = 4
MAX_SUIT_SIZE = 8
SPADES_SYMBOL, CLUBS_SYMBOL, DIAMONDS_SYMBOL, HEARTS_SYMBOL = "\u2660", "\u2663", "\u2666", "\u2665"
SUIT_SYMBOLS = ("\u2660", "\u2663", "\u2666", "\u2665")
RANK_TO_INDEX = {'7': 0, '8': 1, '9': 2, '10': 3, 'T': 3, 'J': 4, 'Q': 5, 'K': 6, 'A': 7}
# RANK_BIG_TO_INDEX = {'7': 7, '8': 6, '9': 5, '10': 4, 'T': 4, 'J': 3, 'Q': 2, 'K': 1, 'A': 0}
INDEX_TO_RANK_CHAR = ('7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
SUIT_SYMBOL_TO_INDEX = {SPADES_SYMBOL: 0, CLUBS_SYMBOL: 1, DIAMONDS_SYMBOL: 2, HEARTS_SYMBOL: 3}


def parse_hand_string(hand_string, delimiter=None):
    """
    :param hand_string: string with hand cards, e.g. ♠KQJ9♣J♦Q♥AJ107 or A.KJT.87.QJ97
    :param delimiter: specifies the delimiter between suits; if None, than suit symbols are used
    :return: np.array of shape (n_suits=4) with suits encoded by uint8
    """
    code = np.zeros(NUMBER_OF_SUITS, dtype=np.uint8)
    suit_index = 0
    repl_suit_string = re.sub('10', 'T', hand_string)
    for ch in repl_suit_string:
        if delimiter is not None and ch == delimiter:
            suit_index += 1
            continue
        if delimiter is None and ch in SUIT_SYMBOL_TO_INDEX:
            suit_index = SUIT_SYMBOL_TO_INDEX[ch]
            continue
        code[suit_index] |= 1 << RANK_TO_INDEX[ch]
    return code


def suit_codes2multiperm(suit_codes):
    """
    :param suit_codes: np.array with dtype=np.uint8, suit_codes[i] encodes suit at ith hand
    :return: np.array of ints, permutation of the multiset {1, 2, 3, 4} (0 if card is missing)
    """
    n_hands = suit_codes.shape[0]
    result = np.dot(np.arange(1, n_hands + 1, dtype=np.int8),
                    np.unpackbits(suit_codes, bitorder='little').reshape((n_hands, -1)))
    return result


def multiperm2suit_codes(multiperm, n_hands):
    """
    :param multiperm: np.array, permutation of the multiset {0, 1, 2, 3}
    :return: codes, np.array of the shape (n_hands,)
    """
    result = np.zeros((n_hands, len(multiperm)), dtype=np.uint8)
    for i in range(n_hands):
        result[i][multiperm == i] = 1
    return np.packbits(result, axis=1, bitorder='little').squeeze()


def get_tightened_matrix(matrix, to_ace: bool, suit_indices=None):
    if suit_indices is None:
        suit_indices = range(matrix.shape[0])
    result = np.zeros_like(matrix)
    for i in suit_indices:
        suit_bits = np.unpackbits(matrix[i], bitorder='little').reshape((matrix.shape[1], -1))
        non_zero_bits = suit_bits.sum(axis=0, dtype=np.uint8) > 0
        result[i, :] = np.packbits(suit_bits[:, non_zero_bits], axis=1, bitorder='little').ravel()
        if to_ace:
            shift = np.sum(1 - non_zero_bits, dtype=np.int8)
            result[i, :] = np.left_shift(result[i, :], shift, dtype=np.uint8)
    return result


def extract_widow(code):
    if code < 3 or code == 7:
        result = np.zeros(NUMBER_OF_SUITS, dtype=np.uint8)
        result[code & 0b11] += 2
        return result
    return np.unpackbits(np.array([code, ], dtype=np.uint8), bitorder='big')[NUMBER_OF_SUITS:]


class PrefDealMatrix:
    def __init__(self, card_matrix: np.array, trump=0):
        """
        :param card_matrix: np.array of shape (n_suits, n_hands), encodes i-th suit of j-th hand
        :param trump: 0 if no trump; 1 if spades is trump, 2 if clubs is trump, 3 if diamonds is trump,
         4 if hearts is trump
        """
        self.card_matrix = np.copy(card_matrix)
        self.trump = trump
        self._hands_perm = np.arange(card_matrix.shape[1])

    @classmethod
    def from_strings(cls, hand_strings: list, delimiter=None, trump=0):
        """
        :param hand_strings: list of strings with hand cards, i.e.
        [♠KQJ9♣J♦Q♥AJ107, ♠8♣AKQ10♦KJ7♥K8, ♠A107♣987♦A10♥Q9]
        :param delimiter: specifies the delimiter between suits; if None, than suit symbols are used
        :param trump: specifies trump suit
        :return: np.array with dtype=np.uint8, hand codes
        """
        result = np.zeros((NUMBER_OF_SUITS, len(hand_strings)), dtype=np.uint8)
        for i, hand_string in enumerate(hand_strings):
            suit_codes = parse_hand_string(hand_string, delimiter)
            result[:, i] = suit_codes
        return cls(result, trump)

    @classmethod
    def from_hash(cls, index, to_ace=True):
        code = index & 0b11
        widow = code == 0
        trump = 0
        if code < 2:
            table = PrefContingencyTable.from_hash(index & 0xFFFFFF)
            if code == 0:
                deal_code = index >> 24
            else:
                deal_code = index >> 28
                trump = (index & 0xF000000) >> 24
        else:
            table = PrefContingencyTable.from_hash(index & 0xFFFFFFFF)
            deal_code = index >> 33
            trump = (index >> 32) % 2
        matrix = np.zeros_like(table.matrix, dtype=np.uint8)

        for i in range(table.matrix.shape[0]):
            multi_coef = multinomial(table.matrix[i])
            multiperm = index2multiperm(deal_code % multi_coef, table.matrix[i])
            deal_code //= multi_coef
            suit_code = multiperm2suit_codes(multiperm, 4 if widow else 3)
            matrix[i, :] = suit_code
        if to_ace:
            matrix = get_tightened_matrix(matrix, True)
        return cls(matrix, trump)

    def __hash__(self):
        table = self.contingency_table()
        result = 0
        trump_code = 0
        if table.sum() == 32:
            offset = 24
        elif table.sum() == 30:
            offset = 28
            trump_code = self.trump << 24
        else:
            offset = 33
            if self.trump:
                trump_code = 1 << 32
        table = table[table.sum(axis=1) > 0]
        table_hash = hash(PrefContingencyTable(table))
        prefix = 1
        for i in range(table.shape[0]):
            mp = suit_codes2multiperm(self.card_matrix[i])
            multi_coef = multinomial(table[i])
            index = multiperm2index(mp[mp > 0] - 1, table[i])
            result += prefix * int(index)
            prefix *= multi_coef
        assert table_hash < 1 << offset
        return (result << offset) + trump_code + table_hash

    def do_move(self, cards):
        """
        :param cards: np.array of shape (n_hands,), cards indices for making moves
        """
        suit_index = cards // MAX_SUIT_SIZE
        rank_index = cards % MAX_SUIT_SIZE
        assert np.min(self.card_matrix[suit_index, self._hands_perm] & (1 << rank_index)) > 0
        self.card_matrix[suit_index, self._hands_perm] -= 1 << rank_index

    def undo_move(self, cards):
        """
        :param cards: np.array of shape (n_hands,), cards indices for making moves
        """
        suit_index = cards // MAX_SUIT_SIZE
        rank_index = cards % MAX_SUIT_SIZE
        self.card_matrix[suit_index, self._hands_perm] += 1 << rank_index

    def __eq__(self, other):
        if self.trump != other.trump:
            return False
        return np.array_equal(
            get_tightened_matrix(self.card_matrix, False), get_tightened_matrix(other.card_matrix, False)
        )

    def __repr__(self):
        result = ""
        for j in range(self.card_matrix.shape[1]):
            for i in range(self.card_matrix.shape[0]):
                # result += SUIT_SYMBOLS[self._suit_perm[i]]
                result += SUIT_SYMBOLS[i]
                # if self.card_matrix[self._suit_perm[i], j] == 0:
                if self.card_matrix[i, j] == 0:
                    suit_string = '-'
                else:
                    suit_string = ""
                    # for index in np.nonzero(np.unpackbits(self.card_matrix[self._suit_perm[i], j], bitorder='little'))[0]:
                    for index in np.nonzero(np.unpackbits(self.card_matrix[i, j], bitorder='little'))[0]:
                        suit_string += INDEX_TO_RANK_CHAR[index]
                result += suit_string[::-1] + ' '
            result += '\n'
        return result.strip()

    def suit_sizes(self):
        return np.unpackbits(self.card_matrix, axis=1).sum(axis=1, dtype=np.uint8)

    def contingency_table(self):
        return np.unpackbits(self.card_matrix).reshape(self.card_matrix.shape + (-1,)).sum(axis=-1, dtype=np.uint8)
