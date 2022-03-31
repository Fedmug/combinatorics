import numpy as np

NUMBER_OF_SUITS=4
SPADES_SYMBOL, CLUBS_SYMBOL, DIAMONDS_SYMBOL, HEARTS_SYMBOL = "\u2660", "\u2663", "\u2666", "\u2665"
SUIT_SYMBOLS = ("\u2660", "\u2663", "\u2666", "\u2665")
RANK_CHAR_TO_INDEX = {'7': 0, '8': 1, '9': 2, '10': 3, 'T': 3, 'J': 4, 'Q': 5, 'K': 6, 'A': 7}
SUIT_SYMBOL_TO_INDEX = {SPADES_SYMBOL: 0, CLUBS_SYMBOL: "\u2663", DIAMONDS_SYMBOL: "\u2666", HEARTS_SYMBOL: "\u2665"}


def parse_hand_string(hand_string):
    """
    :param hand_string: string with hand cards, i.e. ♠KQJ9♣J♦Q♥AJ107
    :return: np.array of shape (n_suits=4) with suits encoded by uint8
    """
    code = np.zeros(NUMBER_OF_SUITS, dtype=np.uint8)
    suit_index = 0
    for ch in hand_string:
        if ch in SUIT_SYMBOL_TO_INDEX:
            suit_index = SUIT_SYMBOL_TO_INDEX[ch]
        else:
            if ch == '1':
                continue
            if ch == '0':
                ch = '10'
            code[suit_index] += 1 << RANK_CHAR_TO_INDEX[ch]
    return code


class PrefDeal:
    def __init__(self, card_matrix: np.array, trump: bool):
        """
        :param card_matrix: np.array of shape (n_suits, n_hands), encodes i-th suit of j-th hand
        :param trump: true, if 0th suit is trump
        """
        self.card_matrix = np.copy(card_matrix)
        self.trump = trump

    @classmethod
    def from_strings(cls, hand_strings: list):
        """
        :param hand_strings: list of strings with hand cards, i.e.
        [♠KQJ9♣J♦Q♥AJ107, ♠8♣AKQ10♦KJ7♥K8, ♠A107♣987♦A10♥Q9]
        :return: np.array with dtype=np.uint8, hand codes
        """
        result = np.zeros((NUMBER_OF_SUITS, len(hand_strings)), dtype=np.uint8)
        for i, hand_string in enumerate(hand_strings):
            suit_codes = parse_hand_string(hand_string)
            result[:, i] = suit_codes
        return cls(result, False)
