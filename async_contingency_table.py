import asyncio
import asyncpg
from combinations import bounded_compositions, contingency_table, check_lex_order_for, deal_variants
import numpy as np


def generate_suit_sizes(predicate, max_suits=4, max_suit_size=8, deck_size=32):
    for n_cards in range(1, deck_size + 1):
        for n_suits in range(1, max_suits + 1):
            for suit_sizes in bounded_compositions(n_cards, n_suits, n_suits * [max_suit_size, ]):
                if predicate(suit_sizes):
                    yield suit_sizes.tolist()


def rows_are_sorted(suit_sizes, table):
    for i in range(1, table.shape[0] - 1):
        if suit_sizes[i] == suit_sizes[i + 1] and not check_lex_order_for(table[i], table[i + 1]):
            return False
    return True


def generate_contingency_tables(predicate, suit_sizes, n_hands=3):
    deck_size = sum(suit_sizes)
    hand_sizes = deck_size // n_hands * np.ones(n_hands, dtype=np.int8)
    hand_sizes[n_hands - deck_size%n_hands:] += 1
    for table in contingency_table(suit_sizes, hand_sizes):
        if len(table.shape) == 1:
            table = table[None, :]
        if predicate(suit_sizes, table):
            yield table


def get_suit_sizes_clause(suit_sizes):
    result = [f"deck_size = {sum(suit_sizes)}", ]
    for i in range(1, len(suit_sizes) + 1):
        result.append(f"sizes[{i}] = ${i}::int")
    return " AND ".join(result)


suit_sizes_by_stage_sorted = sorted(
    generate_suit_sizes(lambda s: s.min() > 0 and s.sum() % 3 == 0 and np.all(s[1:-1] <= s[2:])),
    key=lambda item: [len(item), ] + item
)


async def main():
    conn = await asyncpg.connect('postgres://rafael:VfPLiCASXsMd7Y@localhost/preferance')

    await conn.execute('''
            CREATE TABLE IF NOT EXISTS contingency_tables_async (
                    index               SERIAL PRIMARY KEY,
                    suit_sizes          int2 REFERENCES suit_sizes_v3,
                    matrix              uint1[][] NOT NULL,
                    variants            bigint NOT NULL,
                    reduced_nt          bigint,
                    reduced_trump       bigint
            )
        ''')

    for suit_sizes in suit_sizes_by_stage_sorted:
        n = len(suit_sizes) + 1
        query = f'''
                    INSERT INTO contingency_tables_async (suit_sizes, matrix, variants, reduced_nt, reduced_trump)
                    VALUES ((SELECT index FROM suit_sizes_v3 WHERE {get_suit_sizes_clause(suit_sizes)}),
                    ${n}::int[][], ${n + 1}::bigint, ${n + 2}::bigint, ${n+3}::bigint );
                '''
        stmt = await conn.prepare(query)
        values = []
        for table in generate_contingency_tables(rows_are_sorted, suit_sizes):
            values.append((*suit_sizes, table.tolist(), deal_variants(table),
                           deal_variants(table, reduce=True), deal_variants(table, True, True)))
        await stmt.executemany(values)

    await conn.close()

asyncio.run(main())
