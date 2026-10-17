"""Count how the user's three stream examples move from row to row, and print the table
the Stream style draws its columns from (_STEPS in mapmaker/chart.py).

    .venv/bin/python research/tools/steps.py

The examples are the ones the user listed as streams (research/examples/index.json):
Rise Of The Chaos Wizards [Dark Galactic Skies], ETERNAL DRAIN [Eternal] and Palette
GAMMA [Affection], unpacked by compare.py. Only their streams count (shapes.py), only
stretches of three rows in which no row shares a column with the one before, and each
stretch also counts mirrored. Needs research/tools/compare.py to have unpacked them."""

import collections
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)

from measure import parse  # noqa: E402
from shapes import streams  # noqa: E402

EXAMPLES = (4054401, 1001780, 888774)


def name(cols):
    return "".join(str(c + 1) for c in cols)


def count():
    """(row two before, row before) -> Counter of the next row, rows as column tuples."""
    table = collections.defaultdict(collections.Counter)
    for bid in EXAMPLES:
        chart = parse(os.path.join(RESEARCH, "reference", str(bid), "chart.osu"))
        keys = chart["keys"]
        for run in streams(chart):
            rows = [cols for _, cols, _ in run]
            for a, b, c in zip(rows, rows[1:], rows[2:]):
                if set(a) & set(b) or set(b) & set(c):
                    continue
                for flip in (lambda r: r, lambda r: tuple(sorted(keys - 1 - x for x in r))):
                    table[(flip(a), flip(b))][flip(c)] += 1
    return table


def main():
    table = count()
    print("_STEPS = {")
    for (a, b), nxt in sorted(table.items(), key=lambda kv: (len(kv[0][0]), kv[0][0], len(kv[0][1]), kv[0][1])):
        kept = {k: v for k, v in nxt.items() if v >= 2}
        if kept:
            body = ", ".join(f'"{name(k)}": {v}' for k, v in sorted(kept.items(), key=lambda kv: -kv[1]))
            print(f'    ("{name(a)}", "{name(b)}"): {{{body}}},')
    print("}")


if __name__ == "__main__":
    main()
