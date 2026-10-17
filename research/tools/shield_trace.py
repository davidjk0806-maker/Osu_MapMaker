"""Where a Coordination chart's shields come from: counted after each step of
chart.build_chart (the patterner, _independence, _more_holds).

    .venv/bin/python research/tools/shield_trace.py "Mega Hyper Ultrastorm" 4.5
"""
import collections
import os
import random
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
sys.path.insert(0, os.path.join(os.path.dirname(RESEARCH), "tests"))
import engine  # noqa: E402
from mapmaker import chart  # noqa: E402


def shields(notes, span):
    by_col = collections.defaultdict(list)
    for k, c, e in notes:
        by_col[c].append((k, e))
    n = 0
    for col in by_col.values():
        col.sort()
        n += sum(1 for a, b in zip(col, col[1:]) if a[1] is None and b[1] is not None and b[0] - a[0] <= span)
    return n, sum(1 for n in notes if n[2] is not None)


def main(argv):
    name, stars = argv[0], float(argv[1])
    song, _ = engine.song_for(name)
    style = chart.rated(chart.STYLES["coordination"], stars)
    density, comfort, notes, got = chart._tune(song, stars, 2016, style)
    span = chart._shield_span(song, chart.spacing(song, style))
    print(f"{name}: density {density}, comfort {comfort}, {got:.2f} stars, shield span {span / song['line']:.0f} lines")
    steps = {}
    real_ind, real_more = chart._independence, chart._more_holds

    def ind(notes, *a, **k):
        steps["patterner"] = shields(notes, span)
        out = real_ind(notes, *a, **k)
        steps["after _independence"] = shields(out, span)
        return out

    def more(notes, *a, **k):
        out = real_more(notes, *a, **k)
        steps["after _more_holds"] = shields(out, span)
        return out
    chart._independence, chart._more_holds = ind, more
    chart.build_chart(song, density, random.Random(2016), style, None, comfort)
    for step, (n, holds) in steps.items():
        print(f"  {step:22s} {n} shields on {holds} holds")


if __name__ == "__main__":
    main(sys.argv[1:])
