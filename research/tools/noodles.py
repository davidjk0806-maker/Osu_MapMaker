"""How a chart uses its long notes: how long they are, how many are down at once, and what
follows what in one column (a shield is a column pressed again a quarter beat after it
was let go; see pattern_glossary-osumania.webp).

    .venv/bin/python research/tools/noodles.py chart.osu [more.osu ...]

The two charts the user named for the Coordination style are
research/examples/coordination-*.osu."""


import collections
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
sys.path.insert(0, os.path.dirname(RESEARCH))
sys.path.insert(0, TOOLS)

import numpy as np  # noqa: E402
from measure import parse  # noqa: E402


def beat_of(chart):
    red = chart["red"]
    def at(t):
        cur = red[0]
        for r in red:
            if r[0] <= t + 1: cur = r
        return cur[1]
    return at

def look(path, label=None):
    chart = parse(path)
    beat = beat_of(chart)
    notes = sorted(chart["notes"])
    lns = [n for n in notes if n[2] is not None]; taps = [n for n in notes if n[2] is None]
    f = chart["fields"]
    print(f"\n=== {label or f.get('Title', '')[:40]} [{f.get('Version', '')[:30]}]: {len(notes)} notes, {len(lns)} LNs ({100 * len(lns) / len(notes):.0f}%), {len(taps)} taps")
    q = lambda x: round(x * 8) / 8
    L = collections.Counter(min(4.0, q((e - t) / beat(t))) for t, c, e in lns)
    print("  LN lengths in beats:", ", ".join(f"{k:g}: {100 * v / len(lns):.0f}%" for k, v in sorted(L.items()) if v / len(lns) >= 0.02))
    # how many are held when a note starts
    held_at = collections.Counter()
    for t, c, e in notes:
        h = sum(1 for a, cc, b in lns if a < t < b and cc != c)
        held_at[(e is not None, min(h, 3))] += 1
    print("  others held when an LN starts: " + ", ".join(f"{k}: {100 * held_at[(True, k)] / max(1, len(lns)):.0f}%" for k in range(4)) +
          " | when a tap is hit: " + ", ".join(f"{k}: {100 * held_at[(False, k)] / max(1, len(taps)):.0f}%" for k in range(4)))
    # what follows what in one column, and how soon after the release / the hit
    by = collections.defaultdict(list)
    for n in notes: by[n[1]].append(n)
    pairs = collections.defaultdict(list)
    for col in by.values():
        for a, b in zip(col, col[1:]):
            gap = (b[0] - (a[2] if a[2] is not None else a[0])) / beat(b[0])
            pairs[("LN" if a[2] is not None else "tap") + " then " + ("LN" if b[2] is not None else "tap")].append(gap)
    for kind, gaps in sorted(pairs.items()):
        g = np.array(gaps)
        print(f"  {kind:13s} in the same column: {len(g):4d} | gap (from letting go / the hit) within 1/8 beat {100 * (g <= 0.135).mean():3.0f}%, 1/4 {100 * (g <= 0.26).mean():3.0f}%, 1/2 {100 * (g <= 0.51).mean():3.0f}%, 1 beat {100 * (g <= 1.01).mean():3.0f}% | median {np.median(g):.2f} beats")
    # releases: does something start where a hold ends?
    starts = np.array(sorted({n[0] for n in notes}))
    rel = [np.abs(starts - e).min() <= 6 for t, c, e in lns]
    print(f"  holds that end exactly where another note starts: {100 * np.mean(rel):.0f}%")
    # notes that start while a hold is down, per hold
    per = [sum(1 for n in notes if t < n[0] < e and n[1] != c) for t, c, e in lns]
    print(f"  notes that start in other columns while a hold is down: median {np.median(per):.0f}, 9 in 10 holds at most {np.percentile(per, 90):.0f}")
    same = sum(1 for a, b in zip(lns, lns[1:]) if a[1] == b[1]) / max(1, len(lns) - 1)
    hand = sum(1 for a, b in zip(lns, lns[1:]) if (a[1] < 2) == (b[1] < 2)) / max(1, len(lns) - 1)
    print(f"  the next hold starts in the same column {100 * same:.0f}%, on the same hand {100 * hand:.0f}%")
    rows = sorted({n[0] for n in notes}); gaps = np.diff(rows) / np.array([beat(t) for t in rows[1:]])
    G = collections.Counter(q(min(2.0, g)) for g in gaps)
    print("  rows apart, in beats: " + ", ".join(f"{k:g}: {100 * v / len(gaps):.0f}%" for k, v in sorted(G.items()) if v / len(gaps) >= 0.03))

if __name__ == "__main__":
    for p in sys.argv[1:]:
        look(p)
