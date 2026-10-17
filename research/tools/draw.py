"""Print a stretch of a chart as text, time running down, to see its patterns.

    python research/tools/draw.py chart.osu [first beat] [beats]

Each row shows the beat it falls in, the notes (O = tap, # = start of a long note,
| = held), the snap it sits on and the time since the row before."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import parse  # noqa: E402

SNAPS = (1, 2, 3, 4, 6, 8, 12, 16, 5, 7, 9, 24, 32)


def draw(chart, first=0, beats=8):
    t0, length = chart["red"][0]
    rows = {}
    for t, c, e in chart["notes"]:
        rows.setdefault(t, {})[c] = e
    held, prev = {}, None
    for t in sorted(rows):
        for c, e in rows[t].items():
            if e is not None:
                held[c] = e
        beat = (t - t0) / length
        if not first <= beat < first + beats:
            prev = t
            continue
        cells = "".join(("#" if rows[t][c] is not None else "O") if c in rows[t]
                        else "|" if held.get(c, -1) > t else "." for c in range(chart["keys"]))
        part = beat % 1.0
        snap = next((d for d in SNAPS if abs(part * d - round(part * d)) * length / d <= 2.5), 0)
        gap = "" if prev is None else f"+{t - prev:4d} ms"
        print(f"  {int(beat // 4) + 1:3d}.{int(beat % 4) + 1}  {cells}   "
              + (f"1/{snap:<3d}" if snap else "?    ") + f" {gap}")
        prev = t


if __name__ == "__main__":
    draw(parse(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 0.0,
         float(sys.argv[3]) if len(sys.argv) > 3 else 8.0)
