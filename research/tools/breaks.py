"""The parts of a song without drums: what the app charts there, beside what the mappers
of the example songs charted in the same bars.

    .venv/bin/python research/tools/breaks.py [style] [word from a song's name ...]

For every example: how many rows a second the mapper has in the drumless bars and how
many the app has (at three densities), how many of the app's rows are rows of the
mapper's ("right"), how many of the mapper's the app has ("found"), and where in the
beat the app's rows sit. Needs research/tools/compare.py to have unpacked the examples."""

import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
sys.path.insert(0, os.path.dirname(RESEARCH))
sys.path.insert(0, TOOLS)

import numpy as np  # noqa: E402
import compare  # noqa: E402
from mapmaker.chart import STYLES, plan  # noqa: E402


def main(argv):
    style = STYLES[argv[0]] if argv and argv[0] in STYLES else STYLES["stream"]
    words = [a for a in argv if a not in STYLES]
    total = {}
    for _, _, osu, audio, pack in compare.unpack():
        if words and not any(w.lower() in pack.lower() for w in words):
            continue
        _, rows, _, song, shift = compare.lined_up(osu, audio)
        T0, tick, line, bar = song["T0"], song["line_ms"], song["line"], song["bar"]
        calm = [b for b in song["bars"] if song["section"][b] == "calm"]
        seconds = len(calm) * song["bar_len"] / 1000.0
        inside = lambda t: song["section"].get(int((t - T0) // song["bar_len"])) == "calm"  # noqa: E731
        theirs = np.array([t for t in rows + shift if inside(t)])
        if len(theirs) < 8:
            continue
        print(f"{pack[:44]:44s} {seconds:4.0f} s without drums; the mapper: {len(theirs) / seconds:.1f} rows a second")
        for density in (0.0, 0.5, 1.0):
            taps = plan(song, density, style)
            mine = np.array(sorted(T0 + k / line * tick for k in taps
                                   if song["section"][k // (bar * line)] == "calm"))
            if not len(mine):
                print(f"    density {density}: nothing")
                continue
            right = sum(1 for t in mine if np.abs(theirs - t).min() <= 10.0)
            found = sum(1 for t in theirs if np.abs(mine - t).min() <= 10.0)
            where = [0, 0, 0]
            for k in taps:
                if song["section"][k // (bar * line)] == "calm":
                    part = (k % (song["tpb"] * line)) * 4 // (song["tpb"] * line) if not k % line else -1
                    where[0 if part == 0 else 1 if part == 2 else 2] += 1
            two = sum(1 for k, n in taps.items() if n > 1 and song["section"][k // (bar * line)] == "calm")
            t = total.setdefault(density, [0, 0, 0, 0])
            t[0] += right; t[1] += len(mine); t[2] += found; t[3] += len(theirs)
            print(f"    density {density}: {len(mine) / seconds:4.1f} rows a second, {100 * right / len(mine):3.0f}% right, "
                  f"{100 * found / len(theirs):3.0f}% found | on the beat {where[0]}, half beat {where[1]}, "
                  f"elsewhere {where[2]} | chords {two}")
    for density, (right, mine, found, theirs) in total.items():
        print(f"all songs, density {density}: {100 * right / max(1, mine):.0f}% right, {100 * found / max(1, theirs):.0f}% found")


if __name__ == "__main__":
    main(sys.argv[1:])
