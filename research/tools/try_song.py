"""Chart any song file in one of the app's styles and describe the result in numbers.

    .venv/bin/python research/tools/try_song.py "path/to/song.opus" speedjack 5.9 [first beat] [beats]

Uses the app's own cache, so a song the app has already listened to is quick. With the
last two numbers it also draws that stretch of the chart as text."""

import os
import shutil
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
ROOT = os.path.dirname(RESEARCH)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS)

import mapmaker  # noqa: E402
from draw import draw  # noqa: E402
from measure import measure, parse, show  # noqa: E402


def main(argv):
    path, style, stars = argv[0], argv[1], float(argv[2])
    out = os.path.join(RESEARCH, "out")
    try:
        done = mapmaker.make_map(path, mapmaker.Options(style=style, stars=stars), out)
        chart = parse(done["osu"])
    finally:
        shutil.rmtree(out, ignore_errors=True)
    s = done["stats"]
    show(f"{done['meta']['title']} - {style} at {stars:g} (density {s['density']}, comfort "
         f"{s['comfort']}, {s['bpm']} BPM, real {100 * s['real']:.0f}%)", measure(chart))
    for w in done["warnings"]:
        print("  note:", w)
    if len(argv) > 3:
        draw(chart, float(argv[3]), float(argv[4]) if len(argv) > 4 else 8.0)


if __name__ == "__main__":
    main(sys.argv[1:])
