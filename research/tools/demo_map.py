"""Write a rough demo chart that mixes snaps, to feel what the engine can now place.

    .venv/bin/python research/tools/demo_map.py ["song file"] [stars]

It is a throwaway style, not one of the app's: 1/4 runs with invented 1/8 and 1/6
run-ins into the accents, single notes with jumps on the drum hits. The .osz lands in
research/demos/."""

import glob
import os
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import mapmaker  # noqa: E402
from mapmaker import chart  # noqa: E402

DEMO = replace(chart.STYLES["stream"], label="Engine demo", cap=-1, chord=1.6, apart=1,
               repeat=95.0, fills=(0.3, 0.6), ornaments=(("half", 3), ("triplet", 2)))


def main(argv):
    song = argv[0] if argv else next(
        (p for where in ("songs", os.path.join("library", "songs"))
         for p in sorted(glob.glob(os.path.join(ROOT, where, "**", "d4vd - 2016 *.opus"), recursive=True))), None)
    stars = float(argv[1]) if len(argv) > 1 else 4.5
    chart.STYLES["demo"] = DEMO
    try:
        done = mapmaker.make_map(song, mapmaker.Options(
            style="demo", stars=stars, version=f"Engine demo - mixed snaps {stars:g}"),
            os.path.join(ROOT, "research", "demos"),
            lambda stage, part: print(f"\r{stage} {part * 100:3.0f}%   ", end="", flush=True))
    finally:
        del chart.STYLES["demo"]
    s = done["stats"]
    print(f"\n{done['osz']}\n{s['stars']} stars, {s['objects']} objects, {s['grid']} lines plus "
          f"{' '.join(s['finer']) or 'nothing finer'}")
    for w in done["warnings"]:
        print("note:", w)


if __name__ == "__main__":
    main(sys.argv[1:])
