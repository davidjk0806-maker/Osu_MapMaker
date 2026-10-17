"""
python -m mapmaker                    open the drag-and-drop page in your browser
python -m mapmaker "song.mp3"         chart one song from the command line

Chart options:
    --stars 4.5         star rating to aim for
    --keys 4            4, 5, 6 or 7
    --style stream      one of the styles the page offers (see --help), or "auto"
                        for the one that suits the song best
    --ln 0.5            share of held sung notes charted as LNs (0 = none, 1 = all)
    --seed 7            other patterns on the same rhythm
    --bpm 174           BPM to search around; with --offset it is taken as exact
    --offset 1234       ms of the first beat of any bar (red line)
    --tempo half        halve or double the detected BPM
                        (a song whose tempo changes is followed through its changes
                        unless --bpm, --offset or --tempo holds it to one)
    --nudge -5          shift every note and the red line by this many ms
    --creator, --title, --artist, --name (difficulty name), --bg, --od, --hp, --out
"""

import argparse
import os
import sys

from . import STYLES, MapError, Options, make_map
from .recommend import AUTO


def main():
    ap = argparse.ArgumentParser(prog="python -m mapmaker", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--stars", type=float, default=4.5)
    ap.add_argument("--keys", type=int, default=4)
    ap.add_argument("--style", default="coordination", choices=sorted(STYLES) + [AUTO])
    ap.add_argument("--ln", type=float, default=None)
    ap.add_argument("--seed", type=int, default=2016)
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--offset", type=float, default=None)
    ap.add_argument("--tempo", default="auto", choices=["auto", "half", "double"])
    ap.add_argument("--nudge", type=float, default=0.0)
    ap.add_argument("--creator", default=Options.creator)
    ap.add_argument("--title", default="")
    ap.add_argument("--artist", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--bg", default=None)
    ap.add_argument("--od", type=float, default=8.0)
    ap.add_argument("--hp", type=float, default=8.0)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--comfort", type=int, default=None, choices=range(5))
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    if not args.audio:
        from .server import serve
        return serve(args.port, open_browser=not args.no_browser)
    if not os.path.isfile(args.audio):
        sys.exit(f"Can't find audio file: {args.audio}")

    opts = Options(stars=args.stars, keys=args.keys, style=args.style, ln=args.ln,
                   seed=args.seed, creator=args.creator, title=args.title,
                   artist=args.artist, version=args.name, bpm=args.bpm, offset=args.offset,
                   tempo=args.tempo, nudge=args.nudge, od=args.od, hp=args.hp, bg=args.bg,
                   density=args.density, comfort=args.comfort)
    seen = []

    def progress(stage, fraction):
        if stage not in seen:
            seen.append(stage)
            print(stage + " ...")

    try:
        done = make_map(args.audio, opts, args.out, progress)
    except MapError as err:
        sys.exit(str(err))
    s = done["stats"]
    print(f"\nDone: {done['osz']}")
    print(f"  {s['stars']:.2f} stars, {s['keys']}K {STYLES[s['style']].label}, "
          + (f"{s['tempo']['low']:g}-{s['tempo']['high']:g} BPM ({s['tempo']['sections']} tempo sections)"
             if s["tempo"] else f"{s['bpm']:g} BPM (red line at {s['offset']} ms)"))
    print(f"  {s['objects']} objects ({s['holds']} LNs), avg {s['nps']:.1f} NPS, "
          f"peak {s['nps_peak']} NPS; {s['real'] * 100:.0f}% of the notes sit on a real "
          "hit or held note")
    p = s["play"]
    sizes = "/".join(("single", "jump", "hand", "quad")[:len(p["chords"])])
    print(f"  most of it plays like {p['like']:.2f}x its rating, {p['near']}% near it (ranked 4K "
          f"Insane: about 0.84x and 30%); {sizes} {'/'.join(map(str, p['chords']))}%; rows "
          f"usually {p['snap']} ({p['spacing']} ms) apart; "
          + (f"jacks up to {p['longest_jack']} in a row, fastest {p['jack_ms']} ms" if p["jack_ms"]
             else "no jacks"))
    if s["picked"]:
        print("  recommended: " + s["picked"]["why"])
    for w in done["warnings"]:
        print("  note: " + w)
    print("  Double-click the .osz (or drag it onto osu!) to import.")


if __name__ == "__main__":
    main()
