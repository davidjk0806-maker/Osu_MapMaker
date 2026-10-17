"""What the Recommended style picks for the example songs, against the style their
mappers charted them in.

    .venv/bin/python research/tools/recommend_check.py          # at 4.5 stars
    .venv/bin/python research/tools/recommend_check.py 5.5      # at another rating

Uses the listening cached in research/cache (research/tools/compare.py makes it). Each
example's style is the one research/examples/index.json files it under, except
Palette GAMMA, which the user made the model of Speed. A mapper's style is one good
answer for a song, not the only one, so the second choice is shown too, and the
measures the choice was made from."""

import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import compare  # noqa: E402  (points the cache at research/cache)
from mapmaker import pipeline  # noqa: E402
from mapmaker.recommend import describe, explain  # noqa: E402

MODEL_OF = {888774: "speed"}    # beatmap id -> the style the user modelled on it


def main(argv):
    stars = float(argv[0]) if argv else 4.5
    first = second = n = 0
    for category, bid, _, audio, pack in compare.unpack():
        heard = compare.listen_to(audio, cached_only=True)
        if heard is None:
            print(f"{pack}: not listened to yet (run research/tools/compare.py)")
            continue
        times = pipeline._in_time(heard["timing"], pipeline.Options(), heard["raw"], heard["dur_ms"])
        raw, span, bpm, beat, T0, own, meter, _, clock = times
        ranking = pipeline._ranked(raw, span, bpm, beat, T0, own, meter, clock, 4, stars)
        song = pipeline.on_grid(raw, span, T0, beat, own, 4, "compound" if own == 6 else meter, clock)
        d = describe(raw, song)
        want = MODEL_OF.get(bid, category)
        order = [r["style"] for r in ranking]
        n += 1
        first += order[0] == want
        second += want in order[:2]
        mark = "first " if order[0] == want else "second" if order[1] == want else "      "
        title = pack.split(" ", 1)[1].rsplit(".", 1)[0][:44]
        print(f"{mark} {want:12s} {title:44s} {bpm:5.0f} BPM  "
              + "  ".join(f"{r['style']} {r['score']:.2f}" for r in ranking[:3]))
        print(f"       runs {d['runs']:.2f}  fill {d['fill']:.2f}  fast {d['fast']:.2f}  snaps {d['snaps']}  "
              f"jacks {d['jacks_chordjack']:.2f}/{d['jacks_speedjack']:.2f}  held {d['held']:.2f}  "
              f"{d['loudness']:.1f} dBFS  steady {d['steady']:.2f}")
        print(f"       {explain(ranking)}")
    print(f"\nat {stars:g} stars the mapper's style came first for {first} of {n} songs "
          f"and first or second for {second}")


if __name__ == "__main__":
    main(sys.argv[1:])
