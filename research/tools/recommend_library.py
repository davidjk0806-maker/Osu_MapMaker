"""What the Recommended style picks for the songs in the library, and from what.

    .venv/bin/python research/tools/recommend_library.py [STARS]
"""
import os
import pickle
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
sys.path.insert(0, os.path.dirname(RESEARCH))
sys.path.insert(0, TOOLS)
from balance import library  # noqa: E402
from mapmaker import pipeline  # noqa: E402
from mapmaker.audio import CACHE  # noqa: E402
from mapmaker.recommend import _measures, describe, explain  # noqa: E402


def main(argv):
    stars = float(argv[0]) if argv else 4.5
    for title, audio in library():
        key = pipeline.song_key(audio)
        if not os.path.isfile(os.path.join(CACHE, f"analysis-{key}.pkl")):
            print(f"{title[:44]:44s} not listened to yet (make a map of it first)")
            continue
        with open(os.path.join(CACHE, f"analysis-{key}.pkl"), "rb") as fh:
            heard = pickle.load(fh)
        raw, span, bpm, beat, T0, own, meter, _, clock = pipeline._in_time(
            heard["timing"], pipeline.Options(), heard["raw"], heard["dur_ms"])
        ranking = pipeline._ranked(raw, span, bpm, beat, T0, own, meter, clock, 4, stars)
        song = pipeline.on_grid(raw, span, T0, beat, own, 4, "compound" if own == 6 else meter, clock)
        d = describe(raw, song)
        m = _measures(d)
        print(f"{title[:44]:44s} {bpm:5.0f} BPM  " + "  ".join(
            f"{r['style']} {r['score']:.2f}/{r['fit']:.2f}" for r in ranking[:4]))
        print("       " + "  ".join(f"{k} {v[0]:.2f}" for k, v in m.items()))
        print("       " + explain(ranking))


if __name__ == "__main__":
    main(sys.argv[1:])
