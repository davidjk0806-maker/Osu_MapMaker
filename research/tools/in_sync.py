"""Is a chart in time with its song? Charts a song and counts, thirty seconds at a time,
how many of the drum hits heard in the audio have a row of the chart on them (within 10 ms).

    .venv/bin/python research/tools/in_sync.py "song file" [style] [--one-tempo]

--one-tempo charts the song at the single tempo found for it, as the app did before it
could follow tempo changes, for comparison. Uses research/cache."""

import os
import pickle
import shutil
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
ROOT = os.path.dirname(RESEARCH)
os.environ["MAPMAKER_CACHE"] = os.path.join(RESEARCH, "cache")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache", "models"))
os.environ.setdefault("TORCH_HOME", os.environ["HF_HOME"])
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS)

import numpy as np  # noqa: E402
import mapmaker  # noqa: E402
from measure import parse  # noqa: E402


def main(argv):
    one = "--one-tempo" in argv
    argv = [a for a in argv if not a.startswith("--")]
    path, style = argv[0], argv[1] if len(argv) > 1 else "coordination"
    out = os.path.join(RESEARCH, "out")
    options = mapmaker.Options(style=style, stars=4.5)
    try:
        done = mapmaker.make_map(path, options, out)
        if one:                         # the same again, held to the one tempo first found
            with open(os.path.join(RESEARCH, "cache", f"analysis-{done['song']}.pkl"), "rb") as fh:
                timing = pickle.load(fh)["timing"]
            options.bpm, options.offset = timing["bpm"], timing["T0"]
            done = mapmaker.make_map(path, options, out)
        chart = parse(done["osu"])
    finally:
        shutil.rmtree(out, ignore_errors=True)
    with open(os.path.join(RESEARCH, "cache", f"analysis-{done['song']}.pkl"), "rb") as fh:
        heard = pickle.load(fh)
    rows = np.array(sorted({t for t, _, _ in chart["notes"]}), dtype=float)
    hits = np.array([h[0] for h in heard["raw"]["hits"]])
    # the packed audio may sit a few ms from the analysed one: take that out
    near = rows[np.clip(np.searchsorted(rows, hits), 1, len(rows) - 1)]
    before = rows[np.clip(np.searchsorted(rows, hits) - 1, 0, len(rows) - 1)]
    off = np.where(np.abs(near - hits) < np.abs(before - hits), near - hits, before - hits)
    shift = float(np.median(off[np.abs(off) <= 15])) if (np.abs(off) <= 15).any() else 0.0
    ok = np.abs(off - shift) <= 10.0
    s = done["stats"]
    print(f"{done['meta']['title'][:40]} - {style}: {s['stars']} stars, {s['objects']} notes, {len(chart['red'])} red line(s)")
    row, apart = [], []
    for a in range(0, int(heard["dur_ms"] / 1000), 30):
        m = (hits >= a * 1000) & (hits < (a + 30) * 1000)
        row.append(f"{a // 60}:{a % 60:02d} " + (f"{100 * ok[m].mean():3.0f}%" if m.sum() >= 8 else "  - "))
        m &= ok
        apart.append(f"{a // 60}:{a % 60:02d} " + (f"{np.median(np.abs(off[m] - shift)):3.1f}" if m.sum() >= 8 else "  - "))
    print("  drum hits with a note on them: " + "  ".join(row))
    print("  ms from such a hit to its note: " + "  ".join(apart))
    print(f"  over the whole song: {100 * ok.mean():.0f}% of {len(hits)} hits")
    for w in done["warnings"]:
        print("  note:", w)


if __name__ == "__main__":
    main(sys.argv[1:])
