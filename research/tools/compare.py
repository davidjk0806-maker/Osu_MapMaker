"""Compare what the listening hears in the example songs with what their mappers charted.

    .venv/bin/python research/tools/compare.py            # every example
    .venv/bin/python research/tools/compare.py igaku      # the ones whose file name has this word

Reads the .osz files in "research/Osu Songs For Reference", takes from each the
difficulty listed in research/examples/index.json and its audio, listens to the audio
exactly as the app would, and reports:

  - whether the tempo and the beat found match the mapper's red line
  - how many of the charted rows have a drum hit under them
  - how many of the charted rows off the 1/4 grid were heard as hits off that grid,
    and how many of the hits heard there the mapper charted too

Stems and analysis are cached in research/cache."""

import collections
import glob
import json
import os
import pickle
import re
import sys
import tempfile
import zipfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
ROOT = os.path.dirname(RESEARCH)
os.environ["MAPMAKER_CACHE"] = os.path.join(RESEARCH, "cache")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache", "models"))
os.environ.setdefault("TORCH_HOME", os.environ["HF_HOME"])
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS)

import numpy as np  # noqa: E402
from measure import parse  # noqa: E402
from mapmaker import pipeline  # noqa: E402
from mapmaker.audio import read_audio  # noqa: E402
from mapmaker.listen import RES, prepare  # noqa: E402

PACKS = os.path.join(RESEARCH, "Osu Songs For Reference")
OUT = os.path.join(RESEARCH, "reference")
NEAR = 12.0                     # ms within which a hit and a charted row are the same sound


def unpack():
    """Pull each example's chart and audio out of its .osz -> [(category, id, osu, audio)]."""
    with open(os.path.join(RESEARCH, "examples", "index.json"), encoding="utf-8") as fh:
        wanted = {e["id"]: e["category"] for e in json.load(fh)}
    found = []
    for pack in sorted(glob.glob(os.path.join(PACKS, "*.osz"))):
        with zipfile.ZipFile(pack) as z:
            for name in z.namelist():
                if not name.endswith(".osu"):
                    continue
                text = z.read(name).decode("utf-8", "replace")
                bid = re.search(r"BeatmapID:(\d+)", text)
                if not bid or int(bid.group(1)) not in wanted:
                    continue
                audio = re.search(r"AudioFilename:\s*(.+)", text).group(1).strip()
                folder = os.path.join(OUT, bid.group(1))
                os.makedirs(folder, exist_ok=True)
                osu = os.path.join(folder, "chart.osu")
                song = os.path.join(folder, "audio" + os.path.splitext(audio)[1].lower())
                if not os.path.isfile(osu):
                    with open(osu, "wb") as fh:
                        fh.write(z.read(name))
                if not os.path.isfile(song):
                    with open(song, "wb") as fh:
                        fh.write(z.read(audio))
                found.append((wanted[int(bid.group(1))], int(bid.group(1)), osu, song,
                              os.path.basename(pack)))
    return found


def listen_to(path, cached_only=False):
    """What the app hears in a song. Kept in research/cache; None if `cached_only` and
    the song has not been listened to yet."""
    saved = os.path.join(RESEARCH, "cache", f"analysis-{pipeline.song_key(path)}.pkl")
    if os.path.isfile(saved):
        with open(saved, "rb") as fh:
            done = pickle.load(fh)
        if done.get("version") == pipeline.ANALYSIS_VERSION:
            return done
    if cached_only:
        return None
    with tempfile.TemporaryDirectory(dir=OUT) as work:
        data, sr = read_audio(path, work)
        return pipeline.analyse(data, sr, pipeline.song_key(path), lambda stage, part: None)


def lined_up(osu, audio, cached_only=False):
    """An example's charted rows and what was heard in its audio, on one clock ->
    (chart, rows ms, heard, song on my grid, ms to take off my times), or None."""
    heard = listen_to(audio, cached_only)
    if heard is None:
        return None
    chart = parse(osu)
    rows = np.array(sorted({t for t, _, _ in chart["notes"]}), dtype=float)
    timing = heard["timing"]
    beat = 60000.0 / timing["bpm"]
    T0 = float(round(timing["T0"] % (4 * beat)))
    song = prepare(heard["raw"], heard["dur_ms"], T0, beat, timing["tpb"], 4,
                   timing["info"]["feel"])
    fine = np.array(sorted(h[0] for h in heard["raw"]["fine"]))
    # the decoder that made these samples and the game's may start the song a few ms
    # apart, and mappers place notes a little ahead of the sound; the offset that lines
    # most rows up with a hit is taken out
    diffs = [fine[i] - t for t in rows for i in (np.searchsorted(fine, t) - 1, np.searchsorted(fine, t))
             if 0 <= i < len(fine) and abs(fine[i] - t) <= 40.0]
    hist, edges = np.histogram(diffs, bins=np.arange(-40.0, 41.0, 2.0))
    peak = int(np.argmax(np.convolve(hist, np.ones(3), mode="same")))
    close = [d for d in diffs if abs(d - (edges[peak] + 1.0)) <= 6.0]
    return chart, rows, heard, song, float(np.median(close)) if close else 0.0


def agreement(cached_only=True):
    """Of the hits the listening puts between the lines of the examples, how many did
    their mappers chart? -> {snap: [charted, found]} over every example at hand."""
    out = collections.defaultdict(lambda: [0, 0])
    for _, _, osu, audio, _ in unpack():
        got = lined_up(osu, audio, cached_only)
        if got is None:
            continue
        _, rows, _, song, shift = got
        for p, (_, _, snap) in song["between"].items():
            t = song["T0"] + p / RES * song["beat"] - shift
            out[snap][0] += near(rows, t) <= NEAR
            out[snap][1] += 1
    return dict(out)


def near(times, t):
    """Distance from t to the closest of the sorted `times`."""
    i = np.searchsorted(times, t)
    return min(abs(times[j] - t) for j in (i - 1, i) if 0 <= j < len(times))


def compare(category, bid, osu, audio, pack):
    chart, rows, heard, song, shift = lined_up(osu, audio)
    red = chart["red"]

    def line_of(t):
        cur = red[0]
        for tp in red:
            if tp[0] <= t + 1:
                cur = tp
        return cur

    raw, timing = heard["raw"], heard["timing"]
    beat, T0 = song["beat"], song["T0"]
    fine_at = np.array(sorted(h[0] for h in raw["fine"])) - shift
    fine = fine_at

    # the mapper's rows: on a 1/4 line of their red line, or between two
    main = collections.Counter(round(60000.0 / line_of(t)[1], 2) for t in rows).most_common(1)[0][0]
    off_rows, snaps = [], collections.Counter()
    for t in rows:
        t0, length = line_of(t)
        part = ((t - t0) / length) % 1.0
        if abs(part * 4 - round(part * 4)) * length / 4 > 2.5:
            off_rows.append(t)
            snaps[next((d for d in (8, 6, 3, 12, 16) if abs(part * d - round(part * d)) * length / d <= 2.5), 0)] += 1
    off_rows = np.array(off_rows)

    # mine: the hits put between the lines of my own grid, as times in the song
    mine = np.array(sorted(T0 + p / RES * beat for p in song["between"])) - shift
    on_mine = np.array(sorted(T0 + p / RES * beat for p in song["drums"])) - shift
    ratio = timing["bpm"] / main
    same_tempo = any(abs(ratio / r - 1.0) < 0.003 for r in (0.5, 1.0, 2.0, 1.5, 0.75, 2.0 / 3, 4.0 / 3))
    # does my grid sit on theirs? compare where their 1/4 rows fall on my lines
    lines = [t for t in rows if t not in set(off_rows)]
    miss = [abs((t + shift - T0) / (beat / timing["tpb"]) - round((t + shift - T0) / (beat / timing["tpb"])))
            * beat / timing["tpb"] for t in lines]
    on_grid = sum(1 for m in miss if m <= 8.0) / max(1, len(miss))

    covered = sum(1 for t in rows if near(fine_at, t) <= NEAR) / len(rows)
    off_heard = (sum(1 for t in off_rows if near(fine_at, t) <= NEAR) / len(off_rows)
                 if len(off_rows) else 0.0)
    off_placed = (sum(1 for t in off_rows if len(mine) and near(mine, t) <= NEAR) / len(off_rows)
                  if len(off_rows) else 0.0)
    mine_charted = sum(1 for t in mine if near(rows, t) <= NEAR) / len(mine) if len(mine) else 0.0
    line_charted = (sum(1 for t in on_mine if near(rows, t) <= NEAR) / len(on_mine)
                    if len(on_mine) else 0.0)
    f = chart["fields"]
    print(f"\n=== {category}: {f.get('Artist')} - {f.get('Title')} [{f.get('Version')}]")
    print(f"  tempo     theirs {main:g} BPM ({len(red)} red line{'s' if len(red) > 1 else ''}), mine "
          f"{timing['bpm']:g} BPM {timing['info']['feel']} -> "
          + ("same tempo" if same_tempo else "DIFFERENT TEMPO")
          + f"; {100 * on_grid:.0f}% of their 1/4 rows sit on my lines; audio offset between us {shift:+.1f} ms")
    print(f"  drums     {len(raw['hits'])} hits heard ({len(fine)} when allowed closer together); "
          f"{100 * covered:.0f}% of their {len(rows)} rows have a hit under them; "
          f"{100 * line_charted:.0f}% of my hits on lines have a row of theirs")
    print(f"  between   their rows off the 1/4 grid: {len(off_rows)} ("
          + ", ".join(f"1/{d}: {n}" if d else f"other: {n}" for d, n in snaps.most_common())
          + f"); a hit was heard under {100 * off_heard:.0f}% of them")
    print(f"            hits I put between my lines: {len(mine)} ("
          + (", ".join(f"1/{d}: {n}" for d, n in sorted(song["heard"].items())) or "none")
          + f"); {100 * mine_charted:.0f}% of them are rows of theirs; they cover "
          f"{100 * off_placed:.0f}% of their off-grid rows")
    return {"id": bid, "category": category, "same_tempo": same_tempo, "on_grid": on_grid}


def main(argv):
    todo = [e for e in unpack() if all(w.lower() in e[4].lower() for w in argv)]
    done = [compare(*e) for e in todo]
    print(f"\n{sum(d['same_tempo'] for d in done)} of {len(done)} tempos match the mapper's")
    found = agreement()
    print("hits put between the lines that the mappers charted too: "
          + ", ".join(f"1/{d}: {a} of {n}" for d, (a, n) in sorted(found.items()))
          + f" - {sum(a for a, _ in found.values())} of {sum(n for _, n in found.values())} in all")


if __name__ == "__main__":
    main(sys.argv[1:])
