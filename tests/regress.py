"""Does the engine still write exactly the charts it wrote before?

    .venv/bin/python tests/regress.py              # check every case
    .venv/bin/python tests/regress.py 2016 7K      # only the cases whose name has all these words
    .venv/bin/python tests/regress.py --reference  # only the user-approved 2016 chart
    .venv/bin/python tests/regress.py --record     # accept today's output as the new baseline

The first check is the one that matters most: the style the app began with must still
write `tests/reference/d4vd - 2016 (CASNoob) [Tech].osu` for the 2016 song, the chart
that was played and approved, note for note. That style is no longer one the app
offers (its name, Coordination, now stands for a kind of LN chart); it is kept here
as "original", and its charts stay in the baseline because nothing about breaks or
long notes touches them: if one of them changes, the engine underneath has changed.
The rest compare against charts recorded in tests/golden/.

The songs under tests/songs come from tests/fetch_songs.py. Their separated stems are
kept in tests/cache, apart from the app's own cache, so a baseline stays valid for as
long as that folder does."""

import glob
import gzip
import json
import multiprocessing
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "cache")
GOLDEN = os.path.join(HERE, "golden")
OUT = os.path.join(HERE, "out")
REFERENCE = os.path.join(HERE, "reference", "d4vd - 2016 (CASNoob) [Tech].osu")
MOVING = "Tempo steps"
# (beats, BPM at their start, BPM at their end): steps on bar lines, one slide, a long stretch
MOVING_PARTS = ((48, 120, 120), (16, 125, 125), (16, 130, 130), (32, 140, 140), (32, 140, 160),
                (48, 160, 160), (16, 170, 170), (96, 180, 180))
# the 2016 song: the user's own copy in songs/, or the app's in the library
SONG_2016 = next((p for where in ("songs", os.path.join("library", "songs"))
                  for p in sorted(glob.glob(os.path.join(ROOT, where, "**", "d4vd - 2016 *.opus"), recursive=True))),
                 os.path.join(ROOT, "songs", "d4vd - 2016 (Official Lyric Video) [IqjeqpkQEQ8].opus"))

os.environ["MAPMAKER_CACHE"] = CACHE
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache", "models"))
os.environ.setdefault("TORCH_HOME", os.environ["HF_HOME"])
sys.path.insert(0, ROOT)

from mapmaker import chart as _chart  # noqa: E402

# the app's first style, exactly as it was (it was labelled Coordination, and the notes
# recorded with its charts say so)
ORIGINAL = _chart.Style("Coordination",
                        "Held vocals become LNs with taps under them; drums become chords.",
                        ln=0.5, chord=3.6, runs=(0.55, 0.45), breath=1.3, cap=0, filler=2,
                        comfort=0, bass=False, motifs=_chart._MIXED, tuned=(4, 5, 6, 7))
_chart.STYLES["original"] = ORIGINAL


def songs(made_up=False):
    found = {"2016": SONG_2016}
    for path in sorted(glob.glob(os.path.join(HERE, "songs", "*.mp3"))):
        found[os.path.splitext(os.path.basename(path))[0]] = path
    if made_up:
        found[MOVING] = os.path.join(CACHE, MOVING + ".wav")
    return found


def make_moving():
    """A made-up song whose tempo changes, where the truth is known: drums, bass and a
    lead, each put straight into the cache as its own stem so that nothing has to be
    separated. -> the time of every beat."""
    import numpy as np
    import soundfile as sf
    import synth
    from mapmaker import pipeline
    beats = synth.beat_times(MOVING_PARTS)
    path = os.path.join(CACHE, MOVING + ".wav")
    if os.path.isfile(path) and os.path.isfile(
            os.path.join(CACHE, f"stems-{pipeline.song_key(path)}.npz")):
        return beats
    seconds = beats[-1] / 1000.0 + 2
    drums = synth.render(synth.groove(0, 10**6, beats=beats, seed=1)[0], seconds)
    bass, lead = synth.band(beats, seconds)
    quiet = (1e-4 * np.random.default_rng(5).standard_normal(len(drums))).astype(np.float32)
    drums, bass, lead = 0.5 * drums, 0.5 * bass, 0.5 * lead        # room for all three in the file
    os.makedirs(CACHE, exist_ok=True)
    sf.write(path, drums + bass + lead + quiet, synth.SR, subtype="PCM_16")
    np.savez(os.path.join(CACHE, f"stems-{pipeline.song_key(path)}.npz"),
             drums=drums, bass=bass, other=lead, vocals=quiet)
    return beats


def cases():
    """(name, song, options) for every chart in the baseline."""
    out = []

    def add(song, keys, style, stars, **more):
        extra = "".join(f" {k}={v}" for k, v in more.items())
        out.append((f"{song} {keys}K {style} {stars:g}{extra}", song,
                    dict(keys=keys, style=style, stars=stars, **more)))

    for stars in (1.2, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.5):
        add("2016", 4, "original", stars)
    for stars in (3.5, 4.5, 5.5):
        add("2016", 4, "stream", stars)
    add("2016", 4, "ln", 5.5)
    for stars in (3.0, 4.5):
        add("2016", 4, "chordjack", stars)
    add("2016", 4, "original", 4.5, ln=0.0)
    add("2016", 4, "original", 4.5, ln=1.0)
    add("2016", 4, "original", 4.5, seed=7)
    add("2016", 4, "original", 4.5, density=0.8, comfort=1)
    add("2016", 4, "original", 3.0, tempo="half")
    add("2016", 4, "original", 4.5, tempo="double")
    for keys in (5, 6, 7):
        add("2016", keys, "original", 4.5)
    add("2016", 7, "stream", 4.0)
    add("2016", 6, "ln", 4.0)
    add("2016", 7, "chordjack", 4.0)
    for style in ("tech", "speed", "speedjack", "ln"):
        for stars in (3.5, 4.5):
            add("2016", 4, style, stars)
    for song in songs():
        if song == "2016":
            continue
        for style in ("tech", "speed", "speedjack", "ln"):
            add(song, 4, style, 4.5)
        for stars in (3.0, 4.5, 6.0):
            add(song, 4, "original", stars)
        add(song, 4, "stream", 4.5)
        add(song, 4, "chordjack", 4.0)
        add(song, 4, "ln", 4.0)
        add(song, 7, "original", 4.0)
    for style, stars in (("original", 3.5), ("stream", 4.0), ("tech", 4.0), ("ln", 3.5)):
        add(MOVING, 4, style, stars)
    add(MOVING, 4, "stream", 4.0, bpm=180.0, offset=500.0)      # held to one tempo by hand
    # Coordination as the app has it now: holds that switch columns over single notes
    for stars in (3.0, 4.5, 5.0):
        add("2016", 4, "coordination", stars)
    add("2016", 4, "coordination", 4.5, ln=0.0)       # no long notes at all
    add("2016", 4, "coordination", 4.5, ln=0.25)      # half of its holds
    add("2016", 4, "coordination", 4.0, ln=1.0)       # every note held as well
    add("2016", 7, "coordination", 4.0)
    for song in list(songs()) + [MOVING]:
        if song != "2016":
            add(song, 4, "coordination", 4.5)
    return out


def slug(name):
    return re.sub(r"[^A-Za-z0-9.=]+", "_", name)


def run(case):
    """Chart one case -> everything worth comparing."""
    import mapmaker
    name, song, options = case
    out = os.path.join(OUT, slug(name))
    started = time.time()
    try:
        res = mapmaker.make_map(songs(True)[song], mapmaker.Options(**options), out)
        with open(res["osu"], encoding="utf-8") as fh:
            text = fh.read()
        stats = res["stats"]
        return {"name": name, "notes": [list(n) for n in res["chart"]["notes"]],
                "timing": section(text, "TimingPoints"), "warnings": res["warnings"],
                "stats": {k: stats[k] for k in ("stars", "bpm", "offset", "grid", "density",
                                                "comfort", "objects", "holds")},
                "text": text, "seconds": round(time.time() - started, 1)}
    except Exception as err:            # a case that fails is a result too
        return {"name": name, "error": f"{type(err).__name__}: {err}",
                "seconds": round(time.time() - started, 1)}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def section(text, name):
    body = text.split(f"[{name}]", 1)[1].split("\n[", 1)[0]
    return [line.strip() for line in body.strip().splitlines() if line.strip()]


def check_reference(result):
    """The chart the user played: timing point and every hit object must be unchanged."""
    with open(REFERENCE, encoding="utf-8") as fh:
        want = fh.read()
    if "error" in result:
        return [result["error"]]
    problems = []
    for part in ("TimingPoints", "HitObjects"):
        a, b = section(want, part), section(result["text"], part)
        if a != b:
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            problems.append(f"[{part}] differs: {len(a)} lines approved, {len(b)} now; first "
                            f"difference at line {first}: "
                            f"{a[first] if first < len(a) else '(end)'}  ->  "
                            f"{b[first] if first < len(b) else '(end)'}")
    # the approved file predates the app's naming of difficulties, so its name and tags
    # lines are allowed to differ; nothing else is
    def rest(text):
        return [l for l in text.splitlines() if not l.startswith(("Version:", "Tags:"))]
    if not problems and rest(want) != rest(result["text"]):
        problems.append("notes and timing match, but other lines of the .osu changed")
    return problems


def compare(result, golden):
    if "error" in result or "error" in golden:
        return [] if result.get("error") == golden.get("error") else [
            f"was {golden.get('error', 'a chart')}, now {result.get('error', 'a chart')}"]
    problems = []
    a, b = golden["notes"], result["notes"]
    if a != b:
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        same = len(set(map(tuple, a)) & set(map(tuple, b)))
        problems.append(f"notes differ: {len(a)} before, {len(b)} now, {same} in common; first "
                        f"difference at #{first}: {a[first] if first < len(a) else '(end)'} -> "
                        f"{b[first] if first < len(b) else '(end)'}")
    if golden["timing"] != result["timing"]:
        problems.append(f"timing differs: {golden['timing'][:1]} -> {result['timing'][:1]}")
    for key, was in golden["stats"].items():
        if result["stats"].get(key) != was:
            problems.append(f"{key}: {was} -> {result['stats'].get(key)}")
    if golden["warnings"] != result["warnings"]:
        problems.append(f"warnings: {golden['warnings']} -> {result['warnings']}")
    return problems


def golden_path(name):
    return os.path.join(GOLDEN, slug(name) + ".json.gz")


def prepare_cache():
    """The 2016 song is charted from the very stems the approved chart was made from."""
    os.makedirs(CACHE, exist_ok=True)
    for name in ("stems-af84addce66d.npz", "audio-af84addce66d.ogg",
                 "audio-af84addce66d.ogg.json"):
        src, dst = os.path.join(ROOT, ".cache", name), os.path.join(CACHE, name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copyfile(src, dst)
    make_moving()


def main(argv):
    record = "--record" in argv
    words = [a for a in argv if not a.startswith("--")]
    todo = [c for c in cases() if all(w.lower() in c[0].lower() for w in words)]
    if "--reference" in argv:
        todo = [c for c in todo if c[0] == "2016 4K original 4.5"]
    if not todo:
        print("no case matches")
        return 2
    prepare_cache()
    started = time.time()

    # the first chart of each song runs alone: it separates and listens, and the rest
    # of that song's charts only read what it cached
    first, rest, seen = [], [], set()
    for case in todo:
        (rest if case[1] in seen else first).append(case)
        seen.add(case[1])
    results = [run(c) for c in first]
    if rest:
        with multiprocessing.Pool(min(len(rest), max(1, (os.cpu_count() or 4) - 2))) as pool:
            results += pool.map(run, rest, chunksize=1)
    results.sort(key=lambda r: [c[0] for c in todo].index(r["name"]))

    failed = 0
    os.makedirs(GOLDEN, exist_ok=True)
    for res in results:
        if res["name"] == "2016 4K original 4.5":
            problems = check_reference(res)
            print(("ok       " if not problems else "FAILED   ")
                  + "the approved 2016 chart" + "".join(f"\n           {p}" for p in problems))
            failed += bool(problems)
        path = golden_path(res["name"])
        keep = {k: v for k, v in res.items() if k not in ("text", "seconds")}
        if record:
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(keep, fh)
            shown = (res.get("error") or
                     "{stars:.2f} stars, {objects} objects, {grid}, density {density}, "
                     "comfort {comfort}".format(**res["stats"]))
            print(f"recorded {res['name']:<44} {res['seconds']:>5.1f}s  {shown}")
            continue
        if not os.path.isfile(path):
            print(f"new      {res['name']}  (no baseline; run with --record)")
            failed += 1
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            problems = compare(res, json.load(fh))
        failed += bool(problems)
        print(("ok       " if not problems else "CHANGED  ") + f"{res['name']:<44} "
              f"{res['seconds']:>5.1f}s" + "".join(f"\n           {p}" for p in problems))
    shutil.rmtree(OUT, ignore_errors=True)
    print(f"\n{len(results)} charts in {time.time() - started:.0f}s: "
          + ("baseline recorded" if record else
             "all unchanged" if not failed else f"{failed} CHANGED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
