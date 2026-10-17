"""Chart an example song in one of the app's styles and set it beside the mapper's chart.

    .venv/bin/python research/tools/against.py stream chaos        # my Stream on Rise Of The Chaos Wizards
    .venv/bin/python research/tools/against.py tech igaku 4.5      # at 4.5 stars instead of the example's own
    .venv/bin/python research/tools/against.py stream 2016         # no example to compare: just my chart

The song is picked by a word from its .osz name ("2016" is the user's own song). Both
charts are described with research/tools/measure.py, the mapper's first. Needs
research/tools/compare.py to have unpacked the examples."""

import glob
import json
import os
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

import mapmaker  # noqa: E402
from measure import measure, parse, show  # noqa: E402

# the 2016 song: the user's own copy in songs/, or the app's in the library
SONG_2016 = next((p for where in ("songs", os.path.join("library", "songs"))
                  for p in sorted(glob.glob(os.path.join(ROOT, where, "**", "d4vd - 2016 *.opus"), recursive=True))),
                 os.path.join(ROOT, "songs", "d4vd - 2016 (Official Lyric Video) [IqjeqpkQEQ8].opus"))


def example(word):
    """(their chart or None, audio path, name) for the example whose pack name has `word`."""
    if word == "2016":
        return None, SONG_2016, "d4vd - 2016"
    with open(os.path.join(RESEARCH, "examples", "index.json"), encoding="utf-8") as fh:
        ids = [e["id"] for e in json.load(fh)]
    for bid in ids:
        folder = os.path.join(RESEARCH, "reference", str(bid))
        osu = os.path.join(folder, "chart.osu")
        if not os.path.isfile(osu):
            continue
        chart = parse(osu)
        name = f"{chart['fields'].get('Artist')} - {chart['fields'].get('Title')}"
        if word.lower() in name.lower():
            return chart, glob.glob(os.path.join(folder, "audio.*"))[0], name
    raise SystemExit(f"no example song has '{word}' in its name")


def mine(audio, style, stars, keep=None, **more):
    """Chart `audio` -> (parsed chart, result of make_map). The .osz is thrown away
    unless `keep` names a folder for it."""
    out = keep or os.path.join(RESEARCH, "out")
    try:
        done = mapmaker.make_map(audio, mapmaker.Options(style=style, stars=stars, **more), out)
        return parse(done["osu"]), done
    finally:
        if not keep:
            shutil.rmtree(out, ignore_errors=True)


def brief(m):
    """The numbers that tell the styles apart, on one line."""
    c = m["chord"]
    return (f"{m['stars']:.2f} stars | {m['nps_avg']:4.1f} notes/s, busy second {m['nps_p90']} | "
            f"rows {m['gap_ms_median']:.0f} ms apart (fastest tenth {m['row_gap_p10_ms']:.0f}) | chords {c[1]:.0f}/{c[2]:.0f}/{c[3]:.0f}/{c.get(4, 0):.0f} | "
            f"jacks {m['jack_share']:.0f}% ({m['jack_ms_median']:.0f} ms, longest {m['jack_len_max']}) | "
            f"in runs of 32+ {m['rows_in_runs_32']:.0f}%, 96+ {m['rows_in_runs_96']:.0f}% | "
            f"finer than 1/4: {m['rows_in_bursts']:.0f}% | LN {m['ln_share']:.0f}% | "
            f"column repeat {m['column_repeat_p1']:.0f} ms")


def trial(style, word, stars=None, label="trial", **more):
    """Chart a song with a Style object that is not one of the app's -> (line, result)."""
    from mapmaker import chart as engine
    theirs, audio, _ = example(word)
    if stars is None:
        stars = round(measure(theirs)["stars"], 2) if theirs else 4.5
    engine.STYLES["trial"] = style
    try:
        chart, done = mine(audio, "trial", stars, **more)
    finally:
        del engine.STYLES["trial"]
    s = done["stats"]
    grid = s["grid"] + (" + " + " ".join(s["finer"]) if s["finer"] else "")
    return (f"{label:<22s} {brief(measure(chart))} | density {s['density']}, comfort {s['comfort']}, "
            f"{grid}, real {100 * s['real']:.0f}%"), done


def main(argv):
    style, word = argv[0], argv[1]
    theirs, audio, name = example(word)
    m_theirs = measure(theirs) if theirs else None
    stars = float(argv[2]) if len(argv) > 2 else round(m_theirs["stars"], 2) if theirs else 4.5
    if theirs:
        show(f"{name} [{theirs['fields'].get('Version')}] - the mapper's", m_theirs)
    chart, done = mine(audio, style, stars)
    s = done["stats"]
    show(f"{name} - mine, {style} at {stars:g} (density {s['density']}, comfort {s['comfort']}, "
         f"{s['grid']} lines{' + ' + ' '.join(s['finer']) if s['finer'] else ''})", measure(chart))
    for w in done["warnings"]:
        print("  note:", w)


if __name__ == "__main__":
    main(sys.argv[1:])
