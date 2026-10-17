"""Make the maps for trying the styles out, into playtest/<today>-examples/:

  - the 2016 song in every style, as one .osz with eight difficulties
  - each example song in the style it is an example of, beside nothing else, so it can
    be played against the mapper's own chart of the same song

    .venv/bin/python research/tools/playtest_maps.py [stars]

  - or one song of your own in the styles and ratings you name, as one .osz:

    .venv/bin/python research/tools/playtest_maps.py song "path to audio" speedjack:5.9 ln:5.9 stream

    (style, style:stars, or style:stars:long-notes setting from 0 to 1)

Needs research/tools/compare.py to have unpacked the example songs."""

import os
import shutil
import sys
import time
import zipfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
ROOT = os.path.dirname(RESEARCH)
os.environ["MAPMAKER_CACHE"] = os.path.join(RESEARCH, "cache")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache", "models"))
os.environ.setdefault("TORCH_HOME", os.environ["HF_HOME"])
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS)

import mapmaker  # noqa: E402
from against import SONG_2016, example  # noqa: E402

OUT = os.path.join(ROOT, "playtest", time.strftime("%Y-%m-%d") + "-examples")
EXAMPLES = [("tech", "igaku"), ("tech", "hole dwelling"), ("stream", "chaos wizards"),
            ("speed", "palette"), ("speedjack", "speedjack pack"), ("speedjack", "mi mi"),
            ("chordjack", "sendan"), ("chordjack", "glaza"), ("ln", "on and on"),
            ("ln", "sparkling")]


def one_set(name, audio, styles, stars, **meta):
    """Chart `audio` in each of `styles` and pack the lot as one .osz."""
    work = os.path.join(OUT, ".work")
    shutil.rmtree(work, ignore_errors=True)
    made = []
    for style in styles:
        done = mapmaker.make_map(audio, mapmaker.Options(style=style, stars=stars, **meta), work)
        s = done["stats"]
        made.append(done)
        print(f"  {mapmaker.STYLES[style].label:13s} {s['stars']:.2f} stars, {s['objects']} objects, "
              f"{s['holds']} LNs, {s['grid']} {' '.join(s['finer'])}")
        for w in done["warnings"]:
            print(f"      note: {w}")
    path = os.path.join(OUT, name + ".osz")
    with zipfile.ZipFile(made[0]["osz"]) as first, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("audio.ogg", first.read("audio.ogg"))
        for done in made:
            z.write(done["osu"], os.path.basename(done["osu"]))
    shutil.rmtree(work, ignore_errors=True)
    return path


def song_set(audio, picks):
    """Chart `audio` once for each (style, stars) of `picks` and pack the lot as one .osz."""
    work = os.path.join(OUT, ".work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    made = []
    for style, stars, ln in picks:
        label = mapmaker.STYLES[style].label
        name = f"4K {label} {stars:g}" + ("" if ln is None else f" (LN {100 * ln:.0f}%)")
        done = mapmaker.make_map(audio, mapmaker.Options(style=style, stars=stars, ln=ln,
                                                         version=name), work)
        s = done["stats"]
        made.append(done)
        print(f"  {name[3:]:26s} {s['stars']:.2f} stars, {s['objects']} objects, "
              f"{s['holds']} LNs, {s['grid']} {' '.join(s['finer'])}")
    for w in sorted({w for done in made for w in done["warnings"]}):
        print(f"      note: {w}")
    meta = made[0]["meta"]
    path = os.path.join(OUT, mapmaker.osufile.safe_name(f"{meta['artist']} - {meta['title']}")[:80]
                        + " - mapmaker.osz")
    with zipfile.ZipFile(made[0]["osz"]) as first, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("audio.ogg", first.read("audio.ogg"))
        for done in made:
            z.write(done["osu"], os.path.basename(done["osu"]))
    shutil.rmtree(work, ignore_errors=True)
    print(path)
    return path


def main(argv):
    if argv and argv[0] == "song":
        picks = [(p.split(":") + ["", ""])[:3] for p in argv[2:]]
        picks = [(style, float(stars or 4.5), float(ln) if ln else None) for style, stars, ln in picks]
        return song_set(argv[1], picks or [(style, 4.5, None) for style in mapmaker.STYLES])
    stars = float(argv[0]) if argv else 4.5
    os.makedirs(OUT, exist_ok=True)
    print(f"d4vd - 2016, every style at {stars:g} stars")
    one_set(f"d4vd - 2016 - all styles {stars:g}", SONG_2016, list(mapmaker.STYLES), stars)
    for style, word in EXAMPLES:
        theirs, audio, name = example(word)
        f = theirs["fields"]
        print(f"{name}, {style} at {stars:g} stars")
        one_set(mapmaker.osufile.safe_name(f"{name} - mapmaker {style} {stars:g}"), audio, [style],
                stars, title=f.get("Title", ""), artist=f.get("Artist", ""))


if __name__ == "__main__":
    main(sys.argv[1:])
