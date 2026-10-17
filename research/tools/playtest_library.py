"""Maps of songs in the library (library/songs) for playtesting, one .osz per song with
a difficulty for each style and rating asked, into playtest/<folder>/. Uses the app's
own cache, so songs the app has listened to take seconds.

    .venv/bin/python research/tools/playtest_library.py FOLDER "song words" style:stars ... [-- "other song" ...]

e.g. research/tools/playtest_library.py 2026-10-16 2016 speedjack:4.5 speedjack:5.5 -- Alphabet stream:4.5"""

import os
import shutil
import sys
import zipfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
ROOT = os.path.dirname(RESEARCH)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS)

import mapmaker  # noqa: E402
from balance import library  # noqa: E402


def song_set(folder, words, picks):
    found = library([words])
    if not found:
        print(f"no song in the library matches {words!r}")
        return None
    title, audio = found[0]
    out = os.path.join(ROOT, "playtest", folder)
    work = os.path.join(out, ".work")
    shutil.rmtree(work, ignore_errors=True)
    made = []
    for style, stars in picks:
        label = mapmaker.STYLES[style].label
        done = mapmaker.make_map(audio, mapmaker.Options(style=style, stars=stars,
                                                         version=f"4K {label} {stars:g}"), work)
        made.append(done)
        s, p = done["stats"], done["stats"]["play"]
        print(f"  {label} {stars:g}: {s['stars']:.2f} stars, {s['objects']} objects, {s['holds']} LNs, "
              f"{s['grid']}; plays like {p['like']:.2f}x, {p['near']}% near; chords {p['chords']}; "
              f"rows {p['snap']} {p['spacing']} ms; longest jack {p['longest_jack']}")
        for w in done["warnings"]:
            print(f"      note: {w}")
    meta = made[0]["meta"]
    path = os.path.join(out, mapmaker.osufile.safe_name(f"{meta['artist']} - {meta['title']}")[:80]
                        + " - mapmaker.osz")
    with zipfile.ZipFile(made[0]["osz"]) as first, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("audio.ogg", first.read("audio.ogg"))
        for done in made:
            z.write(done["osu"], os.path.basename(done["osu"]))
    shutil.rmtree(work, ignore_errors=True)
    print(f"{title}: {path}")
    return path


def main(argv):
    folder, rest = argv[0], argv[1:]
    groups, cur = [], []
    for a in rest + ["--"]:
        if a == "--":
            if cur:
                groups.append(cur)
            cur = []
        else:
            cur.append(a)
    for words, *picks in groups:
        song_set(folder, words, [(p.split(":")[0], float(p.split(":")[1]) if ":" in p else 4.5)
                                 for p in picks])


if __name__ == "__main__":
    main(sys.argv[1:])
