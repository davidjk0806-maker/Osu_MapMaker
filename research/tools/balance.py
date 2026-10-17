"""How a chart plays, in the numbers the user's playtest of 2026-10-16 was about:
jacks into long notes (shields), reverse shields and inverse in Coordination; how
steady a stream's spacing is and how big its chords are in Stream; how fast the jacks
come in Speedjack; and how much there is to play for the rating (notes a second, how
much of the chart plays near its rating).

    .venv/bin/python research/tools/balance.py refs                 # every 4K difficulty in the reference packs
    .venv/bin/python research/tools/balance.py mine STYLE STARS [SONG ...]   # the app's maps of the library songs

Songs are the library's (library/songs/<id>), named by any part of their title."""

import collections
import glob
import json
import os
import re
import shutil
import statistics
import sys
import zipfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
ROOT = os.path.dirname(RESEARCH)
sys.path.insert(0, ROOT)
sys.path.insert(0, TOOLS)

from criteria_check import check  # noqa: E402
from measure import parse  # noqa: E402

PACKS = os.path.join(RESEARCH, "Osu Songs For Reference")
SNAPS = (1, 2, 3, 4, 6, 8, 12, 16)


def _beat_at(red):
    def at(t):
        cur = red[0]
        for tp in red:
            if tp[0] <= t + 1:
                cur = tp
            else:
                break
        return cur[1]
    return at


def play(chart):
    """The numbers, from a chart as measure.parse gives it."""
    notes, keys, red = chart["notes"], chart["keys"], chart["red"]
    beat_at = _beat_at(red)
    crit = check(chart)
    rows = collections.defaultdict(list)
    for t, c, e in notes:
        rows[t].append((c, e))
    times = sorted(rows)
    out = {"stars": crit["stars"], "plays_like": crit["moment_median"] / max(crit["stars"], 0.01),
           "near": crit["at_rating"], "nps": crit["nps"], "objects": len(notes)}

    # chords
    sizes = collections.Counter(len(rows[t]) for t in times)
    out["chords"] = [100.0 * sizes.get(n, 0) / len(times) for n in range(1, keys + 1)]

    # spacing from one row to the next while the chart keeps going (a beat or less)
    gaps = collections.Counter()
    for a, b in zip(times, times[1:]):
        beat = beat_at(a)
        if b - a > beat + 3:
            continue
        d = next((d for d in SNAPS if abs((b - a) - beat / d) <= 3.0), None)
        gaps[d or "other"] += 1
    total = sum(gaps.values()) or 1
    out["gaps"] = {k: 100.0 * v / total for k, v in gaps.most_common()}

    # jacks: one column in two rows running, no more than 250 ms apart
    jack_ms, chain, chains = [], [0] * keys, []
    prev, prev_t = set(), None
    for t in times:
        cols = {c for c, _ in rows[t]}
        for c in range(keys):
            if c in cols and c in prev and t - prev_t <= 250:
                jack_ms.append(t - prev_t)
                chain[c] += 1
            else:
                if chain[c]:
                    chains.append(chain[c] + 1)
                chain[c] = 0
        prev, prev_t = cols, t
    chains += [n + 1 for n in chain if n]
    out["jacks"] = 100.0 * len(jack_ms) / len(notes)
    out["jack_ms"] = statistics.median(jack_ms) if jack_ms else 0.0
    out["jack_p10"] = sorted(jack_ms)[len(jack_ms) // 10] if jack_ms else 0.0
    out["jack_long"] = max(chains) if chains else 0
    out["jack_3plus"] = sum(1 for n in chains if n >= 3)
    # two rows running that share two columns or more (jumpjacks, chordjacks)
    out["jumpjacks"] = sum(1 for a, b in zip(times, times[1:])
                           if b - a <= 250 and len({c for c, _ in rows[a]} & {c for c, _ in rows[b]}) >= 2)
    # the same at a quarter beat or less: the fast ones
    fast, chain, chains, fast_jj = 0, [0] * keys, [], 0
    prev, prev_t = set(), None
    for t in times:
        cols = {c for c, _ in rows[t]}
        near = prev_t is not None and t - prev_t <= beat_at(prev_t) / 4 + 3
        fast_jj += near and len(cols & prev) >= 2
        for c in range(keys):
            if near and c in cols and c in prev:
                fast += 1
                chain[c] += 1
            else:
                if chain[c]:
                    chains.append(chain[c] + 1)
                chain[c] = 0
        prev, prev_t = cols, t
    chains += [n + 1 for n in chain if n]
    out["fast_jacks"] = 100.0 * fast / len(notes)
    out["fast_long"] = max(chains) if chains else 0
    out["fast_jj"] = fast_jj

    # long notes
    lns = [(t, c, e) for t, c, e in notes if e is not None]
    out["ln_share"] = 100.0 * len(lns) / len(notes)
    by_col = {c: sorted((t, e) for t, cc, e in notes if cc == c) for c in range(keys)}
    shields = reverse = jump_shields = tight = 0
    shielded_at = collections.Counter()
    for c, col in by_col.items():
        for (t0, e0), (t1, e1) in zip(col, col[1:]):
            q = beat_at(t0) / 4 + 3
            if e1 is not None and e0 is None and t1 - t0 <= q:
                shields += 1
                shielded_at[t1] += 1
            if e0 is not None and e1 is None and t1 - e0 <= q:
                reverse += 1
            if e0 is not None and t1 - e0 <= q:
                tight += 1
    jump_shields = sum(1 for n in shielded_at.values() if n >= 2)
    out["shields"] = 100.0 * shields / max(1, len(lns))
    out["reverse"] = 100.0 * reverse / max(1, len(lns))
    out["n_shields"], out["n_reverse"], out["jump_shields"] = shields, reverse, jump_shields
    out["release_tight"] = 100.0 * tight / max(1, len(lns))
    # how much of the time three columns or more are held at once (inverse)
    edges = sorted([(t, 1) for t, _, _ in lns] + [(e, -1) for _, _, e in lns])
    held, last, three = 0, times[0], 0
    for t, d in edges:
        if held >= 3:
            three += t - last
        held, last = held + d, t
    out["held3"] = 100.0 * three / max(1, times[-1] - times[0])
    out["bpm"] = 60000.0 / statistics.median(beat_at(t) for t in times)
    return out


def line(name, m):
    gaps = " ".join(f"{('1/' + str(k)) if k != 'other' else k}:{v:.0f}" for k, v in list(m["gaps"].items())[:4])
    chords = "/".join(f"{v:.0f}" for v in m["chords"])
    text = (f"{name[:46]:46s} {m['stars']:4.2f}* {m['bpm']:4.0f}bpm {m['nps']:5.1f}nps "
            f"like {m['plays_like']:.2f} near {m['near']:3.0f}% | chords {chords:13s} | "
            f"1/4 jacks {m['fast_jacks']:3.0f}% long {m['fast_long']} jj {m['fast_jj']:3d} | gaps {gaps:28s} | "
            f"jacks {m['jacks']:3.0f}% {m['jack_ms']:3.0f}ms p10 {m['jack_p10']:3.0f} long {m['jack_long']:2d} "
            f"3+ {m['jack_3plus']:3d} jj {m['jumpjacks']:3d}")
    if m["ln_share"] >= 1:
        text += (f" | LN {m['ln_share']:3.0f}% shield {m['shields']:3.0f}% ({m['jump_shields']} jump) "
                 f"rev {m['reverse']:3.0f}% tight {m['release_tight']:3.0f}% held3 {m['held3']:3.0f}%")
    return text


def refs():
    """Every 4K mania difficulty in the reference packs -> [(pack, version, chart)]."""
    out = []
    for pack in sorted(glob.glob(os.path.join(PACKS, "*.osz"))):
        with zipfile.ZipFile(pack) as z:
            for name in z.namelist():
                if not name.endswith(".osu"):
                    continue
                text = z.read(name).decode("utf-8", "replace")
                if not re.search(r"Mode:\s*3", text) or not re.search(r"CircleSize:\s*4\b", text):
                    continue
                path = os.path.join(RESEARCH, "out", "ref.osu")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                version = re.search(r"Version:(.*)", text).group(1).strip()
                out.append((os.path.basename(pack).split(" ", 1)[1][:-4], version, parse(path)))
    shutil.rmtree(os.path.join(RESEARCH, "out"), ignore_errors=True)
    return out


def library(words=()):
    """The library's songs -> [(title, audio path)], those whose title has one of `words`."""
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "library", "songs", "*", "song.json"))):
        with open(path, encoding="utf-8") as fh:
            s = json.load(fh)
        title = f"{s['artist']} - {s['title']}" if s["artist"] else s["title"]
        if not words or any(w.lower() in title.lower() for w in words):
            out.append((title, os.path.join(os.path.dirname(path), s["file"])))
    return out


def mine(audio, style, stars, **more):
    """The app's map of `audio` -> (chart as parse gives it, what make_map returned)."""
    import mapmaker
    out = os.path.join(RESEARCH, "out", f"{os.getpid()}")
    try:
        done = mapmaker.make_map(audio, mapmaker.Options(style=style, stars=stars, **more), out)
        return parse(done["osu"]), done
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _one(job):
    title, audio, style, stars, changes = job
    if changes:                         # try the style with some of its settings changed
        from dataclasses import replace
        from mapmaker import chart as engine
        engine.STYLES[style] = replace(engine.STYLES[style], **changes)
    chart, done = mine(audio, style, stars)
    return title, style, chart, done


def main(argv):
    """`--set name=value` (repeatable) changes a setting of the styles tried, and
    `--brief` prints only the first numbers of each map."""
    if argv[0] == "refs":
        for pack, version, chart in refs():
            print(line(f"{pack[:24]} [{version}]", play(chart)))
        return
    changes, brief, rest = {}, "--brief" in argv, []
    it = iter(argv)
    for a in it:
        if a == "--set":
            name, value = next(it).split("=", 1)
            changes[name] = json.loads(value.replace("(", "[").replace(")", "]"))
            if isinstance(changes[name], list):
                changes[name] = tuple(tuple(v) if isinstance(v, list) else v for v in changes[name])
        elif a != "--brief":
            rest.append(a)
    style, stars = rest[1], float(rest[2])
    import multiprocessing
    jobs = [(title, audio, s, stars, changes) for title, audio in library(rest[3:]) for s in style.split(",")]
    with multiprocessing.Pool(min(8, len(jobs))) as pool:
        for title, s, chart, done in pool.imap(_one, jobs):
            st = done["stats"]
            text = line(f"{title[:30]} {s}", play(chart)) + f" | grid {st['grid']} {st['finer']}"
            print(text[:175] if brief else text)
            for w in ([] if brief else done["warnings"]):
                print("      note:", w[:220])


if __name__ == "__main__":
    main(sys.argv[1:])
