"""Checks of what the engine can do that no finished style uses yet: notes between the
grid lines, spacing rules per style, invented run-ins, hits heard on fine snaps.

    .venv/bin/python tests/engine.py            # every check
    .venv/bin/python tests/engine.py listening  # only the checks whose name has this word

The charts are made with throwaway styles ("probes") defined here. They exist to push
one feature each; they are not meant to play well. Needs the songs and the cache that
tests/regress.py sets up, so run that once first."""

import bisect
import collections
import math
import multiprocessing
import os
import pickle
import random
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regress  # noqa: E402  (points the cache at tests/cache)
import synth  # noqa: E402
import numpy as np  # noqa: E402
from mapmaker import chart, pipeline  # noqa: E402
from mapmaker.chart import FINEST_MS, STYLES, lay_out, spacing, to_ms, tune  # noqa: E402
from mapmaker.listen import RES, between, listen, on_pulse, prepare, raw_hits, snap_of  # noqa: E402
from mapmaker.recommend import RECIPES  # noqa: E402
from mapmaker.stars import star_rating  # noqa: E402
from mapmaker.timing import Clock  # noqa: E402

# The Stream style as it was until 2026-10-16, which the probes below are built on, so
# that each still pushes the one feature it was made for
STREAM_BEFORE = chart.Style("Stream", "Long unbroken 1/4 runs with jumps on the drum hits. No LNs.",
                            hush=0.6, crux=1.5, ln=0.0, chord=4.5, runs=(0.15, 0.30), breath=0.40, cap=0,
                            filler=2, comfort=0, bass=False, motifs=chart._FLOWING, arc=False, layer=False,
                            third=2.5, wide=True, phrase=16, gradual=True, harshest=3, steady=True,
                            suits=(72.0, 100.0, 1), hand_cost=(0.0, 0.5, 2.0, 4.0, 6.0), one_hand=0.0)
JACKY = {"chorus": (["jack", "roll", "split"], [3, 1, 1]),
         "verse": (["jack", "zigzag"], [2, 1]), "calm": (["zigzag"], [1])}
RUN_INS = (("half", 3), ("triplet", 2), ("thirds", 1), ("quarters", 1))
PROBES = {
    # a column may repeat on the very next line
    "jacks": replace(STREAM_BEFORE, apart=1, repeat=70.0, motifs=JACKY),
    # real hits between the lines are charted
    "detail": replace(regress.ORIGINAL, detail=16),
    # invented run-ins on 1/8, 1/6, 1/12 and 1/16, with a repeat limit in ms alone
    "ornate": replace(STREAM_BEFORE, apart=1, repeat=95.0, fills=(0.3, 0.6),
                      ornaments=RUN_INS),
    # the same run-ins under the old two-line repeat rule
    "ornate, strict": replace(STREAM_BEFORE, fills=(0.3, 0.6), ornaments=RUN_INS),
    # a column may be pressed again 40 ms after its long note ends
    "quick release": replace(regress.ORIGINAL, ln=1.0, bass=True, release=40.0, steady=True,
                             hold_taps=(0.6, 0.15, 0.40, 0.45)),
    # everything at once
    "all": replace(regress.ORIGINAL, apart=1, repeat=90.0, release=45.0, detail=16,
                   fills=(0.4, 0.6), ornaments=RUN_INS, motifs=JACKY),
}


def song_for(name, tpb=None, keys=4):
    """A song from the regression cache, put on its own grid (or on `tpb` lines a beat)."""
    key = pipeline.song_key(regress.songs()[name])
    with open(os.path.join(regress.CACHE, f"analysis-{key}.pkl"), "rb") as fh:
        heard = pickle.load(fh)
    timing = heard["timing"]
    beat = 60000.0 / timing["bpm"]
    base = tpb or timing["tpb"]
    meter = "compound" if base == 6 else timing["info"]["feel"]
    return prepare(heard["raw"], heard["dur_ms"], float(round(timing["T0"] % (4 * beat))),
                   beat, base, keys, meter), heard["dur_ms"]


def faults(notes, song, style, length_ms):
    """Everything about a chart that would be wrong whatever the style: a column pressed
    again sooner than the style allows, or while it is held, or rows too close to tell
    apart."""
    line, line_ms, keys = song["line"], song["line_ms"], song["keys"]
    _, gap, release = spacing(song, style)
    out = []
    by_col, seen = collections.defaultdict(list), set()
    for k, c, e in notes:
        if not 0 <= c < keys:
            out.append(f"column {c} does not exist")
        if (k, c) in seen:
            out.append(f"two notes at once in column {c}")
        seen.add((k, c))
        if e is not None and e <= k:
            out.append("a long note that ends before it starts")
        ms = song["T0"] + k / line * line_ms
        if not 0 <= ms <= length_ms:
            out.append(f"a note outside the song, at {ms:.0f} ms")
        by_col[c].append((k, e))
    for c, col in by_col.items():
        col.sort(key=lambda n: n[0])
        for (k1, e1), (k2, _) in zip(col, col[1:]):
            if e1 is None and k2 - k1 < gap:
                out.append(f"column {c} repeats after {(k2 - k1) / line * line_ms:.0f} ms, "
                           f"under the {gap / line * line_ms:.0f} ms the style allows")
            if e1 is not None and k2 - e1 < release:
                out.append(f"column {c} is pressed {(k2 - e1) / line * line_ms:.0f} ms after "
                           "its long note ends")
    rows = sorted({k for k, _, _ in notes})
    slow = song.get("slow", {})         # where the tempo changes: how much longer a bar really lasts
    for a, b in zip(rows, rows[1:]):
        apart = (b - a) / line * line_ms * slow.get(a // (song["bar"] * line), 1.0)
        if (a % line or b % line) and apart < FINEST_MS - 0.01:
            out.append(f"rows {apart:.0f} ms apart, too close to play apart")
    return sorted(set(out))


def shape(notes, song):
    """A few numbers about a chart."""
    line = song["line"]
    rows = collections.defaultdict(set)
    for k, c, _ in notes:
        rows[k].add(c)
    order = sorted(rows)
    jacks = sum(len(rows[a] & rows[b]) for a, b in zip(order, order[1:]) if b - a <= line)
    snaps = collections.Counter(snap_of(k) for k in order)
    off = sum(1 for k in order if k % line)
    cols = collections.defaultdict(list)
    for k, c, e in notes:
        cols[c].append((k, e))
    repeat, let_go = [], []
    for col in cols.values():
        col.sort(key=lambda n: n[0])
        for (k1, e1), (k2, _) in zip(col, col[1:]):
            (repeat if e1 is None else let_go).append(
                ((k2 - k1) if e1 is None else (k2 - e1)) / line * song["line_ms"])
    return {"notes": len(notes), "rows": len(order), "off": off / max(1, len(order)),
            "jacks": jacks / max(1, len(notes)), "snaps": dict(snaps),
            "repeat": min(repeat, default=0.0), "let_go": min(let_go, default=0.0)}


# ------------------------------------------------------------------- the checks

def check_old_spacing():
    """The spacing rule of the existing styles is exactly what it always was: at least
    125 ms and never the very next line, counted in whole lines."""
    import math
    out = []
    for tpb in (3, 4, 6, 8):
        for n in range(240, 1300):
            beat = 60000.0 / (n / 4.0)
            line_ms, line = beat / tpb, RES // tpb
            old = max(2, math.ceil(125.0 / line_ms))
            near, gap, release = spacing({"line": line, "line_ms": line_ms}, regress.ORIGINAL)
            # on the lines, "at least `gap` apart" must mean the same whole number of lines
            if near != old * line or not (old - 1) * line < gap <= near or release != gap:
                out.append(f"{n / 4:g} BPM on 1/{tpb}: was {old} lines, now near {near / line:g}, "
                           f"gap {gap / line:g}, release {release / line:g}")
    return out[:5], "every tempo from 60 to 325 BPM on 1/3, 1/4, 1/6 and 1/8 lines"


def check_times():
    """A note on a line gets the same millisecond it always got."""
    rng, out = random.Random(1), []
    for _ in range(20000):
        tpb = rng.choice((3, 4, 6, 8))
        beat, T0, n = 60000.0 / rng.uniform(60, 330), float(rng.randrange(0, 2000)), rng.randrange(-40, 9000)
        new = to_ms([(n * (RES // tpb), 0, None)], T0, beat / tpb, RES // tpb)[0][0]
        if new != int(round(T0 + n * (beat / tpb))):
            out.append(f"line {n} at {60000 / beat:.3f} BPM")
    return out[:5], "20000 random lines"


def check_listening():
    """In a drum track of known rhythm, 1/8 notes and triplets between the lines are
    found on their snaps. Nothing is found that was not played, and the things that
    only look like notes are left alone: a flam, a hit played late, a lone hit close
    to a line."""
    out, report = [], []
    for bpm in (96, 120, 140, 160, 177):
        beat = 60000.0 / bpm
        fills, decoys = [], []
        for bar in range(2, 46):
            kind = bar % 6
            if kind == 0:
                fills.append((bar, 3, (1 / 8, 5 / 8), "T"))
            elif kind == 1:
                fills.append((bar, 1, (1 / 6, 2 / 6, 4 / 6, 5 / 6), "T"))
            elif kind == 2:
                fills.append((bar, 2, (7 / 8,), "K"))
            elif kind == 3:
                decoys.append((bar, 3, (30.0 / beat,), "S"))            # a flam after the beat
                decoys.append((bar, 1, (0.25 + 17.0 / beat,), "T"))     # a hit played late
                decoys.append((bar, 2, (0.5 + 1 / 12,), "T"))           # alone, close to a line
            elif kind == 4:
                fills.append((bar, 0, (3 / 8,), "S"))
        events, truth = synth.groove(bpm, 48, fills=fills, seed=bpm)
        extra, _ = synth.groove(bpm, 48, fills=decoys, seed=bpm)
        events = sorted(set(events) | {e for e in extra if e[1] in "ST" and e[2] == 0.8})
        audio = synth.render(events, 48 * 4 * beat / 1000.0 + 2)
        fine = raw_hits(audio, (18.0,))[0]
        placed, _ = between(fine, 500.0, beat, 4, 0, 48 * 16)
        want = {int(round((ms - 500.0) / beat * RES)) for ms, _ in truth}
        got = set(placed)
        wrong, missed = got - want, want - got
        report.append(f"{bpm} BPM: {len(got & want)} of {len(want)} found, {len(wrong)} wrong")
        if wrong:
            out.append(f"{bpm} BPM: {len(wrong)} hits placed where no note was played "
                       f"(e.g. 1/{snap_of(min(wrong))})")
        if len(got & want) < 0.6 * len(want):
            by = collections.Counter(snap_of(p) for p in missed)
            out.append(f"{bpm} BPM: only {len(got & want)} of {len(want)} found; missed "
                       + ", ".join(f"{n} on 1/{d}" for d, n in sorted(by.items())))
    # a track with nothing between the lines must stay that way
    events, _ = synth.groove(150, 40, sloppy=4.0, seed=3)
    placed, _ = between(raw_hits(synth.render(events, 66), (18.0,))[0], 500.0, 400.0, 4, 0, 640)
    report.append(f"plain beat played loosely: {len(placed)} found")
    if placed:
        out.append(f"{len(placed)} hits found between the lines of a plain beat")
    return out, "; ".join(report)


def check_mappers():
    """The hits heard between the lines of the example songs are hits their mappers
    charted too. (Needs the examples the user supplied and research/tools/compare.py run
    once; skipped without them.)"""
    sys.path.insert(0, os.path.join(regress.ROOT, "research", "tools"))
    import compare
    found = compare.agreement() if os.path.isdir(compare.PACKS) else {}
    hits, total = sum(a for a, _ in found.values()), sum(n for _, n in found.values())
    if not total:
        return [], "skipped: no example songs listened to yet"
    note = ", ".join(f"1/{d}: {a} of {n}" for d, (a, n) in sorted(found.items()))
    return ([] if hits >= 0.8 * total else [f"only {hits} of {total} were charted by the mappers"],
            f"{hits} of {total} charted by the mappers too ({note})")


def check_real_songs():
    """What the listening finds between the lines of the test songs."""
    out, report = [], []
    for name in regress.songs():
        song, _ = song_for(name)
        report.append(f"{name}: " + (", ".join(f"{n} on 1/{d}" for d, n in sorted(song["heard"].items()))
                                     or "nothing"))
        bad = [k for k in song["between"] if k % song["line"] == 0]
        if bad:
            out.append(f"{name}: {len(bad)} of the hits 'between the lines' sit on a line")
    song, _ = song_for("2016")
    if song["between"]:
        out.append("2016 has no drum hits finer than 1/4, but some were found")
    return out, "; ".join(report)


def _probe(label, style, name, target, expect):
    song, length = song_for(name)
    density, comfort, notes, stars = tune(song, target, 2016, style)
    again = lay_out(song, density, 2016, style, None, comfort)[0]
    out = faults(notes, song, style, length)
    if again != notes:
        out.append("the same settings gave a different chart the second time")
    s = shape(notes, song)
    out += [f"{name}: {text}" for ok, text in expect(s, song, stars, notes) if not ok]
    snaps = ", ".join(f"1/{d} {100 * n / s['rows']:.0f}%" for d, n in sorted(s["snaps"].items()))
    return out, (f"{name}: {stars:.2f} stars, {s['notes']} notes; rows on {snaps}; "
                 f"{100 * s['jacks']:.0f}% jacks; fastest repeat {s['repeat']:.0f} ms")


def check_jacks():
    """A style may let a column repeat on the very next line."""
    return _probe("jacks", PROBES["jacks"], "2016", 4.5, lambda s, song, stars, notes: [
        (s["jacks"] >= 0.10, f"only {100 * s['jacks']:.0f}% of notes are jacks"),
        (abs(s["repeat"] - song["line_ms"]) < 1.0,
         f"fastest repeat is {s['repeat']:.0f} ms, not one line ({song['line_ms']:.0f} ms)"),
        (abs(stars - 4.5) <= 0.05, f"{stars:.2f} stars, asked 4.5")])


def check_detail():
    """Real hits between the lines are charted where they were heard."""
    def expect(s, song, stars, notes):
        rows = {k for k, _, _ in notes}
        off = {k for k in rows if k % song["line"]}
        return [(off <= set(song["between"]), "a note between the lines that was not heard there"),
                (len(off) >= 0.85 * len(song["between"]),
                 f"only {len(off)} of {len(song['between'])} hits between the lines were charted")]
    return _probe("detail", PROBES["detail"], "Electrodoodle", 3.5, expect)


def check_ornaments():
    """Invented run-ins put rows on 1/8, 1/6 and finer, and lift what a song can reach."""
    song, _ = song_for("2016")
    plain = tune(song, 6.0, 2016, STREAM_BEFORE)[3]
    out, text = _probe("ornate", PROBES["ornate"], "2016", 4.5, lambda s, song, stars, notes: [
        (s["off"] >= 0.25, f"only {100 * s['off']:.0f}% of rows are between the lines"),
        (s["snaps"].get(8, 0) > 0 and s["snaps"].get(6, 0) > 0, "no rows on 1/8 or on 1/6"),
        (s["repeat"] >= 95.0 - 1.0, f"a column repeats after {s['repeat']:.0f} ms"),
        (abs(stars - 4.5) <= 0.05, f"{stars:.2f} stars, asked 4.5")])
    high = tune(song, 6.0, 2016, PROBES["ornate"])
    out += faults(high[2], song, PROBES["ornate"], 1e9)
    if high[3] < plain + 0.5:
        out.append(f"run-ins reach {high[3]:.2f} stars, the plain style {plain:.2f}")
    strict, more = _probe("ornate, strict", PROBES["ornate, strict"], "2016", 4.5,
                          lambda s, song, stars, notes: [
        (s["off"] > 0.02, "no run-ins fit under the two-line rule"),
        (s["repeat"] >= 2 * song["line_ms"] - 1.0, f"a column repeats after {s['repeat']:.0f} ms")])
    return out + strict, f"{text}; asked 6: {high[3]:.2f} against {plain:.2f} without | strict: {more}"


def check_release():
    """A style may let a column be pressed again soon after its long note ends."""
    return _probe("quick release", PROBES["quick release"], "2016", 4.0,
                  lambda s, song, stars, notes: [
        (40.0 - 1.0 <= s["let_go"] <= song["line_ms"] + 1.0,
         f"soonest press after a long note is {s['let_go']:.0f} ms")])


def _quarter_lines(beats):
    """The time of every 1/4 line, from the time of every beat."""
    return np.concatenate([beats[i] + (beats[i + 1] - beats[i]) * np.arange(4) / 4.0
                           for i in range(len(beats) - 1)])


def _miss(lines, true):
    """How far each true line is from the nearest of `lines`, in ms."""
    i = np.clip(np.searchsorted(lines, true), 1, len(lines) - 1)
    return np.minimum(np.abs(true - lines[i - 1]), np.abs(true - lines[i]))


def check_tempo_followed():
    """Drum tracks whose tempo steps or slides are followed to within a few ms, their
    steady stretches come out at the tempo they were played at, and their bars start
    where they should. A track that keeps one tempo is left alone, however loosely it
    is played."""
    out, report = [], []
    tracks = (                          # name, parts, looseness, lines within 5 ms, bars to check
        ("steps", [(64, 120, 120), (64, 140, 140), (64, 160, 160)], 0.0, 0.995, True),
        ("a big step", [(64, 130, 130), (96, 200, 200)], 0.0, 0.995, True),
        ("many steps", [(64, 120, 120), (16, 125, 125), (16, 130, 130), (16, 135, 135),
                        (48, 140, 140), (16, 142, 142), (16, 145, 145), (32, 150, 150),
                        (32, 160, 160), (16, 170, 170), (40, 180, 180), (160, 205, 205),
                        (16, 210, 210), (16, 220, 220), (120, 230, 230)], 0.0, 0.995, True),
        ("a slide", [(48, 120, 120), (96, 120, 160), (48, 160, 160)], 0.0, 0.99, False),
        ("a quick slide", [(48, 140, 140), (32, 140, 180), (64, 180, 180)], 0.0, 0.99, False),
        ("a slide, played loosely", [(48, 130, 130), (64, 130, 170), (64, 170, 170)], 5.0, 0.97, False),
        ("one tempo", [(160, 150, 150)], 0.0, None, False),
        ("one tempo, played loosely", [(160, 128, 128)], 6.0, None, False),
    )
    for name, parts, sloppy, wanted, bars in tracks:
        beats = synth.beat_times(parts)
        audio = synth.render(synth.groove(0, 10**6, beats=beats, sloppy=sloppy, seed=1)[0],
                             beats[-1] / 1000.0 + 2)
        dur = len(audio) / synth.SR * 1000.0
        timing = pipeline._measure({"drums": audio}, audio, dur, False,
                                   hits=raw_hits(audio)[0])
        if wanted is None:
            report.append(f"{name}: left alone")
            if timing.get("moving"):
                out.append(f"{name}: taken for a song whose tempo changes")
            continue
        if not timing.get("moving"):
            out.append(f"{name}: not followed")
            continue
        clock, true = Clock(**timing["moving"]), _quarter_lines(beats)
        miss = _miss(clock.lines, true)
        report.append(f"{name}: {100 * (miss <= 5).mean():.1f}% of lines within 5 ms")
        if (miss <= 5).mean() < wanted:
            out.append(f"{name}: only {100 * (miss <= 5).mean():.1f}% of the lines are within "
                       f"5 ms of where they were played (worst {miss.max():.0f} ms)")
        for beats_, start, end in parts:
            if start == end and beats_ >= 32 and not any(
                    abs(15000.0 / line - start) < 0.05 and k1 - k0 >= 3 * beats_
                    for k0, k1, _, line in clock.parts):
                out.append(f"{name}: no stretch at {start} BPM for the {beats_} beats played at it")
        if bars:
            starts = clock.lines[clock.first::16]
            good = np.mean([np.abs(starts - t).min() <= 5 for t in true[::16]])
            if good < 0.95:
                out.append(f"{name}: only {100 * good:.0f}% of the bars start where they were played")
    return out, "; ".join(report)


def check_tempo_charts():
    """A made-up song whose tempo changes, through the whole app in every style: each
    note lands where the song really has that line, the charts have no faults in the
    song's own time either, the .osu has a red line where each tempo starts and keeps
    one scroll speed, and setting BPM and offset by hand still holds one tempo."""
    import mapmaker
    out = []
    beats = regress.make_moving()
    true = _quarter_lines(beats)
    path = regress.songs(True)[regress.MOVING]
    folder = os.path.join(regress.OUT, "engine tempo")
    try:
        done = mapmaker.make_map(path, mapmaker.Options(style="stream", stars=4.0), folder)
        with open(done["osu"], encoding="utf-8") as fh:
            points = [p.split(",") for p in regress.section(fh.read(), "TimingPoints")]
        held = mapmaker.make_map(path, mapmaker.Options(style="stream", stars=4.0, bpm=180.0,
                                                        offset=500.0), folder)
    finally:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)

    # --- what the app says and writes
    tempo = done["stats"]["tempo"]
    if not tempo or (tempo["low"], tempo["high"]) != (120.0, 180.0):
        out.append(f"the tempo range is given as {tempo}, not 120 to 180")
    if not any("follows it" in w for w in done["warnings"]):
        out.append("no note that the map follows the tempo")
    if held["stats"]["tempo"] or not any("holds 180 BPM" in w for w in held["warnings"]):
        out.append("BPM and offset set by hand did not hold the map to one tempo")
    red = [(float(p[0]), float(p[1])) for p in points if p[6] == "1"]
    green = {float(p[0]): float(p[1]) for p in points if p[6] == "0"}
    # the packed audio may start a few ms from the one that was analysed
    lag = float(np.median([min(t - true, key=abs) for t, _ in red]))
    parts, n = regress.MOVING_PARTS, 0
    for i, (count, start, end) in enumerate(parts):
        when, n = beats[n], n + count
        if start != end:
            continue
        stepped = i > 0 and parts[i - 1][1] == parts[i - 1][2]      # not reached by a slide
        near = min(red, key=lambda r: abs(r[0] - lag - when))
        if stepped and abs(near[0] - lag - when) > 4.0:
            out.append(f"no red line where {start} BPM starts ({when:.0f} ms)")
        elif not any(abs(60000.0 / length - start) < 0.05 and at - lag < when + 8 * 60000.0 / start
                     for at, length in red):
            out.append(f"no red line of {start} BPM within two bars of where that tempo starts")
    speeds = {round(-100.0 / green[t] / beat, 9) for t, beat in red if t in green}
    if len(speeds) != 1 or any(t not in green for t, _ in red):
        out.append("the scroll speed is not the same under every red line")

    # --- the charts, against the lines that were really played
    key = pipeline.song_key(path)
    with open(os.path.join(regress.CACHE, f"analysis-{key}.pkl"), "rb") as fh:
        heard = pickle.load(fh)
    clock = Clock(**heard["timing"]["moving"])
    span = float(clock.even_time(heard["dur_ms"]))
    song = pipeline.on_grid(clock.straighten(heard["raw"]), span, clock.T0, clock.beat, 4, 4,
                            "straight", clock)
    first = int(np.argmin(np.abs(clock.lines - true[0])))  # the line the first beat is on
    if (first - clock.first) % 16:
        out.append(f"bars start {(first - clock.first) % 16} lines away from where they were played")
    worst, n = 0.0, 0
    for label, style in STYLES.items():
        for density in (0.3, 1.0):
            notes, stars = lay_out(song, density, 5, style)
            out += [f"{label}: {f}" for f in faults(notes, song, style, span)]
            objects = to_ms(notes, song["T0"], song["line_ms"], song["line"], clock)
            if abs(stars - star_rating(objects, 4)) > 1e-9:
                out.append(f"{label}: the stars are not those of the chart in the song's own time")
            # where the song has each note: its line, counted from the first beat played
            place = np.array([clock.first + k / (song["line"]) - first for k, _, _ in sorted(notes, key=lambda n: (n[0], n[1]))])
            inside = (place >= 0) & (place <= len(true) - 1)
            want = np.interp(place, np.arange(len(true)), true)
            got = np.array(sorted(t for t, _, _ in objects), dtype=float)
            off = np.abs(np.sort(want[inside]) - np.sort(got[inside]))
            worst = max(worst, float(np.percentile(off, 99)))
            if np.percentile(off, 99) > 4.0:
                out.append(f"{label}: notes up to {off.max():.0f} ms from where the song has them "
                           f"(1 in 100 over {np.percentile(off, 99):.1f} ms)")
            _, gap, _ = spacing(song, style)
            floor = gap / song["line"] * song["line_ms"] - 2.0
            last = {}
            for t, c, e in objects:
                if c in last and last[c][1] is None and t - last[c][0] < floor:
                    out.append(f"{label}: column {c} repeats after {t - last[c][0]} ms of the song")
                    break
                last[c] = (t, e)
            n += 1
    return sorted(set(out))[:12], f"{n} charts, 99 in 100 notes within {worst:.1f} ms of the song's lines"


def check_breaks():
    """A song that opens with eight bars of melody and no drums: every style charts
    that melody there, note for note, on the beats and half beats it is played on,
    although its notes come in more softly than a drum and a touch late. Nothing is
    charted that was not played, and nothing between the half beats (but for a run-in
    to the bar where the drums come in, in a style that has those). And where the
    sounds of such an opening keep no time, they are not charted at all."""
    beat, bars, quiet = 500.0, 24, 8
    events, _ = synth.groove(120, bars)
    drums = synth.render([e for e in events if e[0] >= 500.0 + quiet * 4 * beat - 1.0],
                         bars * 4 * beat / 1000.0 + 2)
    scale = (392.0, 440.0, 523.25, 587.33, 659.25, 783.99)
    shapes = ((0, 2, 3, 4, 6), (0, 1, 2, 4, 5, 6), (0, 2, 4, 5, 6, 7), (0, 3, 4, 6))    # half beats of a bar
    starts = [(bar, h) for bar in range(bars) for h in shapes[bar % 4]]
    notes = [(500.0 + 8.0 + (bar * 8 + h) * beat / 2, 0.8 * beat / 2 * (2 if (h + 1) % 8 not in shapes[bar % 4] else 1),
              scale[(bar * 3 + h * 5) % len(scale)]) for bar, h in starts]
    lead = synth.melody(notes, bars * 4 * beat / 1000.0 + 2)[:len(drums)]
    hush = (1e-4 * np.random.default_rng(2).standard_normal(len(drums))).astype(np.float32)
    raw = listen({"drums": drums, "bass": hush, "other": lead, "vocals": hush}, drums + lead + hush)
    song = prepare(raw, len(drums) / synth.SR * 1000.0, 500.0, beat, 4, 4)
    line, out = song["line"], []
    calm = [b for b in song["bars"] if song["section"][b] == "calm" and 0 <= b < quiet]
    if len(calm) != quiet:
        return [f"{len(calm)} of the {quiet} bars without drums were taken for bars without drums"], ""
    want = {(bar * 16 + 2 * h) * line for bar, h in starts if bar < quiet}
    report = []
    for label, style in STYLES.items():
        if label == "original":
            continue
        for density in (0.5, 1.0):
            notes_, _ = lay_out(song, density, 3, style)
            got = {k for k, _, _ in notes_ if 0 <= k < quiet * 16 * line}
            # (a run-in to the first bar with drums may start in the beat before it)
            off = [k for k in got if (k % line or not on_pulse(k // line % 16, 4, "straight"))
                   and k < (quiet * 16 - 4) * line]
            if len(got & want) < 0.9 * len(want):
                out.append(f"{label} at {density}: only {len(got & want)} of the {len(want)} notes of the melody are charted")
            if len({k for k in got - want if k < (quiet * 16 - 4) * line}) > 0.1 * len(want):
                out.append(f"{label} at {density}: {len(got - want)} notes where the melody has none")
            if off:
                out.append(f"{label} at {density}: {len(off)} notes off the beats and half beats")
        report.append(f"{label} {len(got & want)}/{len(want)}")
    # The same opening with sounds that keep no time at all (what an onset detector
    # makes of a held voice or a reverb tail): none of that is charted.
    rng = np.random.default_rng(4)
    stray = sorted(rng.uniform(500.0, 500.0 + quiet * 4 * beat, 110))
    stray = [(t, 90.0, scale[n % len(scale)]) for n, t in enumerate(stray)
             if n == 0 or t - stray[n - 1] > 45.0]
    stray += [n for n in notes if n[0] >= 500.0 + quiet * 4 * beat]
    lead = synth.melody(stray, bars * 4 * beat / 1000.0 + 2)[:len(drums)]
    raw = listen({"drums": drums, "bass": hush, "other": lead, "vocals": hush}, drums + lead + hush)
    heard = sum(1 for h in raw["melody"] if h[0] < 500.0 + quiet * 4 * beat)
    song = prepare(dict(raw, lead=[]), len(drums) / synth.SR * 1000.0, 500.0, beat, 4, 4)
    for label in ("stream", "coordination", "ln"):
        notes_, _ = lay_out(song, 1.0, 3, STYLES[label])
        got = {k for k, _, _ in notes_ if 0 <= k < (quiet * 16 - 4) * line}
        if len(got) > 4:
            out.append(f"{label}: {len(got)} notes charted from {heard} sounds that keep no time")
    report.append(f"of {heard} sounds that keep no time, {len(got)} charted")
    return out[:12], "melody notes charted in the bars without drums: " + ", ".join(report)


def check_crux():
    """A song whose loudest stretch has the fewest drum hits (a half-time drop under a
    wall of sound, as at 4:25 of the user's song): that stretch is found, and every
    style charts it much fuller than its drum hits alone would have it."""
    beat, bars = 400.0, 48
    quiet, drop = range(0, 8), range(32, 40)
    events, _ = synth.groove(150, bars)
    def keep(ms, sound, loud):
        bar, part = divmod((ms - 500.0) / beat, 4)
        if int(bar) in drop:            # half time: a kick on one, a snare on three, nothing else
            return sound in "KS" and abs(part - round(part)) < 0.01 and round(part) in (0, 2)
        return int(bar) not in quiet or sound in "KS"
    seconds = bars * 4 * beat / 1000.0 + 2
    drums = synth.render([e for e in events if keep(*e)], seconds)
    bass, lead = synth.band(500.0 + np.arange(bars * 4 + 1) * beat, seconds)
    gain = np.ones(len(drums), dtype=np.float32)
    for part, level in ((quiet, 0.25), (drop, 3.0)):
        a = int((500.0 + part[0] * 4 * beat) * synth.SR / 1000.0)
        b = int((500.0 + (part[-1] + 1) * 4 * beat) * synth.SR / 1000.0)
        gain[a:b] = level
    bass, lead = bass[:len(drums)] * gain, lead[:len(drums)] * gain
    hush = (1e-4 * np.random.default_rng(2).standard_normal(len(drums))).astype(np.float32)
    raw = listen({"drums": drums, "bass": bass, "other": lead, "vocals": hush}, drums + bass + lead + hush)
    song = prepare(raw, len(drums) / synth.SR * 1000.0, 500.0, beat, 4, 4)
    line, out = song["line"], []
    found = [b for b in drop if song["lift"].get(b, 0.0) >= 0.5]
    stray = [b for b in song["bars"] if b not in drop and song["lift"].get(b, 0.0) >= 0.5]
    if len(found) < 6 or stray:
        return [f"{len(found)} of the 8 loudest bars were found, and {len(stray)} others taken for them"], ""
    def per_bar(notes, which):
        return sum(1 for k, _, _ in notes if k // (16 * line) in which) / len(which)
    usual = range(12, 28)
    report = []
    for label, style in STYLES.items():
        if label == "original":
            continue
        notes, _ = lay_out(song, 0.35, 3, style)
        plain, _ = lay_out(song, 0.35, 3, replace(style, crux=0.0))
        now, before, around = per_bar(notes, drop), per_bar(plain, drop), per_bar(notes, usual)
        report.append(f"{label} {before:.0f} -> {now:.0f} (around it {around:.0f})")
        if now < 1.3 * before:
            out.append(f"{label}: the loudest bars carry {now:.1f} notes each, hardly more than the {before:.1f} their drum hits give")
        if now < 0.4 * around:          # (the bars around have a drum hit on every line here)
            out.append(f"{label}: the loudest bars carry {now:.1f} notes each, the bars around them {around:.1f}")
    return out[:12], "notes a bar in the loudest stretch, without and with the lift: " + ", ".join(report)


def check_coordination(name):
    """Coordination is a kind of LN chart that trains the fingers apart: something is
    held most of the time and never every column at once, a hold seldom follows in
    the column of the one before, and most single notes are played while a hold is
    down. Since the user's playtest of 2026-10-16 ("many scary jacks (even consecutive
    jumps) before lns ... tone it down so jacks before lns are rare, and maybe
    balancing that by adding some reverse shields, as well as inverse lns"): shields
    (a tap a quarter beat before a hold, same column) are rare and never two at once,
    reverse shields (a tap a quarter beat after one) are common, holds are pressed
    again a quarter beat after they were let go (inverse and long-notes-only phrases),
    no column is hit three times running on 1/4 notes, and no two rows a quarter beat
    apart share two columns."""
    song, _ = song_for(name)
    style = STYLES["coordination"]
    _, _, notes, stars = tune(song, 4.5, 2016, style)
    objects = to_ms(notes, song["T0"], song["line_ms"], song["line"])
    holds = [(t, c, e) for t, c, e in objects if e is not None]
    taps = [t for t, _, e in objects if e is None]
    out = []
    if len(holds) < 0.15 * len(objects):
        out.append(f"only {len(holds)} of {len(objects)} notes are long notes")
    edges = sorted([(t, 1) for t, _, _ in holds] + [(e, -1) for _, _, e in holds])
    held = most = 0
    down, last = 0.0, objects[0][0]
    for t, d in edges:
        down += (t - last) if held else 0.0
        held, last = held + d, t
        most = max(most, held)
    share = down / max(1.0, objects[-1][0] - objects[0][0])
    if share < 0.6:
        out.append(f"something is held only {100 * share:.0f}% of the time")
    if most > song["keys"] - 1:
        out.append(f"{most} notes held at once")
    same = sum(1 for a, b in zip(holds, holds[1:]) if a[1] == b[1]) / max(1, len(holds) - 1)
    if same > 0.15:
        out.append(f"{100 * same:.0f}% of the holds are in the same column as the hold before")
    starts, ends = sorted(t for t, _, _ in holds), sorted(e for _, _, e in holds)
    under = sum(1 for t in taps if bisect.bisect_left(starts, t) > bisect.bisect_right(ends, t))
    if under < 0.5 * len(taps):
        out.append(f"only {100 * under / max(1, len(taps)):.0f}% of the single notes are played under a hold")
    # a quarter of a beat, or as soon after as the style lets a column be pressed again
    quarter = song["beat"] / (3 if song["tpb"] % 3 == 0 else 4)
    _, gap, release = spacing(song, style)
    quarter *= max(1, math.ceil(max(gap, release) / song["line"] * song["line_ms"] / quarter - 0.01))
    shields = reverse = again = 0
    shielded = collections.Counter()
    by_col = collections.defaultdict(list)
    for t, c, e in objects:
        by_col[c].append((t, e))
    for col in by_col.values():
        for a, b in zip(col, col[1:]):
            if b[0] - (a[0] if a[1] is None else a[1]) <= quarter + 2.0:
                shields += a[1] is None and b[1] is not None
                shielded[b[0]] += a[1] is None and b[1] is not None
                reverse += a[1] is not None and b[1] is None
                again += a[1] is not None and b[1] is not None
    if shields > 0.1 * len(holds):
        out.append(f"{shields} shields on {len(holds)} holds: not rare")
    if any(n >= 2 for n in shielded.values()):
        out.append(f"{sum(1 for n in shielded.values() if n >= 2)} rows of two holds each right after a tap in its column")
    if reverse < 0.2 * len(holds) or again < 10:
        out.append(f"{reverse} reverse shields on {len(holds)} holds, {again} holds pressed again at once: too few")
    rows = collections.defaultdict(set)
    for t, c, _ in objects:
        rows[t].add(c)
    times, run, longest, doubled = sorted(rows), collections.Counter(), 0, 0
    near = spacing(song, chart.rated(style, 4.5))[0] / song["line"] * song["line_ms"]   # the fastest repeat allowed
    for a, b in zip(times, times[1:]):
        quick = b - a <= near + 2.0
        doubled += quick and len(rows[a] & rows[b]) >= 2
        for c in range(song["keys"]):
            run[c] = run[c] + 1 if quick and c in rows[a] and c in rows[b] else 0
            longest = max(longest, run[c] + 1)
    if longest > 2:
        out.append(f"a column hit {longest} times running as fast as the style allows ({near:.0f} ms)")
    if doubled:
        out.append(f"{doubled} rows {near:.0f} ms apart share two columns")
    return out, (f"{stars:.2f} stars, {len(holds)} holds in {len(objects)} notes, held {100 * share:.0f}% of the "
                 f"time, {shields} shields, {reverse} reverse shields, {again} holds pressed again a quarter beat on")


def check_tempo_guide():
    """Speedjack's jacks may come faster the higher the rating, as in the speedjack
    charts at hand: 119 ms apart at 4.5 stars (Mi Mi Mi), 86-90 ms at 5.5-6.2 (the rest
    of NoNamed's Speedjack Pack). So 1/4 notes at 177 BPM (85 ms) suit it at 5.5 stars
    but not at 4.5, where the user found them "insanely fast"; there the rows go into
    triplets instead (113 ms), as the user asked on 2026-10-16, and back onto the song's
    1/4 notes from about 5.5 stars. A 126 BPM song keeps its 1/4 notes (119 ms)."""
    sj, out = STYLES["speedjack"], []
    for stars, ms, fits in ((4.5, 119.0, True), (4.5, 85.0, False), (5.0, 97.0, True),
                            (5.6, 86.0, True), (6.2, 88.0, True), (6.2, 120.0, False)):
        low, high, _ = chart.suit_range(sj, stars)
        if (low <= ms <= high) != fits:
            out.append(f"rows {ms:g} ms apart at {stars} stars: {'should' if fits else 'should not'} "
                       f"suit Speedjack (it takes {low:.0f}-{high:.0f} ms)")
    for bpm, stars, tpb in ((177.0, 4.5, 3), (177.0, 5.0, 3), (177.0, 5.5, 4), (126.0, 4.5, 4),
                            (126.0, 5.5, 4), (150.0, 4.5, 4), (168.0, 4.5, 3), (88.0, 4.5, 6)):
        got = chart.grid_for(sj, 60000.0 / bpm, 4, stars)
        if got != tpb:
            out.append(f"{bpm:g} BPM at {stars} stars: rows on 1/{got} notes, not 1/{tpb}")
    return out, "Speedjack's row spacing at 4.5 / 5.5 / 6.2 stars: " + ", ".join(
        "{:.0f}-{:.0f} ms".format(*chart.suit_range(sj, s)[:2]) for s in (4.5, 5.5, 6.2))


def check_coordination_ratings():
    """Coordination writes few phrases in long notes only at ordinary ratings and more
    towards 5.5 stars (the user: "they can be rare, but more common in the higher star
    rating (around 5.5 ish)")."""
    ratings = (3.0, 4.0, 4.5, 5.0, 5.5, 6.5)
    shares = [chart.rated(STYLES["coordination"], stars).mix for stars in ratings]
    inverse = [chart.rated(STYLES["coordination"], stars).inv for stars in ratings]
    out = []
    if any(b < a for a, b in zip(shares, shares[1:])):
        out.append(f"the share falls as the rating rises: {shares}")
    if shares[2] > 0.15 or shares[4] < 0.35:
        out.append(f"{shares[2]:.2f} of the phrases at 4.5 stars and {shares[4]:.2f} at 5.5")
    # inverse, which the user had not met yet: "not too many though, these are known to
    # be pretty challenging": none at 3 stars, about one phrase in eight at 4.5
    if any(b < a for a, b in zip(inverse, inverse[1:])) or inverse[0] > 0.0 or not 0.08 <= inverse[2] <= 0.16:
        out.append(f"inverse phrases at those ratings: {inverse}")
    return out, ("share of phrases in long notes only at 3, 4, 4.5, 5, 5.5, 6.5 stars: " + ", ".join(f"{x:.2f}" for x in shares)
                 + "; inverse: " + ", ".join(f"{x:.2f}" for x in inverse))


def check_coordination_all_held(name):
    """Coordination with its Long notes setting at the top: long notes are used as
    normal notes, as in the two charts the user named (research/examples/
    coordination-*.osu). Most notes are held, two at once much of the time and never
    four; a hold is often let go just a quarter of a beat before its column is
    pressed again (a shield) and never right before."""
    song, length = song_for(name)
    style = STYLES["coordination"]
    _, _, notes, stars = tune(song, 4.0, 2016, style, 1.0)
    out = [f for f in faults(notes, song, style, length)]
    objects = to_ms(notes, song["T0"], song["line_ms"], song["line"])
    holds = [(t, c, e) for t, c, e in objects if e is not None]
    if len(holds) < 0.7 * len(objects):
        out.append(f"only {len(holds)} of {len(objects)} notes are long notes")
    edges = sorted([(t, 1) for t, _, _ in holds] + [(e, -1) for _, _, e in holds])
    held = most = 0
    two, last = 0.0, objects[0][0]
    for t, d in edges:
        two += (t - last) if held >= 2 else 0.0
        held, last = held + d, t
        most = max(most, held)
    two /= max(1.0, objects[-1][0] - objects[0][0])
    if two < 0.3:
        out.append(f"two notes are held at once only {100 * two:.0f}% of the time")
    if most > 3:
        out.append(f"{most} notes held at once")
    quarter = song["beat"] / (3 if song["tpb"] % 3 == 0 else 4)
    _, gap, release = spacing(song, style)
    quarter *= max(1, math.ceil(max(gap, release) / song["line"] * song["line_ms"] / quarter - 0.01))
    again, by_col = [], collections.defaultdict(list)
    for t, c, e in objects:
        by_col[c].append((t, e))
    for col in by_col.values():
        again += [b[0] - a[1] for a, b in zip(col, col[1:]) if a[1] is not None]
    if again and min(again) < style.release - 2.0:
        out.append(f"a column is pressed {min(again):.0f} ms after its hold ends")
    shields = sum(1 for g in again if g <= quarter + 2.0) / max(1, len(again))
    if shields < 0.15:
        out.append(f"only {100 * shields:.0f}% of the holds are let go a quarter beat before the next press")
    return out[:8], (f"{stars:.2f} stars, {len(holds)} holds in {len(objects)} notes, two at once "
                     f"{100 * two:.0f}% of the time, {100 * shields:.0f}% let go a quarter beat before the next press")


def check_stream(name):
    """Stream as the user described it on 2026-10-16: "a stream ... that come in a
    consistent time (like around 1/4 or 1/8 depending on the song bpm) with not much
    complex division of notes", whose rating comes from "the stream section": single
    and light-jump streams at low ratings, handstream about 4.5, quads higher up.
    So: rows on one grid (the song's 1/4, or 1/8 below about 106 BPM), almost all
    gaps inside a run that grid's spacing, more and bigger chords as the rating rises,
    no column three times running."""
    out, said = [], []
    song, length = song_for(name)
    style = STYLES["stream"]
    tpb = chart.grid_for(style, song["beat"], song["tpb"])
    if tpb != song["tpb"]:
        song, length = song_for(name, tpb)
    before = None
    for target in (3.0, 4.5, 5.5):
        _, _, notes, stars = tune(song, target, 2016, style)
        out += [f"{target}: {f}" for f in faults(notes, song, chart.rated(style, target), length)]
        rows = collections.defaultdict(set)
        for k, c, _ in notes:
            rows[k].add(c)
        order = sorted(rows)
        gaps = [b - a for a, b in zip(order, order[1:]) if b - a <= song["tpb"] * song["line"]]
        steady = sum(1 for g in gaps if g == song["line"]) / max(1, sum(1 for g in gaps if g < 2 * song["line"]))
        off = sum(1 for k in order if k % song["line"])
        if off or steady < 0.97:
            out.append(f"{target}: {off} rows off the 1/{tpb} lines, {100 * steady:.0f}% of quick gaps on one line")
        chords = sum(len(rows[k]) for k in order) / len(order)
        if before is not None and chords < before - 0.02 and abs(stars - target) <= 0.05:
            out.append(f"{target}: {chords:.2f} notes a row, fewer than the {before:.2f} at a lower rating")
        before = chords
        run, longest = collections.Counter(), 1
        for a, b in zip(order, order[1:]):
            for c in range(song["keys"]):
                run[c] = run[c] + 1 if b - a <= song["line"] and c in rows[a] and c in rows[b] else 0
                longest = max(longest, run[c] + 1)
        if longest > 2:
            out.append(f"{target}: a column hit {longest} times running")
        # no chordjack (the user, after the first version: "the streams have chordjack
        # during the chorus section which doesn't fit the style"): no column on two rows
        # running except next to a quad, and no two chords half a beat apart with nothing
        # between them that share a column
        full = {k for k in order if len(rows[k]) >= song["keys"]}
        jacks = sum(1 for a, b in zip(order, order[1:]) if b - a <= song["line"]
                    and rows[a] & rows[b] and not {a, b} & full)
        slow = sum(1 for a, b in zip(order, order[1:]) if song["line"] < b - a <= 2 * song["line"]
                   and len(rows[a]) > 1 and len(rows[b]) > 1 and rows[a] & rows[b] and not {a, b} & full)
        if jacks or slow > 0.01 * len(order):
            out.append(f"{target}: {jacks} rows repeat a column of the row before, "
                       f"{slow} chords one of the chord half a beat before")
        quads = sum(1 for k in order if len(rows[k]) == 4)
        if quads and target < 5.0:
            out.append(f"{target}: {quads} quads")
        # not one figure over and over (the user, 2026-10-17: "the pattern of a jump
        # followed by a short one hand trill just repeats over and over. a stream should
        # be diverse"): in the beats run through on four 1/4 rows, no one way of sharing
        # out the notes takes most of them, and a beat seldom repeats the one before
        if song["keys"] == 4 and song["meter"] == "straight" and target == 4.5:
            line, beats = song["line"], []
            for k in order:
                if k // line % 4 == 0 and all(k + i * line in rows for i in range(4)):
                    beats.append((k, tuple(frozenset(rows[k + i * line]) for i in range(4))))
            layouts = collections.Counter("".join(str(len(r)) for r in b) for _, b in beats)
            again = sum(1 for (a, x), (b, y) in zip(beats, beats[1:]) if b - a == 4 * line and x == y)
            if beats and (layouts.most_common(1)[0][1] > 0.6 * len(beats) or again > 0.08 * len(beats)):
                out.append(f"{target}: {layouts.most_common(1)[0][0]} on {layouts.most_common(1)[0][1]} "
                           f"of {len(beats)} beats, {again} beats the same as the one before")
        said.append(f"{target}: {stars:.2f} stars, {chords:.2f} notes a row")
    return out[:12], f"on 1/{tpb} notes; " + ", ".join(said)


def check_jack_limits(name):
    """No style hits a column more times running at its fastest spacing than it says
    (Style.jacks), whatever the rating (the user: "limiting time of consecutive
    same-column hits to prevent unfair or downright harsh playability")."""
    out, said = [], []
    for label, style in STYLES.items():
        if not style.jacks:
            continue
        for target in (4.5, 5.8):
            rated = chart.rated(style, target)
            song, _ = song_for(name, chart.grid_for(style, song_for(name)[0]["beat"],
                                                         song_for(name)[0]["tpb"], target))
            notes = tune(song, target, 2016, style)[2]
            near = spacing(song, rated)[0]
            cols, longest = collections.defaultdict(list), 1
            for k, c, _ in sorted(notes):
                cols[c].append(k)
            for ks in cols.values():
                run = 1
                for a, b in zip(ks, ks[1:]):
                    run = run + 1 if b - a <= near else 1
                    longest = max(longest, run)
            if longest > rated.chain:
                out.append(f"{label} at {target}: {longest} presses running, the limit is {rated.chain}")
            said.append(f"{label} {target}: {longest}")
    return out, "longest jack: " + ", ".join(said)


def check_recommended():
    """The Recommended style ranks every style for each test song, picks one whose
    tempo suits the song and says why, the same way each time; the made-up song, a
    drum track on every 1/4 line whose tempo steps and slides, gets Stream; and asking
    for "auto" charts exactly what asking for the picked style does."""
    import shutil
    import mapmaker
    out, picks = [], []
    songs = dict(regress.songs(), **{regress.MOVING: regress.songs(True)[regress.MOVING]})
    for name, path in songs.items():
        key = pipeline.song_key(path)
        ranking = pipeline.recommend_for(key, 4.5)
        if ranking is None:
            out.append(f"{name}: not listened to yet (run tests/regress.py first)")
            continue
        scores = [r["score"] for r in ranking]
        if sorted(r["style"] for r in ranking) != sorted(RECIPES):
            out.append(f"{name}: ranks {[r['style'] for r in ranking]}")
        if scores != sorted(scores, reverse=True) or not 0.0 < scores[0] <= 1.0:
            out.append(f"{name}: scores {scores}")
        if ranking[0]["fit"] < 0.3:
            out.append(f"{name}: picks {ranking[0]['style']}, which its tempo does not suit")
        if not ranking[0]["why"]:
            out.append(f"{name}: no reason given for {ranking[0]['style']}")
        if pipeline.recommend_for(key, 4.5) != ranking:
            out.append(f"{name}: a different ranking the second time")
        picks.append(f"{name} {ranking[0]['style']}")
        if name == regress.MOVING and ranking[0]["style"] != "stream":
            out.append(f"{name}: picks {ranking[0]['style']}, not stream")
    folder = os.path.join(regress.OUT, "engine recommended")
    try:
        auto = mapmaker.make_map(songs["2016"], mapmaker.Options(style="auto", stars=4.0), folder)
        plain = mapmaker.make_map(songs["2016"], mapmaker.Options(style=auto["stats"]["style"], stars=4.0),
                                  folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    if auto["chart"]["notes"] != plain["chart"]["notes"]:
        out.append(f"2016: \"auto\" charts other notes than {auto['stats']['style']} itself")
    if not auto["stats"]["picked"] or plain["stats"]["picked"]:
        out.append("2016: the map does not say which style was picked, or says so when none was")
    return out, "picks at 4.5 stars: " + ", ".join(picks)


def check_everything(name):
    """Every feature at once on one song, at several ratings, key counts and grids:
    the charts must have no faults."""
    out, n = [], 0
    straight = song_for(name)[0]["meter"] == "straight"
    for keys, target, tpb in ((4, 3.0, None), (4, 4.5, None), (4, 6.0, None), (7, 4.5, None),
                              (4, 4.5, 8), (4, 4.5, 6)):
        if tpb and not straight:
            continue
        song, length = song_for(name, tpb, keys)
        for label in ("all", "ornate", "jacks"):
            notes = tune(song, target, 7, PROBES[label])[2]
            out += [f"{keys}K {label} {target}" + (f" on 1/{tpb} lines" if tpb else "") + f": {f}"
                    for f in faults(notes, song, PROBES[label], length)]
            n += 1
    return out[:12], f"{n} charts"


def check_app_styles(name):
    """The charts of the styles the app offers have no faults (they are compared note
    for note by tests/regress.py; this looks at them from the other side)."""
    out, n = [], 0
    song, length = song_for(name)
    for label, style in STYLES.items():
        notes = tune(song, 4.5, 2016, style)[2]
        out += [f"{label}: {f}" for f in faults(notes, song, style, length)]
        off = sum(1 for k, _, _ in notes if k % song["line"])
        if off and not (style.ornaments or style.detail):
            out.append(f"{label}: {off} notes between the lines")
        n += 1
    return out[:12], f"{n} charts"


def tasks():
    """(label, function name, arguments) for every check; the two slow ones once a song."""
    out = []
    for name, fn in globals().items():
        if not name.startswith("check_"):
            continue
        label = name[6:].replace("_", " ")
        if fn.__code__.co_argcount:
            out += [(f"{label}, {song}", name, (song,)) for song in regress.songs()]
        else:
            out.append((label, name, ()))
    return out


def run(task):
    label, name, args = task
    try:
        problems, note = globals()[name](*args)
    except Exception as err:
        problems, note = [f"{type(err).__name__}: {err}"], "crashed"
    return label, problems, note


def main(argv):
    todo = [t for t in tasks() if not argv or any(word in t[0] for word in argv)]
    failed = 0
    with multiprocessing.Pool(min(len(todo), max(1, (os.cpu_count() or 4) - 2))) as pool:
        for label, problems, note in pool.imap(run, todo):
            failed += bool(problems)
            print(("ok      " if not problems else "FAILED  ") + f"{label}: {note}"
                  + "".join(f"\n          {p}" for p in problems), flush=True)
    print("\nall checks passed" if not failed else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
