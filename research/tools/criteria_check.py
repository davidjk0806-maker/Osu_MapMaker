"""Hold a 4K chart against the osu!mania ranking criteria's difficulty-specific
guidelines (research/notes/osu-wiki/ranking_criteria_mania.md) and against the shape of its
star rating. See research/notes/star_rating_reference.md for what came of it.

    .venv/bin/python research/tools/criteria_check.py some.osu ...

The criteria are written for 180 BPM and scale with tempo (Scaling_BPM: at 120 BPM a
guideline about 1/4 reads like one about 1/2, at 90 BPM density doubles), so every
gap here is measured in milliseconds against 180 BPM snaps: 1/1 = 333 ms, 1/2 = 167,
1/4 = 83, 1/6 = 56, 1/8 = 42. A gap counts as a snap's class when it is nearer to it
than to the next one (edges at the geometric midpoints)."""

import collections
import math
import os
import statistics
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.dirname(RESEARCH))
from measure import parse  # noqa: E402
from mapmaker.stars import _legacy_sort  # noqa: E402

B = 60000.0 / 180                      # the criteria's beat
EDGE = {1: 471.4, 2: 235.7, 4: 117.9, 6: 68.0, 8: 48.1, 12: 34.0}
LEVELS = ["Easy", "Normal", "Hard", "Insane", "Expert"]
BAND = [(2.0, "Easy"), (2.7, "Normal"), (4.0, "Hard"), (5.3, "Insane"), (6.5, "Expert"), (99, "Expert+")]


def band(stars):
    return next(name for top, name in BAND if stars < top)


def strain_peaks(objects, keys):
    """The 400 ms section peaks of lazer's mania strain, in time order (same loop as
    mapmaker/stars.py), so the rating can be taken apart."""
    objs = [(float(t), c, float(t if e is None else e)) for t, c, e in objects]
    objs.sort(key=lambda o: o[0])
    _legacy_sort(objs, key=lambda o: int(round(o[0])))
    prev, col_start, individual = [None] * keys, [None] * keys, [0.0] * keys
    highest, overall, current = 0.0, 1.0, 0.0
    peaks, section_peak, section_end, last = [], 0.0, 0.0, None
    for i in range(1, len(objs)):
        start, col, end = objs[i]
        delta = start - objs[i - 1][0]
        if last is not None:
            prev[last[1]] = (last[0], last[2])
        col_time = start if col_start[col] is None else start - col_start[col]
        col_start[col] = start
        if i == 1:
            section_end = math.ceil(start / 400.0) * 400.0
        while start > section_end:
            peaks.append(section_peak)
            gone = (section_end - last[0]) / 1000.0
            section_peak = highest * 0.125 ** gone + overall * 0.30 ** gone
            section_end += 400.0
        hold_factor, overlapping, closest_end = 1.0, False, abs(end - start)
        for p in prev:
            if p is None:
                continue
            p_start, p_end = p
            overlapping |= p_end - start > 1 and end - p_end > 1 and start - p_start > 1
            if p_end - end > 1 and start - p_start > 1:
                hold_factor = 1.25
            closest_end = min(closest_end, abs(end - p_end))
        hold_addition = 1.0 / (1.0 + math.exp(0.27 * (30.0 - closest_end))) if overlapping else 0.0
        individual[col] = individual[col] * 0.125 ** (col_time / 1000.0) + 2.0 * hold_factor
        highest = max(highest, individual[col]) if delta <= 1 else individual[col]
        overall = overall * 0.30 ** (delta / 1000.0) + (1.0 + hold_addition) * hold_factor
        current += highest + overall - current
        section_peak = max(current, section_peak)
        last = (start, col, end)
    peaks.append(section_peak)
    ranked = sorted((p for p in peaks if p > 0), reverse=True)
    stars = 0.018 * sum(p * 0.9 ** i for i, p in enumerate(ranked))
    return peaks, stars


def chains(times, edge):
    out, cur = [], [times[0]]
    for a, b in zip(times, times[1:]):
        if b - a < edge:
            cur.append(b)
        else:
            out.append(cur)
            cur = [b]
    out.append(cur)
    return out


def check(chart):
    keys, notes = chart["keys"], chart["notes"]
    rows = collections.defaultdict(list)
    for t, c, e in notes:
        rows[t].append((c, e))
    times = sorted(rows)
    size = {t: len(rows[t]) for t in times}
    minutes = max((times[-1] - times[0]) / 60000.0, 0.1)
    v = collections.OrderedDict()     # name -> (count, first level it is fine at, words)

    def add(name, count, level, words):
        v[name] = (count, level, words)

    def interior_chords(edge):
        return sum(1 for ch in chains(times, edge) if len(ch) >= 3 for t in ch[1:-1] if size[t] >= 2)

    # ---- rhythm: runs of rows at a snap or faster
    c4, c2, c6, c8 = (chains(times, EDGE[d]) for d in (4, 2, 6, 8))
    add("consecutive 1/4", sum(1 for ch in c4 if len(ch) >= 3), "Normal", "runs of 3+ rows at 1/4 (Easy: none)")
    add("more than 5 rows of 1/2", sum(1 for ch in c2 if len(ch) > 5), "Normal", "Easy: at most 5 in a row")
    add("chords inside 1/2 runs", interior_chords(EDGE[2]), "Normal", "Easy: chords only at the ends")
    add("consecutive 1/6", sum(1 for ch in c6 if len(ch) >= 3), "Hard", "Normal: none")
    add("more than 5 rows of 1/4", sum(1 for ch in c4 if len(ch) > 5), "Hard", "Normal: at most 5 in a row")
    long_half = [ch for ch in chains(times, 260) if ch[-1] - ch[0] > 16 * B]
    add("1/2 streams with no 1/1 pause for a phrase", len(long_half), "Hard", "Normal: a 1/1 pause every 4 bars")
    add("chords inside 1/4 runs", interior_chords(EDGE[4]), "Hard", "Normal: chords only at the ends")
    add("consecutive 1/8 (past 3 grace notes)", sum(1 for ch in c8 if len(ch) >= 4), "Insane", "Hard: none")
    long_quarter = [ch for ch in chains(times, 140) if ch[-1] - ch[0] > 16 * B]
    add("1/4 streams with no 1/2 pause for a phrase", len(long_quarter), "Insane", "Hard: a 1/2 pause every 4 bars")
    close = 0
    for ch in c4:
        cr = [t for t in ch if size[t] >= 2]
        close += sum(1 for a, b in zip(cr, cr[1:]) if b - a < 0.9 * B)
    add("chords under 1/1 apart inside 1/4 streams", close, "Insane", "Hard: at least 1/1 apart")
    add("chords inside 1/6 runs", interior_chords(EDGE[6]), "Insane", "Hard: only at the ends")
    add("1/8 streams longer than 4 beats", sum(1 for ch in c8 if ch[-1] - ch[0] > 4 * B), "Expert", "Insane: avoid")
    add("chords inside 1/8 runs", interior_chords(EDGE[8]), "Expert", "4K Insane: only at the ends")

    # ---- chords
    add("hands (3-note chords)", sum(1 for t in times if size[t] == 3), "Hard", "4K Easy/Normal: 2 notes at most")
    add("quads (4-note chords)", sum(1 for t in times if size[t] >= 4), "Insane", "4K Hard: 3 notes at most")

    # ---- columns: jacks, shields, anchors
    by_col = collections.defaultdict(list)
    for t, c, e in notes:
        by_col[c].append((t, e))
    mini, jack3, shield4, jack_ms = 0, 0, 0, []
    for c, objs in by_col.items():
        run = 1
        for (t1, e1), (t2, e2) in zip(objs, objs[1:]):
            if e1 is None and e2 is None and t2 - t1 < EDGE[4]:
                run += 1
                jack_ms.append(t2 - t1)
                continue
            if e1 is None and e2 is not None and t2 - t1 < EDGE[4]:
                shield4 += 1
            if run == 2:
                mini += 1
            elif run >= 3:
                jack3 += 1
            run = 1
        if run == 2:
            mini += 1
        elif run >= 3:
            jack3 += 1
    add("1/4 minijacks", mini, "Hard", "Normal: none; Hard: only with ample rest")
    add("1/4 jacks of 3+ notes", jack3, "Insane", "Hard: discouraged")
    add("1/4 shields", shield4, "Hard", "Normal: none")
    anchors3 = anchors5 = longest_anchor = 0
    for ch in c4:                     # anchors: one column every other row of a 1/4 stream
        if len(ch) < 4:
            continue
        g = statistics.median(b - a for a, b in zip(ch, ch[1:]))
        lo, hi = ch[0], ch[-1]
        for c, objs in by_col.items():
            ts = [t for t, _ in objs if lo <= t <= hi]
            n = 1
            for a, b in zip(ts, ts[1:]):
                if 1.2 * g < b - a <= 2.4 * g:
                    n += 1
                else:
                    anchors3 += n >= 3
                    anchors5 += n >= 5
                    longest_anchor = max(longest_anchor, n)
                    n = 1
            anchors3 += n >= 3
            anchors5 += n >= 5
            longest_anchor = max(longest_anchor, n)
    add("anchors of 3+ notes", anchors3, "Hard", "Normal: avoid")
    add("anchors of 5+ notes", anchors5, "Insane", "Hard: avoid")

    # ---- trills and split-jumptrills inside 1/4 runs
    long_trills, longest_trill, split9, longest_split = 0, 0, 0, 0
    for ch in c4:
        cols = [frozenset(c for c, _ in rows[t]) for t in ch]
        n = 0
        for i, s in enumerate(cols):
            ok = len(s) == 1 and i >= 1 and len(cols[i - 1]) == 1 and s != cols[i - 1] and (i < 2 or cols[i - 2] == s or n < 2)
            n = n + 1 if ok else 1 if len(s) == 1 else 0
            longest_trill = max(longest_trill, n)
            if n == 10:
                long_trills += 1
        n = 0
        for i, s in enumerate(cols):
            split = len(s) == 2 and min(s) < keys / 2 <= max(s)
            ok = split and i >= 1 and len(cols[i - 1]) == 2 and not (s & cols[i - 1]) and (i < 2 or cols[i - 2] == s or n < 2)
            n = n + 1 if ok else 1 if split else 0
            longest_split = max(longest_split, 2 * n)
            if n == 5:
                split9 += 1
    add("trills longer than 9 notes", long_trills, "Insane", "Hard: 9 notes at most")
    add("split-jumptrills longer than 9 notes", split9, "Expert", "4K Insane: 9 notes at most")

    # ---- long notes
    lns = [(t, c, e) for t, c, e in notes if e is not None]
    if lns:
        lengths = [e - t for t, _, e in lns]
        add("LNs shorter than 1/1", sum(1 for x in lengths if x < 0.9 * B), "Normal", "Easy: held 1 beat at least")
        add("LNs shorter than 1/2", sum(1 for x in lengths if x < 0.45 * B), "Hard", "Normal: held 1/2 at least")
        add("LNs shorter than 1/4", sum(1 for x in lengths if x < 0.22 * B), "Insane", "Hard: held 1/4 at least")
        ends = sorted({e for _, _, e in lns})
        add("LN releases under 1/1 apart", sum(1 for a, b in zip(ends, ends[1:]) if 1 < b - a < 0.9 * B), "Normal", "Easy: 1 beat apart")
        add("LN releases under 1/2 apart", sum(1 for a, b in zip(ends, ends[1:]) if 1 < b - a < 0.45 * B), "Hard", "Normal: 1/2 apart")
        inside_1 = sum(1 for t, c, e in lns if e - t <= 1.25 * B for t2, c2, _ in notes if c2 != c and t + 1 < t2 < e - 1)
        inside_half = sum(1 for t, c, e in lns if e - t <= 0.6 * B for t2, c2, _ in notes if c2 != c and t + 1 < t2 < e - 1)
        add("notes during a 1/1 hold", inside_1, "Normal", "Easy: none")
        add("notes during a 1/2 hold", inside_half, "Hard", "Normal: none")
        heads = sorted({t for t, _, _ in lns})
        add("LN streams at 1/6 or faster", sum(1 for ch in chains(heads, EDGE[6]) if len(ch) >= 3), "Expert", "Insane: avoid")
    add("LNs held under 1/12 (28 ms)", sum(1 for t, _, e in lns if e - t < B / 12), "Expert", "every level: avoid")

    # ---- where the rating comes from
    peaks, stars = strain_peaks(notes, keys)
    level = [p * 0.18 for p in peaks]          # what the map would rate if it were all like this
    first = int(times[0] // 400)
    active = [x for x in level if x > 0.05]
    nps_win = collections.Counter(int(t // 1000) for t, _, _ in notes)
    secs = [nps_win.get(s, 0) for s in range(int(times[0] // 1000), int(times[-1] // 1000) + 1)]
    return {
        "violations": v,
        "stars": stars,
        "moment_median": statistics.median(active) if active else 0.0,
        "moment_p90": sorted(active)[int(0.9 * (len(active) - 1))] if active else 0.0,
        "moment_max": max(active) if active else 0.0,
        "at_rating": 100.0 * sum(1 for x in active if x >= 0.9 * stars) / max(1, len(active)),
        "nps": len(notes) / max(1.0, (times[-1] - times[0]) / 1000.0),
        "nps_peak": max(secs) if secs else 0,
        "jack_p5_ms": sorted(jack_ms)[int(0.05 * (len(jack_ms) - 1))] if jack_ms else None,
        "jack_share": 100.0 * len(jack_ms) / len(notes),
        "longest_trill": longest_trill, "longest_split": longest_split, "longest_anchor": longest_anchor,
        "ln_share": 100.0 * len(lns) / len(notes),
        "minutes": minutes, "rows": len(times), "first_section": first,
    }


def slack(minutes):
    """Slips of one kind let through: guidelines say "avoid", not "never"."""
    return max(2.0, 1.0 * minutes)


def criteria_level(viol, minutes):
    """The lowest level whose guidelines the chart keeps to."""
    need = 0
    for count, level, _ in viol.values():
        if count > slack(minutes):
            need = max(need, LEVELS.index(level))
    return LEVELS[need]


def why(viol, level, minutes):
    """The guidelines of `level` the chart goes past."""
    i = LEVELS.index(level)
    return [(name, count) for name, (count, lv, _) in viol.items()
            if count > slack(minutes) and LEVELS.index(lv) > i]


if __name__ == "__main__":
    for path in sys.argv[1:]:
        r = check(parse(path))
        print(f"\n{os.path.basename(path)}")
        print(f"  {r['stars']:.2f} stars: {band(r['stars'])} by its rating, "
              f"{criteria_level(r['violations'], r['minutes'])} by its patterns")
        print(f"  most of it plays like {r['moment_median']:.2f} stars ({r['moment_median'] / max(r['stars'], 0.01):.2f}x the rating); "
              f"{r['at_rating']:.0f}% of it is near the rating; {r['nps']:.1f} notes a second")
        for name, (count, lv, words) in r["violations"].items():
            if count:
                print(f"   {name:45s} {count:5d}   (fine from {lv}; {words})")
