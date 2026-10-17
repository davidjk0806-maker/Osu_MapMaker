"""How varied a chart's streams are: where the chords sit in the beat, what follows a
chord, how much runs as a staircase, and how often a beat repeats the one before.
Written after the user's playtest of 2026-10-17 ("a jump followed by a short one hand
trill just repeats over and over. a stream should be diverse").

    .venv/bin/python research/tools/shapes.py refs [pack words ...]       # the reference packs' 4K difficulties
    .venv/bin/python research/tools/shapes.py mine STYLE STARS [SONG ...]  # the app's maps of the library songs
    .venv/bin/python research/tools/shapes.py file some.osu ...           # any chart

Only the streams are looked at: eight or more rows in a row, evenly spaced, at most a
quarter beat apart. `mine` takes `--set name=value` like balance.py."""

import collections
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

from balance import library, refs  # noqa: E402
from measure import parse  # noqa: E402


def streams(chart):
    """The chart's stream runs -> [[(time, columns, place in the beat 0-3), ...], ...]."""
    rows = collections.defaultdict(list)
    for t, c, _ in chart["notes"]:
        rows[t].append(c)
    times = sorted(rows)
    red = chart["red"] or [(0.0, 500.0)]

    def line(t):
        cur = red[0]
        for r in red:
            if r[0] <= t + 2:
                cur = r
        return cur

    runs, cur = [], [times[0]] if times else []
    for a, b in zip(times, times[1:]):
        offset, length = line(a)
        gap = b - a
        if gap <= length / 4 + 3 and (len(cur) < 2 or abs(gap - (cur[-1] - cur[-2])) <= 3):
            cur.append(b)
        else:
            if len(cur) >= 8:
                runs.append(cur)
            cur = [b]
    if len(cur) >= 8:
        runs.append(cur)
    out = []
    for run in runs:
        offset, length = line(run[0])
        gap = (run[-1] - run[0]) / (len(run) - 1)
        out.append([(t, tuple(sorted(rows[t])), round((t - offset) / gap) % 4) for t in run])
    return out


def mirror(shape, keys):
    return tuple(tuple(sorted(keys - 1 - c for c in row)) for row in shape)


def describe(chart):
    keys = chart["keys"]
    hand = {c: (0 if c < keys // 2 else 1) for c in range(keys)}
    runs = streams(chart)
    n = sum(len(r) for r in runs)
    m = {"rows": n}
    if not n:
        return m
    size = collections.Counter(min(len(cols), 4) for run in runs for _, cols, _ in run)
    m["chords"] = "/".join(f"{100 * size[k] / n:.0f}" for k in (1, 2, 3, 4))
    # where the chords sit: on the beat, a quarter after, half beat, three quarters
    place = collections.Counter(p for run in runs for _, cols, p in run if len(cols) > 1)
    total = sum(place.values()) or 1
    m["chord_place"] = "/".join(f"{100 * place[p] / total:.0f}" for p in range(4))
    # rows from one chord to the next
    gaps = collections.Counter()
    for run in runs:
        at = [i for i, (_, cols, _) in enumerate(run) if len(cols) > 1]
        for a, b in zip(at, at[1:]):
            gaps[min(b - a, 6)] += 1
    total = sum(gaps.values()) or 1
    m["chord_gaps"] = " ".join(f"{k if k < 6 else '6+'}:{100 * v / total:.0f}"
                               for k, v in sorted(gaps.items()))
    # the three rows after a chord, when they are single notes
    after = collections.Counter()
    for run in runs:
        for i, (_, cols, _) in enumerate(run[:-3]):
            if len(cols) < 2:
                continue
            nxt = [run[i + j][1] for j in (1, 2, 3)]
            if any(len(x) > 1 for x in nxt[:2]):
                after["chord again"] += 1
                continue
            a, b = nxt[0][0], nxt[1][0]
            c = nxt[2][0] if len(nxt[2]) == 1 else None
            if c == a and hand[a] == hand[b]:
                after["one-hand trill"] += 1
            elif c == a:
                after["two-hand trill"] += 1
            elif c is not None and (a < b < c or a > b > c):
                after["stair"] += 1
            else:
                after["other"] += 1
    total = sum(after.values()) or 1
    m["after_chord"] = " ".join(f"{k}:{100 * v / total:.0f}" for k, v in after.most_common())
    # staircases: rows each wholly above (or below) the row before, three or more long
    in_stair, stair_rows_chord, stairs = 0, 0, collections.Counter()
    for run in runs:
        cols = [c for _, c, _ in run]
        steps = []
        for a, b in zip(cols, cols[1:]):
            steps.append(1 if min(b) > max(a) else -1 if max(b) < min(a) else 0)
        i = 0
        while i < len(steps):
            j = i
            while j < len(steps) and steps[j] == steps[i] != 0:
                j += 1
            if j - i >= 2:                      # three rows or more
                part = cols[i:j + 1]
                in_stair += len(part)
                stairs[min(len(part), 5)] += 1
                stair_rows_chord += any(len(x) > 1 for x in part)
                i = j
            else:
                i += 1
    m["stair_rows"] = round(100 * in_stair / n)
    m["stairs"] = sum(stairs.values())
    m["stair_with_chord"] = round(100 * stair_rows_chord / (sum(stairs.values()) or 1))
    # beats (four rows from a beat) and what they repeat
    beats, pairs = [], 0                # (run, row it starts on, its rows)
    for r, run in enumerate(runs):
        for i, (_, _, p) in enumerate(run):
            if p == 0 and i + 4 <= len(run):
                beats.append((r, i, tuple(c for _, c, _ in run[i:i + 4])))
    same = same_m = same2 = 0
    for k in range(1, len(beats)):
        (ra, ia, a), (rb, ib, b) = beats[k - 1], beats[k]
        if (ra, ia + 4) != (rb, ib):    # only a beat and the one straight after it
            continue
        pairs += 1
        same += a == b
        same_m += a == b or mirror(a, keys) == b
        if k >= 2 and beats[k - 2][:2] == (ra, ia - 4):
            same2 += beats[k - 2][2] == b and a != b
    nb = max(1, pairs)
    m["beats"] = len(beats)
    m["repeat_beat"] = round(100 * same / nb)
    m["repeat_mirror"] = round(100 * same_m / nb)
    m["repeat_every_2"] = round(100 * same2 / nb)
    # how many kinds of beat, by their shape up to which columns (moved to start at 0)
    kinds = collections.Counter()
    for _, _, b in beats:
        kinds[min(b, mirror(b, keys))] += 1
    m["kinds_per_100"] = round(100 * len(kinds) / max(1, len(beats)))
    m["top_kinds"] = ", ".join(f"{''.join(fmt(r) for r in k)}:{100 * v / len(beats):.0f}%"
                               for k, v in kinds.most_common(4))
    return m


def fmt(cols):
    s = "".join(str(c + 1) for c in cols)
    return s if len(cols) == 1 else f"[{s}]"


def show(name, m):
    if not m.get("rows"):
        print(f"{name}: no streams")
        return
    print(f"{name}: {m['rows']} stream rows, {m['beats']} beats | chords 1/2/3/4 {m['chords']} | "
          f"chords on beat/e/and/a {m['chord_place']} | rows between chords {m['chord_gaps']}")
    print(f"    after a chord: {m['after_chord']} | stairs: {m['stair_rows']}% of rows "
          f"({m['stairs']} stairs, {m['stair_with_chord']}% with a chord)")
    print(f"    beat = beat before {m['repeat_beat']}% (or its mirror {m['repeat_mirror']}%), "
          f"= two before {m['repeat_every_2']}% | {m['kinds_per_100']} kinds per 100 beats | "
          f"most: {m['top_kinds']}")


def main(argv):
    if argv[0] == "refs":
        for pack, version, chart in refs():
            if len(argv) == 1 or any(w.lower() in pack.lower() for w in argv[1:]):
                show(f"{pack[:30]} [{version}]", describe(chart))
    elif argv[0] == "file":
        for path in argv[1:]:
            show(os.path.basename(path)[:60], describe(parse(path)))
    else:
        import json
        import multiprocessing
        from balance import _one
        style, stars, rest, changes = argv[1], float(argv[2]), [], {}
        it = iter(argv[3:])
        for a in it:
            if a == "--set":            # as in balance.py
                name, value = next(it).split("=", 1)
                changes[name] = json.loads(value.replace("(", "[").replace(")", "]"))
                if isinstance(changes[name], list):
                    changes[name] = tuple(tuple(v) if isinstance(v, list) else v for v in changes[name])
            else:
                rest.append(a)
        jobs = [(title, audio, style, stars, changes) for title, audio in library(rest)]
        with multiprocessing.Pool(min(8, len(jobs))) as pool:
            for title, s, chart, done in pool.imap(_one, jobs):
                show(f"{title[:30]} {s} {done['stats']['stars']:.2f}", describe(chart))


if __name__ == "__main__":
    main(sys.argv[1:])
