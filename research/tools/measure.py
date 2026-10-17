"""Measure what a chart is made of, so styles can be described in numbers:
snaps, speed, chords, jacks, bursts, long notes.

    python research/tools/measure.py                 # every chart in research/examples
    python research/tools/measure.py some.osu ...    # specific charts
"""

import collections
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mapmaker.stars import star_rating  # noqa: E402

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
SNAPS = (1, 2, 4, 8, 16, 3, 6, 12)     # checked coarse to fine, straight before triplet


def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
    sec, cur = collections.defaultdict(list), None
    for line in text.split("\n"):
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1]
        elif cur and line.strip() and not line.startswith("//"):
            sec[cur].append(line)
    fields = {}
    for name in ("General", "Metadata", "Difficulty"):
        for line in sec[name]:
            k, _, v = line.partition(":")
            fields[k.strip()] = v.strip()
    keys = int(round(float(fields["CircleSize"])))
    red, green = [], []
    for line in sec["TimingPoints"]:
        f = line.split(",")
        t, length = float(f[0]), float(f[1])
        if (f[6] if len(f) > 6 else "1") == "1" and length > 0:
            red.append((t, length))
        elif length < 0:
            green.append((t, -100.0 / length))
    notes = []
    for line in sec["HitObjects"]:
        f = line.split(",")
        col = min(keys - 1, max(0, int(int(f[0]) * keys / 512)))
        hold = int(f[3]) & 128
        notes.append((int(f[2]), col, int(f[5].split(":")[0]) if hold else None))
    notes.sort(key=lambda n: (n[0], n[1]))
    return {"fields": fields, "keys": keys, "red": sorted(red), "green": sorted(green),
            "notes": notes}


def pct(x, total):
    return 100.0 * x / total if total else 0.0


def quant(values, q):
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def measure(chart):
    notes, keys, red = chart["notes"], chart["keys"], chart["red"]
    hand = [0 if c < keys / 2 - 0.25 else 1 if c > keys / 2 - 0.75 else 2 for c in range(keys)]
    out = {}

    def beat_at(t):
        cur = red[0]
        for tp in red:
            if tp[0] <= t + 1:
                cur = tp
            else:
                break
        return cur

    rows = collections.defaultdict(list)
    for t, c, e in notes:
        rows[t].append((c, e))
    times = sorted(rows)
    span = (times[-1] - times[0]) / 1000.0
    out["objects"], out["rows"], out["length_s"] = len(notes), len(times), span
    out["stars"] = star_rating(notes, keys)
    # main tempo = the one most of the notes are played under
    by_bpm = collections.Counter(round(60000.0 / beat_at(t)[1], 1) for t in times)
    out["bpm"], out["bpm_share"] = by_bpm.most_common(1)[0][0], pct(by_bpm.most_common(1)[0][1], len(times))
    out["red_lines"], out["sv_points"] = len(red), len(chart["green"])
    sv = [m for _, m in chart["green"]]
    out["sv_range"] = (min(sv), max(sv)) if sv else (1.0, 1.0)

    # --- which grid line each row sits on
    snap = collections.Counter()
    for t in times:
        t0, length = beat_at(t)
        pos = ((t - t0) / length) % 1.0
        for d in SNAPS:
            if abs(pos * d - round(pos * d)) * length / d <= 2.5:
                snap[d] += 1
                break
        else:
            snap["other"] += 1
    out["snap"] = {str(k): pct(v, len(times)) for k, v in snap.items()}

    # --- time between one row and the next, as a fraction of the beat
    gaps = collections.Counter()
    ms_gaps = []
    for a, b in zip(times, times[1:]):
        length = beat_at(a)[1]
        ms_gaps.append(b - a)
        for d in (1, 2, 3, 4, 6, 8, 12, 16):
            if abs((b - a) - length / d) <= 2.5:
                gaps[d] += 1
                break
        else:
            gaps["slower" if b - a > length else "other"] += 1
    out["gap"] = {str(k): pct(v, len(times) - 1) for k, v in gaps.items()}
    out["gap_ms_median"] = statistics.median(ms_gaps)

    # --- density through the song
    per_sec = collections.Counter(int((t - times[0]) / 1000.0) for t, _, _ in notes)
    seconds = [per_sec.get(s, 0) for s in range(int(span) + 1)]
    out["nps_avg"], out["nps_peak"] = len(notes) / max(span, 1.0), max(seconds)
    out["nps_p50"], out["nps_p90"] = quant(seconds, 0.5), quant(seconds, 0.9)
    out["quiet_time"] = pct(sum(1 for s in seconds if s <= 0.25 * out["nps_p90"]), len(seconds))

    # --- runs: rows that follow each other with no gap longer than a 1/4 beat
    runs, cur = [], 1
    for a, b in zip(times, times[1:]):
        if b - a <= beat_at(a)[1] / 4 + 3:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    in_runs = sum(r for r in runs)
    out["run_median"], out["run_p90"], out["run_max"] = (quant([r for r in runs if r >= 3], 0.5),
                                                          quant([r for r in runs if r >= 3], 0.9), max(runs))
    for n in (8, 32, 96):
        out[f"rows_in_runs_{n}"] = pct(sum(r for r in runs if r >= n), in_runs)
    out["breaks_per_min"] = 60.0 * sum(1 for a, b in zip(times, times[1:]) if b - a >= 2 * beat_at(a)[1]) / max(span, 1.0)
    # the same in plain time: stretches where rows keep coming at least every 200 ms
    stretches, start = [], times[0]
    for a, b in zip(times, times[1:]):
        if b - a > 200:
            stretches.append((a - start) / 1000.0)
            start = b
    stretches.append((times[-1] - start) / 1000.0)
    busy = [x for x in stretches if x >= 0.3]
    out["stretch_median_s"], out["stretch_p90_s"], out["stretch_max_s"] = quant(busy, 0.5), quant(busy, 0.9), max(stretches)
    out["time_in_stretches"] = pct(sum(busy), max(span, 1.0))
    out["row_gap_p10_ms"], out["row_gap_p25_ms"] = quant(ms_gaps, 0.10), quant(ms_gaps, 0.25)
    # bursts: rows in a row spaced finer than 1/4 of a beat (1/6, 1/8, 1/12, 1/16)
    bursts, cur, changes, prev_kind = [], 0, 0, None
    for a, b in zip(times, times[1:]):
        length = beat_at(a)[1]
        kind = next((d for d in (1, 2, 3, 4, 6, 8, 12, 16) if abs((b - a) - length / d) <= 2.5), 0)
        if b - a < length / 4 - 3:
            cur += 1
        else:
            if cur:
                bursts.append(cur + 1)
            cur = 0
        if prev_kind is not None and kind != prev_kind and b - a <= length:
            changes += 1
        prev_kind = kind
    if cur:
        bursts.append(cur + 1)
    out["bursts_per_min"] = 60.0 * len(bursts) / max(span, 1.0)
    out["burst_median"], out["burst_p90"], out["burst_max"] = quant(bursts, 0.5), quant(bursts, 0.9), max(bursts, default=0)
    out["rows_in_bursts"] = pct(sum(bursts), len(times))
    out["spacing_changes_per_beat"] = changes / max(1.0, sum((b - a) / beat_at(a)[1] for a, b in zip(times, times[1:]) if b - a <= beat_at(a)[1]))

    # --- chords
    sizes = collections.Counter(len(v) for v in rows.values())
    out["chord"] = {n: pct(sizes.get(n, 0), len(times)) for n in range(1, keys + 1)}
    out["notes_per_row"] = len(notes) / len(times)

    # --- jacks: the same column in two rows running
    jack_ms, chain, chains = [], collections.Counter(), []
    prev_cols = set()
    prev_t = None
    for t in times:
        cols = {c for c, _ in rows[t]}
        for c in range(keys):
            if c in cols and c in prev_cols and t - prev_t <= 250:
                jack_ms.append(t - prev_t)
                chain[c] += 1
            else:
                if chain[c]:
                    chains.append(chain[c] + 1)
                chain[c] = 0
        prev_cols, prev_t = cols, t
    chains += [n + 1 for n in chain.values() if n]
    out["jack_share"] = pct(len(jack_ms), len(notes))
    out["jack_ms_median"] = statistics.median(jack_ms) if jack_ms else 0.0
    out["jack_fast_share"] = pct(sum(1 for m in jack_ms if m <= 110), len(notes))
    out["jack_len"] = {"2": sum(1 for n in chains if n == 2), "3-4": sum(1 for n in chains if 3 <= n <= 4),
                       "5+": sum(1 for n in chains if n >= 5)}
    out["jack_len_max"] = max(chains) if chains else 0
    # the fastest any column is ever asked to repeat, ignoring the rarest 1%
    col_gaps = []
    for c in range(keys):
        ts = [t for t, cc, _ in notes if cc == c]
        col_gaps += [b - a for a, b in zip(ts, ts[1:])]
    out["column_repeat_p1"], out["column_repeat_p10"] = quant(col_gaps, 0.01), quant(col_gaps, 0.10)

    # --- how single notes move inside runs: trills, rolls, hand changes
    moves = collections.Counter()
    singles = [(t, rows[t][0][0]) for t in times if len(rows[t]) == 1]
    for (t1, a), (t2, b), (t3, c) in zip(singles, singles[1:], singles[2:]):
        length = beat_at(t1)[1]
        if t2 - t1 > length / 4 + 3 or t3 - t2 > length / 4 + 3:
            continue
        if a == c and a != b:
            moves["one-hand trill" if hand[a] == hand[b] else "two-hand trill"] += 1
        elif a < b < c or a > b > c:
            moves["roll"] += 1
        elif a == b or b == c:
            moves["jack"] += 1
        else:
            moves["other"] += 1
    total = sum(moves.values())
    out["moves"] = {k: pct(v, total) for k, v in moves.items()}
    out["single_triples"] = total

    # --- long notes
    lns = [(t, c, e) for t, c, e in notes if e is not None]
    out["ln_share"] = pct(len(lns), len(notes))
    if lns:
        out["ln_beats_median"] = statistics.median((e - t) / beat_at(t)[1] for t, _, e in lns)
        edges = sorted([(t, 1) for t, _, _ in lns] + [(e, -1) for _, _, e in lns])
        held, last, cover = 0, times[0], collections.Counter()
        for t, d in edges:
            cover[held] += t - last
            held, last = held + d, t
        total_ms = max(1, times[-1] - times[0])
        out["held_1"], out["held_2"], out["held_3"] = (pct(sum(v for k, v in cover.items() if k >= n), total_ms)
                                                       for n in (1, 2, 3))
        # taps that land while something is being held in another column
        taps = [(t, c) for t, c, e in notes if e is None]
        under = sum(1 for t, c in taps if any(s < t < e for s, cc, e in lns if cc != c and s > t - 20000))
        out["taps_under_holds"] = pct(under, len(taps)) if taps else 0.0
        # inverse: how soon after letting go the same column is pressed again
        release_gap = []
        for c in range(keys):
            col = [(t, e) for t, cc, e in notes if cc == c]
            for (t, e), (t2, _) in zip(col, col[1:]):
                if e is not None:
                    release_gap.append((t2 - e) / beat_at(t)[1])
        out["release_gap_median_beats"] = statistics.median(release_gap) if release_gap else 0.0
        out["release_tight"] = pct(sum(1 for g in release_gap if g <= 0.26), len(release_gap))
    return out


def show(name, m, extra=""):
    def top(d, n=4):
        return " ".join(f"1/{k}:{v:.0f}%" if str(k).isdigit() else f"{k}:{v:.0f}%"
                        for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:n] if v >= 0.5)
    print(f"\n=== {name}{extra}")
    print(f"  {m['stars']:.2f} stars by the port | {m['bpm']:g} BPM ({m['bpm_share']:.0f}% of rows; "
          f"{m['red_lines']} red lines, {m['sv_points']} SV points, SV {m['sv_range'][0]:.2f}-{m['sv_range'][1]:.2f}x)"
          f" | {m['length_s']:.0f} s | {m['objects']} objects in {m['rows']} rows")
    print(f"  density   avg {m['nps_avg']:.1f}/s, typical second {m['nps_p50']}, busy second {m['nps_p90']}, "
          f"peak {m['nps_peak']} | quiet for {m['quiet_time']:.0f}% of the time | {m['breaks_per_min']:.1f} breaks/min")
    print(f"  on lines  {top(m['snap'], 8)}")
    print(f"  row gaps  {top(m['gap'], 8)} | median {m['gap_ms_median']:.0f} ms")
    print(f"  runs      typical {m['run_median']} rows, long {m['run_p90']}, longest {m['run_max']} | "
          f"rows inside runs of 8+: {m['rows_in_runs_8']:.0f}%, 32+: {m['rows_in_runs_32']:.0f}%, 96+: {m['rows_in_runs_96']:.0f}%")
    print(f"  shape     keeps going for typically {m['stretch_median_s']:.1f} s, long {m['stretch_p90_s']:.1f} s, longest {m['stretch_max_s']:.0f} s "
          f"({m['time_in_stretches']:.0f}% of the song) | fastest tenth of row gaps {m['row_gap_p10_ms']:.0f} ms, fastest quarter {m['row_gap_p25_ms']:.0f} ms")
    print(f"  bursts    finer than 1/4: {m['rows_in_bursts']:.0f}% of rows, {m['bursts_per_min']:.0f} per minute, typically {m['burst_median']} rows, "
          f"long {m['burst_p90']}, longest {m['burst_max']} | spacing changes {m['spacing_changes_per_beat']:.2f} times per beat")
    print(f"  chords    " + " ".join(f"{n}:{v:.0f}%" for n, v in m["chord"].items()) + f" | {m['notes_per_row']:.2f} notes per row")
    print(f"  jacks     {m['jack_share']:.0f}% of notes repeat the column of the row before (median {m['jack_ms_median']:.0f} ms; "
          f"{m['jack_fast_share']:.0f}% of notes at 110 ms or faster) | runs of 2: {m['jack_len']['2']}, 3-4: {m['jack_len']['3-4']}, "
          f"5+: {m['jack_len']['5+']}, longest {m['jack_len_max']} | fastest column repeat {m['column_repeat_p1']:.0f} ms (1%), {m['column_repeat_p10']:.0f} ms (10%)")
    print(f"  movement  {top(m['moves'], 5)} (from {m['single_triples']} single-note triples in runs)")
    if m["ln_share"] >= 0.5:
        print(f"  holds     {m['ln_share']:.0f}% of objects, median {m['ln_beats_median']:.2f} beats | something held {m['held_1']:.0f}% of the time, "
              f"2+ held {m['held_2']:.0f}%, 3+ held {m['held_3']:.0f}% | taps under a hold {m['taps_under_holds']:.0f}% | "
              f"release to next press: median {m['release_gap_median_beats']:.2f} beats, {m['release_tight']:.0f}% within 1/4")
    else:
        print("  holds     none")


def main():
    paths = sys.argv[1:]
    index = {}
    if not paths:
        paths = sorted(glob.glob(os.path.join(RESEARCH, "examples", "*.osu")))
        with open(os.path.join(RESEARCH, "examples", "index.json"), encoding="utf-8") as fh:
            index = {e["id"]: e for e in json.load(fh)}
    for path in paths:
        chart = parse(path)
        m = measure(chart)
        f = chart["fields"]
        name = f"{f.get('Artist', '?')} - {f.get('Title', '?')} [{f.get('Version', '?')}] ({chart['keys']}K)"
        base = os.path.basename(path)[:-4]
        entry = index.get(int(base.rsplit("-", 1)[-1])) if base.rsplit("-", 1)[-1].isdigit() else None
        extra = f"   <- your example of: {entry['category']}; site says {entry['site_stars']:.2f} stars" if entry else ""
        show(name, m, extra)


if __name__ == "__main__":
    main()
