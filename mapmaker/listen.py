"""What is actually played and sung: drum hits, held notes, and how full each bar is."""

import bisect
import collections
import math

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .audio import SR
from .timing import FRAME_MS, N_FFT, attack_slope, env_mean, flux, stft_power

# Positions in a chart are whole numbers of these per beat. It divides evenly by every
# snap a chart might use (1/2 to 1/9, 1/12, 1/16, 1/24, 1/32), so notes on different
# snaps can share one song and still be compared exactly.
RES = 10080
CLEAR = 40.0                  # ms a hit between the lines must keep from other hits
MIN_HOLD_MS = 330.0           # shortest sung note worth holding
MAX_HOLD_MS = 2700.0          # longest single hold
SNAPS = (1, 2, 3, 4, 6, 8, 12, 16, 5, 7, 9, 24, 32)     # rows per beat, plainest first


def snap_of(position):
    """The plainest snap a position sits on, as rows per beat (4 = a 1/4 line)."""
    return next((d for d in SNAPS if position * d % RES == 0), RES)


def raw_hits(stem, apart=(40.0,)):
    """Percussive hits in a stem -> [(start_ms, kind, strength)], start sample-accurate.
    kind: K = kick (mostly bass), S = snare / clap, H = something lighter.
    One list per entry of `apart`, the closest (ms) two hits in that list may be."""
    power = stft_power(stem)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    env = flux(librosa.feature.melspectrogram(S=power, sr=SR, n_mels=40))
    env = env / (np.percentile(env, 99.5) + 1e-9)
    slope = attack_slope(stem, SR)
    low = power[(freqs >= 30) & (freqs < 150)].sum(axis=0)
    mid = power[(freqs >= 150) & (freqs < 1500)].sum(axis=0)
    high = power[freqs >= 3000].sum(axis=0)
    out = []
    for ms in apart:
        peaks, _ = find_peaks(env, height=0.25, prominence=0.15,
                              distance=max(1, int(ms / FRAME_MS)))
        hits = []
        for p in peaks:
            t = p * FRAME_MS
            lo = max(0, int((t - 25.0) * SR / 1000.0))
            hi = min(len(slope), int((t + 15.0) * SR / 1000.0))
            if hi - lo < 8:
                continue
            start = (lo + int(np.argmax(slope[lo:hi]))) * 1000.0 / SR
            w = slice(int(start / FRAME_MS), int((start + 60.0) / FRAME_MS))
            lo_e, mid_e, hi_e = low[w].sum(), mid[w].sum(), high[w].sum()
            total = lo_e + mid_e + hi_e + 1e-12
            kind = "K" if lo_e / total > 0.5 else ("S" if mid_e / total > 0.35 else "H")
            hits.append((start, kind, float(env[p])))
        out.append(hits)
    return out


def melodic_notes(stem, fmin, fmax):
    """Notes of a single-voice stem as (start_ms, end_ms, midi pitch), read off its
    pitch track. Works for the vocal stem and for the bass."""
    hop = 256
    fms = hop / SR * 1000.0
    f0, voiced, _ = librosa.pyin(stem, fmin=fmin, fmax=fmax, sr=SR,
                                 frame_length=2048, hop_length=hop)
    midi = librosa.hz_to_midi(f0)
    rms = librosa.feature.rms(y=stem, frame_length=1024, hop_length=hop)[0][:len(midi)]
    gate = np.percentile(rms, 95) * 0.08
    active = voiced & (rms > gate) & np.isfinite(midi)
    # a new syllable on an unchanged pitch still shows as a jump in loudness
    level = np.log(gaussian_filter1d(rms, 1.5) + gate * 0.5)
    rise = np.zeros(len(level))
    rise[3:] = np.maximum(level[3:] - level[:-3], 0.0)
    syllables = {int(i) - 2 for i in find_peaks(rise, height=0.35,
                                                distance=max(1, int(110.0 / fms)))[0]}

    notes, cur = [], None               # cur = [start frame, pitches so far]

    def close(end):
        if cur and (end - cur[0]) * fms >= 70.0:
            notes.append((cur[0] * fms, end * fms, float(np.median(cur[1]))))

    i = 0
    while i < len(midi):
        if not active[i]:
            j = i
            while j < len(midi) and not active[j]:
                j += 1
            # a dropout under 45 ms inside one pitch is still the same note
            if (cur and (j - i) * fms <= 45.0 and j < len(midi)
                    and abs(midi[j] - np.median(cur[1][-8:])) < 0.8):
                i = j
                continue
            close(i)
            cur, i = None, j
            continue
        if cur is None:
            cur = [i, [midi[i]]]
        else:
            ahead = midi[i:i + 4][active[i:i + 4]]
            moved = (len(ahead) >= 3 and (i - cur[0]) * fms >= 70.0
                     and abs(np.median(ahead) - np.median(cur[1][-10:])) > 0.8)
            if moved or (i in syllables and (i - cur[0]) * fms >= 150.0):
                close(i)
                cur = [i, [midi[i]]]
            else:
                cur[1].append(midi[i])
        i += 1
    close(len(midi))
    return notes


def listen(stems, mix, progress=None):
    """Everything heard in the song, in milliseconds and independent of any grid, so
    it can be cached and re-used for every key count, style and star rating."""
    step = progress or (lambda f: None)
    loud = float(np.sqrt(np.mean(mix ** 2))) + 1e-9
    share = {k: float(np.sqrt(np.mean(v ** 2))) / loud for k, v in stems.items()}
    no_drums = share["drums"] < 0.08
    # with hardly any drums, the hits of everything that isn't the voice stand in
    # `hits` keeps hits at least 40 ms apart, which is all a 1/4 grid can use; `fine`
    # lets them come closer, for rolls and grace notes between the lines
    hits, fine = raw_hits(stems["drums"] + stems["bass"] + stems["other"] if no_drums
                          else stems["drums"], (40.0, 18.0))
    # where anything that is not a drum starts a note: what a bar without drums is
    # charted from
    melody = raw_hits(stems["other"] + stems["vocals"] + stems["bass"])[0]
    step(0.25)
    # the notes to hold: the voice, or the lead instrument of an instrumental
    lead = melodic_notes(stems["vocals"], 70, 1000) if share["vocals"] >= 0.03 else []
    lead_from = "voice"
    if len(lead) < len(mix) / SR / 6.0 and share["other"] >= 0.03:
        lead, lead_from = melodic_notes(stems["other"], 80, 1600), "instrument"
    step(0.65)
    bass = melodic_notes(stems["bass"], 30, 400) if share["bass"] >= 0.03 else []
    step(0.95)
    band = (np.sqrt(stft_power(stems["bass"]).mean(axis=0))
            + np.sqrt(stft_power(stems["other"]).mean(axis=0)))
    level = np.sqrt(stft_power(mix).mean(axis=0))
    return {"hits": hits, "fine": fine, "melody": melody, "no_drums": no_drums, "lead": lead,
            "lead_from": lead_from,
            "bass": bass, "band": band.astype(np.float32), "level": level.astype(np.float32)}


def between(fine, T0, beat, tpb, k_min, k_max, slow=None):
    """Hits that fall between two grid lines, each put on the snap it was played on:
    ({position: (kind, strength, rows per beat)}, {rows per beat: hits found}).

    Most of what a detector finds between the lines is not a note: a hit played a
    little late, the second crack of a clap, the same hit found twice, a stretch where
    the song runs a touch off the grid. Measured against hand-made charts of eleven
    songs (research/tools/compare.py), nine in ten hits found midway between two lines were
    charted by the mapper too, but hardly any of those found close to a line. So a
    hit counts only if
      - no other hit of any weight is within CLEAR ms of it,
      - it lands within 8 ms of a snap,
      - that snap is CLEAR ms or more from the lines on both sides, or the hit is one
        of a triplet whose other note between the lines was heard as well,
      - and the snap holds clearly more hits across the song than stray hits would.
    `slow` is for a song whose tempo changes and has been straightened (timing.Clock):
    how many of the song's ms one ms lasts around a time, the ms above being the song's."""
    real = slow or (lambda at: 1.0)
    tick, line = beat / tpb, RES // tpb
    per_ms = RES / beat
    reach = 8.0 * per_ms
    guard = max(12.0, 0.2 * tick)       # this near a line, a hit was meant for the line
    spots = {}                          # places inside one beat -> their snap
    for d in (2, 3, 4, 6, 8):
        for j in range(1, d):
            if (j * RES // d) % line:
                spots.setdefault(j * RES // d, d)
    if not spots:
        return {}, {}
    grid = sorted(spots)

    def off_line(p):                    # how far a place is from the nearest line
        return min(p % line, line - p % line)

    times = [h[0] for h in fine]
    found = {}                          # position -> (kind, strength, snap), before the tests
    spread = np.zeros(max(1, int((tick - 2 * guard) // 4.0)))   # where in a line gap, 4 ms bins
    for i, (start, kind, strength) in enumerate(fine):
        k = int(round((start - T0) / tick))
        wide = real(start)
        if abs(start - T0 - k * tick) <= max(12.0 / wide, 0.2 * tick):
            continue
        lo = bisect.bisect_left(times, start - CLEAR / wide)
        hi = bisect.bisect_right(times, start + CLEAR / wide)
        if any(j != i and fine[j][2] >= 0.5 * strength for j in range(lo, hi)):
            continue                    # two detections of one sound, or a flam
        spread[max(0, min(len(spread) - 1, int(((start - T0) % tick - guard) // 4.0)))] += 1
        b = math.floor((start - T0) / beat)
        x = (start - T0 - b * beat) * per_ms
        p = min(grid, key=lambda g: abs(g - x))
        pos = b * RES + p
        if abs(p - x) > reach / wide or not k_min * line <= pos <= k_max * line:
            continue
        if pos not in found or strength > found[pos][1]:
            found[pos] = (kind, strength, spots[p])
    placed, count = {}, collections.Counter()
    for pos, hit in found.items():
        if off_line(pos) < CLEAR * per_ms / real(T0 + pos / per_ms):
            # close to a line: only as part of a triplet, 1/6 or 1/3 of a beat from
            # another hit between the lines
            if hit[2] not in (3, 6) or not any(
                    pos + step in found and found[pos + step][2] in (3, 6)
                    for step in (-RES // 3, -RES // 6, RES // 6, RES // 3)):
                continue
        placed[pos] = hit
        count[hit[2]] += 1
    # stray hits fall anywhere in the gap between two lines; real ones pile up on a few
    # places. The typical 4 ms of the gap says how many hits a snap would catch by chance.
    room = collections.Counter(spots[p] for p in grid)
    chance = float(np.median(spread)) / 4.0 * 16.0 / tpb       # per place of a snap in a beat
    heard = {d: n for d, n in count.items() if n >= 3.0 * chance * room[d] + 5.0}
    return {p: hit for p, hit in placed.items() if hit[2] in heard}, heard


def on_pulse(i, tpb, meter):
    """Is line `i` of a bar on the pulse a melody is counted in: the beats and half
    beats (in triplet meters the eighths, or the beat and its swung off-beat)?"""
    if meter == "swing":
        return i % 3 != 1
    if meter == "compound":
        return i % 2 == 0
    return (2 * i) % tpb == 0


def _keeps_time(found, bars, pulses, beats, lines, bar_len):
    """Where does the melody keep time? What an onset detector finds in a voice, a pad
    or a reverb tail is often no note at all, and such finds fall anywhere: as many of
    them near a beat as chance puts there. So each bar is judged by the onsets of the
    five bars around it (the clear ones, if there are enough): 2 if distinctly more
    of them start on the beats and half beats than chance would have it, or right on
    the lines (a melody running in 1/4 notes or faster), 1 if that is so only for the
    beats, 0 otherwise.
    found: (bar, strength, starts within 25 ms of a beat or half beat, ... of a beat,
    how many ms of the song one ms there lasts, starts within 12 ms of a line);
    pulses, beats, lines: how many beat-and-half-beat lines, beats and lines a bar has."""
    by_bar = {}
    for item in found:
        by_bar.setdefault(item[0], []).append(item)

    def stands_out(hits, n, chance):
        return n >= 8 and hits / n >= max(1.6 * chance, chance + 3.0 * math.sqrt(chance * (1.0 - chance) / n))

    out = {}
    for b in bars:
        around = [item for j in range(b - 2, b + 3) for item in by_bar.get(j, ())]
        clear = [item for item in around if item[1] >= 0.5]
        use = clear if len(clear) >= 8 else around
        if len(use) < 8:
            out[b] = 0
            continue
        ms = bar_len * sum(item[4] for item in use) / len(use)      # a bar, in the song's own time
        on_pulse_ = stands_out(sum(item[2] for item in use), len(use), min(0.9, pulses * 50.0 / ms))
        on_line = stands_out(sum(item[5] for item in use), len(use), min(0.9, lines * 24.0 / ms))
        on_beat = stands_out(sum(item[3] for item in use), len(use), min(0.9, beats * 50.0 / ms))
        out[b] = 2 if on_pulse_ or on_line else 1 if on_beat else 0
    return out


def _snap_notes(raw, T0, tick, step, k_min, k_max):
    """Held notes onto every `step`-th grid line; two syllables on one line are one note."""
    notes = []
    for on, off, pitch in raw:
        s = step * int(round((on - T0) / (step * tick)))
        e = max(s + step, step * int(round((off - T0) / (step * tick))))
        if not k_min <= s <= k_max - step:
            continue
        if notes and s <= notes[-1][0]:
            notes[-1][1] = max(notes[-1][1], e)
            continue
        if notes:
            notes[-1][1] = max(notes[-1][0] + step, min(notes[-1][1], s))
        notes.append([s, min(e, k_max), pitch])
    return notes


def _lean(notes, width):
    """{start: (end, pitch, column the note leans to)}: where its pitch sits among its
    neighbours, 0 = leftmost column .. width = rightmost."""
    out = {}
    pitches = [n[2] for n in notes]
    for i, (s, e, pitch) in enumerate(notes):
        near = pitches[max(0, i - 6):i + 7]
        lo, hi = min(near), max(near)
        out[s] = (e, pitch, width / 2 if hi - lo < 2.0 else width * (pitch - lo) / (hi - lo))
    return out


def _split(values):
    """Threshold between a quiet and a loud group of bars (Otsu), or None when the
    song has no such split."""
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) < 12:
        return None
    best, cut = 0.0, None
    for i in range(max(4, len(v) // 12), len(v) - max(4, len(v) // 12) + 1):
        lo, hi = v[:i], v[i:]
        between = len(lo) * len(hi) * (hi.mean() - lo.mean()) ** 2
        if between > best and hi.mean() - lo.mean() >= 2.0:
            best, cut = between, (v[i - 1] + v[i]) / 2.0
    return cut


def prepare(raw, dur_ms, T0, beat, tpb, keys, meter="straight", slow=None):
    """Put what was heard on the grid: drum hits, held notes, song sections.
    `tpb` is grid lines per beat; a charting bar is 16 lines, or 12 in triplet meters.
    Lines and bars are counted in lines; where things sit ("positions": the keys of
    drums, sung, bass and stabs, and their ends) is counted in RES-ths of a beat, so
    that a note may also sit between two lines. `slow`: see between()."""
    real = slow or (lambda at: 1.0)
    tick = beat / tpb                   # ms per line
    line = RES // tpb                   # positions per line
    bar = 16 if meter == "straight" else 12
    bar_len = bar * tick
    k_min = -int(T0 // tick)
    k_max = int((dur_ms - 150.0 - T0) // tick)
    bars = list(range(k_min // bar, k_max // bar + 1))

    drums = {}
    for start, kind, strength in raw["hits"]:
        k = int(round((start - T0) / tick))
        if abs(start - T0 - k * tick) * real(start) > 12.0 or not k_min <= k <= k_max:
            continue                    # a flam or bleed from another stem, not a grid hit
        if k * line not in drums or strength > drums[k * line][1]:
            drums[k * line] = (kind, strength)
    off_line, heard = between(raw.get("fine", ()), T0, beat, tpb, k_min, k_max, slow)
    # `tune`: where a note of the melody starts, for charting the bars without drums.
    # An instrument or a voice comes in more softly than a drum and mappers follow it
    # on the beats and half beats (research/tools/breaks.py), so there it may be 25 ms from
    # the line; on the lines between it has to be as exact as a drum hit.
    melody, tune, found = {}, {}, []
    for start, _, strength in raw.get("melody", ()):
        k = int(round((start - T0) / tick))
        wide = real(start)
        off = abs(start - T0 - k * tick) * wide
        if off <= 12.0 and k_min <= k <= k_max:
            melody[k * line] = max(strength, melody.get(k * line, 0.0))
        near = min((j for j in range(k - tpb, k + tpb + 1) if on_pulse(j % bar, tpb, meter)),
                   key=lambda j: abs(start - T0 - j * tick))
        close = abs(start - T0 - near * tick) * wide <= 25.0
        found.append((int((start - T0) // bar_len), strength, close,
                      close and near % tpb == 0, wide, off <= 12.0))
        if off > 12.0:                  # not on a line: the nearest beat or half beat, if close
            if not close:
                continue
            k = near
        if k_min <= k <= k_max:
            tune[k * line] = max(strength, tune.get(k * line, 0.0))
    # ... but only where the melody keeps time (see _keeps_time): on beats and half
    # beats where it does, on the beats alone (and only its clear notes) where just
    # those stand out, and not at all where its notes fall anywhere
    trust = _keeps_time(found, bars, sum(on_pulse(i, tpb, meter) for i in range(bar)),
                        bar // tpb, bar, bar_len)
    for k in list(tune):
        level, beat_line = trust.get(k // line // bar, 0), k // line % tpb == 0
        if level < 2 and not (level == 1 and beat_line and tune[k] >= 0.5):
            del tune[k]

    # held notes go on lines at least ~1/10 s apart: singing is loose, nobody means finer
    steps = {"straight": (1, 2, 4, 8), "compound": (1, 2, 6), "swing": (1,)}[meter]
    step = next((n for n in steps if n * tick >= 105.0), steps[-1])
    def held(notes, width):
        found = _lean(_snap_notes(notes, T0, tick, step, k_min, k_max), width)
        return {s * line: (e * line, pitch, lean) for s, (e, pitch, lean) in found.items()}

    sung = held(raw["lead"], keys - 1)
    bass = held(raw["bass"], (keys - 1) * 0.35)

    # sections. A bar without drums is calm (in a song that has drums at all; in one
    # that mostly doesn't, only its near-silent bars are). Otherwise how full the bar
    # sounds (whole mix plus the band, in dB) splits the quieter verses from the choruses.
    band, level = raw["band"], raw["level"]
    loud_band, loud_level = np.percentile(band, 90), np.percentile(level, 95)

    def db(x, ref):
        return 20.0 * math.log10(max(x, 1e-3 * ref) / ref)

    strong, energy, release = {}, {}, {}
    q = bar // 4
    for b in bars:
        t = T0 + b * bar_len
        strong[b] = sum(1 for i in range(bar)
                        if drums.get((b * bar + i) * line, ("", 0.0))[1] >= 0.45)
        energy[b] = (db(env_mean(level, t, t + bar_len), loud_level)
                     + 0.5 * db(env_mean(band, t, t + bar_len), loud_band))
        # a chord that rings from the bar line and is cut off mid-bar becomes a held note
        first = env_mean(band, t, t + q * tick)
        release[b] = next((r for r in range(q, bar - q + 1, 2 if bar == 16 else 1)
                           if env_mean(band, t + r * tick, t + (r + q) * tick) < 0.3 * first),
                          None) if first > 0.25 * loud_band else None
    active = [b for b in bars if strong[b] >= 2]
    drummed = len(active) >= 0.5 * len(bars)
    if not drummed:
        floor = float(np.median([energy[b] for b in bars])) - 12.0
        active = [b for b in bars if energy[b] >= floor]
    cut = _split([energy[b] for b in active])
    busy = 0.6 * float(np.median([strong[b] for b in active])) if drummed and active else 0.0
    live, section, stabs = set(active), {}, {}
    for b in bars:
        full = cut is None or energy[b] >= cut
        if b not in live:
            section[b] = "calm"
        elif full and (strong[b] >= busy or section.get(b - 1) == "chorus"):
            section[b] = "chorus"
        else:
            section[b] = "verse"
        if release[b] is not None:
            stabs[b * bar * line] = (b * bar + release[b]) * line
    # A song's loudest stretches (a drop) are its high point, however few drum hits
    # they have. `lift`: how far each full bar stands above the song's usual full level,
    # 0 .. 1, in songs where some bars clearly do (2 dB or more), and not for a bar
    # that does so alone.
    lift, full = {}, [b for b in bars if section[b] == "chorus"]
    if len(full) >= 8:
        usual = float(np.median([energy[b] for b in full]))
        top = float(np.percentile([energy[b] for b in full], 95))
        if top - usual >= 2.0:
            above = {b: min(1.0, max(0.0, (energy[b] - usual) / (top - usual))) for b in full}
            for b in full:
                lift[b] = max(min(above[b], above.get(b - 1, 0.0)), min(above[b], above.get(b + 1, 0.0)))
    # `rank`: how loud each chorus bar's four-bar phrase is among the song's choruses,
    # 0 (the quietest) .. 1 (the loudest), for styles that chart the quieter ones lighter
    rank, phrases = {}, {}
    for b in full:
        phrases.setdefault(b // 4, []).append(energy[b])
    if len(phrases) >= 3:
        order = sorted(phrases, key=lambda p: float(np.mean(phrases[p])))
        for b in full:
            rank[b] = order.index(b // 4) / (len(order) - 1)
    return {"T0": T0, "beat": beat, "line_ms": tick, "line": line, "tpb": tpb,
            "meter": meter, "bar": bar, "bar_len": bar_len,
            "k_min": k_min, "k_max": k_max, "bars": bars, "keys": keys,
            "min_hold": math.ceil(MIN_HOLD_MS / tick) * line,
            "max_hold": int(round(MAX_HOLD_MS / tick)) * line,
            "drums": drums, "between": off_line, "heard": heard, "melody": melody, "tune": tune,
            "in_time": trust, "lift": lift, "rank": rank,
            "sung": sung, "bass": bass, "section": section, "stabs": stabs}
