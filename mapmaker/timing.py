"""Tempo, offset and downbeat, measured from the hits themselves."""

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .audio import SR, MapError

N_FFT = 512                   # 23 ms window: short, so hits stay sharp in time
HOP = 64
FRAME_MS = HOP / SR * 1000.0  # ~2.9 ms per analysis frame
CHROMA_HOP = 512
CHROMA_MS = CHROMA_HOP / SR * 1000.0


# ------------------------------------------------------------------ helpers

def smooth3(env):
    out = env.copy()
    out[1:] = np.maximum(out[1:], env[:-1])
    out[:-1] = np.maximum(out[:-1], env[1:])
    return out


def env_at(env, t_ms, frame_ms=FRAME_MS):
    idx = np.rint(np.asarray(t_ms, dtype=float) / frame_ms).astype(int)
    return env[np.clip(idx, 0, len(env) - 1)]


def env_mean(env, t0_ms, t1_ms):
    lo = max(0, int(np.floor(t0_ms / FRAME_MS)))
    hi = min(len(env), int(np.ceil(t1_ms / FRAME_MS)))
    return float(env[lo:hi].mean()) if hi > lo else 0.0


def stft_power(y):
    return np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2


def flux(power, lag=3):
    """Onset envelope: how fast each band's level (in dB) is rising, summed over bands."""
    db = 10.0 * np.log10(np.maximum(power, 1e-12))
    db = np.maximum(db, db.max() - 80.0)
    rise = np.zeros_like(db)
    rise[:, lag:] = db[:, lag:] - db[:, :-lag]
    return np.maximum(rise, 0.0).mean(axis=0)


def onset_envelope(power):
    env = flux(librosa.feature.melspectrogram(S=power, sr=SR, n_mels=40))
    return env / (np.percentile(env, 99) + 1e-9)


def attack_slope(mono, sr):
    """Per-sample steepness of the amplitude rise over 3 ms. Its peak marks the
    first sample of a drum hit, which is far more exact than an FFT frame."""
    pre = np.diff(mono.astype(np.float64), prepend=float(mono[0]))
    n = len(pre)
    h = max(1, int(round(0.0015 * sr)))
    cs = np.concatenate([[0.0], np.cumsum(pre * pre)])
    i = np.arange(n)
    amp = np.sqrt((cs[np.minimum(i + h, n)] - cs[np.maximum(i - h, 0)]) / (2 * h))
    slope = np.zeros(n)
    slope[h:n - h] = amp[2 * h:] - amp[:n - 2 * h]
    return slope


# ------------------------------------------------------------------- timing

def fold_scan(env, bpms, bins=96):
    """Fold the envelope at each candidate beat length; the true tempo stacks every
    beat on one spot. Returns (score, time of a beat) per candidate."""
    t = np.arange(len(env)) * FRAME_MS
    scores, phases = np.zeros(len(bpms)), np.zeros(len(bpms))
    for n, bpm in enumerate(bpms):
        beat = 60000.0 / bpm
        idx = np.minimum(((t % beat) * (bins / beat)).astype(int), bins - 1)
        prof = (np.bincount(idx, weights=env, minlength=bins)
                / np.maximum(np.bincount(idx, minlength=bins), 1))
        j = int(np.argmax(prof))
        scores[n], phases[n] = prof[j], (j + 0.5) / bins * beat
    return scores, phases


def fold_search(env, bpms):
    scores, phases = fold_scan(env, bpms)
    j = int(np.argmax(scores))
    return float(bpms[j]), float(phases[j])


def fit_grid(times, weights, bpm, off, lock_bpm):
    """Robust straight-line fit of hit times to a 1/4 grid (outliers get weight 0)."""
    tick = 15000.0 / bpm
    for cut in (30.0, 20.0, 14.0, 10.0, 10.0, 10.0, 10.0):
        n = np.rint((times - off) / tick)
        r = times - off - n * tick
        w = weights * np.clip(1.0 - (r / cut) ** 2, 0.0, None) ** 2
        if np.count_nonzero(w) < 8:
            break
        if lock_bpm:
            off += float((w * r).sum() / w.sum())
        else:
            nm, tm = (w * n).sum() / w.sum(), (w * times).sum() / w.sum()
            tick = float((w * (n - nm) * (times - tm)).sum() / (w * (n - nm) ** 2).sum())
            off = float(tm - tick * nm)
    n = np.rint((times - off) / tick)
    r = times - off - n * tick
    return 15000.0 / tick, off, r


def sharp_hits(env, slope, sr):
    """Sample-accurate start of each strong hit -> (times ms, weights, envelope peak ms)."""
    peaks, _ = find_peaks(env, height=np.percentile(env, 85),
                          distance=max(1, int(50.0 / FRAME_MS)))
    times, sharp, frame_t = [], [], []
    for p in peaks:
        t = p * FRAME_MS
        lo = max(0, int((t - 30.0) * sr / 1000.0))
        hi = min(len(slope), int((t + 15.0) * sr / 1000.0))
        if hi - lo < 16:
            continue
        i = lo + int(np.argmax(slope[lo:hi]))
        times.append(i * 1000.0 / sr)
        sharp.append(slope[i])
        frame_t.append(t)
    times, sharp, frame_t = np.array(times), np.array(sharp), np.array(frame_t)
    if len(times) < 40:
        raise MapError("Found too few hits to time this - is the file silent or very short?")
    return times, np.clip(sharp / (np.median(sharp) + 1e-12), 0.1, 3.0), frame_t


def _in_range(beat_ms):
    """Fold a beat length into 120-240 BPM, the range straight songs are charted in."""
    while beat_ms <= 250.5:
        beat_ms *= 2
    while beat_ms > 501.0:
        beat_ms /= 2
    return beat_ms


def _stack(env, beat_ms):
    """How well the beats pile up when the song is folded at this beat length."""
    return fold_scan(env, np.array([60000.0 / beat_ms]))[0][0]


def find_timing(env, slope, sr, hint=None):
    """BPM, grid lines per beat, and the time of one beat - measured from the hits.

    Without a hint every grid from 120 to 240 BPM is tried (a slower song simply
    counts at double speed). The few that stack the beats best are each fitted
    against the sample-accurate hit times, and the grid the hits actually sit on wins.
    Then the beat is read off that grid. Usually it is four lines long ("straight").
    When the beats pile up clearly better six lines apart than four or eight, the song
    is in threes: "swing" (three lines to a beat) or "compound" (six, as in 6/8)."""
    times, weights, frame_t = sharp_hits(env, slope, sr)
    wide = gaussian_filter1d(env, 20.0 / FRAME_MS)
    mid = gaussian_filter1d(env, 8.0 / FRAME_MS)
    bpms = (np.arange(hint * 0.95, hint * 1.05, 0.02) if hint
            else np.arange(119.0, 241.0, 0.02))
    scores, _ = fold_scan(wide, bpms)
    picks = []
    for i in np.argsort(-scores):
        if all(abs(bpms[i] / bpms[j] - 1.0) > 0.01 for j in picks):
            picks.append(int(i))
        if len(picks) == (1 if hint else 5):
            break
    best = None
    for i in picks:
        bpm, off = fold_search(mid, np.arange(bpms[i] - 0.06, bpms[i] + 0.06, 0.004))
        bpm, off = fold_search(smooth3(env), np.arange(bpm - 0.012, bpm + 0.012, 0.001))
        free_bpm, off, r = fit_grid(times, weights, bpm, off, lock_bpm=False)
        # share of the hits on this grid, beyond what any grid this fine catches by chance
        on_grid = weights[np.abs(r) < 8.0].sum() / weights.sum() - 16.0 * free_bpm / 15000.0
        if best is None or on_grid > best[0]:
            best = (on_grid, free_bpm, off)
    line = 15000.0 / best[1]            # ms between grid lines
    off = best[2]

    # which multiple of the grid is the beat?
    feel, lines, waltz = "straight", 4, False
    if not hint:
        six, three = _stack(wide, 6 * line), _stack(wide, 3 * line)
        if six > 1.10 * max(_stack(wide, 4 * line), _stack(wide, 8 * line)):
            fast_ok = 3 * line > 250.5 and six <= 1.25 * three
            feel, lines = ("swing", 3) if fast_ok else ("compound", 6)
        else:
            lines = int(round(_in_range(4 * line) / line))
            waltz = _stack(wide, 12 * line) > 1.10 * max(_stack(wide, 8 * line),
                                                          _stack(wide, 16 * line))
    free_bpm = 60000.0 / (lines * line)
    # studio tracks are almost always whole BPMs; keep a fractional one only if
    # the hits clearly ask for it
    whole = abs(free_bpm - round(free_bpm)) <= 0.02
    bpm = float(round(free_bpm)) if whole else round(free_bpm, 3)
    beat = 60000.0 / bpm
    _, off, r = fit_grid(times, weights, 60000.0 / (beat / lines) / 4, off, lock_bpm=True)
    if lines != 4:
        # the beat is not the grid's own "beat": find which grid line the beats stack on
        _, phase = fold_scan(wide, np.array([bpm]))
        off += round((phase[0] - off) / (beat / lines)) * (beat / lines)

    good = np.abs(r) < 10.0
    lag = float(np.median((frame_t - times)[good])) if good.any() else 0.0
    spread = float(np.sqrt(np.mean(r[good] ** 2))) if good.any() else 0.0
    edges = np.linspace(0, times.max(), 7)
    parts = [np.median(r[good & (times >= a) & (times < b)])
             for a, b in zip(edges[:-1], edges[1:])
             if np.count_nonzero(good & (times >= a) & (times < b)) >= 6]
    drift = float(max(parts) - min(parts)) if len(parts) >= 2 else 0.0
    info = {"hits": int(good.sum()), "of": int(len(r)), "spread": spread, "drift": drift,
            "free_bpm": float(free_bpm), "feel": feel, "waltz": bool(waltz),
            "on_grid": float(weights[good].sum() / weights.sum())}
    # a straight song charts on 4 lines per beat whatever grid won the fit
    return bpm, (4 if feel == "straight" else lines), off, lag, info


# ------------------------------------------------- a tempo that changes

LINE_LO, LINE_HI = 62.5, 125.0          # ms from one 1/4 line to the next at 240 and at 120 BPM


def _period_scores(times, weights, centres, half, periods):
    """For the hits within `half` ms of each centre: how well they sit on a grid of each
    period (1 = every hit on it, whatever its phase)."""
    out = np.zeros((len(centres), len(periods)))
    for n, c in enumerate(centres):
        near = np.abs(times - c) <= half
        if near.sum() < 6:
            continue
        t, w = times[near], weights[near]
        turns = np.exp(2j * np.pi * t[None, :] / periods[:, None])
        out[n] = np.abs((turns * w[None, :]).sum(axis=1)) / w.sum()
    return out


def _steady_path(scores, periods, free=0.004, stiff=4.0):
    """The path through the windows that scores best while moving smoothly: up to
    `free` (as a ratio) from one window to the next costs nothing, a jump costs a lot.
    A period and its double, the two ends of the octave, count as neighbours."""
    log = np.log(periods)
    apart = np.abs(log[:, None] - log[None, :])
    apart = np.minimum(apart, np.log(2.0) - apart)
    cost = stiff * np.clip((apart - free) / 0.03, 0.0, 1.0) + np.where(apart > 0.03 + free, 2.0, 0.0)
    best, back = -scores[0].copy(), []
    for row in scores[1:]:
        total = best[:, None] + cost
        back.append(np.argmin(total, axis=0))
        best = total.min(axis=0) - row
    path = [int(np.argmin(best))]
    for step in reversed(back):
        path.append(int(step[path[-1]]))
    return periods[np.array(path[::-1])]


def _whole_path(path):
    """The path knows each tempo only up to doubling or halving, and comes back with
    every period inside one octave. Here a tempo that moves by less than a quarter
    keeps its lines (crossing the end of the octave is not a jump), and each run of
    such moves is then put in the octave that keeps most of it between LINE_LO and
    LINE_HI, and next to that the jumps between runs small."""
    path = path.copy()
    runs = [0]                          # where each run starts
    for n in range(1, len(path)):
        same = path[n] * 2.0 ** np.rint(np.log2(path[n - 1] / path[n]))
        if abs(np.log(same / path[n - 1])) < np.log(1.25):
            path[n] = same
        else:
            runs.append(n)
    runs.append(len(path))
    shifts = (0.25, 0.5, 1.0, 2.0, 4.0)
    lo, hi = np.log(LINE_LO), np.log(LINE_HI)
    best, back = None, []
    for a, b in zip(runs, runs[1:]):
        at = np.log(path[a:b])[None, :] + np.log(shifts)[:, None]
        out = (np.clip(lo - at, 0.0, None) + np.clip(at - hi, 0.0, None)).sum(axis=1)
        if best is None:
            best = out
            continue
        jump = np.abs(np.log(path[a] * np.array(shifts))[None, :]
                      - np.log(path[a - 1] * np.array(shifts))[:, None])       # [before, after]
        total = best[:, None] + 0.5 * jump
        back.append(np.argmin(total, axis=0))
        best = total.min(axis=0) + out
    pick = [int(np.argmin(best))]
    for step in reversed(back):
        pick.append(int(step[pick[-1]]))
    for (a, b), k in zip(zip(runs, runs[1:]), pick[::-1]):
        path[a:b] *= shifts[k]
    return path


def _lines_across(times, weights, at, before, to, after):
    """How many lines lie between a line at `at` and a line at `to` when the tempo
    steps from one period to another somewhere between them: the old grid is run
    forward and the new one back, and the step is put on the line where the two
    meet (within 12 ms) with the most hits on the right grid either side. None if
    they never meet, as when the tempo slides instead of stepping."""
    near = (times > at) & (times < to)
    t, w = times[near], weights[near]
    on_old = np.abs((t - at) / before - np.rint((t - at) / before)) * before <= 8.0
    on_new = np.abs((to - t) / after - np.rint((to - t) / after)) * after <= 8.0
    best, count = None, None
    for a in range(1, int((to - at) / before) + 1):
        meet = at + a * before
        b = int(round((to - meet) / after))
        apart = abs(to - b * after - meet)
        if b < 1 or apart > 12.0:
            continue
        score = w[on_old & (t <= meet)].sum() + w[on_new & (t >= meet)].sum() - 0.1 * apart
        if best is None or score > best:
            best, count = score, a + b
    if best is None or best < 0.7 * w.sum():        # too many hits on neither grid: a slide
        return None
    return count


def follow_lines(times, weights, dur_ms, step=500.0, half=1500.0, good=0.5):
    """Follow a tempo that moves: the time (ms) of every 1/4 line, start to end.

    The song is looked at through three-second windows half a second apart. In each,
    the period the hits fit best is found; a smooth path through those gives a first
    guess, and a straight-line fit of the window's hits makes it exact. Every window
    whose hits really do sit on its grid then pins one line near its centre, and the
    lines from one pin to the next are spaced as a tempo sliding evenly from the one
    window's to the other's would space them. So a ramp is followed, and a stretch
    without drums, or a change too quick to catch, is bridged.
    Returns (line times, share of the windows that could be used), or (None, 0.0)."""
    centres = np.arange(half, dur_ms - half, step)
    periods = np.exp(np.arange(np.log(LINE_LO), np.log(LINE_HI), 0.001))
    if len(centres) < 4:
        return None, 0.0
    path = _whole_path(_steady_path(_period_scores(times, weights, centres, half, periods),
                                    periods))

    pins = []                           # (line number, its time, the period there)
    for c, period in zip(centres, path):
        near = np.abs(times - c) <= half
        if near.sum() < 8:
            continue
        t, w = times[near], weights[near]
        turn = (w * np.exp(2j * np.pi * t / period)).sum()
        off = np.angle(turn) / (2 * np.pi) * period
        off += np.round((c - off) / period) * period
        bpm, fitted, r = fit_grid(t, w, 15000.0 / period, off, lock_bpm=False)
        if abs(np.log(15000.0 / bpm / period)) <= 0.02:     # else the fit ran away: keep the scan's
            period, off = 15000.0 / bpm, fitted
        else:
            r = t - off - np.rint((t - off) / period) * period
        fits = np.abs(r) < 8.0
        if w[fits].sum() < good * w.sum():
            continue
        # pin a line among the hits that fit: next to a tempo step the window's centre
        # may lie where this grid no longer holds
        off += np.round((np.average(t[fits], weights=w[fits]) - off) / period) * period
        if not pins:
            pins.append((0, off, period))
            continue
        number, at, before = pins[-1]
        ahead = int(round(2.0 * (off - at) / (before + period)))
        if (off - at) * abs(1.0 / before - 1.0 / period) > 0.3:
            ahead = _lines_across(times, weights, at, before, off, period) or ahead
        if ahead >= 1:
            pins.append((number + ahead, off, period))
    if len(pins) < 4:
        return None, 0.0

    lines = []
    number, at, period = pins[0]
    back = int(np.ceil(at / period))
    lines += [at - n * period for n in range(back, 0, -1)]
    for (n0, t0, p0), (n1, t1, p1) in zip(pins, pins[1:]):
        count = n1 - n0
        h = np.arange(count) / count    # an even slide from one period to the other
        lines += list((2 * h**3 - 3 * h**2 + 1) * t0 + (h**3 - 2 * h**2 + h) * count * p0
                      + (3 * h**2 - 2 * h**3) * t1 + (h**3 - h**2) * count * p1)
    number, at, period = pins[-1]
    lines += [at + n * period for n in range(int(np.ceil((dur_ms - at) / period)) + 1)]
    lines = np.array(lines)
    if np.any(np.diff(lines) <= 0.4 * LINE_LO):
        return None, 0.0
    return lines, len(pins) / len(centres)


def _on_lines(lines, times, reach):
    """For each hit, the number of the line nearest to it, and whether it is within reach."""
    at = np.clip(np.searchsorted(lines, times), 1, len(lines) - 1)
    at = np.where(np.abs(times - lines[at - 1]) < np.abs(times - lines[at]), at - 1, at)
    return at, np.abs(times - lines[at]) <= reach


def _knot_times(knots, n, t, w):
    """A broken line through the points (line number n, time t): straight between the
    knots, no jumps. Solves the times at the knots, robustly -> (times, cost)."""
    k = np.array(knots, dtype=float)
    seg = np.clip(np.searchsorted(k, n, side="right") - 1, 0, len(k) - 2)
    u = (n - k[seg]) / (k[seg + 1] - k[seg])
    rows = np.arange(len(n))
    design = np.zeros((len(n), len(k)))
    design[rows, seg], design[rows, seg + 1] = 1.0 - u, u
    use = w
    for cut in (30.0, 20.0, 14.0, 10.0, 8.0, 8.0):
        root = np.sqrt(use)
        at = np.linalg.lstsq(design * root[:, None], t * root, rcond=None)[0]
        r = t - design @ at
        use = w * np.clip(1.0 - (r / cut) ** 2, 0.0, None) ** 2
    return at, float((w * np.minimum(r * r, 64.0)).sum())


def tidy_lines(lines, times, weights, price=40.0, starts=None):
    """Straighten followed lines: a song's tempo mostly holds still and then steps, so
    the lines are refitted as a broken line through the hits, straight (one steady
    tempo) between knots. Knots start wherever the followed spacing has shifted; their
    times are solved together; then each knot is slid to the line where it fits best,
    two that sit close together are replaced by one, and any that buys less than
    `price` is dropped. With `starts` (the number of a line that starts a bar) the
    given knots are instead only nudged onto bar lines where that costs next to
    nothing. Returns [(line number, ms)] for the knots, ends included."""
    at, on = _on_lines(lines, times, 14.0)
    n, t, w = at[on].astype(float), times[on], weights[on]
    last = len(lines) - 1
    if len(n) < 20:
        return [(0, float(lines[0])), (last, float(lines[-1]))]
    centre = float(t.mean())
    t = t - centre

    def cost(knots, lo, hi):
        """What a set of knots costs, counted from its knot `lo` to its knot `hi`."""
        part = knots[max(0, lo):hi + 1]
        near = (n >= part[0]) & (n <= part[-1])
        return _knot_times(part, n[near], t[near], w[near])[1] if near.sum() > len(part) else 0.0

    if starts is not None:
        knots, bar = starts
        knots = [k for k, _ in knots]
        for j in range(1, len(knots) - 1):
            to = knots[j] + (bar - knots[j] + 8) % 16 - 8
            if to != knots[j] and abs(to - knots[j]) <= 2 and knots[j - 1] + 8 <= to <= knots[j + 1] - 8:
                moved = knots[:j] + [to] + knots[j + 1:]
                if cost(moved, j - 2, j + 2) - cost(knots, j - 2, j + 2) < 0.4 * price:
                    knots = moved
        at = _knot_times(knots, n, t, w)[0] + centre
        return [(int(k), float(x)) for k, x in zip(knots, at)]

    span = 8
    spacing = np.log((lines[2 * span:] - lines[:-2 * span]) / (2 * span))
    knots, held = [0], spacing[0]
    for i in range(1, len(spacing)):
        if abs(spacing[i] - held) > 0.003:
            if i + span - knots[-1] >= 8 and last - i - span >= 8 and ((n >= knots[-1]) & (n < i + span)).sum() >= 3:
                knots.append(i + span)
            held = spacing[i]
    knots.append(last)
    for _ in range(5):
        before = list(knots)
        for j in range(1, len(knots) - 1):          # slide each knot to where it fits best
            now, best = cost(knots, j - 2, j + 2), None
            for d in (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16):
                if knots[j - 1] + 8 <= knots[j] + d <= knots[j + 1] - 8:
                    moved = knots[:j] + [knots[j] + d] + knots[j + 1:]
                    c = cost(moved, j - 2, j + 2)
                    if c < now - 1e-6:
                        now, best = c, moved
            knots = best or knots
        j = 1
        while j < len(knots) - 2:                   # two knots close together: one will do
            if knots[j + 1] - knots[j] <= 24:
                now, best = cost(knots, j - 2, j + 3) + price, None
                for m in range(knots[j], knots[j + 1] + 1):
                    if knots[j - 1] + 8 <= m <= knots[j + 2] - 8:
                        one = knots[:j] + [m] + knots[j + 2:]
                        c = cost(one, j - 2, j + 2)
                        if c < now:
                            now, best = c, one
                if best:
                    knots = best
                    continue
            j += 1
        j = 1
        while j < len(knots) - 1:                   # a knot that buys too little goes
            without = knots[:j] + knots[j + 1:]
            if cost(without, j - 2, j + 1) - cost(knots, j - 2, j + 2) < price:
                knots = without
            else:
                j += 1
        if knots == before:
            break
    j = 0
    while j < len(knots) - 1 and len(knots) > 2:    # a stretch with next to no hits in it says nothing,
        inside = n[(n >= knots[j]) & (n < knots[j + 1])]
        edge = j == 0 or j == len(knots) - 2        # nor do a few hits at one end of the song's first or last
        if len(inside) < 4 or (edge and inside.max() - inside.min() < min(128, 0.5 * (knots[j + 1] - knots[j]))):
            knots = knots[:j] + knots[j + 1:] if 0 < j else knots[:1] + knots[2:]
            j = max(0, j - 1)
        else:
            j += 1
    knots = knots[:-1] + [last] if knots[-1] != last else knots
    at = _knot_times(knots, n, t, w)[0] + centre
    return [(int(k), float(x)) for k, x in zip(knots, at)]


def lines_through(knots, count):
    """The time of lines 0 .. count-1 on the broken line through the knots."""
    k = np.array([x[0] for x in knots], dtype=float)
    at = np.array([x[1] for x in knots])
    n = np.arange(count, dtype=float)
    out = np.interp(n, k, at)
    out = np.where(n < k[0], at[0] + (n - k[0]) * (at[1] - at[0]) / (k[1] - k[0]), out)
    return np.where(n > k[-1], at[-1] + (n - k[-1]) * (at[-1] - at[-2]) / (k[-1] - k[-2]), out)


def in_time(hits, lines, dur_ms):
    """Where a grid holds: the song in ten-second stretches, each marked by whether
    most of its drum hits sit on the lines -> (share of the stretches that do, the
    times in seconds where they do as [start, end] runs). `lines` is the time of every
    line, or (time of one line, ms from line to line) for a steady grid."""
    starts = np.array([h[0] for h in hits])
    if isinstance(lines, tuple):
        first, step = lines
        while step < 60.0:              # a grid this fine catches hits by chance
            step *= 2.0
        off = np.abs((starts - first) / step - np.rint((starts - first) / step)) * step
    else:
        off = np.abs(starts - lines[_on_lines(lines, starts, 0.0)[0]])
    runs, good, total = [], 0, 0
    for start in range(0, int(dur_ms), 10000):
        here = off[(starts >= start) & (starts < start + 10000)]
        if len(here) < 8:
            continue
        total += 1
        if (here <= 12.0).sum() >= 0.6 * len(here):
            good += 1
            if runs and runs[-1][1] == start // 1000:
                runs[-1][1] = start // 1000 + 10
            else:
                runs.append([start // 1000, start // 1000 + 10])
    return (good / total if total else 1.0), runs


def downbeat_among(low, chroma, beats, lag):
    """Which of every four beats is beat 1, from the kick drum and the chord changes,
    for beats that need not be evenly spaced."""
    kick_env = smooth3(smooth3(low))
    kick = env_at(kick_env, beats + lag)
    nov = chroma_novelty(chroma, beats, 2.0 * float(np.median(np.diff(beats))))
    score = kick / (kick.mean() + 1e-9) + nov / (nov.mean() + 1e-9)
    return int(np.argmax([score[p::4].mean() for p in range(4)]))


def follow_tempo(env, slope, sr, dur_ms, low, chroma):
    """Time a song whose tempo changes -> {"lines": the time of every 1/4 line,
    "knots": [(line number, ms)] where the tempo changes, "down": the number of a line
    that starts a bar, "used": share of the song the beat could be read from}, or None.

    The beat is followed from the hits (follow_lines), straightened into steady
    stretches (tidy_lines), and then the lines are sorted into beats and bars: the
    beat is where the hits weigh most, and a bar starts where the tempo changes, if
    the changes agree on that, or else where kick and chord changes say."""
    times, weights, frame_t = sharp_hits(env, slope, sr)
    followed, used = follow_lines(times, weights, dur_ms)
    if followed is None:
        return None
    knots = tidy_lines(followed, times, weights)
    lines = lines_through(knots, len(followed))
    if np.any(np.diff(lines) <= 0.4 * LINE_LO):
        return None
    at, on = _on_lines(lines, times, 10.0)
    loud = env_at(env, frame_t)         # how hard each hit is, not how sharp: kick and snare mark the beat
    beat = int(np.argmax([loud[on & (at % 4 == p)].sum() for p in range(4)]))
    lag = float(np.median((frame_t - times)[on])) if on.any() else 0.0
    down = beat + 4 * downbeat_among(low, chroma, lines[beat::4], lag)
    # tempo changes come on bar lines: let them say where bars start. Only a step from
    # one stretch of two bars or more to another counts; a slide is cut up anywhere.
    steps = [b[0] for a, b, c in zip(knots, knots[1:], knots[2:])
             if min(b[0] - a[0], c[0] - b[0]) >= 32]
    if len(knots) > 2:
        votes = np.bincount([k % 16 for k in steps], minlength=16).astype(float)
        votes = (votes + 0.5 * np.roll(votes, 1) + 0.5 * np.roll(votes, -1))[beat % 4::4]
        first, second = np.sort(votes)[::-1][:2]
        if first >= 1.0 and first >= 2.0 * second:
            down = beat % 4 + 4 * int(np.argmax(votes))
        knots = tidy_lines(followed, times, weights, starts=(knots, down))
        lines = lines_through(knots, len(followed))
    return {"lines": lines, "knots": knots, "down": int(down % 16), "used": float(used)}


class Clock:
    """A song whose tempo changes, set against a straightened copy of its time in which
    every 1/4 line is equally long (as long as where the song is fastest, so that
    nothing charted there comes out closer together than it was meant to). The song
    is charted in the straightened time, where one tempo holds, and the notes are
    then put back where they belong."""

    def __init__(self, knots, down, lines=None, **_):
        knots = [(int(k), float(t)) for k, t in knots]
        count = knots[-1][0] + 1 if lines is None else len(lines)
        self.parts = []                 # steady stretches: [first line, the line after, ms of the first, ms a line]
        for (k0, t0), (k1, t1) in zip(knots, knots[1:]):
            line = (t1 - t0) / (k1 - k0)
            # a whole number of BPM, if that moves no line of the stretch by more than 1.5 ms
            whole = 15000.0 / max(1.0, round(15000.0 / line))
            if abs(whole - line) * (k1 - k0) / 2.0 <= 1.5:
                t0, line = (t0 + t1) / 2.0 - whole * (k1 - k0) / 2.0, whole
            self.parts.append([k0, k1, t0, line])
        k0, _, t0, line = self.parts[0]
        self.parts[0][:3] = 0, self.parts[0][1], t0 - k0 * line
        self.parts[-1][1] = count
        self.lines = np.array([t0 + (n - k0) * line for k0, k1, t0, line in self.parts
                               for n in range(k0, k1)])
        self.line = float(np.percentile(np.diff(self.lines), 5))
        self.beat = 4.0 * self.line
        self.first = int(down) % 16     # the first line that starts a bar
        self.T0 = float(round(self.first * self.line))
        self.even = self.T0 + (np.arange(len(self.lines)) - self.first) * self.line

    @staticmethod
    def _carry(x, here, there):
        x = np.asarray(x, dtype=float)
        out = np.interp(x, here, there)
        before = there[0] + (x - here[0]) * (there[1] - there[0]) / (here[1] - here[0])
        after = there[-1] + (x - here[-1]) * (there[-1] - there[-2]) / (here[-1] - here[-2])
        return np.where(x < here[0], before, np.where(x > here[-1], after, out))

    def even_time(self, ms):
        """A time in the song -> the same moment in the straightened time."""
        return self._carry(ms, self.lines, self.even)

    def real_time(self, ms):
        """A time in the straightened time -> when that is in the song."""
        return self._carry(ms, self.even, self.lines)

    def stretch(self, ms):
        """How many ms of the song one ms of the straightened time lasts, around a time there."""
        i = int(min(max((ms - self.T0) // self.line + self.first, 0), len(self.lines) - 2))
        return float(self.lines[i + 1] - self.lines[i]) / self.line

    def straighten(self, raw):
        """What was heard in the song (see listen.listen), moved into the straightened time."""
        out = dict(raw)
        for name in ("hits", "fine", "melody"):
            if raw.get(name):
                at = self.even_time([h[0] for h in raw[name]])
                out[name] = [(float(t),) + tuple(h[1:]) for t, h in zip(at, raw[name])]
        for name in ("lead", "bass"):
            if raw.get(name):
                on = self.even_time([n[0] for n in raw[name]])
                off = self.even_time([n[1] for n in raw[name]])
                out[name] = [(float(a), float(b), n[2]) for a, b, n in zip(on, off, raw[name])]
        for name in ("band", "level"):
            frames = int(np.ceil(float(self.even_time(len(raw[name]) * FRAME_MS)) / FRAME_MS))
            there = self.real_time(np.arange(frames) * FRAME_MS) / FRAME_MS
            out[name] = np.interp(there, np.arange(len(raw[name])), raw[name]).astype(np.float32)
        return out

    def red_lines(self):
        """[(ms, beat length ms, whether a bar starts there)]: the red lines of the song.
        One where each steady stretch starts; the game counts its bars from there, so
        a stretch that starts inside a bar gets a second one on its first bar line.
        The first of all is put on a bar line, at or after the start of the audio."""
        out = []
        for n, (k0, k1, t0, line) in enumerate(self.parts):
            into = (k0 - self.first) % 16
            if n == 0:
                k0 = next((k for every in (16, 4, 1) for k in range(k1)
                           if (k - self.first) % every == 0 and t0 + k * line >= 0.0), 0)
                t0, into = t0 + k0 * line, (k0 - self.first) % 16
            out.append((t0, 4.0 * line, into == 0))
            if into and k0 + 16 - into < k1:
                out.append((t0 + (16 - into) * line, 4.0 * line, True))
        return out

    def tempo(self):
        """(the BPM held for longest, the lowest, the highest), leaving out stretches under two bars."""
        spans = ([((k1 - k0) * line, 15000.0 / line) for k0, k1, _, line in self.parts
                  if k1 - k0 >= 32] or [(1.0, 15000.0 / self.line)])
        spans = [(d, float(round(b)) if abs(b - round(b)) < 0.05 else round(b, 1)) for d, b in spans]
        time = {}
        for d, b in spans:
            time[b] = time.get(b, 0.0) + d
        return max(time, key=time.get), min(b for _, b in spans), max(b for _, b in spans)


def chroma_novelty(chroma, times, span):
    n = chroma.shape[1]
    cs = np.concatenate([np.zeros((chroma.shape[0], 1)), np.cumsum(chroma, axis=1)], axis=1)

    def seg(t0, t1):
        a = int(np.clip(t0 / CHROMA_MS, 0, n))
        b = int(np.clip(t1 / CHROMA_MS, 0, n))
        if b <= a:
            return None
        v = (cs[:, b] - cs[:, a]) / (b - a)
        return v / (np.linalg.norm(v) + 1e-9)

    out = np.zeros(len(times))
    for i, t in enumerate(times):
        p, q = seg(t - span, t), seg(t, t + span)
        if p is not None and q is not None:
            out[i] = 1.0 - float(p @ q)
    return out


def find_downbeat(low, chroma, dur_ms, offset, beat, lag):
    """Pick which beat is beat 1, using kick drum + chord changes. The beat itself
    moves by half a beat only if the kicks clearly sit there instead (a song counted
    at half its speed can come out with the snare on the beat)."""
    n = int((dur_ms - offset) // beat) - 1
    if n < 16:
        return offset
    bt = offset + np.arange(n) * beat
    kick_env = smooth3(smooth3(low))
    kick, nov = env_at(kick_env, bt + lag), chroma_novelty(chroma, bt, 2 * beat)
    score = kick / (kick.mean() + 1e-9) + nov / (nov.mean() + 1e-9)
    on = [score[p::4].mean() for p in range(4)]
    half = bt + beat / 2
    shifted = (env_at(kick_env, half + lag) / (kick.mean() + 1e-9)
               + chroma_novelty(chroma, half, 2 * beat) / (nov.mean() + 1e-9))
    off = [shifted[p::4].mean() for p in range(4)]
    if max(off) > 1.25 * max(on):
        return offset + (int(np.argmax(off)) + 0.5) * beat
    return offset + int(np.argmax(on)) * beat
