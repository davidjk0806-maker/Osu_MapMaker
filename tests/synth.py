"""Drum tracks with a known rhythm, for checking that the listening finds every hit
on the snap it was played on. Nothing here is music; it only has to be unambiguous."""

import numpy as np

SR = 22050


def _kick(rng):
    t = np.arange(int(0.11 * SR)) / SR
    return np.sin(2 * np.pi * (48.0 * t + 55.0 * (1 - np.exp(-t * 28.0)) / 28.0 * 2.2)) * np.exp(-t * 26.0)


def _snare(rng):
    t = np.arange(int(0.09 * SR)) / SR
    return (0.7 * rng.standard_normal(len(t)) + 0.5 * np.sin(2 * np.pi * 190.0 * t)) * np.exp(-t * 38.0)


def _hat(rng):
    t = np.arange(int(0.03 * SR)) / SR
    noise = np.diff(rng.standard_normal(len(t) + 1))        # crude high-pass
    return 0.5 * noise * np.exp(-t * 160.0)


def _tom(rng):
    t = np.arange(int(0.06 * SR)) / SR
    return (np.sin(2 * np.pi * 240.0 * t) + 0.25 * rng.standard_normal(len(t))) * np.exp(-t * 55.0)


SOUNDS = {"K": _kick, "S": _snare, "H": _hat, "T": _tom}


def render(events, seconds, seed=0):
    """events: [(ms, sound, loudness)] -> mono float32 at SR."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * SR) + SR)
    for ms, sound, loud in events:
        wave = SOUNDS[sound](rng) * loud
        wave[-64:] *= np.linspace(1.0, 0.0, 64)             # no click where the sound stops
        i = int(round(ms * SR / 1000.0))
        out[i:i + len(wave)] += wave[:len(out) - i]
    out += 0.002 * rng.standard_normal(len(out))            # a noise floor
    return (0.6 * out / np.abs(out).max()).astype(np.float32)


def beat_times(parts, offset=500.0):
    """When each beat falls for a tempo that moves. parts: [(beats, BPM at the start, BPM
    at the end)]; within a part the tempo slides evenly from one to the other."""
    times = [offset]
    for beats, start, end in parts:
        for i in range(beats):
            times.append(times[-1] + 60000.0 / (start + (end - start) * i / beats))
    return np.array(times)


def band(beats, seconds):
    """Something to play along with a groove on the same `beats`: a bass note on every
    beat and a lead note held over every two -> (bass, lead), mono float32 at SR."""
    bass, lead = np.zeros(int(seconds * SR) + SR), np.zeros(int(seconds * SR) + SR)

    def note(out, start, end, hz, loud, overtones):
        t = np.arange(int((end - start) * SR / 1000.0)) / SR
        wave = sum(np.sin(2 * np.pi * hz * n * t) / n for n in overtones)
        wave *= np.minimum(1.0, t / 0.006) * np.minimum(1.0, (t[-1] - t) / 0.02) * np.exp(-1.5 * t)
        i = int(round(start * SR / 1000.0))
        out[i:i + len(wave)] += loud * wave[:len(out) - i]

    roots = (55.0, 55.0, 73.42, 65.41)                      # A, A, D, C: a bar each
    tune = (440.0, 523.25, 587.33, 659.25, 587.33, 523.25, 440.0, 392.0)
    for i in range(len(beats) - 1):
        note(bass, beats[i], beats[i] + 0.8 * (beats[i + 1] - beats[i]), roots[(i // 4) % 4], 0.5, (1, 2))
        if i % 2 == 0 and i + 2 < len(beats):
            note(lead, beats[i], beats[i] + 0.9 * (beats[i + 2] - beats[i]),
                 tune[(i // 2) % len(tune)], 0.25, (1, 2, 3))
    return bass.astype(np.float32), lead.astype(np.float32)


def melody(notes, seconds, attack=15.0):
    """A lead line that comes in softly, as an instrument does: notes = [(start ms,
    length ms, Hz)] -> mono float32 at SR."""
    out = np.zeros(int(seconds * SR) + SR)
    for start, length, hz in notes:
        t = np.arange(int(length * SR / 1000.0)) / SR
        wave = sum(np.sin(2 * np.pi * hz * n * t) / n for n in (1, 2, 3))
        wave *= np.minimum(1.0, t / (attack / 1000.0)) * np.minimum(1.0, (t[-1] - t) / 0.03)
        i = int(round(start * SR / 1000.0))
        out[i:i + len(wave)] += 0.25 * wave[:len(out) - i]
    return out.astype(np.float32)


def groove(bpm, bars, offset=500.0, fills=(), swing=0.0, sloppy=0.0, seed=0, beats=None):
    """A rock beat with ornaments: kick on 1 and 3, snare on 2 and 4, a hat on every 1/4
    line. `fills` lists (bar, beat, fractions of the beat, sound) for extra hits, e.g.
    (2, 3, (1/8, 3/8), "S"). `sloppy` is the spread (ms) of every hit around its place.
    `beats`, the time of every beat, replaces the steady `bpm` (see beat_times).
    Returns (events, truth) where truth = [(ms, fraction of the beat)] for the fills."""
    rng = np.random.default_rng(seed)
    if beats is None:
        beats = offset + np.arange(bars * 4 + 1) * 60000.0 / bpm
    bars = min(bars, (len(beats) - 1) // 4)
    events, truth = [], []

    def add(ms, sound, loud):
        events.append((ms + (rng.normal(0.0, sloppy) if sloppy else 0.0), sound, loud))

    for bar in range(bars):
        for b in range(4):
            start, beat = beats[bar * 4 + b], beats[bar * 4 + b + 1] - beats[bar * 4 + b]
            add(start, "K" if b % 2 == 0 else "S", 1.0)
            for q in (1, 2, 3):
                add(start + q * beat / 4, "H", 0.5)
    for bar, b, fractions, sound in fills:
        start, beat = beats[bar * 4 + b], beats[bar * 4 + b + 1] - beats[bar * 4 + b]
        for f in fractions:
            add(start + f * beat, sound, 0.8)
            truth.append((start + f * beat, f))
    return sorted(events), sorted(truth)
