"""Which style suits a song best: the "Recommended" choice.

The user asked for an option that "automatically chooses the best fit based on the
music ... it should obviously rely on bpm, but also rely on the instruments and
volume". Each style charts something in particular, so a song suits a style when it
has that something:

  - Stream: drums that keep going on back-to-back 1/4 notes through the loud parts
    (or, counting for less, a steady eighth-note beat, which the stream fills in).
  - Speed: fast notes between the beats (or 1/4 notes that are already that fast).
  - Tech: hits between the beats on more than one snap (1/8 and 1/6, say).
  - Chordjack and Speedjack: the same drum hit again on back-to-back rows of the
    style's grid, which is what a jack is made of; Chordjack from a loud, hard master,
    Speedjack also from a voice that sings one note again straight away (Mi Mi Mi).
  - LN and Coordination: a voice or lead that holds its notes. These two can chart any
    song, so they are also what a song gets when nothing else stands out;
    Coordination for a lighter rhythm and ordinary ratings (the user: "kinda of low
    bpm and star usually because it's more control than stamina or reading").

How loud the song is plays in too: a song that stays loud nearly all the way through
suits the stamina styles, one with quiet parts LN. All of this is measured in the
song's loud bars, beyond what onsets falling anywhere would give by chance, and the
style's own tempo range (pipeline._fit) weighs the result: a style fits a song best
at the middle of the row spacing it is meant for.

Checked against the eleven example songs whose mappers chose a style
(research/tools/recommend_check.py: six first, two more second, as of 2026-10-16). Two
limits show there: Tech is only recommended where the app hears its fast hits, and the
fast notes of the two Tech examples are in the voice and melody, which it does not hear
as notes; and two songs can suit Chordjack and Speedjack alike. Since 2026-10-16 Stream
and Speedjack fit most tempos (they pick their grid for it), so the tempo no longer
rules them out on its own."""

import math

import numpy as np

from .chart import STYLES, grid_for, suit_range
from .listen import MIN_HOLD_MS
from .timing import FRAME_MS

AUTO = "auto"                   # the style asked for when the app is to choose

# raw["level"] is the mix's spectrum magnitude per frame (listen.listen). Its RMS over a
# song stands 23.0 dB above the RMS of the mix itself, the same for every song
# (measured on 23 songs, within 0.2 dB; a 512-sample Hann window predicts 22.8).
LEVEL_DB = 23.0


def _ramp(x, lo, hi):
    """0 at `lo`, 1 at `hi`, straight in between."""
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def _hit(onsets, times, within=12.0):
    """Which of `times` have one of the sorted `onsets` within `within` ms."""
    if not len(onsets) or not len(times):
        return np.zeros(len(times), dtype=bool)
    i = np.searchsorted(onsets, times)
    before = np.abs(onsets[np.clip(i - 1, 0, len(onsets) - 1)] - times)
    after = np.abs(onsets[np.clip(i, 0, len(onsets) - 1)] - times)
    return np.minimum(before, after) <= within


def _beyond_chance(share, rows, step):
    """share(rows) beyond what the same rows shifted off the grid get: what onsets
    falling anywhere would put there. 0 = no more than chance, 1 = all of them."""
    if len(rows) < 2:
        return 0.0
    got = share(rows)
    chance = float(np.mean([share(rows + shift * step) for shift in (0.29, -0.29, 0.41, -0.41)]))
    return max(0.0, (got - chance) / (1.0 - chance + 1e-9))


def describe(raw, song):
    """What the recommendation is made from, all of it from what was heard and none of
    it from the rating: `raw` as listen.listen gives it (straightened, for a song whose
    tempo changes), `song` the same put on the song's own grid (pipeline.on_grid)."""
    T0, beat, own, bar_len = song["T0"], song["beat"], song["tpb"], song["bar_len"]
    tick, section = song["line_ms"], song["section"]
    loud = [b for b in song["bars"] if section[b] == "chorus"]
    if len(loud) < 8:
        loud = [b for b in song["bars"] if section[b] != "calm"]
    drums = np.array(sorted(h[0] for h in raw["fine"]))
    anything = np.array(sorted([h[0] for h in raw["fine"]] + [h[0] for h in raw["melody"]]))

    def rows(step):                     # every `step` ms through the loud bars
        n = int(round(bar_len / step))
        return np.array([T0 + b * bar_len + k * step for b in loud for k in range(n)])

    def on(onsets):
        return lambda at: float(_hit(onsets, at).mean())

    def pairs(onsets):                  # back-to-back rows that both have one
        def share(at):
            hit = _hit(onsets, at)
            return float((hit[:-1] & hit[1:]).mean())
        return share

    lines = rows(tick)
    half = beat / 2 if own == 4 else beat / 3       # the eighth notes (in threes: the triplets)
    d = {"beat": beat, "own": own,
         "runs": _beyond_chance(pairs(drums), lines, tick),
         "drive": _beyond_chance(pairs(drums), rows(half), half),
         "fill": _beyond_chance(on(anything), lines, tick)}

    # fast notes: on the rows Speed charts, about 57 ms apart
    if grid_for(STYLES["speed"], beat, own) == 2 * own:     # between the lines
        d["fast"] = _beyond_chance(on(anything), lines + tick / 2, tick / 2)
        d["fast_ms"], d["fast_between"] = tick / 2, True
    else:                                                   # the lines themselves, if quick enough
        d["fast"] = d["fill"] if tick <= 70.0 else 0.0
        d["fast_ms"], d["fast_between"] = tick, False

    # jacks: the same kind of drum hit (kick, snare, other) on back-to-back rows
    hits = sorted((h[0], h[1]) for h in raw["hits"] if h[2] >= 0.3)
    at = np.array([h[0] for h in hits])

    def same_kind(times):
        if not len(at):
            return 0.0
        i = np.clip(np.searchsorted(at, times), 1, len(at) - 1)
        nearest = np.where(np.abs(at[i - 1] - times) <= np.abs(at[i] - times), i - 1, i)
        kinds = [hits[j][1] if abs(at[j] - t) <= 12.0 else None for j, t in zip(nearest, times)]
        return sum(1 for a, b in zip(kinds, kinds[1:]) if a and a == b) / max(1, len(kinds) - 1)

    d["jack_ms"] = {}
    for name in ("chordjack", "speedjack"):
        style = STYLES[name]
        step = d["jack_ms"][name] = style.rows[1] * beat / grid_for(style, beat, own)
        d["jacks_" + name] = _beyond_chance(same_kind, rows(step), step)

    d["snaps"] = sorted(song["heard"])
    span = (song["k_max"] - song["k_min"]) * tick
    d["between"] = len(song["between"]) / max(1.0, span / 60000.0)    # per minute
    d["held"] = sum(e - s for s, e, _ in raw["lead"] if e - s >= MIN_HOLD_MS) / max(1.0, span)
    d["voice"] = raw["lead_from"] == "voice"
    # the voice (or lead) singing one note again straight away, as Mi Mi Mi's "mi mi mi"
    lead = sorted(raw["lead"])
    d["repeats"] = sum(1 for (s1, _, p1), (s2, _, p2) in zip(lead, lead[1:])
                       if abs(p1 - p2) < 0.6 and s2 - s1 <= 0.6 * beat) / max(1.0, span / 60000.0)

    # how loud: the master, and how much of the song stays near its loudest
    level = np.asarray(raw["level"], dtype=np.float64)
    d["loudness"] = 20.0 * math.log10(float(np.sqrt(np.mean(level ** 2))) + 1e-12) - LEVEL_DB
    frames = int(round(1000.0 / FRAME_MS))
    seconds = np.array([level[i:i + frames].mean() for i in range(0, len(level) - frames, frames)])
    if len(seconds):
        d["steady"] = float(np.mean(20.0 * np.log10(seconds / np.percentile(seconds, 95) + 1e-12) > -3.0))
    else:
        d["steady"] = 0.0
    return d


def _fit(style, d, felt, stars):
    """How well a style's row spacing suits the song's tempo: 1 at the middle of the
    spacing it is meant for at this rating (chart.suit_range), 0.7 at its edges, and
    falling off fast outside them. A style for any tempo gets 0.85."""
    span = suit_range(style, stars)
    if not span:
        return 0.85
    low, high, lines = span
    spacing = lines * felt / grid_for(style, felt, d["own"], stars)
    middle, half = math.sqrt(low * high), math.log(high / low) / 2.0
    off = abs(math.log(spacing / middle))
    if off <= half:
        return 1.0 - 0.3 * (off / half) ** 2
    return 0.7 * math.exp(-((off - half) / 0.06) ** 2)


# What each style needs from a song, how much of its fit that is, what else adds to it,
# and what any song gets: fit = any + (1 - any) * need * (share + sum of weight * other).
RECIPES = {
    "stream": ("streams", 0.6, (("fill", 0.2), ("steady", 0.2)), 0.0),
    "speed": ("fast", 0.7, (("steady", 0.3),), 0.0),
    "tech": ("mixed", 1.0, (), 0.0),
    "chordjack": ("jacks_chordjack", 0.3, (("hard", 0.5), ("steady", 0.2)), 0.0),
    "speedjack": ("jacky", 0.4, (("hard", 0.25), ("busy", 0.2), ("steady", 0.15)), 0.0),
    "ln": ("held", 0.7, (("quiet", 0.3),), 0.2),
    "coordination": ("held", 0.6, (("light", 0.4),), 0.2),
}


def _measures(d):
    """How clearly the song has each thing a style may need, 0..1, and how to say so."""
    busy = _ramp(d["fill"], 0.4, 0.8)
    steady = _ramp(d["steady"], 0.45, 0.85)
    snaps = ", ".join(f"1/{s}" for s in d["snaps"])
    jacks = "the same drum hit comes again on back-to-back rows {:.0f} ms apart"
    m = {
        "runs": (_ramp(d["runs"], 0.1, 0.5), "the drums keep going on back-to-back 1/4 notes in the loud parts"),
        "fill": (_ramp(d["fill"], 0.5, 0.85), "something plays on nearly every 1/4 note"),
        "busy": (busy, "something plays on most 1/4 notes"),
        "light": (1.0 - busy, "the rhythm around those notes is light"),
        "fast": (_ramp(d["fast"], 0.05, 0.3),
                 f"it has fast notes between the beats ({d['fast_ms']:.0f} ms apart)" if d["fast_between"]
                 else f"its 1/4 notes are already {d['fast_ms']:.0f} ms apart"),
        "mixed": (_ramp(len(d["snaps"]), 1, 3) * _ramp(d["between"], 3.0, 15.0),
                  f"it has hits between the beats on more than one snap ({snaps})"),
        "jacks_chordjack": (_ramp(d["jacks_chordjack"], 0.05, 0.3), jacks.format(d["jack_ms"]["chordjack"])),
        "jacks_speedjack": (_ramp(d["jacks_speedjack"], 0.05, 0.3), jacks.format(d["jack_ms"]["speedjack"])),
        # a loud master, in dBFS over the whole song: the two chordjack examples are
        # -7.4 and -7.0, the other examples -8.7 to -15.6
        "hard": (_ramp(d["loudness"], -13.0, -8.0), "it is mastered loud and hard"),
        "steady": (steady, "it stays loud nearly all the way through"),
        "quiet": (1.0 - steady, "it has quiet parts as well as loud ones"),
        "held": (_ramp(d["held"], 0.05, 0.4),
                 f"the {'voice' if d['voice'] else 'lead instrument'} holds its notes "
                 f"(about {100 * d['held']:.0f}% of the song)"),
        # a steady eighth-note beat counts for Stream too, though less than 1/4 runs: the
        # stream then follows the rest of the band between the drums
        "drive": (0.6 * _ramp(d["drive"], 0.6, 0.95),
                  "the drums keep a steady eighth-note beat through the loud parts"),
        # jacks can follow a voice that sings one note again and again, as in Mi Mi Mi
        # (17 times a minute; most songs 2 to 10)
        "repeats": (_ramp(d["repeats"], 8.0, 18.0),
                    f"the {'voice' if d['voice'] else 'lead'} sings the same note again straight "
                    f"away (about {d['repeats']:.0f} times a minute)"),
    }
    m["streams"] = max(m["runs"], m["drive"], key=lambda v: v[0])
    m["jacky"] = max(m["jacks_speedjack"], m["repeats"], key=lambda v: v[0])
    return m


def _evidence(name, m, stars):
    """How much of what a style charts the song has, 0..1, and the reasons worth
    giving for it: [(how much it counted, reason)]. A reason is given only where the
    song clearly has the thing: its main need from 0.3, and then the rest from 0.5."""
    need, share, others, anyway = RECIPES[name]
    have, said = m[need]
    rest = sum(w * m[o][0] for o, w in others)
    fit = anyway + (1.0 - anyway) * have * (share + rest)
    reasons = []
    if have >= 0.3:                             # the rest only say more about the main need
        reasons.append(((1.0 - anyway) * have * share, said))
        reasons += [((1.0 - anyway) * have * w * m[o][0], m[o][1]) for o, w in others if m[o][0] >= 0.5]
    if anyway:
        reasons.append((anyway, f"{STYLES[name].label} can chart any song"))
    if name == "coordination":                  # it is "low bpm and star usually"
        lower = 1.0 - 0.4 * _ramp(stars, 4.5, 6.0)
        fit, reasons = fit * lower, [(w * lower, r) for w, r in reasons]
    return fit, reasons


def rank(raw, song, felt, stars):
    """Every style, best fit first: [{"style", "label", "score", "fit", "why"}].
    `felt` is the beat the song is mostly in, in ms (pipeline.make_map's), `stars` the
    rating asked for: Speedjack's tempo depends on it, and Coordination is for ordinary
    ratings more than high ones."""
    d = describe(raw, song)
    m = _measures(d)
    out = []
    for name, style in STYLES.items():
        if name not in RECIPES:         # a style kept for the tests only
            continue
        fit = _fit(style, d, felt, stars)
        evidence, reasons = _evidence(name, m, stars)
        why = [r for w, r in sorted(reasons, key=lambda x: -x[0]) if w >= 0.05][:3]
        bpm = 60000.0 / felt
        if suit_range(style, stars) and fit >= 0.9:
            why.append(f"{bpm:.0f} BPM is right in its range")
        elif suit_range(style, stars) and fit >= 0.65:
            why.append(f"{bpm:.0f} BPM is inside its range")
        elif suit_range(style, stars) and fit >= 0.3:
            why.append(f"{bpm:.0f} BPM is a little outside its range")
        out.append({"style": name, "label": style.label, "score": round(fit * evidence, 4),
                    "fit": round(fit, 3), "why": why})
    out.sort(key=lambda r: -r["score"])
    return out


def explain(ranking):
    """The ranking in a sentence or two, for the map's card."""
    best, others = ranking[0], [r["label"] for r in ranking[1:3] if r["score"] >= 0.5 * ranking[0]["score"]]
    why = [r for r in best["why"] if not r.endswith("can chart any song")]
    if best["style"] in ("ln", "coordination") and not any("holds its notes" in r for r in why):
        # nothing else stood out: one of the two styles that suit any song
        text = (f"Nothing in this song stands out for a particular style (no drums running "
                f"through the loud parts, no fast notes, no repeated hits), so {best['label']}, "
                "which can chart any song")
    else:
        text = f"{best['label']} suits this song best"
    if why:
        text += ": " + (why[0] if len(why) == 1 else ", ".join(why[:-1]) + " and " + why[-1])
    text += "."
    if others:
        text += " Close behind: " + " and ".join(others) + "."
    return text
