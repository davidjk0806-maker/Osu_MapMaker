"""Turning what was heard into a chart: how many notes each grid line carries, which
columns they go to, and how dense it all has to be to hit a star rating.

Where a note sits is a "position": a whole number, `song["line"]` of them from one grid
line to the next. Rows of notes mostly sit on the lines, but nothing here needs them
to, so a chart can mix snaps. Spacing rules are therefore written as distances
between positions, with one line as the yardstick."""

import bisect
import itertools
import math
import random
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from .listen import MAX_HOLD_MS, MIN_HOLD_MS, on_pulse
from .stars import star_rating

# which hand plays each column (2 = thumb)
HANDS = {4: [0, 0, 1, 1], 5: [0, 0, 2, 1, 1], 6: [0, 0, 0, 1, 1, 1], 7: [0, 0, 0, 2, 1, 1, 1]}
BIGGEST_CHORD = {4: 3, 5: 3, 6: 4, 7: 4}
MOST_HOLDS = {4: 2, 5: 2, 6: 3, 7: 3}

# comfort ladder, strictest first: (longest run of fastest-allowed repeats on one
# column, columns a chord inside a run may reuse from the last chord, same on a drum
# accent, most grid lines in a row one hand may play on)
COMFORT = [(3, 1, 2, 4), (4, 1, 3, 4), (5, 2, 3, 5), (6, 2, 3, 6), (99, 4, 4, 99)]


@dataclass(frozen=True)
class Style:
    label: str
    blurb: str
    ln: float           # share of held sung notes charted as LNs (longest first)
    chord: float        # how fast chords grow as the level rises
    runs: tuple         # level at which 1/4 runs begin, and the span over which they fill
    breath: float       # level at which the breath before each accent starts to fill
    cap: int            # biggest chord, relative to the usual for the key count
    filler: int         # biggest chord on a line with no drum hit on it
    comfort: int        # first rung of the comfort ladder to try
    bass: bool          # hold the bass notes as well as the voice
    motifs: dict
    repeat: float = 125.0       # ms: a column is never asked to repeat faster than this,
    apart: int = 2              # nor sooner than this many lines (1 = the very next line)
    release: float = None       # ms from letting an LN go to pressing its column again
                                # (None: the same as `repeat`)
    detail: int = 0             # chart real hits between the lines down to this snap
                                # (8 = 1/8 notes ...; 0 = only what sits on a line)
    fills: tuple = (99.0, 1.0)  # level at which invented ornaments begin, the span over
                                # which they fill and, if given, the share of rows that
                                # get one at most
    ornaments: tuple = ()       # (name, weight) pairs the ornaments are drawn from
    verse: float = 0.6          # a verse (a bar with drums, quieter than the choruses) is
                                # charted at this share of a chorus's level
    contour: float = 0.0        # the quietest of the song's choruses are charted this much
                                # lighter than the loudest (by four-bar phrase, in between
                                # by how loud they are), so the rating is reached in the
                                # loudest and the rest plays easier, as in ranked maps
    hush: float = 0.0           # a bar without drums is charted at this share of a
                                # chorus's level, on the lines where the melody moves
                                # (0: such a bar carries only its real hits and sung notes)
    arc: bool = True            # phrases breathe: the first bar of four is thinner, the
                                # last one runs into the next
    layer: bool = True          # a sung note on a drum hit adds a note to its chord
    third: float = 0.0          # how much later than usual chords grow past two notes
    wide: bool = False          # chords may grow past two notes on any strong line with a
                                # kick or snare, not only the backbeat and the downbeat
    fill_at: float = 3.0        # how far chords must have grown before a line with no
                                # drum hit on it gets a second note
    thick: float = 99.0         # ... and before the lines in between, where runs go, do
    sustain: float = 0.0        # on lines closer together than this (ms) runs fill at
                                # most half a bar, so they stay short bursts
    gather: bool = False        # a bar's run grows backwards from the next downbeat as
                                # one piece, instead of from every accent at once
    phrase: int = 0             # charting bars from one breather to the next (0 = none):
                                # the rest of the beat after such a downbeat is left empty
    gradual: bool = False       # lines fill a few bars at a time as the level rises,
                                # not in every bar at once (finer steps in the rating)
    harshest: int = 4           # last rung of the comfort ladder the style may use
    steady: bool = False        # never leave a planned row out for comfort's sake
    hand_cost: tuple = (0.0, 0.0, 1.0, 3.0, 5.0)    # cost of a row for a hand that has
                                # already played the 0, 1, 2, 3, 4+ rows before it
    one_hand: float = 1.0       # cost of a chord played by one hand alone
    chord_hands: float = 1.0    # share of `hand_cost` a chord pays (in handstream both hands
                                # are kept busy anyway, and the full cost would put every
                                # jump on the hand that rested)
    repeat_cost: tuple = (0.4, 2.5, 1.5)    # cost of repeating a column as fast as allowed:
                                # the first time the motif asks for it, otherwise, and
                                # what each further repeat in a row adds
    short: tuple = None         # when set, `ln` is the share of all notes held as short
                                # long notes, and these are the odds of one lasting a
                                # quarter of a beat, half a beat, a whole beat
    long: bool = False          # with `short`: what is sung (and with `bass`, the bass) is
                                # still held for as long as it sounds
    suits: tuple = None         # (from ms, to ms, lines): how far apart its rows should
                                # be for the style to play as meant, and how many lines
                                # apart they sit (None: any tempo will do)
    quick: tuple = None         # (stars, ms per star): with `suits`, rows as close as its
                                # `from` suit only that rating; for each star less they
                                # should come this much further apart (see suit_range)
    reach: tuple = None         # (ms x stars, the same with quads): with `suits`, how far
                                # apart its rows may be and still reach a rating: for a
                                # rating of s, at most this / s ms (see suit_range)
    hold_taps: tuple = None     # (share, quarter, half, whole): besides the sung notes,
                                # this share of the taps is held too, for a quarter, half
                                # or whole beat by those odds, so a song with little that
                                # is sung still gets its long notes
    beside: float = 0.0         # cost of a note on a hand that is holding a long note
    noodles: tuple = None       # (share, quarter, half, three quarters, whole): with `weave`,
                                # the chart is written to train the fingers apart (see
                                # _independence): shields and reverse shields around the
                                # holds, and some phrases in long notes only, where this
                                # share of the single notes is held for that many beats by
                                # those odds. `ln` then runs from no long notes (0) through
                                # the style's own mix (0.5) to long notes only (1)
    phrases: tuple = None       # (stars, share, stars, share): with `noodles`, the share of
                                # the phrases written in long notes only, rising from the
                                # first rating to the second (see rated())
    mix: float = None           # ... that share, for the rating being charted
    weave: tuple = None         # (share, one beat, a beat and a half, two beats): whenever
                                # nothing is held, a note on a beat or half beat is held
                                # this often, for that long by those odds, each in a
                                # column of the other hand than the hold before it, while
                                # the other notes go on in the columns left
    crux: float = 0.0           # how much fuller the song's loudest stretches are charted
                                # than its other choruses (1.5: up to 2.5 times as full)
    tuned: tuple = (4,)         # the key counts it has been tuned for
    rows: tuple = None          # (ms, lines): the row spacing the style is built for and
                                # how many lines apart its rows sit; the grid is then
                                # chosen to come closest (None: the song's own 1/4 grid,
                                # and a finer one only if the rating needs it). With
                                # `quick` the grid is chosen by the rating instead (see
                                # grid_for), and may then be in triplets
    finer: bool = True          # without `rows`: a grid twice as fine may be used where the
                                # song's own can't reach the rating
    ladder: tuple = None        # (from, step): chords grow by these steps of `chord` x
                                # level, the same for every kind of line, offset by how
                                # strong the line is (see _rung): first jumps on the
                                # accents, then on every line, hands on the accents, and
                                # so on, so that a stream gets denser as the rating rises
                                # while its spacing stays the same
    quads: float = None         # chords of four only from this rating up
    jacks: tuple = None         # ((stars, presses), ...): the most presses in a row one
                                # column may take at the fastest spacing allowed, from
                                # each rating up (None: as the comfort ladder allows)
    chain: int = 99             # ... that number, for the rating being charted
    reuse: tuple = None         # a row may press at most this many of the columns of the row
                                # just before (at the fastest spacing allowed), at each rung
                                # of the comfort ladder (None: as the ladder says)
    shields: tuple = (0.5, 0.5) # with `noodles`: how often a hold gets a shield, and how
                                # often a reverse shield, in the phrases written that way
    shield_cost: float = 0.0    # cost of starting a hold in a column pressed the line
                                # before (a shield that was not asked for)
    inverse: tuple = None       # (stars, share, stars, share): with `noodles`, the share of
                                # the phrases written as inverse, from the first rating to
                                # the second (None: a third of the long-notes-only ones)
    inv: float = None           # ... that share, for the rating being charted
    vary: int = 0               # lines: a chord costs more where the same chord was played
                                # this close before (0: only the chord just before, within
                                # two lines)
    steps: bool = False         # 4K, with `ladder`: each beat's chords are laid out as in
                                # the user's stream examples (_figures), and each row's
                                # columns follow on from the rows before as there (_STEPS)


_MIXED = {"chorus": (["roll", "split", "zigzag", "trill", "anchor"], [3, 3, 2, 2, 2]),
          "verse": (["zigzag", "split", "roll", "trill", "anchor"], [3, 2, 2, 2, 1]),
          "calm": (["zigzag"], [1])}
_WOVEN = {"chorus": (["roll", "split", "zigzag", "trill", "anchor"], [3, 3, 2, 2, 2]),
          "verse": (["zigzag", "split", "roll", "trill"], [3, 2, 2, 2]),
          "calm": (["zigzag"], [1])}
_FLOWING = {"chorus": (["roll", "split", "zigzag", "trill"], [4, 3, 3, 1]),
            "verse": (["zigzag", "roll", "split"], [3, 3, 2]),
            "calm": (["zigzag"], [1])}
_JACKS = {"chorus": (["jack", "roll", "split", "zigzag"], [4, 2, 2, 1]),
          "verse": (["jack", "zigzag", "roll"], [3, 2, 2]),
          "calm": (["zigzag"], [1])}
_RUNS = {"chorus": (["roll", "split", "zigzag"], [4, 3, 3]),
         "verse": (["zigzag", "roll", "split"], [3, 3, 2]),
         "calm": (["zigzag"], [1])}
_HEAVY = {"chorus": (["anchor", "trill", "split", "zigzag"], [3, 2, 2, 1]),
          "verse": (["anchor", "zigzag", "split"], [2, 2, 2]),
          "calm": (["zigzag"], [1])}
# "flow": each single note in a stream is drawn the way Palette GAMMA's mapper wrote
# them (4K only; elsewhere these fall back to _RUNS)
_SPEEDY = {"chorus": (["flow"], [1]), "verse": (["flow"], [1]), "calm": (["zigzag"], [1])}

# 4K, the three columns just played -> how often each column came next, counted in the
# single-note streams of Palette GAMMA's Affection, Gratitude and Abstract (Cadmium-113,
# research/reference/888774) and their mirror images. Those streams roll three columns
# and turn, or switch hands, far more than they roll all four: a full roll is 2-5% of
# their four-note stretches. A column comes back two notes later about half the time
# (1231), but almost never one note later (121), so they hold no plain trills.
_FLOW = {(0, 1, 0): (0, 2, 4, 6), (0, 1, 2): (45, 0, 0, 47), (0, 1, 3): (39, 0, 45, 0),
         (0, 2, 0): (0, 2, 5, 4), (0, 2, 1): (43, 0, 2, 75), (0, 2, 3): (26, 50, 2, 0),
         (0, 3, 0): (0, 1, 4, 3), (0, 3, 1): (30, 0, 75, 9), (0, 3, 2): (26, 52, 0, 1),
         (1, 0, 1): (3, 0, 2, 4), (1, 0, 2): (1, 39, 0, 53), (1, 0, 3): (4, 44, 32, 0),
         (1, 2, 0): (0, 48, 2, 83), (1, 2, 1): (0, 0, 1, 1), (1, 2, 3): (46, 43, 1, 0),
         (1, 3, 0): (0, 55, 67, 7), (1, 3, 1): (0, 0, 1, 2), (1, 3, 2): (52, 40, 0, 1),
         (2, 0, 1): (1, 0, 40, 52), (2, 0, 2): (2, 1, 0, 0), (2, 0, 3): (7, 67, 55, 0),
         (2, 1, 0): (0, 1, 43, 46), (2, 1, 2): (1, 1, 0, 0), (2, 1, 3): (83, 2, 48, 0),
         (2, 3, 0): (0, 32, 44, 4), (2, 3, 1): (53, 0, 39, 1), (2, 3, 2): (4, 2, 0, 3),
         (3, 0, 1): (1, 0, 52, 26), (3, 0, 2): (9, 75, 0, 30), (3, 0, 3): (3, 4, 1, 0),
         (3, 1, 0): (0, 2, 50, 26), (3, 1, 2): (75, 2, 0, 43), (3, 1, 3): (4, 5, 2, 0),
         (3, 2, 0): (0, 45, 0, 39), (3, 2, 1): (47, 0, 0, 45), (3, 2, 3): (6, 4, 2, 0)}

# 4K, the two rows just played -> how often each row came next (columns counted from 1),
# in the streams of the three maps the user listed as streams: Rise Of The Chaos
# Wizards [Dark Galactic Skies], ETERNAL DRAIN [Eternal] and Palette GAMMA [Affection]
# (research/tools/steps.py, which prints this). Only stretches that repeat no column
# from one row to the next, and their mirror images. They run in staircases that turn
# (1 2 3 then 4 or back), with a jump at the foot of the stair or in it ([12] 3 4,
# 1 [23] 4), and seldom come back to a column on the next row but one.
_STEPS = {
    ("1", "2"): {"3": 190, "34": 84, "4": 74, "134": 5, "13": 3, "14": 2},
    ("1", "3"): {"2": 146, "4": 115, "24": 35, "14": 7, "1": 5, "124": 4, "12": 3},
    ("1", "4"): {"3": 121, "2": 66, "23": 20, "13": 9, "12": 3, "1": 3},
    ("1", "23"): {"4": 25},
    ("1", "24"): {"3": 53},
    ("1", "34"): {"2": 59, "12": 11, "1": 8},
    ("1", "234"): {"1": 9},
    ("2", "1"): {"3": 189, "4": 116, "34": 73, "234": 6, "24": 4, "2": 4},
    ("2", "3"): {"4": 192, "1": 130, "14": 74, "2": 38, "24": 21, "12": 9, "124": 7},
    ("2", "4"): {"3": 154, "1": 76, "13": 44, "2": 4, "12": 4, "23": 2, "123": 2},
    ("2", "13"): {"4": 44, "2": 12},
    ("2", "14"): {"3": 75, "2": 5},
    ("2", "34"): {"1": 61, "2": 19, "12": 11},
    ("2", "134"): {"2": 10},
    ("3", "1"): {"2": 154, "4": 76, "24": 44, "3": 4, "34": 4, "23": 2, "234": 2},
    ("3", "2"): {"1": 192, "4": 130, "14": 74, "3": 38, "13": 21, "34": 9, "134": 7},
    ("3", "4"): {"2": 189, "1": 116, "12": 73, "123": 6, "13": 4, "3": 4},
    ("3", "12"): {"4": 61, "3": 19, "34": 11},
    ("3", "14"): {"2": 75, "3": 5},
    ("3", "24"): {"1": 44, "3": 12},
    ("3", "124"): {"3": 10},
    ("4", "1"): {"2": 121, "3": 66, "23": 20, "24": 9, "34": 3, "4": 3},
    ("4", "2"): {"3": 146, "1": 115, "13": 35, "14": 7, "4": 5, "134": 4, "34": 3},
    ("4", "3"): {"2": 190, "12": 84, "1": 74, "124": 5, "24": 3, "14": 2},
    ("4", "12"): {"3": 59, "34": 11, "4": 8},
    ("4", "13"): {"2": 53},
    ("4", "23"): {"1": 25},
    ("4", "123"): {"4": 9},
    ("12", "3"): {"4": 61, "14": 8, "1": 7, "2": 6, "12": 5},
    ("12", "4"): {"3": 52, "1": 11, "2": 10, "23": 4, "13": 4, "12": 2},
    ("12", "34"): {"12": 65, "1": 12, "2": 2},
    ("13", "2"): {"4": 44, "3": 16, "34": 7},
    ("13", "4"): {"2": 34, "12": 6, "3": 4, "1": 3},
    ("14", "2"): {"3": 60, "4": 16, "34": 6},
    ("14", "3"): {"2": 60, "1": 16, "12": 6},
    ("23", "1"): {"4": 11, "3": 6, "34": 6, "2": 3},
    ("23", "4"): {"1": 11, "2": 6, "12": 6, "3": 3},
    ("24", "1"): {"3": 34, "34": 6, "2": 4, "4": 3},
    ("24", "3"): {"1": 44, "2": 16, "12": 7},
    ("34", "1"): {"2": 52, "4": 11, "3": 10, "23": 4, "24": 4, "34": 2},
    ("34", "2"): {"1": 61, "14": 8, "4": 7, "3": 6, "34": 5},
    ("34", "12"): {"34": 65, "4": 12, "3": 2},
    ("123", "4"): {"3": 4, "12": 3, "2": 3, "123": 2},
    ("124", "3"): {"1": 7, "2": 6},
    ("134", "2"): {"4": 7, "3": 6},
    ("234", "1"): {"2": 4, "34": 3, "3": 3, "234": 2},
}


def _one_back(steps):
    out = {}
    for (_, row), seen in steps.items():
        for nxt, n in seen.items():
            out.setdefault(row, {})
            out[row][nxt] = out[row].get(nxt, 0) + n
    return out


_STEPS1 = _one_back(_STEPS)     # ... going by the row just played alone

# 4K streams, the notes a beat carries -> how its four rows (beat, quarter, half beat,
# three quarters) may share them out, and how often. The user's Stream example, Rise
# Of The Chaos Wizards, plays a jump on the beat and three single notes (55% of its
# stream beats), single notes only (29%), now and then a hand on the beat or a jump
# moved off it, and short jumptrills ([12][34]) into the next phrase; the other
# examples share their notes out the same ways, with more chords to the beat where they
# are harder. Fours (quads) only where the rating allows them.
_FIGURES = {
    4: {"1111": 1},
    5: {"2111": 75, "1121": 11, "1211": 7, "1112": 7},
    6: {"3111": 50, "2121": 18, "2211": 8, "2112": 8, "1221": 6, "1212": 5, "1122": 5},
    7: {"3121": 55, "3112": 12, "2221": 12, "2122": 6, "2212": 6, "1222": 4, "4111": 50},
    8: {"3122": 40, "2222": 30, "4121": 40, "4112": 20},
}

STYLES = {
    # A kind of LN chart: something is held most of the time, a beat or two at a stretch
    # and each hold in another column than the last, while the other fingers play the
    # song's rhythm as single notes, quick repeats and light chords. Voice or lead is
    # held where it sounds; where nothing is, holds are woven in on the beat. The user
    # played that and called it good, and then said what sets the style apart from LN:
    # it "uses lns to make finger independence the focus, so this is the style that
    # uses a lot of focus on shields, reverse shields, and inverse", with "comparably
    # less notes usually to train finger independence and not stamina and reading".
    # So most phrases get shields and reverse shields around their holds, and some are
    # written in long notes only (see _independence): long notes used as normal notes,
    # as in the two charts the user named (research/examples/coordination-*.osu;
    # "sometimes a mapper uses lns as regular notes to make sure they are really using
    # coordination"), or inverse. The Long notes setting moves the mix from no long
    # notes (0) through this (0.5) to long notes only (1). It suits tempos up to about
    # 185 BPM.
    # After the user's playtest of 2026-10-16 ("many scary jacks (even consecutive jumps)
    # before lns ... tone it down so jacks before lns are rare, and maybe balancing that
    # by adding some reverse shields, as well as inverse lns"): a hold never starts in a
    # column pressed just before it unless as one of the rare shields placed on purpose
    # (shield_cost, shields), reverse shields stay at about every other hold, inverse
    # phrases come from 3.5 stars (inverse), two rows a quarter beat apart never share
    # two columns and share one only where the rating needs it (reuse), no column is
    # hit three times running (jacks), and the song's own grid is kept (finer=False).
    # Without those jacks the rating is reached with more jumps instead (chord 4.5, was
    # 2.8); songs under about 130 BPM may still stop short of 4.5 stars.
    # (Until 2026-10-13 this name stood for what tests/regress.py keeps as "original":
    # only held vocals as LNs.)
    "coordination": Style("Coordination",
                          "Holds that keep switching columns over single notes, with shields, reverse shields and inverse.",
                          hush=0.6, crux=1.5, contour=0.3, ln=0.5, chord=4.5, runs=(0.15, 0.30), breath=0.6, cap=0, filler=2,
                          comfort=0, bass=False, motifs=_WOVEN, third=1.5, gradual=True, steady=True,
                          harshest=3, apart=1, repeat=70.0, repeat_cost=(2.0, 2.5, 3.0),
                          hand_cost=(0.0, 0.3, 1.2, 3.0, 5.0), beside=1.0,
                          release=60.0, weave=(0.8, 0.45, 0.2, 0.35),
                          noodles=(0.9, 0.25, 0.4, 0.15, 0.2), suits=(80.0, 250.0, 1),
                          phrases=(4.5, 0.1, 5.5, 0.4), inverse=(3.5, 0.0, 5.5, 0.25),
                          shields=(0.15, 0.6), shield_cost=3.0, reuse=(0, 0, 1, 1, 1), jacks=((0.0, 2),),
                          finer=False),
    # After Igaku and Hole Dwelling: the spacing between rows keeps changing. Real hits
    # between the lines are charted where the song has them; beyond that, short run-ins
    # on 1/8, 1/6 and finer are woven in front of about half the rows, at every
    # difficulty, while density decides how full the 1/4 lines are and how many rows
    # are jumps. A column may repeat after 88 ms, so quick trills and the odd jack occur.
    "tech": Style("Tech", "Row spacing keeps changing: 1/4 notes with 1/8 and 1/6 bursts woven in.",
                  hush=0.6, crux=1.5, contour=0.3, ln=0.0, chord=3.5, runs=(0.2, 0.4), breath=0.8, cap=-1, filler=2,
                  comfort=0, bass=False, motifs=_FLOWING, arc=False, layer=False,
                  gradual=True, steady=True, phrase=16, harshest=3,
                  hand_cost=(0.0, 0.5, 2.0, 4.0, 6.0), one_hand=0.0, apart=1, repeat=88.0,
                  repeat_cost=(0.4, 1.2, 1.5), detail=8, fills=(0.05, 0.3, 0.45),
                  suits=(72.0, 125.0, 1),
                  ornaments=(("half", 3), ("triplet", 2), ("thirds", 1), ("quarters", 1))),
    # Streams at one steady spacing, as the user described them on 2026-10-16: "a stream
    # ... that come in a consistent time (like around 1/4 or 1/8 depending on the song
    # bpm) with not much complex division of notes ... Notes weren't supposed to be hit
    # rapidly, but consistently", where "the real complexity comes from the stream
    # section": "Lighter stars would use single to light-jump streams ... 4.5 gets
    # around light handstream to dense handstream, and higher stars (around 5.3) would
    # get quad streams". So the grid is the song's 1/4 (its 1/8 only for songs slower
    # than about 106 BPM), never a finer one to reach a rating; a bar's run grows back
    # from the next downbeat in one piece, so there are no holes in it; and the rating
    # is reached by chords (see _rung): jumps on the accents, then on most rows, then
    # hands, and quads from 5 stars. Until 2026-10-16 the style kept to jumps and
    # switched to 1/8 notes when 1/4 jumpstream fell short of the rating, which made
    # most 4.5-star maps of songs under 150 BPM 1/8 bursts "mapped like speed maps".
    # The first version of the chords let a row share a column with the one before it,
    # and on slower songs, where the rating took chords on every row, that played as
    # chordjack (the user: "the streams have chordjack during the chorus section which
    # doesn't fit the style ... curse these metal hands ... literally starts as
    # chordjack"). Now no column is pressed on two rows running (reuse 0, _apart), as in
    # the Stream example and Palette GAMMA, except next to a quad; hands alternate with
    # single notes and jumps, chords change from beat to beat (vary) and the hands take
    # turns at them (chord_hands). A jack-free stream tops out at about 0.026 stars per
    # BPM of its 1/4 notes, so a song needs about 175 BPM for 4.5 stars (reach); a
    # slower one stops short of the rating rather than turn into chordjack.
    # After the user's next playtest (2026-10-17): "the pattern of a jump followed by a
    # short one hand trill just repeats over and over. a stream should be diverse ...
    # the jump can be inserted in other places as well as there being chances for a
    # staircase pattern (with jumps in different places)". The chords had come on every
    # beat and half beat (jump, single, jump, single), where the user's stream examples
    # mostly play a jump on the beat and a staircase of three single notes after it.
    # Now each beat's notes are laid out as in those examples (steps: _figures), and
    # each row's columns follow on from the two before as theirs do (_STEPS). Laid out
    # like that, a stream rates a little higher for its notes, so the ceiling is now
    # about 0.027 stars per BPM (reach): 4.5 stars from about 167 BPM, 4 from 148.
    "stream": Style("Stream", "Steady 1/4 streams that get denser with the rating: jumpstream, then handstream. No LNs.",
                    hush=0.6, crux=1.5, contour=0.3, ln=0.0, chord=4.5, runs=(0.15, 0.30), breath=0.40, cap=0, filler=2,
                    comfort=0, bass=False, motifs=_FLOWING, arc=False, layer=False,
                    third=2.5, wide=True, phrase=16, gradual=True, harshest=3,
                    steady=True, gather=True, ladder=(0.5, 2.5), quads=5.0, jacks=((0.0, 2),),
                    reuse=(0, 0, 0, 0, 0), vary=4, chord_hands=0.25, steps=True, apart=1, repeat=60.0, repeat_cost=(3.0, 3.0, 4.0),
                    rows=(100.0, 1), suits=(70.0, 145.0, 1), reach=(405.0, 440.0),
                    hand_cost=(0.0, 0.5, 2.0, 4.0, 6.0), one_hand=0.0),
    # After Palette GAMMA: single notes about 57 ms apart, in long runs where the song
    # is loud and a plain 1/4 pulse where it is not. A column is never hit again on the
    # next row but one (so no plain trills) and the hands take turns. The grid is the one
    # whose lines come closest to that spacing; where even those are too fast to keep up
    # for long, the fast notes are gathered into one run per bar, into the downbeat.
    # Which column comes next follows Palette GAMMA's own streams (_FLOW): three-column
    # rolls that turn back and hand-to-hand switches. Until 2026-10-15 it was rolls,
    # fixed four-note orders and zigzags two bars at a time, and the user found "the
    # speed maps repeated rolls way too often" and asked for "a combination of trills
    # and partial rolls ... super synonymous to the palette gamma map".
    "speed": Style("Speed", "Fast single notes in long runs, hands taking turns; few chords.",
                   hush=0.6, crux=1.5, contour=0.3, ln=0.0, chord=2.0, runs=(0.30, 0.10), breath=0.12, cap=-1, filler=2,
                   comfort=0, bass=False, motifs=_SPEEDY, arc=False, layer=False,
                   third=99.0, phrase=8, gradual=True, harshest=3, steady=True,
                   hand_cost=(0.0, 1.0, 4.0, 8.0, 12.0), one_hand=0.0, repeat_cost=(0.0, 2.5, 1.5),
                   sustain=48.0, gather=True, rows=(57.0, 1), suits=(48.0, 66.0, 1)),
    # After Kijibato and Mi Mi Mi: the Stream style's runs, but a column may be hit
    # again on the very next row, and is, a couple of times a bar. Jumps spread to
    # every row before any chord grows into a hand. How fast those jacks may come
    # depends on the rating: in the speedjack charts at hand they are 119 ms apart at
    # 4.5 stars (Mi Mi Mi, 126 BPM) and 86-90 ms apart at 5.5-6.2 stars (the rest of
    # NoNamed's Speedjack Pack). A 4.5-star chart with 85 ms jacks reaches its rating
    # with fewer notes, but its jacks feel like 5.5 stars (the user, of one at 177 BPM:
    # "insanely fast for 4.5 stars"). So the grid is picked for the rating (grid_for):
    # the song's 1/4 where that is about the speed the rating is written for, and
    # otherwise triplets (1/3 or 1/6 of the beat) where those come closer, as the user
    # asked on 2026-10-16, with more hands to make up the rating. A jack runs four
    # presses at most at 4.5 stars, more from 5.5.
    "speedjack": Style("Speedjack", "Runs of light chords in which a column is often hit twice running.",
                       hush=0.6, crux=1.5, contour=0.3, ln=0.0, chord=4.5, runs=(0.15, 0.30), breath=0.40, cap=0, filler=2,
                       comfort=0, bass=False, motifs=_JACKS, arc=False, layer=False,
                       third=2.5, wide=True, thick=3.0, phrase=16, gradual=True, harshest=3,
                       steady=True, hand_cost=(0.0, 0.5, 2.0, 4.0, 6.0), one_hand=0.0,
                       apart=1, repeat=60.0, repeat_cost=(0.0, 0.3, 0.4), rows=(100.0, 1),
                       suits=(72.0, 100.0, 1), quick=(6.0, 22.0),
                       jacks=((0.0, 3), (4.25, 4), (5.5, 5), (6.0, 6))),
    # After Sendan Life and Glaza: rows about 130 ms apart, chords on the drum hits,
    # and columns free to repeat from one row to the next. The grid is picked so the
    # rows come out near that spacing; a few 1/4 notes slip in between at the top end.
    "chordjack": Style("Chordjack", "Chords about 130 ms apart whose columns repeat row after row.",
                       hush=0.6, crux=1.5, contour=0.3, ln=0.0, chord=5.0, runs=(0.8, 0.8), breath=99.0, cap=1, filler=3,
                       comfort=1, bass=False, motifs=_HEAVY, arc=False, layer=False,
                       wide=True, gradual=True, steady=True, apart=1, repeat=95.0,
                       repeat_cost=(0.0, 0.6, 0.6), rows=(130.0, 2), suits=(105.0, 170.0, 2)),
    # After On And On!! and Sparkling Starlight: the first style's rhythm with more than
    # half of its notes held for a quarter or half a beat, so that fingers let go and
    # press in quick succession and two notes are often held at once. Since 2026-10-13
    # it also holds what is sung for as long as it sounds. Until then long sustained
    # notes were a style of their own ("Holds"); the user grouped the two: "holds
    # imply holding notes anyway".
    "ln": Style("LN", "Sung notes are held while they sound; over half the other notes are short holds.",
                hush=0.6, crux=1.5, contour=0.3, ln=0.6, chord=3.6, runs=(0.55, 0.45), breath=1.3, cap=0, filler=2,
                comfort=0, bass=False, motifs=_MIXED, layer=False, third=1.5,
                gradual=True, harshest=2, steady=True, release=40.0,
                short=(0.55, 0.30, 0.15), long=True),
}


# Invented rows between the lines. Each is a short run-in to the row on a line: how
# far before that line its rows sit, in lines. A "triplet" spreads three rows over two
# lines, so it takes the place of the line in between.
ORNAMENTS = {
    "half": (Fraction(1, 2),),
    "third": (Fraction(1, 3),),
    "thirds": (Fraction(2, 3), Fraction(1, 3)),
    "quarter": (Fraction(1, 4),),
    "quarters": (Fraction(1, 2), Fraction(1, 4)),
    "triplet": (Fraction(4, 3), Fraction(2, 3)),
}
FINEST_MS = 36.0              # invented rows are never closer together than this


def spacing(song, style):
    """How close together a style lets one column be pressed, in positions:
    (near, gap, release). `gap` is the least distance between two presses. `near` is
    that rounded up to whole lines: the closest two rows on the lines can repeat a
    column, and what counts as repeating it "as fast as allowed". `release` is the
    least distance from the end of a long note to the next press of its column."""
    line, per_ms = song["line"], song["line"] / song["line_ms"]
    near = max(style.apart, math.ceil(style.repeat / song["line_ms"])) * line
    gap = max(1, min(near, math.ceil(style.repeat * per_ms)))
    if style.apart > 1:
        gap = max(gap, style.apart * line)
    release = gap if style.release is None else max(1, math.ceil(style.release * per_ms))
    return near, gap, release


def _hold(song, k, ms):
    """`ms` of the song in positions, for a long note starting at k: what song["min_hold"]
    and song["max_hold"] hold, but by the tempo around k where the tempo changes."""
    slow = song.get("slow")
    if not slow:
        return song["min_hold"] if ms == MIN_HOLD_MS else song["max_hold"]
    lines = ms / (song["line_ms"] * slow.get(k // (song["bar"] * song["line"]), 1.0))
    return (math.ceil(lines) if ms == MIN_HOLD_MS else int(round(lines))) * song["line"]


def _splits(keys):
    """Hand-alternating orders through all the columns, for split rolls."""
    if keys == 4:
        return [[0, 2, 1, 3], [3, 1, 2, 0], [0, 3, 1, 2], [3, 0, 2, 1], [1, 3, 0, 2], [2, 0, 3, 1]]
    hand = HANDS[keys]
    left = [c for c in range(keys) if hand[c] == 0]
    right = [c for c in range(keys) if hand[c] == 1]
    thumb = [c for c in range(keys) if hand[c] == 2]
    out = []
    for a, b in ((left, right), (left, right[::-1]), (left[::-1], right), (left[::-1], right[::-1])):
        weave = [c for pair in zip(a, b) for c in pair]
        out += [weave + thumb, [c for pair in zip(b, a) for c in pair] + thumb]
    return out


def _pairs(keys):
    """Two-hand column pairs, for trills."""
    if keys == 4:
        return [(0, 2), (1, 3), (0, 3), (1, 2)]
    hand = HANDS[keys]
    return [(a, b) for a in range(keys) for b in range(keys) if hand[a] == 0 and hand[b] == 1]


class Patterner:
    """Picks columns. Every option is scored: comfort first (no fast jacks, no long
    one-hand runs), then the current motif, then how well a sung note's column
    matches its pitch."""

    def __init__(self, keys, line, spaces, rng, comfort, style):
        self.keys, self.line, self.rng, self.motifs = keys, line, rng, style.motifs
        self.near, self.gap, self.release = spaces
        self.hand_cost, self.one_hand, self.beside = style.hand_cost, style.one_hand, style.beside
        self.chord_hands = style.chord_hands
        self.shield_cost = style.shield_cost
        self.shield_span = self.near     # how soon after a press a hold in its column is a shield
        self.repeat_cost = style.repeat_cost
        self.hand = HANDS[keys]
        self.splits, self.pairs = _splits(keys), _pairs(keys)
        self.limits = COMFORT[comfort]
        if style.chain < self.limits[0]:
            self.limits = (style.chain,) + self.limits[1:]
        self.longest = style.chain
        self.reuse = style.reuse[comfort] if style.reuse else 99
        self.stream = style.ladder is not None
        long_ago = -99 * line
        self.last = [long_ago] * keys   # position of each column's latest press
        self.free = [long_ago] * keys   # first position each column may be pressed again
        self.held = [long_ago] * keys   # position a running LN lets go
        self.chain = [0] * keys         # presses in a row at the fastest spacing, per column
        self.cursor = rng.randrange(keys)
        self.dir = rng.choice([1, -1])
        self.tick = long_ago            # position of the latest row
        self.busy = [0, 0, 0]           # rows in a row, no more than a line apart, per hand
        self.chord = (None, -99)
        self.vary = style.vary
        self.chords = {}                # chord -> position it was last played, with `vary`
        self.full = long_ago            # position of the latest row that took every column
        self.trail = []                 # the single notes of the stream going on, latest last
        self.flow_pick = None           # the "flow" motif's choice for the next row
        self.steps = style.steps and style.ladder is not None and keys == 4
        self.rows = []                  # with `steps`: the last two rows, a line apart at most
        self.new_motif("calm")

    def new_motif(self, section):
        names, w = self.motifs[section]
        if "flow" in names and self.keys != 4:
            names, w = _RUNS[section]
        self.motif = self.rng.choices(names, weights=w)[0]
        self.step = 0
        self.seq = None
        if self.motif == "split":
            self.seq = self.rng.choice(self.splits)
        elif self.motif == "trill":
            a, b = self.rng.sample(self.pairs, 2)
            self.seq = list(a) * 2 + list(b) * 2
        elif self.motif == "anchor":
            a = self.rng.randrange(self.keys)
            o = [c for c in range(self.keys) if c != a]
            self.rng.shuffle(o)
            self.seq = [a, o[0], a, o[1], a, o[2]]
        elif self.motif == "jack":
            o = list(range(self.keys))
            self.rng.shuffle(o)
            # every column two or three times running
            self.seq = [c for c in o for _ in range(self.rng.choice((2, 2, 3)))]

    def steer(self, direction):
        """Rolls follow the melody: up when the singer goes up."""
        self.dir = direction

    def flow(self):
        """The next column as Palette GAMMA's streams would have it (see _FLOW), going
        by as many of the notes just played as the table knows; drawn once a row."""
        if self.flow_pick is None:
            for n in (3, 2, 1):
                if len(self.trail) >= n:
                    tail = tuple(self.trail[-n:])
                    w = [sum(ws[c] for ctx, ws in _FLOW.items() if ctx[3 - n:] == tail)
                         for c in range(4)]
                    if sum(w):
                        self.flow_pick = self.rng.choices(range(4), weights=w)[0]
                        return self.flow_pick
            self.flow_pick = self.rng.choice([c for c in range(4) if c != self.cursor])
        return self.flow_pick

    def follow(self, options):
        """With `steps`: one of `options` drawn the way the user's stream examples go on
        from the two rows just played, blended with how they go on from the last one
        alone (_STEPS) the more, the fewer times the examples were at those two rows,
        so that a rare spot has more than one way on; None where the examples never
        went on to any of them."""
        rows = ["".join(str(c + 1) for c in r) for r in self.rows]
        names = ["".join(str(c + 1) for c in o) for o in options]
        two = _STEPS.get(tuple(rows), {}) if len(rows) == 2 else {}
        one = _STEPS1.get(rows[-1], {}) if rows else {}
        n2, n1 = sum(two.get(n, 0) for n in names), sum(one.get(n, 0) for n in names)
        if not n1:
            return None
        trust = n2 / (n2 + 20.0)
        w = [trust * two.get(n, 0) / max(1, n2) + (1.0 - trust) * one.get(n, 0) / n1 + 0.01
             for n in names]
        return self.rng.choices(options, weights=w)[0]

    def want(self):
        if self.seq:
            return self.seq[self.step % len(self.seq)]
        if self.motif == "flow":
            return self.flow()
        nxt = self.cursor + self.dir
        if self.motif == "roll":
            return nxt % self.keys
        return nxt if 0 <= nxt < self.keys else self.cursor - self.dir     # zigzag bounces

    def free_columns(self, k):
        return [c for c in range(self.keys) if self.free[c] <= k]

    def holding(self, k):
        return [c for c in range(self.keys) if self.held[c] > k]

    def worn(self, c, k):
        """Would pressing column c at k make its jack longer than the style allows?"""
        return k - self.last[c] <= self.near and self.chain[c] >= self.longest

    def comfortable(self, cols, k, fast, accent):
        """Hard limits (see COMFORT): no endless hammering of one column, no chord
        inside a run that repeats most of the chord before it, and no hand left
        playing every line for long."""
        chain, reuse, reuse_accent, run = self.limits
        again = [c for c in cols if k - self.last[c] <= self.near]
        if any(self.chain[c] >= chain for c in again):
            return False
        if k - self.tick <= self.line and any(self.busy[self.hand[c]] >= run for c in cols):
            return False
        most = min(self.reuse, reuse_accent if accent or not fast else reuse)
        if len(cols) >= self.keys or k - self.full <= self.near:
            most = max(most, 1)         # a chord of every column can't help sharing one
        elif (self.stream and len(cols) > 1 and self.chord[1] == self.tick
              and k - self.tick <= 2 * self.line and set(cols) & set(self.chord[0])):
            return False                # nor a chord with the chord half a beat before
        return len(again) <= most

    def assign(self, cols, starts, k):
        """Give each long note starting now one of `cols`: near its pitch, and not on a
        hand that is already holding one. With a `shield_cost`, never in a column
        pressed just before (a shield nobody asked for): such a long note stays a tap.
        Returns ({column: end tick}, cost)."""
        busy = {self.hand[c] for c in self.holding(k)}
        ends, cost, left = {}, 0.0, list(cols)
        if self.shield_cost:
            left = [c for c in left if k - self.last[c] > self.shield_span] or left
        for end, target in starts:
            if not left:
                break
            col = min(left, key=lambda c: 1.2 * abs(c - target) + 2.0 * (self.hand[c] in busy))
            if self.shield_cost and k - self.last[col] <= self.shield_span:
                cost += self.shield_cost
                continue
            cost += 1.2 * abs(col - target) + 2.0 * (self.hand[col] in busy)
            busy.add(self.hand[col])
            ends[col] = end
            left.remove(col)
        return ends, cost

    def cost(self, cols, k, target, starts):
        want = self.want()
        c = self.rng.random() * 0.3 + self.assign(cols, starts, k)[1]
        for col in cols:
            gap = k - self.last[col]
            if gap <= self.near:                            # fastest-allowed repeat of a column
                # (Palette GAMMA's streams come back to a column every third note for
                # three or more notes a third of the time, so "flow" may ask again)
                first = col == want and (self.chain[col] == 1 or self.motif == "flow")
                wanted, other, more = self.repeat_cost
                c += (wanted if first else other) + more * (self.chain[col] - 1)
            elif gap <= self.near + self.line:
                c += 0.8
            c -= 0.04 * min(gap / self.line, 16)            # favour columns left alone longest
        hands = {self.hand[col] for col in cols}
        if k - self.tick <= self.line:                      # a hand kept busy line after line
            c += (sum(self.hand_cost[min(self.busy[h], 4)] for h in hands)
                  * (1.0 if len(cols) == 1 else self.chord_hands))
        if self.beside:                                     # the free fingers of a holding hand
            holding = {self.hand[col] for col in self.holding(k)}
            c += self.beside * sum(1 for col in cols if self.hand[col] in holding
                                   and col not in dict(starts and self.assign(cols, starts, k)[0]))
        if len(cols) == 1:
            if cols[0] == want:
                c -= 2.0
        else:
            if len(hands) == 1:
                c += self.one_hand
            if (k - self.chords.get(cols, -10**9) <= self.vary * self.line if self.vary
                    else cols == self.chord[0] and k - self.chord[1] <= 2 * self.line):
                c += 2.5
            if want in cols:
                c -= 1.0
        if target is not None and not starts:
            c += 1.2 * min(abs(col - target) for col in cols)
        return c

    def press(self, cols, k, ends):
        if len(cols) == 1 and k - self.tick <= 2 * self.line:
            self.trail = self.trail[-2:] + [cols[0]]
        else:                           # a chord, or the first note after a pause
            self.trail = [cols[0]] if len(cols) == 1 else []
        self.rows = (self.rows[-1:] if k - self.tick <= self.line else []) + [tuple(cols)]
        self.flow_pick = None
        for col in cols:
            self.chain[col] = self.chain[col] + 1 if k - self.last[col] <= self.near else 1
            self.last[col] = k
            end = ends.get(col)
            self.free[col] = k + self.gap if end is None else end + self.release
            if end is not None:
                self.held[col] = end
        hands = {self.hand[col] for col in cols}
        for h in (0, 1, 2):
            self.busy[h] = ((self.busy[h] + 1 if k - self.tick <= self.line else 1)
                            if h in hands else 0)
        if len(cols) == 1:
            self.cursor = cols[0]
        else:
            self.chord = (cols, k)
            if self.vary:
                self.chords[cols] = k
            self.cursor = max(cols) if self.dir > 0 else min(cols)
            if len(cols) >= self.keys:
                self.full = k
        if self.motif == "zigzag" and self.cursor in (0, self.keys - 1):
            self.dir = 1 if self.cursor == 0 else -1
        self.tick = k
        self.step += 1


INTENSITY = {"calm": 0.0, "verse": 0.6, "chorus": 1.0}


@dataclass(frozen=True)
class Meter:
    """How one charting bar fills up as the level rises."""
    first: dict         # the lines that fill first -> their order, 0 (soonest) .. 1
    runs: dict          # section -> orders in which the remaining lines fill
                        # (even bars, odd bars, phrase-ending bars)
    breaths: int        # how many of those lines are the breath before an accent
    back: int           # the line the backbeat (snare) accent sits on
    beat: int           # lines per pulse


METERS = {
    # 16 lines, two to a 1/2 beat. The 1/2 lines fill first, then 1/4 runs that grow
    # backwards from the snare and the downbeat; the line just before each of those
    # stays empty longest, so the accent gets a breath in front of it.
    "straight": Meter(
        first={0: 0.0, 8: 0.0, 4: 0.1, 12: 0.2, 14: 0.3, 6: 0.5, 2: 0.75, 10: 1.0},
        runs={"chorus": ([13, 5, 11, 3, 9, 1, 7, 15], [5, 13, 3, 11, 1, 9, 15, 7],
                         [13, 15, 11, 9, 5, 3, 7, 1]),
              "verse": ([13, 15, 11, 9, 5, 7, 3, 1], [13, 11, 15, 9, 7, 5, 1, 3],
                        [13, 15, 11, 9, 5, 7, 3, 1])},
        breaths=2, back=8, beat=4),
    # 12 lines = two beats of six (6/8, 12/8, shuffle): beats on 0 and 6, eighths between
    "compound": Meter(
        first={0: 0.0, 6: 0.0, 4: 0.3, 10: 0.3, 2: 0.7, 8: 0.7},
        runs={"chorus": ([9, 3, 7, 1, 5, 11], [3, 9, 1, 7, 11, 5], [9, 11, 7, 3, 5, 1]),
              "verse": ([9, 11, 7, 3, 5, 1], [9, 7, 11, 5, 3, 1], [9, 11, 7, 3, 5, 1])},
        breaths=2, back=6, beat=6),
    # 12 lines = four beats of three (swing): beats on 0 3 6 9, the swung offbeat on 2 5 8 11
    "swing": Meter(
        first={0: 0.0, 6: 0.0, 3: 0.1, 9: 0.2, 11: 0.4, 5: 0.5, 2: 0.75, 8: 1.0},
        runs={"chorus": ([10, 4, 7, 1], [4, 10, 1, 7], [10, 7, 4, 1]),
              "verse": ([10, 7, 4, 1], [10, 4, 7, 1], [10, 7, 4, 1])},
        breaths=1, back=6, beat=3),
}


def plan(song, density, style):
    """How many notes each grid line carries. Real hits always play; the section's
    level (density x intensity) then decides how much is woven between them: the
    emptier strong lines first, then runs on the lines between, with chords growing
    on the accents. Below zero, density thins the real hits out instead, least
    important first."""
    drums, sung, stabs = song["drums"], song["sung"], song["stabs"]
    bass = song["bass"] if style.bass else {}
    cap = max(1, BIGGEST_CHORD[song["keys"]] + style.cap)
    meter, bar, line = METERS[song["meter"]], song["bar"], song["line"]
    n_runs = len(meter.runs["chorus"][0])
    arc = (0, 0, 0, 0) if not style.arc else (-2, 0, 0, 1) if n_runs == 8 else (-1, 0, 0, 1)
    # chords arrive a little at a time rather than on every hit at once: the stronger
    # a hit is among its kind, the slightly sooner it grows (and so the rating can be
    # tuned finely instead of jumping)
    early = {}
    for kind in "KSH":
        ks = sorted((k for k in drums if drums[k][0] == kind), key=lambda k: drums[k][1])
        for n, k in enumerate(ks):
            early[k] = 0.1 * ((n + 0.5) / len(ks) - 0.5)
    taps, topped = {}, set()            # topped: stream rows the rating would make a four
    slow = song.get("slow", {})         # how much longer a bar really lasts, where the tempo changes
    tune, passing = song["tune"], []
    for b in song["bars"]:
        section = song["section"][b]
        quiet = section == "calm" and style.hush > 0
        level = max(0.0, density) * (style.hush if quiet else style.verse if section == "verse"
                                     else INTENSITY[section])
        if style.contour and section == "chorus":
            level *= 1.0 - style.contour * (1.0 - song["rank"].get(b, 1.0))
        # the loudest stretches are charted fuller, though no fuller than a full chart
        level = max(level, min(level * (1.0 + style.crux * song["lift"].get(b, 0.0)), 1.1))
        clear = max(0.3, 0.5 - 0.83 * level)        # how clearly a note of the melody must start
        chord = style.chord * level
        even = (level - 0.3) / 0.3
        # a four-bar arc: the first bar of a phrase breathes, the last one runs into the next
        dither = (b * 0.618034) % 1.0
        if style.gradual:
            even += 0.3 * (dither - 0.5)
        run = min(1.0, max(0.0, (level - style.runs[0]) / style.runs[1]))
        n_odd = int((n_runs - meter.breaths) * run + dither)
        n_odd += int(meter.breaths * min(1.0, max(0.0, (level - style.breath) / 0.3)) + dither)
        n_odd = max(0, min(n_runs, n_odd + arc[b % 4])) if n_odd else 0
        if song["line_ms"] * slow.get(b, 1.0) < style.sustain:
            n_odd = min(n_odd, n_runs // 2)
        orders = meter.runs.get("verse" if quiet else section)
        if style.gather and orders:
            odd = set(sorted(orders[0], reverse=True)[:n_odd])
        else:
            odd = set(orders[2 if b % 4 == 3 else b % 2][:n_odd]) if orders else set()
        breather = style.phrase and b % style.phrase == 0
        for i in range(bar):
            if not song["k_min"] <= b * bar + i <= song["k_max"]:
                continue
            if breather and 0 < i < song["tpb"]:
                continue
            k = (b * bar + i) * line
            hit = drums.get(k)
            grow = chord + early.get(k, 0.1 * (dither - 0.5))
            big = grow - style.third
            if hit and i not in meter.first:
                n = 1
            elif hit and hit[0] == "S":
                top = i == meter.back or (style.wide and big >= 2.0 + meter.first[i])
                n = 1 + (grow >= 0.5) + (big >= 2.0 and top) + (big >= 4.0 and top)
            elif hit and hit[0] == "K":
                top = i == 0 or (style.wide and big >= 2.5 + meter.first[i])
                n = 1 + (grow >= 1.0) + (big >= 2.5 and top) + (big >= 4.5 and top)
            elif hit:
                n = 1 + (grow >= 1.5)
            elif i in meter.first:
                n = min(style.filler, (level > 0 and even >= meter.first[i])
                        + (grow >= style.fill_at + meter.first[i])
                        + (big >= style.fill_at + 2.0 + meter.first[i]))
                if style.gather and i - 1 in odd and i + 1 in odd:
                    n = max(n, 1)       # no hole inside a gathered run
            else:
                n = (i in odd) + (i in odd and grow >= style.thick + 0.5 * dither)
            if style.ladder and n and not quiet:
                n = _rung(style.ladder, grow, i, hit, meter, b * bar + i)
                if n > cap:
                    topped.add(k)
            if quiet and not hit:
                # Without drums the melody is played instead: its notes that start on a
                # beat or half beat, the clearer ones first (a second note on a beat
                # only once chords have grown a good deal). Notes between those lines
                # are left to the end: they play only inside a run.
                heard = tune.get(k, 0.0)
                if on_pulse(i, song["tpb"], song["meter"]):
                    n = (heard >= clear) + (heard >= 0.5 and i % meter.beat == 0
                                            and grow >= 2.5 and style.filler > 2)
                else:
                    n = 0
                    if heard >= 0.5:
                        passing.append(k)
            n = min(cap, n)
            if k in sung:
                n = min(cap, n + 1) if hit and grow >= 0.5 and style.layer else max(n, 1)
            elif k in bass:
                n = max(n, 1)
            if density < 0 and k in stabs:
                n = max(n, 1)
            if density < 0 and n:
                held = k in stabs or (k in sung and sung[k][0] - k >= _hold(song, k, MIN_HOLD_MS))
                on_beat = hit and hit[0] in "KS" and i % meter.beat == 0
                weight = (1.0 if held or on_beat else
                          0.75 if hit and hit[0] in "KS" else 0.5 if k in sung else 0.25)
                n = n if weight > -density else 0
            if n:
                taps[k] = int(n)
    for k in passing:
        if density >= 0 and k - line in taps and k + line in taps:
            taps.setdefault(k, 1)
    if style.steps and style.ladder and song["keys"] == 4 and meter.beat == 4:
        _figures(song, taps, topped, cap)
    if style.ladder:
        _apart(song, taps)
    if style.detail:
        _detail(song, density, style, taps)
    if style.ornaments and density > 0:
        _ornament(song, density, style, taps)
    return taps


def _weak(i, hit, meter, n):
    """How weak line i of a bar (line n of the song) is, for a stream's chords: 0 for a
    drum accent on the beat, then the other beats and accented half beats, then the
    other half beats, and the 1/4 lines in between weakest."""
    accent = bool(hit) and hit[0] in "KS"
    if i % meter.beat == 0:
        return 0.0 if accent else 0.8 + meter.first.get(i, 0.0)
    if i in meter.first:
        return 1.0 if accent else 1.5 + meter.first[i]
    return 2.5 if hit else 3.0 + 0.5 * ((n * 0.618034) % 1.0)


def _rung(ladder, grow, i, hit, meter, n):
    """How many notes a row of a stream gets (Style.ladder), on line i of its bar (line
    n of the song). The beats and half beats climb the same ladder, a jump, then (on
    the beats) a hand, then a quad, one `step` of growth apart, the stronger lines
    first (_weak). The 1/4 lines between them stay single notes: a stream repeats no
    column from one row to the next (the user's Stream example, Rise Of The Chaos
    Wizards, never does, nor does Palette GAMMA), and four columns leave room for a
    hand next to a single note only. So a stream goes from single notes to jumps on the
    accents, to jumpstream, to handstream, without its spacing ever changing."""
    if i not in meter.first:
        return 1
    start, step = ladder
    w = _weak(i, hit, meter, n)
    top = 4 if i % meter.beat == 0 else 2
    return min(top, 1 + sum(grow >= start + j * step + w for j in range(3)))


def _figures(song, taps, topped, cap):
    """Share out the notes of each full beat of a 4K stream (Style.steps) the way the
    user's stream examples do (_FIGURES). A beat keeps as many notes as _rung gave its
    rows, plus one where the rating would have made a four it doesn't allow yet, but
    they no longer sit as a chord on every beat and half beat: a jump on the beat and
    three single notes, a hand and three, a jump moved off the beat, a short jumptrill.
    Each bar favours one layout for each number of notes, so a phrase keeps its shape
    while its columns move on; a chord is likelier on a line the drums accent, and two
    chords running likelier at the end of a phrase. No row gets more notes than the
    row before leaves columns for, and no two hands come within half a beat."""
    line, bar, keys, drums = song["line"], song["bar"], song["keys"], song["drums"]

    def dice(n, salt):
        return ((n * 2654435761 + salt * 40503) % 4294967296) / 4294967296.0

    def pick(weights, u):
        total = sum(weights.values())
        for f, x in weights.items():
            u -= x / total
            if u < 0:
                break
        return f

    def fits(before, figure):
        sizes = before + [int(c) for c in figure]
        for j in range(len(before), len(sizes)):
            for d in (1, 2):
                x, y = sizes[j - d], sizes[j]
                if d == 1 and x + y > keys and not (max(x, y) >= keys and min(x, y) == 1):
                    return False
                if x >= 3 and y >= 3 and max(x, y) < keys:
                    return False
        return max(sizes[len(before):]) <= cap

    for b in song["bars"]:
        for j in range(0, bar, 4):
            ks = [(b * bar + j + i) * line for i in range(4)]
            if not all(k in taps for k in ks):
                continue
            before = [taps.get(ks[0] - 2 * line, 0) if ks[0] - line in taps else 0,
                      taps.get(ks[0] - line, 0)]
            notes = min(8, sum(taps[k] for k in ks) + sum(k in topped for k in ks))
            weights = {}
            while notes >= 4 and not weights:
                weights = {f: x for f, x in _FIGURES[notes].items() if fits(before, f)}
                notes -= not weights
            if not weights:
                continue
            theme = pick({f: x for f, x in _FIGURES[notes].items() if fits([0, 0], f)},
                         dice(b, notes))
            accents = [i for i, k in enumerate(ks) if drums.get(k) and drums[k][0] in "KS"]
            for f in weights:
                weights[f] *= 4.0 if f == theme else 1.0
                if accents:             # up to three times as likely with every accent a chord
                    weights[f] *= 1.0 + 2.0 * sum(f[i] != "1" for i in accents) / len(accents)
                if b % 4 == 3 and j == bar - 4 and "22" in f:
                    weights[f] *= 4.0
            for c, k in zip(pick(weights, dice(b * bar + j, notes + 7)), ks):
                taps[k] = int(c)


def _apart(song, taps):
    """Make room in a stream (Style.ladder) for every row to keep clear of the columns
    of the row before it: two rows with nothing between them, up to half a beat apart,
    have no more notes together than there are columns (only a quad, from the rating
    that allows one, is followed and led into by single notes, and so by a minijack
    either side), and two rows half a beat apart are not both hands, which would leave
    the second no columns but those of the first. Each time the weaker line gives way
    (_weak)."""
    line, bar, keys = song["line"], song["bar"], song["keys"]
    meter, drums = METERS[song["meter"]], song["drums"]

    def weak(k):
        n = k // line
        return _weak(n % bar, drums.get(k), meter, n)

    rows = sorted(taps)
    for a, b in zip(rows, rows[1:]):
        if b - a > 2 * line:
            continue
        x, y = taps[a], taps[b]
        if x + y > keys and max(x, y) < keys:
            loser = b if weak(b) >= weak(a) else a
            taps[loser] = max(1, keys - (y if loser == a else x))
        elif max(x, y) >= keys:
            taps[b if x >= keys else a] = 1
    for i, a in enumerate(rows):
        for b in rows[i + 1:i + 3]:
            if b - a <= 2 * line and taps[a] >= 3 and taps[b] >= 3 and max(taps[a], taps[b]) < keys:
                taps[b if weak(b) >= weak(a) else a] = 2


def _sounds(song, style):
    """Every position on a line where something real is played."""
    real = set(song["drums"]) | set(song["sung"]) | set(song["stabs"])
    return real | set(song["bass"]) if style.bass else real


def _finest(song):
    """FINEST_MS in positions."""
    return math.ceil(FINEST_MS * song["line"] / song["line_ms"])


def _around(rows, k, reach):
    """The rows closer to `k` than `reach`, from the sorted list `rows`."""
    return rows[bisect.bisect_right(rows, k - reach):bisect.bisect_left(rows, k + reach)]


def _detail(song, density, style, taps):
    """Add the real hits that fall between two lines, down to the style's finest snap,
    loudest first. An invented note less than half a line away would only smear such
    a hit, so it makes way; a real one too close to play apart from it stays, and the
    hit is left out."""
    line, real, finest = song["line"], _sounds(song, style), _finest(song)
    rows = sorted(taps)
    hits = sorted(song["between"].items(), key=lambda hit: -hit[1][1])
    for k, (kind, _, snap) in hits:
        if snap > style.detail or (0.75 if kind in "KS" else 0.25) <= -density:
            continue
        near = _around(rows, k, max(finest, (line + 1) // 2))
        if any(r in real and abs(r - k) < finest for r in near):
            continue
        for r in near:
            if r not in real:
                del taps[r]
                rows.remove(r)
        taps[k] = 1
        real.add(k)
        bisect.insort(rows, k)


def _room(rows, taps, lo, hi, gap, keys):
    """Can every note from `lo` to `hi` still get a column of its own? No two notes
    closer together than `gap` may share one. `rows` is the sorted list of taps."""
    i = bisect.bisect_right(rows, lo - gap)
    while i < len(rows) and rows[i] <= hi:
        total, j = 0, i
        while j < len(rows) and rows[j] - rows[i] < gap:
            total += taps[rows[j]]
            j += 1
        if total > keys:
            return False
        i += 1
    return True


def _ornament(song, density, style, taps):
    """Weave invented run-ins between the lines: as the level rises, more rows get one
    leading into them, the accents first. One is only placed where the notes around it
    leave enough columns free."""
    line, bar, keys = song["line"], song["bar"], song["keys"]
    meter, gap = METERS[song["meter"]], spacing(song, style)[1]
    real, first, finest = _sounds(song, style), song["k_min"] * line, _finest(song)
    slow = song.get("slow", {})         # how much longer a bar really lasts, where the tempo changes
    every = []                          # (rows, weight, is a triplet, its closest spacing in ms)
    for name, weight in style.ornaments:
        back = ORNAMENTS[name]
        closest = min(a - b for a, b in zip(back, back[1:] + (0,)))
        if all((f * line).denominator == 1 for f in back):
            every.append(([int(f * line) for f in back], weight, name == "triplet",
                          float(closest) * song["line_ms"]))
    rows = sorted(taps)
    for b in song["bars"]:
        shapes = [s[:3] for s in every if s[3] * slow.get(b, 1.0) >= FINEST_MS]
        total = sum(weight for _, weight, _ in shapes)
        if not total:
            continue
        section = song["section"][b]
        level = density * (style.verse if section == "verse" else INTENSITY[section])
        if style.contour and section == "chorus":
            level *= 1.0 - style.contour * (1.0 - song["rank"].get(b, 1.0))
        level = max(level, min(level * (1.0 + style.crux * song["lift"].get(b, 0.0)), 1.1))
        share = min(style.fills[2] if len(style.fills) > 2 else 1.0,
                    max(0.0, (level - style.fills[0]) / style.fills[1]))
        # which rows get one: the accents and the rows on a real sound sooner, but not
        # the same ones in every bar
        def rank(i):
            n = b * bar + i
            return (meter.first.get(i, 1.2) + 0.8 * ((n * 0.549703) % 1.0)
                    - 0.3 * (n * line in real))
        targets = sorted((i for i in range(bar) if (b * bar + i) * line in taps), key=rank)
        dither = (b * 0.618034) % 1.0
        for i in targets[:int(len(targets) * share + dither) if share else 0]:
            k = (b * bar + i) * line
            pick = ((b * bar + i) * 0.381966) % 1.0 * total
            for back, weight, triplet in shapes:
                pick -= weight
                if pick < 0:
                    break
            new = [k - d for d in back]
            between = k - line          # a triplet takes this line's place
            if (new[0] < first or (triplet and between in real)
                    or any(r != (between if triplet else k) for n in new
                           for r in _around(rows, n, finest))):
                continue
            moved = taps.pop(between, None) if triplet else None
            if moved is not None:
                rows.remove(between)
            for r in new:
                taps[r] = 1
                bisect.insort(rows, r)
            if not _room(rows, taps, new[0], k, gap, keys):
                for r in new:           # no room: put things back as they were
                    del taps[r]
                    rows.remove(r)
                if moved is not None:
                    taps[between] = moved
                    bisect.insort(rows, between)


def build_chart(song, density, rng, style, ln=None, comfort=0):
    """-> [(start position, column, end position or None)]"""
    ln = style.ln if ln is None else ln
    special = 0.0                       # share of the phrases written wholly in long notes
    inverse = None                      # ... and as inverse
    if style.weave and style.noodles:
        mix = rated(style, None).mix if style.mix is None else style.mix
        special = 2.0 * ln * mix if ln <= 0.5 else mix + (1.0 - mix) * (2.0 * ln - 1.0)
        if style.inverse:
            inv = rated(style, None).inv if style.inv is None else style.inv
            inverse = 2.0 * ln * inv if ln <= 0.5 else inv
        ln = min(1.0, 2.0 * ln)
    keys, line = song["keys"], song["line"]
    taps = plan(song, density, style)
    sung, stabs, section = song["sung"], song["stabs"], song["section"]
    bass = song["bass"] if style.bass and ln > 0 else {}
    # long notes: the longest `ln` share of the held sung notes, plus the cut-off chords
    short, ln = (ln, float(style.long and ln > 0)) if style.short else (0.0, ln)
    held = [e - k for k, (e, _, _) in sung.items() if e - k >= _hold(song, k, MIN_HOLD_MS)]
    min_hold = min((n for n in set(held) if sum(h >= n for h in held) <= ln * len(held)),
                   default=10**9)
    spaces = spacing(song, style)
    pat = Patterner(keys, line, spaces, rng, comfort, style)
    pat.shield_span = max(pat.near, _shield_span(song, spaces))
    off_line = song["between"] if style.detail else {}
    notes = []
    rows = sorted(set(taps) | (set(stabs) if density >= 0 else set()))
    phrase, prev_pitch = None, None
    weave = style.weave if style.weave and ln > 0 else None
    meter, last_hold = METERS[song["meter"]], None
    spans = ([meter.beat, 3 * meter.beat // 2, 2 * meter.beat] if meter.beat % 2 == 0
             else [meter.beat, 2 * meter.beat, 2 * meter.beat])     # in lines
    if song["tpb"] == 8:                # a charting bar of sixteen lines is then two beats
        spans = [2 * n for n in spans]
    for i, k in enumerate(rows):
        b = k // (song["bar"] * line)
        if b // 2 != phrase:
            phrase = b // 2
            pat.new_motif(section[b])
        into = i + 1 < len(rows) and rows[i + 1] - k <= line    # a row within the next line
        tight = i > 0 and k - rows[i - 1] <= line               # a row within the line before
        fast = into or tight
        free = pat.free_columns(k)
        n = min(taps.get(k, 1), len(free))
        # rows that come too soon after this one to reuse its columns: leave each a column
        j = i + 1
        while j < len(rows) and rows[j] - k < spaces[1]:
            n = min(n, len(pat.free_columns(rows[j])) - (j - i))
            j += 1
        holding = pat.holding(k)
        if weave and holding and density <= 1.3:
            n = min(n, max(1, len(free) - 1))       # under a hold, one column is left alone
                                                    # (until the chart is packed wall to wall)
        hit = song["drums"].get(k) or off_line.get(k) or ("", 0.0)
        real = k in song["drums"] or k in sung or k in stabs or k in bass or k in off_line
        accent = hit[0] in ("K", "S")
        if len(holding) >= 2 and fast and len({pat.hand[c] for c in free}) < 2:
            n = n if real else 0        # every free column on one hand: no trilling there
        options = []
        while n > 0 and not options:    # thin the chord until it sits comfortably
            options = [cols for cols in itertools.combinations(free, n)
                       if pat.comfortable(cols, k, tight, accent)]
            if not options and n == 1 and (real or style.steady):
                options = [(c,) for c in free if not pat.worn(c, k)]
                if style.ladder:        # a stream would rather keep a hand busy than repeat a column
                    options = [o for o in options if k - pat.last[o[0]] > pat.near] or options
                if not options and style.chain >= 99:   # a style without a jack limit keeps the row
                    options = list(itertools.combinations(free, 1))
            n -= not options
        if not options:
            continue
        starts, target = [], None       # long notes beginning here: (end tick, wanted column)
        if k in sung:
            end, pitch, target = sung[k]
            if prev_pitch is not None and abs(pitch - prev_pitch) >= 0.7:
                pat.steer(1 if pitch > prev_pitch else -1)
            prev_pitch = pitch
            if end - k >= max(min_hold, _hold(song, k, MIN_HOLD_MS)):
                starts.append((min(end, k + _hold(song, k, MAX_HOLD_MS)), target))
        if k in bass and bass[k][0] - k >= _hold(song, k, MIN_HOLD_MS):
            starts.append((min(bass[k][0], k + _hold(song, k, MAX_HOLD_MS)), bass[k][2]))
        elif k in stabs and ln > 0:
            far = target is None or target >= (keys - 1) / 2
            starts.append((stabs[k], 0.0 if far else float(keys - 1)))
        # a second hold may begin when what is held has half a beat or less to go
        handing = max((pat.held[c] for c in holding), default=k) - k <= meter.beat * line // 2
        if (weave and not starts and k % line == 0 and handing and len(holding) < 2
                and on_pulse(k // line % song["bar"], song["tpb"], song["meter"])):
            # nothing is held, or what is held is about to be let go: hold this note, in
            # a column of the other hand than last time (taking over before the other
            # hold ends only now and then)
            dice = (k // line * 0.618034) % 1.0
            if holding and (dice * 7.0) % 1.0 >= 0.4:
                dice = 1.0
            if dice < weave[0] * ln:
                span = spans[0 if dice < weave[0] * ln * weave[1] else
                             1 if dice < weave[0] * ln * (weave[1] + weave[2]) else 2]
                end = min(k + span * line, song["k_max"] * line)
                other = [c for c in range(keys) if last_hold is None
                         or pat.hand[c] != pat.hand[last_hold]]
                switch = [cols for cols in options if set(cols) & set(other)]
                if end - k >= meter.beat * line // 2 and switch:
                    options = switch
                    starts.append((end, float(other[int(dice * 997) % len(other)])))
        starts = starts[:max(0, min(n, MOST_HOLDS[keys] - len(holding)))]
        if weave and not handing:
            starts = []                 # one hold at a time there: the next may only overlap its end
        elif weave:
            starts = starts[:1]
        cols = ((pat.follow(options) if pat.steps else None)
                or min(options, key=lambda cols: pat.cost(cols, k, target, starts)))
        ends = pat.assign(cols, starts, k)[0]
        pat.press(cols, k, ends)
        last_hold = next(iter(ends), last_hold)
        for col in cols:
            notes.append((k, col, ends.get(col)))
    if short:
        return _short_holds(notes, song, spaces[2], style.short, short, rng, keys - 1)
    if style.hold_taps and ln > 0:
        return _short_holds(notes, song, spaces[2], style.hold_taps[1:],
                            style.hold_taps[0] * ln, rng, MOST_HOLDS[keys])
    if weave and style.noodles:
        notes = _independence(notes, song, spaces, special, ln, style.noodles, rng,
                              inverse, style.shields, bool(style.shield_cost))
    if weave and density > 0.9:
        return _more_holds(notes, song, spaces[2], min(0.6, density - 0.9) * ln, rng,
                           _shield_span(song, spaces) if style.shield_cost else 0)
    return notes


def _shield_span(song, spaces):
    """How far before a hold a shield sits, in positions: a quarter of a beat (a third
    in triplet meters), or as soon as the style lets a column be pressed again."""
    tpb, line = song["tpb"], song["line"]
    unit = (tpb // 3 if tpb % 3 == 0 else max(1, tpb // 4)) * line
    return unit * max(1, math.ceil(max(spaces[1], spaces[2]) / unit))


def _independence(notes, song, spaces, special, hold, noodles, rng, inverse=None,
                  shields=(0.5, 0.5), careful=False):
    """What makes a chart of woven holds train the fingers apart (the user's words for
    Coordination: "a lot of focus on shields, reverse shields, and inverse"). Each
    four-bar phrase is written one of three ways:

    shields (the usual one): a hold gets a tap in its own column a quarter beat before
      it starts (a shield), and one a quarter beat after it ends (a reverse shield),
      by the odds in `shields`, where the column and the row have room (and not in a
      bar without drums, which is left to its melody). `careful`: a shield only before
      a hold that starts alone and not at the end of a jack, and no hold made below
      that starts right after a press of its own column;
    noodles (`special` of the phrases, less the inverse ones; few at ordinary ratings,
      more towards 5.5 stars, see rated()): long notes used as normal
      notes. Most of the taps (`noodles[0]`) are held, a quarter, a half, three
      quarters of a beat or a whole one by the odds that follow, or one time in three
      for as long as their column allows, so that the finger lets go a quarter beat
      before it presses again. A third note is held at once only now and then;
    inverse (`inverse` of the phrases; None: about a third of the special ones): the
      phrase is thinned to its beats and half beats, and every note is held until a
      quarter beat before its column is pressed again, so nearly every finger is down
      and it is the gaps one plays.

    A hold ends on a line, never later than a quarter beat before the next press of its
    column, and the last column is never held with all the others."""
    line, tpb, keys, bar = song["line"], song["tpb"], song["keys"], song["bar"]
    _, gap, release = spaces
    share, odds = noodles[0], noodles[1:]
    unit = (tpb // 3 if tpb % 3 == 0 else max(1, tpb // 4)) * line      # a quarter (third) of a beat
    apart = _shield_span(song, spaces)
    phrase = 4 * bar * line

    if inverse is None:
        inverse, noodled = 0.35 * special, special
    else:
        noodled = max(special, inverse + 0.65 * special)

    def kind(k):
        dice = (k // phrase * 0.618034) % 1.0
        return "inverse" if dice < inverse else "noodles" if dice < noodled else "shields"

    notes = sorted((n for n in notes if n[2] is not None or kind(n[0]) != "inverse"
                    or (n[0] % line == 0 and on_pulse(n[0] // line % bar, tpb, song["meter"]))),
                   key=lambda n: (n[0], n[1]))
    presses = {c: sorted(k for k, col, _ in notes if col == c) for c in range(keys)}
    free = {c: sorted((k + gap if e is None else e + release) for k, col, e in notes if col == c)
            for c in range(keys)}       # when each column may be pressed again, press by press
    held = sorted((k, e) for k, _, e in notes if e is not None)
    rows = {}
    for k, _, _ in notes:
        rows[k] = rows.get(k, 0) + 1

    def down(a, b):                     # how many holds are down at some time between a and b
        at = bisect.bisect_left(held, (a, 0))
        return sum(1 for h in held[max(0, at - 32):at + 32] if h[0] < b and h[1] > a)

    def tap(p, c):                      # room for one more tap there?
        i = bisect.bisect_left(presses[c], p)
        before = free[c][i - 1] if i else p
        after = presses[c][i] if i < len(presses[c]) else p + gap
        return (before <= p and p + gap <= after and rows.get(p, 0) < 2
                and rows.get(p, 0) + down(p, p + 1) < keys - 1
                and song["k_min"] * line <= p <= song["k_max"] * line          # inside the song
                and song["section"].get(p // (bar * line)) in ("verse", "chorus"))    # not in a break

    out, tapped = [], set()             # (position, column) of every press left a tap
    for k, c, e in notes:
        what = kind(k)
        if e is None and k % line == 0 and what != "shields":
            after = bisect.bisect_right(presses[c], k)
            if (careful and after >= 2 and k - presses[c][after - 2] <= apart
                    and (presses[c][after - 2], c) in tapped):
                tapped.add((k, c))              # held, it would make a shield: left a tap
                out.append((k, c, e))
                continue
            room = (presses[c][after] - apart - k if after < len(presses[c]) else 8 * unit) // unit * unit
            length = 0
            if what == "inverse":
                length = min(room, 8 * unit)
                while length >= unit and down(k, k + length) >= keys - 1:
                    length -= unit
            elif rng.random() < share:
                want = unit * rng.choices((1, 2, 3, 4), weights=odds)[0]
                length = min(room, 4 * unit) if rng.random() < 0.35 else min(want, room)
                third, brief = rng.random() < 0.5, rng.random() < 0.7
                for length in (length, unit, 0):    # if too much would be held at once: a short one
                    beside = down(k, k + length)
                    if length >= unit and (beside < 2 or (beside == 2 and (third or (brief and length == unit)))):
                        break
            if length >= unit:
                e = k + length
                bisect.insort(held, (k, e))
                i = bisect.bisect_left(presses[c], k)
                free[c][i] = e + release
        elif e is not None and what == "shields":
            for p, wanted in ((k - apart, rng.random() < shields[0] * hold),
                              (e + apart, rng.random() < shields[1] * hold)):
                if careful:
                    i = bisect.bisect_left(presses[c], p)
                    if p < k and (rows.get(k, 0) > 1 or (i and p - presses[c][i - 1] <= apart)):
                        continue                # a shield only before a lone hold, and not at
                                                # the end of a jack
                    if p > k and i < len(presses[c]) and presses[c][i] - p <= apart:
                        continue                # nor a reverse shield at the start of one
                if wanted and tap(p, c):
                    i = bisect.bisect_left(presses[c], p)
                    presses[c].insert(i, p)
                    free[c].insert(i, p + gap)
                    rows[p] = rows.get(p, 0) + 1
                    tapped.add((p, c))
                    out.append((p, c, None))
        if e is None:
            tapped.add((k, c))
        out.append((k, c, e))
    return sorted(out, key=lambda n: (n[0], n[1]))


def _more_holds(notes, song, release, share, rng, careful=0):
    """For a chart of woven holds that is full: there is no room left for more notes,
    so `share` of its taps become short holds instead, a quarter or half a beat long.
    Never a third note held at once, never into the next press of its own column,
    and never in the column of the hold before (`careful`: nor right after a press of
    its own column, within `careful` positions, which would make a shield)."""
    beat = song["tpb"] * song["line"]
    lengths = [beat // 4, beat // 2] if song["tpb"] % 3 else [beat // 3, 2 * beat // 3]
    presses = {c: sorted(k for k, col, _ in notes if col == c) for c in range(song["keys"])}
    held = sorted((k, e, c) for k, c, e in notes if e is not None)
    tapped = {(k, c) for k, c, e in notes if e is None}
    out = []
    for k, c, e in sorted(notes, key=lambda n: (n[0], n[1])):
        if e is None and rng.random() < share:
            end = k + rng.choice(lengths)
            after = bisect.bisect_right(presses[c], k)
            if after < len(presses[c]):
                end = min(end, presses[c][after] - release)
            at = bisect.bisect_left(held, (k, 0, 0))
            before = held[at - 1][2] if at else None
            beside = [h for h in held[max(0, at - 16):at + 16] if h[0] < end and h[1] > k]
            shield = (careful and after >= 2 and k - presses[c][after - 2] <= careful
                      and (presses[c][after - 2], c) in tapped)
            if end - k >= lengths[0] // 2 and len(beside) < 2 and before != c and not shield:
                e = end
                bisect.insort(held, (k, e, c))
                tapped.discard((k, c))
        out.append((k, c, e))
    return out


def _short_holds(notes, song, release, odds, share, rng, most):
    """Turn `share` of the taps into short long notes: a quarter, a half or a whole
    beat long (thirds of a beat in triplet meters), by `odds`. Never into the next
    press of its own column, and never with `most` notes already held."""
    line, keys, beat = song["line"], song["keys"], song["tpb"] * song["line"]
    lengths = ([beat // 4, beat // 2, beat] if song["tpb"] % 3
               else [beat // 3, 2 * beat // 3, beat])
    presses = {c: sorted(k for k, col, _ in notes if col == c) for c in range(keys)}
    out, running = [], []
    for k, c, e in sorted(notes, key=lambda n: (n[0], n[1])):
        if e is not None or rng.random() >= share:
            out.append((k, c, e))
            running += [e] if e is not None else []
            continue
        end = k + rng.choices(lengths, weights=odds)[0]
        mine = presses[c]
        after = bisect.bisect_right(mine, k)
        if after < len(mine):
            end = min(end, mine[after] - release)
        running = [r for r in running if r > k]
        if end - k < min(lengths[0], line) // 2 or len(running) >= most:
            out.append((k, c, None))
        else:
            out.append((k, c, end))
            running.append(end)
    return out


def validate(notes, keys, gap, release):
    """Safety net: drop anything that would overlap or repeat a column too fast."""
    by_col = {c: [] for c in range(keys)}
    for k, c, e in notes:
        by_col[c].append((k, e))
    bad = set()
    for c, lst in by_col.items():
        lst.sort(key=lambda n: n[0])
        free = None
        for k, e in lst:
            if free is not None and k < free:
                bad.add((k, c))
            else:
                free = k + gap if e is None else e + release
    return [n for n in notes if (n[0], n[1]) not in bad], len(bad)


def lay_out(song, density, seed, style, ln=None, comfort=0):
    """One finished chart at a fixed density -> (notes, stars)."""
    _, gap, release = spacing(song, style)
    notes, _ = validate(build_chart(song, float(density), random.Random(seed), style, ln,
                                    comfort), song["keys"], gap, release)
    stars = (star_rating(to_ms(notes, song["T0"], song["line_ms"], song["line"],
                               song.get("clock")), song["keys"]) if notes else 0.0)
    return notes, stars


def to_ms(notes, T0, line_ms, line, clock=None):
    """Positions -> the integer millisecond times written to the .osu, in file order.
    With a `clock` (a song whose tempo changes) the positions are in its straightened
    time and are put back into the song's own."""
    if clock is not None and notes:
        starts = clock.real_time([T0 + k / line * line_ms for k, _, _ in notes])
        ends = clock.real_time([T0 + (k if e is None else e) / line * line_ms for k, _, e in notes])
        out = [(int(round(s)), c, None if e is None else int(round(x)))
               for (_, c, e), s, x in zip(notes, starts, ends)]
        out.sort(key=lambda n: (n[0], n[1]))
        return out
    out = [(int(round(T0 + k / line * line_ms)), c,
            None if e is None else int(round(T0 + e / line * line_ms)))
           for k, c, e in notes]
    out.sort(key=lambda n: (n[0], n[1]))
    return out


def suit_range(style, stars):
    """How far apart (ms) a style's rows should come for it to play as meant at a
    rating of `stars` -> (from, to, lines from one row to the next), or None if any
    tempo will do. A Speedjack's jacks are as fast as its rating (`quick`); a stream,
    which never repeats a column on the next row, reaches a rating only on rows close
    enough together (`reach`)."""
    if not style.suits:
        return None
    low, high, lines = style.suits
    if style.quick:
        top, per = style.quick
        low = max(low, low + per * (top - stars))
        high = max(high, low + 17.0)
    if style.reach:
        plain, quads = style.reach
        high = max(low + 5.0, min(high, (quads if style.quads and stars >= style.quads
                                         else plain) / stars))
    return low, high, lines


def grid_for(style, beat, base, stars=None):
    """Lines per beat a style charts a song on: the song's own (`base`), unless the
    style is built for rows a certain time apart and twice as fine a grid comes closer
    to it. A style whose row spacing depends on the rating (`quick`, Speedjack) takes,
    for a rating of `stars`, the grid that puts its rows inside the spacing written for
    that rating (suit_range): a straight song's 1/4 or 1/8, or its 1/3 or 1/6
    (triplets) where those come clearly closer, so that its jacks are as fast as the
    rating and no faster (the user: "divide into triplets the time of the hits to
    balance it to 4.5 stars ... until around 5 stars, it would be slower than the rapid
    subdivisions of the song")."""
    return grids_for(style, beat, base, stars)[0]


def grids_for(style, beat, base, stars=None):
    """The grids grid_for chooses from, best first."""
    if not style.rows or base not in (3, 4):
        return [base]
    want, lines = style.rows
    own = (base, 2 * base)
    if style.quick and stars is not None:
        low, high, lines = suit_range(style, stars)
        middle = math.sqrt(low * high)

        def off(tpb):                   # how far outside that spacing, and from its middle
            ms = lines * beat / tpb
            return (max(0.0, math.log(low / ms), math.log(ms / high)) + (0.0 if tpb in own else 0.04),
                    abs(math.log(ms / middle)))
        # triplets only ever to slow it down: 1/3 where the song's 1/4 is too fast for
        # the rating, 1/6 where its 1/4 is far too slow (and its 1/8 would be twice as fast)
        quarter = lines * beat / 4
        extra = (((3,) if quarter < low else ()) + ((6,) if quarter > 1.25 * high else ())
                 if base == 4 else ())
        return sorted(own + extra, key=off)
    return sorted(own, key=lambda tpb: abs(math.log(lines * beat / tpb / want)))


def meter_for(tpb, own, meter):
    """The meter a song is charted in on a grid of `tpb` lines a beat, where its own grid
    is `own` lines a beat in `meter`: six lines always chart as compound, and three as
    four beats of three (a straight song taken in triplets)."""
    if tpb == 6:
        return "compound"
    if tpb == 3 and own != 3:
        return "swing"
    if tpb in (4, 8) and own == 3:
        return "straight"
    return meter


def rated(style, stars):
    """The style as it is charted for a rating of `stars` (None: an ordinary 4.5).
    Coordination writes some phrases in long notes only, and how many depends on the
    rating: rare at ordinary ones, more common towards 5.5 stars (the user: "i think
    they can be rare, but more common in the higher star rating (around 5.5 ish)").
    Chords of four come only from `quads` up, and how long a jack may run depends on
    the rating where the style says so (`jacks`)."""
    s, changes = 4.5 if stars is None else stars, {}
    if style.phrases:
        low, rare, high, common = style.phrases
        part = min(1.0, max(0.0, (s - low) / (high - low)))
        changes["mix"] = rare + (common - rare) * part
    if style.inverse:
        low, rare, high, common = style.inverse
        part = min(1.0, max(0.0, (s - low) / (high - low)))
        changes["inv"] = rare + (common - rare) * part
    if style.quads is not None and s >= style.quads:
        changes["cap"] = style.cap + 1
    if style.jacks:
        changes["chain"] = next((n for at, n in reversed(style.jacks) if s >= at), style.jacks[0][1])
    return replace(style, **changes) if changes else style


def _peaks(song, notes, bars=8):
    """How hard the chart is at its hardest, over any `bars` bars, in the song's loudest
    stretches and outside them -> (stars, stars), or None if it has no such stretches."""
    lift, line, bar = song["lift"], song["line"], song["bar"]
    if not any(v >= 0.5 for v in lift.values()):
        return None
    by_bar = {}
    for n in notes:
        by_bar.setdefault(n[0] // (bar * line), []).append(n)
    loud, rest = [], []
    for b in range(song["bars"][0], song["bars"][-1] + 1, bars // 2):
        part = [n for j in range(b, b + bars) for n in by_bar.get(j, ())]
        high = sum(lift.get(j, 0.0) >= 0.5 for j in range(b, b + bars))
        if len(part) >= 16 and (high >= 0.75 * bars or high == 0):
            (loud if high else rest).append(star_rating(
                to_ms(part, song["T0"], song["line_ms"], line, song.get("clock")), song["keys"]))
    return (max(loud), max(rest)) if loud and rest else None


def tune(song, target, seed, style, ln=None, progress=None):
    """The chart for a star rating (see _tune). The song's loudest stretches are charted
    fuller than the rest (Style.crux), which makes them its hardest part; if that
    leaves them more than half a star above everything else, they are eased until
    they are not, so that they are the high point of the chart and not all of it."""
    style = rated(style, target)
    best = _tune(song, target, seed, style, ln, progress)
    peaks = _peaks(song, best[2]) if style.crux else None
    if not peaks or peaks[0] - peaks[1] <= 0.5:
        return best
    low, high, best = 0.0, style.crux, None
    for _ in range(4):                  # the most lift that keeps them within half a star
        crux = (low + high) / 2.0
        trial = _tune(song, target, seed, replace(style, crux=crux), ln, progress)
        peaks = _peaks(song, trial[2])
        if peaks and peaks[0] - peaks[1] > 0.5:
            high = crux
        else:
            low, best = crux, trial
    return best or _tune(song, target, seed, replace(style, crux=0.0), ln, progress)


def _tune(song, target, seed, style, ln=None, progress=None):
    """Find the sparsest, most comfortable chart whose osu! star rating hits the target.
    Comfort is given up before the chart is packed wall to wall; only if no comfort
    level gets there are the densest settings tried, and if even those fall short the
    closest chart from the gentler half of the ladder is kept. If the bare song is
    already harder than asked, its hits are thinned out instead.
    Returns (density, comfort, notes, stars)."""
    def trial(d, comfort):
        return lay_out(song, d, seed, style, ln, comfort)

    rungs = list(range(style.comfort, min(len(COMFORT), style.harshest + 1)))
    if trial(0.0, rungs[0])[1] > target + 0.005:
        passes, past = [(rungs[0], np.arange(-0.01, -1.001, -0.01))], lambda s: s <= target + 0.005
    else:
        passes = [(c, np.arange(0.0, 1.301, 0.01)) for c in rungs]
        passes.append((rungs[-1], np.arange(1.31, 1.801, 0.01)))
        past = lambda s: s >= target - 0.005
    best = gentle = None
    for n, (comfort, densities) in enumerate(passes):
        if progress:
            progress(n / len(passes))
        for d in densities:
            notes, stars = trial(d, comfort)
            if best is None or abs(stars - target) < abs(best[3] - target) - 1e-9:
                best = (round(float(d), 3), comfort, notes, stars)
                if comfort <= max(2, rungs[0]):
                    gentle = best
            if past(stars):
                break
        else:
            continue
        break
    if abs(best[3] - target) > 0.05 and gentle is not None:
        best = gentle                   # out of reach anyway: no point in the harsh patterns
    # the rating moves in small jumps; a fine sweep around the crossing gets closest
    lo = best[0] - 0.02 if best[0] < 0 else max(0.0, best[0] - 0.02)
    for d in np.arange(lo, best[0] + 0.0201, 0.001):
        notes, stars = trial(d, best[1])
        if abs(stars - target) < abs(best[3] - target) - 1e-9:
            best = (round(float(d), 3), best[1], notes, stars)
    if style.reach and best[3] < target - 0.05 and best[0] > 0:
        # a song too slow for the rating (a stream that never repeats a column tops out
        # lower): the lightest chart within a tenth of a star of the most it reaches, so
        # that its verses stay lighter than its choruses instead of all of it packed
        low, high = 0.0, best[0]
        for _ in range(7):
            d = round((low + high) / 2.0, 3)
            notes, stars = trial(d, best[1])
            if stars >= best[3] - 0.1:
                high, best = d, (d, best[1], notes, stars)
            else:
                low = d
    return best
