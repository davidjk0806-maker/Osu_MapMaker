"""One song in, one .osz out."""

import collections
import hashlib
import math
import os
import pickle
import re
import shutil
import tempfile
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

from .audio import CACHE, SR, MapError, pack_audio, preview_wav, read_audio, separate
from .chart import (HANDS, STYLES, grid_for, grids_for, lay_out, meter_for, rated, suit_range,
                    to_ms, tune)
from .recommend import AUTO, explain, rank
from .listen import RES, listen, prepare, snap_of
from .osufile import make_osu, pack_osz, safe_name
from .stars import shape, star_rating
from .timing import (CHROMA_HOP, N_FFT, Clock, attack_slope, find_downbeat, find_timing, flux,
                     follow_tempo, in_time, onset_envelope, stft_power)

ANALYSIS_VERSION = 7          # bump when what gets cached per song changes


@dataclass
class Options:
    stars: float = 4.5
    keys: int = 4
    style: str = "coordination"
    ln: float = None            # share of held notes charted as LNs; None = the style's own
    seed: int = 2016            # another seed = other patterns on the same rhythm
    creator: str = "CASNoob"
    title: str = ""             # blank = guessed from the file name
    artist: str = ""
    version: str = ""           # difficulty name; blank = e.g. "4K Stream"
    bpm: float = None           # a BPM to search around; with `offset`, taken as exact
    offset: float = None        # ms of the first beat of any bar
    tempo: str = "auto"         # "half" / "double" the detected BPM
    nudge: float = 0.0          # shift every note and the red line by this many ms
    od: float = 8.0
    hp: float = 8.0
    bg: str = None              # background image path
    density: float = None       # fixed note density instead of tuning to `stars`
    comfort: int = None         # with `density`: rung of the comfort ladder


def guess_tags(filename):
    """("artist", "title") from a file name like 'Artist - Title (Official Video) [id].mp3'."""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"\s*\[[\w-]{6,}\]\s*$", "", name)                    # a video id
    junk = r"official|lyric|video|audio|visuali[sz]er|hd|4k|hq|mv|remaster"
    name = re.sub(rf"\s*[\(\[][^\)\]]*\b(?:{junk})\b[^\)\]]*[\)\]]", "", name, flags=re.I)
    name = re.sub(r"[_\s]+", " ", name).strip(" -")
    artist, sep, title = name.partition(" - ")
    return (artist.strip(), title.strip()) if sep and title.strip() else ("", name)


def song_key(path):
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()[:12]


def _measure(stems, mix, dur_ms, no_drums, hint=None, hits=None):
    """BPM, beat phase and downbeat. Drums are timed on their own stem; a song without
    drums is timed on the whole mix. With the drum `hits` at hand, a song that one
    tempo does not fit is followed through its tempo changes as well ("moving")."""
    source = mix if no_drums else stems["drums"]
    env, slope = onset_envelope(stft_power(source)), attack_slope(source, SR)
    bpm, tpb, off, lag, info = find_timing(env, slope, SR, hint)
    power = stft_power(mix)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    low = flux(power[(freqs > 30) & (freqs < 180)].sum(axis=0, keepdims=True))
    low = low / (np.percentile(low, 99) + 1e-9)
    chroma = librosa.feature.chroma_stft(y=mix, sr=SR, n_fft=4096, hop_length=CHROMA_HOP)
    T0 = find_downbeat(low, chroma, dur_ms, off, 60000.0 / bpm, lag)
    timing = {"bpm": bpm, "tpb": tpb, "T0": T0, "info": info}
    if hits and hint is None and info["feel"] == "straight":
        steady = in_time(hits, (T0, 60000.0 / bpm / tpb), dur_ms)[0]
        if steady < 0.7:
            moving = follow_tempo(env, slope, SR, dur_ms, low, chroma)
            if moving and in_time(hits, moving["lines"], dur_ms)[0] >= max(0.6, steady + 0.2):
                timing["moving"] = moving
    return timing


def on_grid(raw, span, T0, beat, tpb, keys, meter, clock=None):
    """What was heard, put on the grid (listen.prepare). With a `clock`, `raw` and the
    rest are in the straightened time of a song whose tempo changes, and the song is
    told how much longer each of its bars really lasts ("slow")."""
    song = prepare(raw, span, T0, beat, tpb, keys, meter, clock.stretch if clock else None)
    if clock:
        edges = clock.real_time([T0 + b * song["bar_len"]
                                 for b in song["bars"] + [song["bars"][-1] + 1]])
        song["clock"] = clock
        song["slow"] = {b: float(edges[n + 1] - edges[n]) / song["bar_len"]
                        for n, b in enumerate(song["bars"])}
    return song


def analyse(data, sr, key, say):
    """Everything about a song that doesn't depend on the chart settings. Cached, so
    asking for another key count, style or star rating skips straight to charting."""
    path = os.path.join(CACHE, f"analysis-{key}.pkl")
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            done = pickle.load(fh)
        if done.get("version") == ANALYSIS_VERSION:
            return done
        heard = done.get("raw") if done.get("version") in (5, 6) else None   # only the timing is new
    else:
        heard = None
    mono = data.mean(axis=1).astype(np.float32)
    dur_ms = len(mono) / sr * 1000.0
    say("Separating the instruments", 0.0)
    stems = separate(data, sr, key, lambda f: say("Separating the instruments", f))
    mix = librosa.resample(mono, orig_sr=sr, target_sr=SR) if sr != SR else mono
    say("Listening to drums, voice and bass", 0.0)
    raw = heard or listen(stems, mix, lambda f: say("Listening to drums, voice and bass", f))
    say("Measuring the tempo", 0.0)
    done = {"version": ANALYSIS_VERSION, "dur_ms": dur_ms, "raw": raw,
            "timing": _measure(stems, mix, dur_ms, raw["no_drums"], hits=raw["hits"])}
    preview_wav(mix, key)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(done, fh, protocol=4)
    return done


def _kiai(song):
    """Runs of chorus at least four musical bars long."""
    T0, bar_len, section = song["T0"], song["bar_len"], song["section"]
    runs, start = [], None
    for b in song["bars"] + [song["bars"][-1] + 1]:
        hot = section.get(b) == "chorus"
        if hot and start is None:
            start = b
        elif not hot and start is not None:
            runs.append((start, b))
            start = None
    return [(int(round(T0 + s * bar_len)), int(round(T0 + e * bar_len)))
            for s, e in runs if (e - s) * song["bar"] >= 16 * song["tpb"]]


def _snaps(notes, song):
    """Which snaps the rows sit on -> (the editor grid that shows most of them, the
    snaps finer than the chart's own lines as ["1/8", ...], most used first)."""
    rows = {k for k, _, _ in notes}
    finer = collections.Counter(snap_of(k) for k in rows if k % song["line"])
    shown = {d: sum(1 for k in rows if k * d % RES == 0)
             for d in [song["tpb"]] + [d for d in (6, 8, 12, 16) if d > song["tpb"]]}
    best = next((d for d, n in shown.items() if n >= 0.95 * len(rows)),
                max(shown, key=shown.get))
    return best, [f"1/{d}" for d, n in finer.most_common() if n >= 0.01 * len(rows)]


def _fit(style, beat, tpb, stars):
    """How far apart (ms) a style's rows sit at this tempo, whether that is the spacing
    it is meant for at a rating of `stars`, and that spacing (chart.suit_range)."""
    span = suit_range(style, stars)
    if not span:
        return None, True, None
    low, high, lines = span
    spacing = lines * beat / tpb
    return spacing, low <= spacing <= high, span


def _stats(objects):
    times = [t for t, _, _ in objects]
    span = max(1.0, (times[-1] - times[0]) / 1000.0)
    peak, j = 0, 0
    for i in range(len(times)):
        while times[i] - times[j] >= 1000:
            j += 1
        peak = max(peak, i - j + 1)
    return len(times) / span, peak


def _play(objects, keys, stars, beat):
    """How a chart plays, in the numbers the page shows with it: chord sizes, how far
    apart its rows usually are, its jacks, and how much of it plays near its rating
    (stars.shape)."""
    rows = collections.defaultdict(set)
    for t, c, _ in objects:
        rows[t].add(c)
    times = sorted(rows)
    sizes = collections.Counter(min(len(cols), 4) for cols in rows.values())
    gaps = [b - a for a, b in zip(times, times[1:]) if b - a <= beat + 3]
    spacing = float(np.median(gaps)) if gaps else beat
    snap = min((1, 2, 3, 4, 6, 8, 12, 16), key=lambda d: abs(beat / d - spacing))
    # jacks: a column in rows that follow one another, no slower than the usual spacing
    jack_ms, longest, run = [], 1, [1] * keys
    for a, b in zip(times, times[1:]):
        quick = b - a <= max(1.15 * spacing, spacing + 3)
        for c in range(keys):
            if quick and c in rows[a] and c in rows[b]:
                jack_ms.append(b - a)
                run[c] += 1
                longest = max(longest, run[c])
            elif c in rows[b]:
                run[c] = 1
    like, near = shape(objects, keys, stars)
    return {"chords": [round(100.0 * sizes.get(n, 0) / len(times)) for n in range(1, min(keys, 4) + 1)],
            "spacing": round(spacing), "snap": f"1/{snap}",
            "jack_ms": round(sorted(jack_ms)[len(jack_ms) // 10]) if jack_ms else None,
            "jacks": round(100.0 * len(jack_ms) / max(1, len(objects))),
            "longest_jack": longest if jack_ms else 0,
            "like": round(like, 2), "near": round(100.0 * near)}


def _in_time(timing, opts, raw, dur_ms):
    """The song's time as it is charted -> (raw, span, bpm, beat, T0, lines per beat,
    meter, info, clock). `timing` is what was measured (or set by hand). A song whose
    tempo changes is charted in a straightened time where one tempo holds
    (timing.Clock), unless BPM, offset or half/double tempo hold it to one; `raw` and
    `span` are then in that time, and `bpm` is the tempo held longest."""
    clock = None
    if timing.get("moving") and not opts.bpm and opts.offset is None and opts.tempo == "auto":
        clock = Clock(**timing["moving"])
    bpm, T0, info, base = timing["bpm"], timing["T0"], timing["info"], timing["tpb"]
    meter = info["feel"] if info else "straight"
    if opts.offset is not None:
        T0 = float(opts.offset)
    if opts.tempo in ("half", "double"):
        bpm = bpm / 2 if opts.tempo == "half" else bpm * 2
    beat = 60000.0 / bpm
    T0 = float(round(T0 % (4 * beat)))
    span = dur_ms
    if clock:
        # the tempo changes: chart the song in a straightened time, where it holds
        raw, span = clock.straighten(raw), float(clock.even_time(dur_ms))
        beat, T0, base, meter = clock.beat, clock.T0, 4, "straight"
        bpm = clock.tempo()[0]
    return raw, span, bpm, beat, T0, base, meter, info, clock


def _ranked(raw, span, bpm, beat, T0, own, meter, clock, keys, stars):
    """The styles, best fit for the song first (recommend.rank), from its time as it
    is charted (_in_time)."""
    song = on_grid(raw, span, T0, beat, own, keys, "compound" if own == 6 else meter, clock)
    return rank(raw, song, 60000.0 / bpm if clock else beat, stars)


def recommend_for(key, stars, keys=4, tempo="auto"):
    """The styles ranked for a song that has been listened to already, at the tempo
    found (or half or double it); None if it hasn't been."""
    try:
        with open(os.path.join(CACHE, f"analysis-{key}.pkl"), "rb") as fh:
            heard = pickle.load(fh)
    except (OSError, EOFError, pickle.UnpicklingError):     # not yet, or still being written
        return None
    if heard.get("version") != ANALYSIS_VERSION:
        return None
    raw, span, bpm, beat, T0, own, meter, _, clock = _in_time(
        heard["timing"], Options(tempo=tempo), heard["raw"], heard["dur_ms"])
    return _ranked(raw, span, bpm, beat, T0, own, meter, clock, keys, stars)


def make_map(audio_path, opts, out_dir, progress=None):
    """Chart `audio_path` with `opts` and write the .osz (and its .osu) into `out_dir`.
    `progress(stage, fraction)` is called as work goes on. Returns a dict describing
    the result: file paths, stats, warnings, and the chart itself for previewing.
    Style "auto" charts the song in the style that suits it best (recommend.rank)."""
    say = progress or (lambda stage, fraction: None)
    if opts.keys not in HANDS:
        raise MapError(f"Keys must be one of {sorted(HANDS)}.")
    if opts.style not in STYLES and opts.style != AUTO:
        raise MapError(f"Style must be one of {sorted(STYLES)}, or {AUTO} to have one picked.")
    if not 0.5 <= opts.stars <= 10.0:
        raise MapError("Stars should be between 0.5 and 10.")
    if opts.bg and not os.path.isfile(opts.bg):
        raise MapError(f"Can't find background image: {opts.bg}")
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix=".build-", dir=out_dir)
    try:
        say("Reading the audio", 0.0)
        key = song_key(audio_path)
        data, sr = read_audio(audio_path, work)
        if len(data) < sr * 10:
            raise MapError("That audio is shorter than 10 seconds - is it the right file?")
        heard = analyse(data, sr, key, say)
        raw, dur_ms = heard["raw"], heard["dur_ms"]
        if not os.path.isfile(os.path.join(CACHE, f"preview-{key}.wav")):
            mono = data.mean(axis=1).astype(np.float32)
            preview_wav(librosa.resample(mono, orig_sr=sr, target_sr=SR), key)

        # --- timing
        timing = heard["timing"]
        if opts.bpm and opts.offset is not None:
            timing = {"bpm": float(opts.bpm), "tpb": 4, "T0": float(opts.offset), "info": None}
        elif opts.bpm:
            say("Measuring the tempo", 0.0)
            mono = data.mean(axis=1).astype(np.float32)
            mix = librosa.resample(mono, orig_sr=sr, target_sr=SR) if sr != SR else mono
            timing = _measure(separate(data, sr, key), mix, dur_ms, raw["no_drums"], opts.bpm)
        heard_raw = raw
        raw, span, bpm, beat, T0, base, meter, info, clock = _in_time(timing, opts, raw, dur_ms)
        own = base                      # the song's own lines per beat
        felt = 60000.0 / bpm if clock else beat     # for a tempo that changes: the one held longest

        # --- style: the one asked for, or the one that suits the song best
        chosen, picked = opts.style, None
        if chosen == AUTO:
            picked = _ranked(raw, span, bpm, beat, T0, own, meter, clock, opts.keys, opts.stars)
            chosen = picked[0]["style"]
        style = STYLES[chosen]

        # --- chart; twice as fine a grid only if the usual one can't reach the rating
        def chart(tpb):
            # six lines to a beat always chart as compound, whether detected or refined to,
            # and three as four beats of three
            song = on_grid(raw, span, T0, beat, tpb, opts.keys, meter_for(tpb, own, meter), clock)
            if opts.density is None:
                density, comfort, notes, stars = tune(
                    song, opts.stars, opts.seed, style, opts.ln,
                    lambda f: say("Writing the chart", f / 2 if tpb == base else 0.5 + f / 2))
            else:
                density, comfort = opts.density, opts.comfort or style.comfort
                notes, stars = lay_out(song, density, opts.seed, rated(style, opts.stars),
                                       opts.ln, comfort)
            return song, density, comfort, notes, stars

        say("Writing the chart", 0.0)
        grids = grids_for(style, beat, own, opts.stars)
        base = grids[0]
        song, density, comfort, notes, stars = chart(base)
        if (opts.density is None and stars < opts.stars - 0.05 and base in (3, 4)
                and beat / (2 * base) >= 45.0 and not style.rows and style.finer):
            finer = chart(2 * base)
            if abs(finer[4] - opts.stars) < abs(stars - opts.stars):
                song, density, comfort, notes, stars = finer
        elif (opts.density is None and stars < opts.stars - 0.15 and style.quick
              and len(grids) > 1 and grids[1] > base
              and beat / grids[1] >= 0.9 * suit_range(style, opts.stars)[0]):
            # rows as far apart as the rating is written for fell well short of it: the
            # next closest grid, if it is only a little faster, and gets nearer
            faster = chart(grids[1])
            if abs(faster[4] - opts.stars) < abs(stars - opts.stars):
                song, density, comfort, notes, stars = faster
                base = grids[1]
        if len(notes) < 20:
            raise MapError("Found almost nothing to chart - is the file silent?")

        # --- audio for the package
        say("Packing the .osz", 0.0)
        probe = sf.info(audio_path) if audio_path.lower().endswith(".ogg") else None
        if probe is not None and probe.subtype == "VORBIS":
            audio_file, lag_ms = audio_path, 0.0
        else:
            audio_file, lag_ms = pack_audio(data, sr, key)
        shift = int(round(lag_ms + opts.nudge))
        if clock:
            def in_song(ms):          # a time on the straightened chart -> ms in the packed audio
                return int(round(float(clock.real_time(ms)))) + shift
            objects = [(t + shift, c, None if e is None else e + shift) for t, c, e
                       in to_ms(notes, T0, song["line_ms"], song["line"], clock)]
            red = [(int(round(at)) + shift, length, bar) for at, length, bar in clock.red_lines()]
            kiai = [(in_song(s), in_song(e)) for s, e in _kiai(song)]
        else:
            T0 += shift
            objects = to_ms(notes, T0, song["line_ms"], song["line"])
            red = [(T0, beat, True)]
            kiai = [(s + shift, e + shift) for s, e in _kiai(song)]
        divisor, finer = _snaps(notes, song)
        preview_ms = kiai[0][0] if kiai else int(0.4 * dur_ms)
        guess = guess_tags(audio_path)

        def line(text):                 # metadata lands inside the .osu: one line each
            return re.sub(r"[\x00-\x1f]+", " ", text or "").strip()

        meta = {"title": line(opts.title) or guess[1] or "Untitled",
                "artist": line(opts.artist) or guess[0] or "Unknown Artist",
                "creator": line(opts.creator) or "mapmaker",
                "version": line(opts.version) or f"{opts.keys}K {style.label}",
                "tags": f"{opts.keys}k {chosen}", "keys": opts.keys,
                "od": opts.od, "hp": opts.hp, "divisor": divisor}
        bg_name = ("bg" + os.path.splitext(opts.bg)[1].lower()) if opts.bg else None
        text = make_osu(meta, "audio.ogg", bg_name, red, objects, kiai, preview_ms)
        stem = safe_name(f"{meta['artist']} - {meta['title']} ({meta['creator']}) "
                         f"[{meta['version']}]")
        osu_path = os.path.join(out_dir, stem + ".osu")
        with open(osu_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        osz_path = os.path.join(out_dir, stem + ".osz")
        files = {stem + ".osu": osu_path, "audio.ogg": audio_file}
        if bg_name:
            files[bg_name] = opts.bg
        pack_osz(osz_path, files)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # --- what to tell the person
    real = set(song["drums"]) | set(song["sung"]) | set(song["stabs"])
    if style.bass:
        real |= set(song["bass"])
    if style.detail:
        real |= set(song["between"])
    if style.hush:
        real |= set(song["tune"])
    kinds = [song["section"][b] for b in song["bars"]]
    nps_avg, nps_peak = _stats(objects)
    warnings = []

    def minutes(s):
        return f"{s // 60}:{s % 60:02d}"

    if clock:
        _, low, high = clock.tempo()
        steady, where = in_time(heard_raw["hits"], clock.lines, dur_ms)
        warnings.append(
            f"This song changes tempo, from {low:g} to {high:g} BPM. The map follows it, in "
            f"{len(clock.parts)} timing sections."
            + ("" if steady >= 0.9 else " The beat could be read clearly only at "
               + (", ".join(f"{minutes(a)}-{minutes(b)}" for a, b in where[:5]) or "few places")
               + "; elsewhere the notes may be a little off."))
        info = None                     # what was measured against one tempo does not apply
    else:
        if info and info["drift"] > 12.0:
            warnings.append(f"The tempo wanders ({info['drift']:.0f} ms from start to end against "
                            "a steady beat). This song needs more than one timing section, so "
                            "parts will feel off.")
        steady, where = in_time(heard_raw["hits"], (T0 - shift, beat / own), dur_ms)
        fits = ", ".join(f"{minutes(a)}-{minutes(b)}" for a, b in where[:4]) or "nowhere for long"
        if heard["timing"].get("moving"):       # it could have been followed, but was held to one tempo
            warnings.append(
                f"This song changes tempo, but this map holds {bpm:g} BPM throughout because "
                "BPM, offset or half/double tempo was set. That fits only about "
                f"{steady * 100:.0f}% of the song ({fits}). Leave those settings alone and the "
                "map follows the tempo changes.")
        elif info and steady < 0.7:
            warnings.append(
                f"This song changes tempo. The beat that was found ({bpm:g} BPM) fits only "
                f"about {steady * 100:.0f}% of it ({fits}); everywhere else the notes will be "
                "out of time with the music, because its changes could not be followed.")
    if info and info["on_grid"] < 0.3:
        warnings.append(f"Only {info['on_grid'] * 100:.0f}% of the hits sit on a steady beat, so "
                        "the BPM may be wrong. If it feels off, enter the BPM yourself.")
    if meter != "straight":
        warnings.append("This song is in threes (swing, shuffle or 6/8), so it is charted "
                        f"on a 1/{song['tpb']} grid. That path is newer and less proven "
                        "than straight 4/4.")
    if info and info.get("waltz"):
        warnings.append("This song seems to be in 3/4. Notes are on the beat, but bar lines "
                        "and phrasing assume 4/4.")
    if raw["no_drums"]:
        warnings.append("No drums found - accents and timing come from the other instruments.")
    if (style.ln > 0 and not style.short and (opts.ln is None or opts.ln > 0)
            and raw["lead_from"] != "voice"):
        warnings.append("No vocals found - the long notes follow the lead instrument instead.")
    if opts.density is None and abs(stars - opts.stars) > 0.05 and style.reach and stars < opts.stars:
        warnings.append(f"{stars:.2f} stars is about as far toward {opts.stars:g} as this song goes "
                        f"in the {style.label} style: packing every part of the song would add "
                        "only about a tenth of a star.")
    elif opts.density is None and abs(stars - opts.stars) > 0.05:
        warnings.append(f"{stars:.2f} stars is as close to {opts.stars:g} as this song gets "
                        f"in the {style.label} style.")
    if song["tpb"] != base:
        warnings.append(f"1/{base} notes alone couldn't reach the rating, so this chart "
                        f"uses 1/{song['tpb']} notes.")
    spacing, fits, span = _fit(style, felt, song["tpb"], opts.stars)
    if not fits and style.quick:        # its grid was picked for the rating: only say so when well off
        fits = 0.95 * span[0] <= spacing <= 1.05 * span[1]
    if not fits:
        low, high, lines = span
        better = [s.label for name, s in STYLES.items()
                  if s.suits and name != chosen
                  and _fit(s, felt, grid_for(s, felt, own, opts.stars), opts.stars)[1]]
        tpb = song["tpb"]                       # the songs it is meant for, read the way this one is
        songs = f"{60000.0 * lines / (high * tpb):.0f}-{60000.0 * lines / (low * tpb):.0f} BPM"
        rushed = spacing < low
        if style.quick:
            text = (f"At {opts.stars:g} stars, {style.label} rows are written to come {low:.0f} to "
                    f"{high:.0f} ms apart. Here they come {spacing:.0f} ms apart")
            if rushed:
                like = style.quick[0] - (spacing - style.suits[0]) / style.quick[1]
                text += (f", so its jacks are as fast as in {style.label.lower()} charts of about "
                         f"{like:.1f} stars and it will play "
                         f"{'a little ' if like - opts.stars < 0.4 else 'much '}harder than its rating.")
            else:
                text += ", so it makes up its rating with chords and will feel slow."
        elif style.reach and not rushed:
            text = (f"At {opts.stars:g} stars, {style.label} rows are written to come {low:.0f} to "
                    f"{high:.0f} ms apart (songs of about {songs}). Here they come {spacing:.0f} ms "
                    f"apart, and a {style.label.lower()} that slow can't reach the rating without "
                    "repeating a column from one row to the next, which it never does.")
        else:
            text = (f"At this tempo {style.label} rows come {spacing:.0f} ms apart, and the "
                    f"style is meant for {low:g} to {high:g} ms (songs of about {songs}), so it "
                    f"will feel {'rushed' if rushed else 'slow'}.")
        rate = math.floor(20.0 * spacing / low) / 20.0 if rushed else 1.0
        slow = ""
        if style.quick and 0.5 <= rate <= 0.95:
            # played at `rate`, every gap is 1/rate as long: rate exactly that chart
            slowed = star_rating([(t / rate, c, None if e is None else e / rate)
                                  for t, c, e in objects], opts.keys)
            # a denser chart loses a little more to the same slow-down; the power 1.1 was
            # measured on 2016 (0.80x, 0.65x), Griphop, Captain Scurvy (0.75x), Rollin at 5
            # (0.90x) and Mega Hyper Ultrastorm (0.60x): asked for, it plays within 0.1 stars
            # of the rating wanted
            ask = opts.stars * (stars / slowed) ** 1.1
            slow = (f" Or play it slowed to {rate:.2f}x in osu! (osu!lazer's Half Time can be "
                    f"set to that speed). At {rate:.2f}x this chart rates {slowed:.2f} stars; to "
                    f"play about {opts.stars:g} stars at that speed, ask for about {ask:.1f} here.")
        warnings.append(
            text + (f" This song's tempo suits {', '.join(better)} better." if better else "")
            + slow)
    if style.quick and own == 4 and song["tpb"] in (3, 6):
        # taken in triplets (grid_for): say why, and from which rating its own grid is used
        mine = 4 if song["tpb"] == 3 else 8
        quarter = felt / mine
        top, per = style.quick
        like = top - (quarter - style.suits[0]) / per
        warnings.append(
            f"This map's rows are 1/{song['tpb']} of a beat apart ({felt / song['tpb']:.0f} ms): "
            f"triplets, slower than this song's 1/{mine} notes ({quarter:.0f} ms), whose jacks would "
            f"be as fast as in {style.label.lower()} charts of about {like:.1f} stars. From about "
            f"{math.ceil(like * 10) / 10:.1f} stars it uses the song's 1/{mine} notes.")
    if style.ornaments:
        between = {k for k, _, _ in notes if k % song["line"]}
        if len(between & set(song["between"])) < 0.3 * len(between):
            warnings.append(
                f"Few fast notes were heard between this song's beats "
                f"({sum(song['heard'].values())}), so the bursts in this chart are invented "
                "to fit the style rather than taken from the music.")
    if opts.keys not in style.tuned:
        warnings.append(f"The {style.label} style has only been tuned for 4K so far.")

    runs = []                           # sections as [start ms, end ms, kind] for the preview
    for b in song["bars"]:
        s = T0 + b * song["bar_len"]
        if runs and runs[-1][2] == song["section"][b]:
            runs[-1][1] = s + song["bar_len"]
        else:
            runs.append([s, s + song["bar_len"], song["section"][b]])
    if clock:
        runs = [[in_song(s), in_song(e), kind] for s, e, kind in runs]
        tempo = {"low": clock.tempo()[1], "high": clock.tempo()[2], "sections": len(clock.parts)}
    else:
        tempo = None
    first = int(red[0][0])              # the first red line
    return {
        "osz": osz_path, "osu": osu_path, "song": key, "warnings": warnings,
        "meta": {k: meta[k] for k in ("title", "artist", "creator", "version")},
        "stats": {"stars": round(stars, 2), "target": opts.stars, "keys": opts.keys,
                  "style": chosen, "bpm": bpm, "offset": first, "tempo": tempo,
                  "picked": {"why": explain(picked), "ranking": [[r["style"], r["score"]] for r in picked]}
                  if picked else None,
                  "grid": f"1/{song['tpb']}", "finer": finer,
                  "heard": {f"1/{d}": n for d, n in sorted(song["heard"].items())},
                  "density": density, "comfort": comfort,
                  "objects": len(objects), "holds": sum(o[2] is not None for o in objects),
                  "play": _play(objects, opts.keys, stars, 60000.0 / bpm if clock else beat),
                  "nps": round(nps_avg, 1), "nps_peak": nps_peak,
                  "real": round(sum(1 for k, _, _ in notes if k in real) / len(notes), 2),
                  "length": round(dur_ms / 1000.0, 1),
                  "drum_hits": len(song["drums"]), "sung_notes": len(song["sung"]),
                  "bars": {s: kinds.count(s) for s in ("calm", "verse", "chorus")},
                  "timing": info},
        "chart": {"keys": opts.keys, "notes": objects, "bpm": bpm, "offset": first,
                  "beat": 60000.0 / bpm if clock else beat, "kiai": kiai, "length": dur_ms,
                  "lines": [[float(at), float(length)] for at, length, _ in red],
                  "sections": [[int(s), int(e), k] for s, e, k in runs]},
    }
