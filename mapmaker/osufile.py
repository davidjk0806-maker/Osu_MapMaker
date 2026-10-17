"""Writing the .osu chart and zipping it with its audio into an importable .osz."""

import bisect
import re
import zipfile


def safe_name(text):
    return re.sub(r'[\\/:*?"<>|]', "_", text).strip() or "untitled"


def _scroll(red, last):
    """osu!mania scrolls faster where the tempo is faster, measured against the tempo
    that is held for longest. -> for each red line, the green-line value that undoes
    this, so that the notes come down at one speed all song long."""
    held = {}                           # as the game counts it: time under each beat length
    for n, (at, beat, _) in enumerate(red):
        start = 0.0 if n == 0 else at
        end = red[n + 1][0] if n + 1 < len(red) else last
        held[round(beat, 3)] = held.get(round(beat, 3), 0.0) + (max(0.0, end - start) if at <= last else 0.0)
    base = max(held, key=held.get)
    return [-100.0 * base / beat for _, beat, _ in red]


def make_osu(meta, audio_name, bg_name, red, objects, kiai, preview_ms):
    """meta: title, artist, creator, version, tags, keys, od, hp, divisor.
    red: [(ms, beat length, whether a bar starts there)], the red lines: one for a song
    of one tempo, several for one whose tempo changes (timing.Clock.red_lines)."""
    keys = meta["keys"]
    L = []
    L += ["osu file format v14", "",
          "[General]", f"AudioFilename: {audio_name}", "AudioLeadIn: 2000",
          f"PreviewTime: {preview_ms}", "Countdown: 0", "SampleSet: Soft",
          "StackLeniency: 0.7", "Mode: 3", "LetterboxInBreaks: 0",
          "SpecialStyle: 0", "WidescreenStoryboard: 0", "",
          "[Editor]", "DistanceSpacing: 1", f"BeatDivisor: {meta['divisor']}", "GridSize: 4",
          "TimelineZoom: 1", "",
          "[Metadata]", f"Title:{meta['title']}", f"TitleUnicode:{meta['title']}",
          f"Artist:{meta['artist']}", f"ArtistUnicode:{meta['artist']}",
          f"Creator:{meta['creator']}", f"Version:{meta['version']}", "Source:",
          f"Tags:{meta['tags']}", "BeatmapID:0", "BeatmapSetID:-1", "",
          "[Difficulty]", f"HPDrainRate:{meta['hp']:g}", f"CircleSize:{keys}",
          f"OverallDifficulty:{meta['od']:g}",
          "ApproachRate:5", "SliderMultiplier:1.4", "SliderTickRate:1", "",
          "[Events]", "//Background and Video events"]
    if bg_name:
        L.append(f'0,0,"{bg_name}",0,0')
    L += ["//Break Periods", "//Storyboard Layer 0 (Background)",
          "//Storyboard Layer 1 (Fail)", "//Storyboard Layer 2 (Pass)",
          "//Storyboard Layer 3 (Foreground)", "//Storyboard Layer 4 (Overlay)",
          "//Storyboard Sound Samples", "",
          "[TimingPoints]"]
    if len(red) == 1:
        L.append(f"{int(red[0][0])},{red[0][1]:.15g},4,2,0,40,1,0")
        for s, e in kiai:
            L.append(f"{s},-100,4,2,0,40,0,1")
            L.append(f"{e},-100,4,2,0,40,0,0")
    else:
        # a green line after every red one keeps the scroll speed even, and both carry
        # on whatever kiai time they fall in
        last = max((t if te is None else te for t, _, te in objects), default=0)
        starts = [int(round(at)) for at, _, _ in red]
        speed = _scroll(red, last)
        for at in sorted(set(starts) | {t for pair in kiai for t in pair}):
            lit = int(any(s <= at < e for s, e in kiai))
            n = max(0, bisect.bisect_right(starts, at) - 1)
            if at in starts:            # 8: no bar line here, for a tempo that changes inside a bar
                L.append(f"{at},{red[n][1]:.15g},4,2,0,40,1,{lit + (0 if red[n][2] else 8)}")
            L.append(f"{at},{speed[n]:.15g},4,2,0,40,0,{lit}")
    L += ["", "", "[HitObjects]"]
    for t, c, te in objects:
        x = int(512 * (c + 0.5) / keys)     # osu! reads the column back from x
        if te is None:
            L.append(f"{x},192,{t},1,0,0:0:0:0:")
        else:
            L.append(f"{x},192,{t},128,0,{te}:0:0:0:0:")
    return "\r\n".join(L) + "\r\n"


def pack_osz(path, files):
    """files: {name inside the archive: path on disk}"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, src in files.items():
            z.write(src, name)
