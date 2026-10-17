"""Port of osu!lazer's mania difficulty calculator (Strain skill, version 20241007),
so a chart can be tuned to a star rating before it ever reaches the game."""

import math


def _legacy_sort(keys, key):
    """osu! sorts hit objects with the old unstable .NET 4.0 sort, and the order it
    leaves chord notes in feeds into the rating, so it is reproduced exactly."""

    def swap_if_greater(a, b):
        if a != b and key(keys[a]) > key(keys[b]):
            keys[a], keys[b] = keys[b], keys[a]

    def down_heap(i, n, lo):
        d = keys[lo + i - 1]
        while i <= n // 2:
            child = 2 * i
            if child < n and key(keys[lo + child - 1]) < key(keys[lo + child]):
                child += 1
            if not key(d) < key(keys[lo + child - 1]):
                break
            keys[lo + i - 1] = keys[lo + child - 1]
            i = child
        keys[lo + i - 1] = d

    def heapsort(lo, hi):
        n = hi - lo + 1
        for i in range(n // 2, 0, -1):
            down_heap(i, n, lo)
        for i in range(n, 1, -1):
            keys[lo], keys[lo + i - 1] = keys[lo + i - 1], keys[lo]
            down_heap(1, i - 1, lo)

    def quicksort(left, right, depth):
        while True:
            if depth == 0:
                heapsort(left, right)
                return
            i, j = left, right
            middle = i + ((j - i) >> 1)
            swap_if_greater(i, middle)
            swap_if_greater(i, j)
            swap_if_greater(middle, j)
            x = key(keys[middle])
            while True:
                while key(keys[i]) < x:
                    i += 1
                while x < key(keys[j]):
                    j -= 1
                if i > j:
                    break
                if i < j:
                    keys[i], keys[j] = keys[j], keys[i]
                i += 1
                j -= 1
                if i > j:
                    break
            depth -= 1
            if j - left <= right - i:
                if left < j:
                    quicksort(left, j, depth)
                left = i
            else:
                if i < right:
                    quicksort(i, right, depth)
                right = j
            if left >= right:
                return

    if keys:
        quicksort(0, len(keys) - 1, 32)


def star_rating(objects, keys=4):
    """objects: (start_ms, column, end_ms or None) in .osu file order."""
    total, weight = 0.0, 1.0
    for p in sorted((p for p in section_peaks(objects, keys) if p > 0), reverse=True):
        total += p * weight
        weight *= 0.9
    return total * 0.018


def section_peaks(objects, keys=4):
    """The strain at its highest in each 400 ms of the chart, in time order: what the
    rating is made of (the hardest of them, weighted 1, 0.9, 0.81, ... and summed)."""
    objs = [(float(t), c, float(t if e is None else e)) for t, c, e in objects]
    objs.sort(key=lambda o: o[0])
    _legacy_sort(objs, key=lambda o: int(round(o[0])))
    if len(objs) < 2:
        return []

    prev = [None] * keys            # latest earlier object per column: (start, end)
    col_start = [None] * keys
    individual = [0.0] * keys
    highest, overall, current = 0.0, 1.0, 0.0
    peaks, section_peak, section_end = [], 0.0, 0.0
    last = None
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

        hold_factor = 1.0           # something else is held across this whole note
        overlapping = False         # this note outlasts a hold it started under
        closest_end = abs(end - start)
        for p in prev:
            if p is None:
                continue
            p_start, p_end = p
            overlapping |= p_end - start > 1 and end - p_end > 1 and start - p_start > 1
            if p_end - end > 1 and start - p_start > 1:
                hold_factor = 1.25
            closest_end = min(closest_end, abs(end - p_end))
        hold_addition = (1.0 / (1.0 + math.exp(0.27 * (30.0 - closest_end)))
                         if overlapping else 0.0)

        individual[col] = individual[col] * 0.125 ** (col_time / 1000.0) + 2.0 * hold_factor
        highest = max(highest, individual[col]) if delta <= 1 else individual[col]
        overall = overall * 0.30 ** (delta / 1000.0) + (1.0 + hold_addition) * hold_factor
        current += highest + overall - current
        section_peak = max(current, section_peak)
        last = (start, col, end)

    peaks.append(section_peak)
    return peaks


def shape(objects, keys=4, stars=None):
    """How the rating is spread over the chart -> (what most of it plays like, as a
    share of the rating; the share of it that plays within 10% of the rating). Each
    400 ms is taken as the rating a chart would have if it were all like that (its
    strain x 0.18). Ranked 4K Insane maps come to about 0.84 and 30% (see
    research/notes/star_rating_reference.md)."""
    level = sorted(p * 0.18 for p in section_peaks(objects, keys) if p * 0.18 > 0.05)
    stars = star_rating(objects, keys) if stars is None else stars
    if not level or stars <= 0:
        return 0.0, 0.0
    middle = len(level) // 2
    median = level[middle] if len(level) % 2 else (level[middle - 1] + level[middle]) / 2.0
    return median / stars, sum(1 for x in level if x >= 0.9 * stars) / len(level)
