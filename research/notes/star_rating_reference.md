# Star rating and the osu!mania ranking criteria

A reference for judging whether a chart plays like its star rating. Part 1–3 are what
the osu! wiki says; part 4–5 are what came of holding the reference sets and the app's
own maps against it (2026-10-15). Nothing in the app was changed for this.

The wiki pages themselves are saved in `research/notes/osu-wiki/` (fetched 2026-10-15 from
github.com/ppy/osu-wiki, the source of osu.ppy.sh/wiki):

| File | Page |
| --- | --- |
| `star_rating.md` | [Beatmap/Star rating](https://osu.ppy.sh/wiki/en/Beatmap/Star_rating) (a stub: no formula) |
| `difficulty.md` | [Beatmap/Difficulty](https://osu.ppy.sh/wiki/en/Beatmap/Difficulty) |
| `ranking_criteria_mania.md` | [Ranking criteria/osu!mania](https://osu.ppy.sh/wiki/en/Ranking_criteria/osu%21mania), with its glossary |
| `ranking_criteria_general.md` | [Ranking criteria](https://osu.ppy.sh/wiki/en/Ranking_criteria) (all modes) |
| `scaling_bpm.md` | [Scaling BPM on the ranking criteria](https://osu.ppy.sh/wiki/en/Ranking_criteria/Scaling_BPM) |
| `difficulty_naming.md`, `difficulty_spread.md`, `drain_time.md` | naming, spread, drain time |
| `pattern_*.md` | [osu!mania patterns](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu%21mania): chord, stream, jack, anchor, trill, hold note |
| `mania_mapping_guide.md`, `mania_modding_guide.md` | the community mapping and modding guides |
| `half_time_lazer.md`, `double_time_lazer.md` | rate mods in osu!lazer |
| `overall_difficulty.md`, `judgement_mania.md`, `game_mode_mania.md`, `performance_points.md` | hit windows, judgements, pp |
| `news/*.md` | every news post since 2022 that changed star rating |

## 1. What the star rating measures

**The number.** osu! works it out from the notes alone. For mania it is a "strain"
model; the wiki has no formula (the page is marked TODO), so the source is lazer's code,
which `mapmaker/stars.py` ports (lazer's version 20241007). The port gives the same
number as osu!'s website, to two decimals, for all 55 4K difficulties in the reference
sets (section 4).

**Changes to it.** Mania's rating changed in October 2022 (fast LN releases no longer
over-rewarded; chords judged as a whole) and on 28 October 2024 (the LN overlap bonus
needs longer overlaps; LNs starting together in a chord are no longer buffed). The
October 2025 update only restructured the code, and the March 2025 and July 2026
updates did not touch mania. So the port is current.

**How it is built.** Every note adds strain to its column (which fades to 1/8 in a
second) and to the whole chart (fading to 0.3 in a second); a note under a held LN
counts 1.25x, and an LN that starts under another and outlasts it gets a small bonus. The song is
cut into 0.4 s sections and each keeps its highest strain. The sections are sorted
hardest first and added up with weights 1, 0.9, 0.81, …, then multiplied by 0.018.

**What follows from that.**

- **The rating is the hardest ten seconds.** The 22 hardest sections (8.8 s of play,
  anywhere in the song) carry 90% of the weight, so the rest of a map barely counts.
  Two maps of the same rating can differ a lot in how hard they are overall. A map
  that keeps up its hardest density all song is much more tiring than one that
  reaches it in a few bursts. The modding guide warns about this: "Don't ever use Star
  Rating as a spread measure. It is currently wildly inaccurate in osu!mania as it only
  focuses on note density to calculate Star Rating, which can cause one dense section
  to spike the rest of the rating".
- **Length barely matters.** A three-minute map and a five-minute map with the same
  hardest stretch rate about the same.
- **What it can't see.** Pattern type, reading, scroll speed, and the speed of a jack
  beyond its share of the density are all invisible to it. The 2016 Speedjack at 4.5★
  has jacks 85 ms apart, a speed that charts only reach at about 5.4★. The rating
  stays at 4.5 because the map has fewer notes around them. The wiki says the same in
  general terms: difficulty "may deviate significantly" from the rating, "this is why
  the ranking criteria offers more decisive ground rules to determine a beatmap's
  actual difficulty level … regardless of its star rating."
- **Long notes.** The LN bonuses rate LN-heavy charts higher for the same density. In
  the reference sets, most of an LN chart above 4★ plays at 0.61–0.71x its rating,
  against about 0.8–0.9x for rice charts.

**Rate mods** (`half_time_lazer.md`). osu!lazer's Half Time slows a map to 0.75x by
default and can be set anywhere from 0.50x to 0.99x. The rating is worked out again
at the new speed. Scores stay ranked at the default 0.75x (with or without "Adjust
pitch"); any other speed makes them unranked.

## 2. Difficulty names by rating

From `star_rating.md` / `difficulty.md` (the website's colours and AiMod use these):

| Name | Stars |
| --- | --- |
| Easy | 0–1.99 |
| Normal | 2.0–2.69 |
| Hard | 2.7–3.99 |
| Insane | 4.0–5.29 |
| Expert | 5.3–6.49 |
| Expert+ | 6.5 and up |

4.5★ is Insane. In ranked sets, names drift below these bands: maps named "Insane"
sit at 3.47 (GLORYHAMMER), 3.67 (Igaku) and 4.22 (Sendan Life). Names such as Hyper
(between Hard and Insane), Another and Advanced fill the gaps.

## 3. The ranking criteria, level by level

**Scaling with BPM** (`scaling_bpm.md`). Every rhythm guideline is written for about
180 BPM in 4/4. Halfway to 120 BPM it loosens, and by 120 BPM a guideline about 1/4
reads like one about 1/2; at 90 BPM, double the density is fine. Above 180 it
tightens the same way, so by 240 BPM a guideline about 1/2 reads like one about
1/4. In other words, the guidelines are really about time. The table uses 180-BPM
snaps in milliseconds:

| Snap at 180 BPM | 1/1 | 1/2 | 1/4 | 1/6 | 1/8 | 1/12 |
| --- | --- | --- | --- | --- | --- | --- |
| ms | 333 | 167 | 83 | 56 | 42 | 28 |

So a 126 BPM song's 1/4 (119 ms) is about 180's 1/2-and-a-bit, and 270 BPM's 1/2
(111 ms) counts as 1/4.

**Every level.**

- Every note should follow a sound in the music. Simplify where following it exactly
  would be unreasonable or would cause "an inappropriate difficulty spike".
- Hold LNs for at least 1/12 of a beat.
- Never place two notes in one column at the same moment.
- Leave no column empty.
- Press at most 6 notes at once in Insane or lower.
- When the BPM changes, keep the scroll speed the same with SV (the app does this).
- Use breaks only where the music is "extremely calm or silent".

**Per level, for 4K.** Each guideline names the patterns a level should avoid. A
pattern a level avoids is fine from the next level up.

| Level | Guidelines (4K) |
| --- | --- |
| Easy | No runs of consecutive 1/4 notes. Mostly 1/1, with occasional 1/2. At most 5 consecutive 1/2 notes. Chords only at the start or end of a 1/2 pattern. Chords of 2 notes at most. LNs held 1 beat or more. LN releases at least 1 beat apart. No notes during a 1/1 LN. HP and OD 7 at most. |
| Normal | No consecutive 1/6 or faster. Mostly 1/1 and 1/2, with occasional 1/4. At most 5 consecutive 1/4 notes. A 1/1 pause after every phrase (4 bars) of 1/2. Chords only at the ends of 1/4 runs. Chords of 2 notes at most (a jump on the other hand from an LN is fine). Anchors under 3 notes. No 1/4 minijacks or jacks. No 1/4 shields. LNs held 1/2 beat or more. LN releases 1/2 beat apart. No notes during a 1/2 LN. Nothing faster than 1/2 during a 1/1 LN. HP and OD 7.5 at most. |
| Hard | No consecutive 1/8, except 3 grace notes. A 1/2 pause after every phrase of 1/4. Chords inside 1/4 streams at least 1/1 apart. No chords in the middle of 1/6 streams. Anchors under 5 notes. 1/4 jacks discouraged (minijacks with ample rest are fine). Trills of 9 notes at most. Chords of 3 notes at most. LNs held 1/4 beat or more. HP and OD 8 at most. |
| Insane | "Avoid unjustified spikes in difficulty. Difficulty should be representative of the song's intensity." No split rolls or complicated 1/8 streams longer than 4 beats. No LN streams at 1/6 or faster. No chords in the middle of 1/8 streams. Split-jumptrills of 9 notes at most. Split-jumptrills of 2+ beats leading into a hand are discouraged. |
| Expert | Avoid unjustified spikes. That is the only pattern guideline. |

The guidelines are strict up to Hard and loose from Insane up. An Insane map can use
nearly anything at 1/4, including 1/4 jacks, jumpstreams, chordjacks and quads.

**Spread.** A set with drain time under 2:00 needs a Normal or easier difficulty, or a
proper spread of 4. From 2:00 to 2:45 it needs a Hard or easier, or 3 difficulties.
From 2:45 to 3:30 it needs an Insane or easier, or 2.

**Glossary, as the wiki uses it** (it agrees with the pattern glossary in
`research/notes/pattern_glossary-osumania.webp`):

- **Jack:** 3 or more notes in a row in one column. A **minijack** is 2; a
  **longjack** is 4 or more, or the same chord repeated; a **chordjack** mixes jacks
  with chords and is denser than a quadstream.
- **Anchor:** one column hit far more often than the others inside a stream, usually
  every 1/2.
- **Shield:** a note just before an LN in the same column (it plays like a minijack).
  A **reverse shield** is a note just after an LN's release. An **inverse** fills a
  column with LNs that end just before the next note.
- **Release:** an LN's end, judged on its own timing. **Grace:** notes in quick
  succession, at 1/6 or finer.
- **Trill:** 3 or more notes alternating between two columns. A **jumptrill** is
  chords played with one hand each; a **split-jumptrill** is chords that each need
  both hands.
- **Bracket:** several trills at once (not used for 4K).
- **Stream:** a steady run of notes at one snap, at least a beat long, with no
  minijack. **Jumpstream**, **handstream** and **quadstream** mix in chords of 2, 3
  and 4; in 4K, a quad makes a minijack with its neighbours. A **burst** is a short
  run faster than the notes around it.
- **Keysounding:** hitsounds taken from the song itself.
- **Long-term SV:** SV held for more than 4 bars. **Short-term SV:** under 1/2 beat.

## 4. What the reference sets show

**The app's star numbers match osu!'s.** For all 55 4K difficulties in the reference
.osz sets, `mapmaker/stars.py` gives the website's rating to two decimals. The sets are
GLORYHAMMER, ETERNAL DRAIN, Palette GAMMA, Sendan Life, Glaza, Sparkling Starlight,
Igaku, Hole Dwelling, On And On!!, Mi Mi Mi and the Speedjack Pack.

**A checker for the criteria.** `research/tools/criteria_check.py` turns each guideline in
section 3 into a measurement, in 180-BPM milliseconds as above. It then names the
lowest level whose guidelines a chart keeps to. Guidelines say "avoid", so it lets
through up to 2 slips of each kind, or 1 a minute in longer maps. On recently
ranked sets (GLORYHAMMER, Glaza, Igaku, Sparkling Starlight):

- Maps named Easy or Normal come out at the level they're named.
- Of three maps named Hard, GLORYHAMMER's Hard comes out as Hard. Glaza's HARD and
  Igaku's Hard go past Hard on one guideline only (chords less than 1/1 apart inside
  1/4 streams, 3 and 12 times).
- Maps named Insane come out as Insane.
- Expert maps often come out as Insane, because Expert adds no pattern guidelines.
- The older ETERNAL DRAIN set (ranked under older rules) and Palette GAMMA (loved,
  so no criteria) run a level above their names.

**How much of a map plays near its rating.** Each 0.4 s section's strain × 0.18 is the
rating the map would have if it were like that section all the way through. Two
numbers sum up the shape:

- **"plays like":** the middle (median) section's level.
- **"near the rating":** the share of sections within 10% of the rating or above it.

| Ranked maps by band | Plays like (× rating) | Near the rating | Notes/s |
| --- | --- | --- | --- |
| Easy (7) | 0.77 | 16% | 2.9 |
| Normal (4) | 0.83 | 18% | 7.0 |
| Hard (14) | 0.81 | 22% | 11.3 |
| Insane (15) | 0.84 (0.64–0.92) | 30% (3–63%) | 16.1 |
| Expert (15) | 0.80 | 26% | 19.0 |

So a ranked map's rating sits well above most of its play: for a ranked 4.5★, most
of it plays like about 3.8★. The approved 2016 chart (the "original" style, 4.50★)
fits this well: Insane by its patterns, most of it like 3.83★ (0.85x), and 33% near
the rating.

## 5. The app's maps against it (engine of 2026-10-15)

76 maps were tested:

- every 4K difficulty of the reference sets, charted by the app at that difficulty's
  website rating in the set's own style
- 2016 in all seven styles at 2.3, 3.3, 4.5 and 5.8★

| App maps by band | Patterns of a higher level | Plays like | Near the rating | Notes/s |
| --- | --- | --- | --- | --- |
| Easy (5) | 3 of 5 (ranked: 3 of 7) | 0.76 | 12% | 3.2 |
| Normal (11) | 10 of 11 (ranked: 1 of 4) | 0.74 | 11% | 5.9 |
| Hard (21) | 19 of 21 (ranked: 12 of 14) | 0.77 | 13% | 9.9 |
| Insane (30) | 4 of 30 (ranked: 2 of 15) | 0.91 | 56% | 16.4 |

**At Insane (the 4–5.29★ band)**

- **The patterns fit.** 26 of the 30 Insane-band maps keep to the Insane guidelines.
  The four that don't:
  - Speed puts chords inside its 1/8 bursts (2016 at 4.5★: 48 times). 4K Insane
    says chords only at a burst's ends.
  - Two Chordjack and Stream maps run split-jumptrills past 9 notes (Sendan Life at
    4.21★: 7; ETERNAL DRAIN at 4.20★: 4).
- **The maps are flatter than ranked ones.** Most of an app Insane map plays at 0.91x
  its rating, and 56% of it is near the rating; ranked maps are at 0.84x and 30%. On
  2016 at 4.5★, Tech, Stream and Chordjack play at about 0.92x and 57–61% (the approved
  chart: 0.85x, 33%). Speed, Speedjack and LN are near ranked (0.80–0.84x, 16–31%), and
  Coordination is spikier (0.75x, 7%).
  - The clearest case is GLORYHAMMER at about 4.78★. The ranked "Dark Galactic
    Skies" averages 12.3 notes a second and reaches its rating in a few bursts (3%
    near it). The app's 4.76★ averages 20.2 notes a second, and 79% of it is near the
    rating.
  - Same rating, much more to play. A flat map is a stamina test the number doesn't
    show, so it will feel harder than a ranked map of the same rating.
- **Speedjack's jack speed is unchanged.** On 2016, Speedjack's jacks are 84 ms apart
  at every rating from 2.3 to 5.8★ (see the tempo guide; ranked 4.5★ speedjack, Mi Mi
  Mi, has none under 118 ms).

**Below Insane**

- **The app thins its patterns where ranked maps simplify them.** Ranked Easy and
  Normal maps switch to slower snaps, smaller chords, longer LNs and no jacks. The app
  keeps each style's patterns and uses fewer of them, so its lower-rated maps carry
  patterns of a higher level. Each example below is the app's map of 2016 at 2.3★,
  which is in the Normal band:

  | Style | What it does at 2.3★ | What Normal allows |
  | --- | --- | --- |
  | Speed | 7 runs of consecutive 1/8 | Normal stops at consecutive 1/6; consecutive 1/8 is past Hard |
  | Coordination | 85 shields at 1/4 | none |
  | Chordjack | 40 hands | 2-note chords at most |
  | LN | 172 LNs held under 1/2 beat, and 103 releases under 1/2 beat apart | LNs held 1/2 beat; releases 1/2 beat apart |
  | Speedjack | 36 minijacks at 1/4 | none |
  | Tech | 51 runs of consecutive 1/6 | none |

  In the Hard band the most common excess is many chords less than 1/1 apart inside
  1/4 streams, for example 340 on ETERNAL DRAIN at 3.65★. Ranked maps in that band
  slip on that guideline too, but far less often.
- **Ratings it can't reach.** Some songs and styles top out below the rating asked
  for, and the app's note on the map says so each time. On 2016: Stream 4.54★,
  Chordjack 4.57★, Speed 4.91★, Tech 5.03★, LN 5.62★. Elsewhere: Palette GAMMA in
  Speed 4.74★, Sparkling Starlight in LN 5.35★. At the low end, GLORYHAMMER in Stream
  gives 1.22★ when asked for 1.45 and 2.69★ when asked for 2.12.

## 6. Checking a map

    .venv/bin/python research/tools/criteria_check.py "some map.osu" ...

For each map it prints:

- its rating, the band that rating falls in, and the lowest level its patterns keep to
- what most of it plays like, and how much of it is near the rating
- every guideline it goes past, with a count and the level that allows it

Its limits:

- It handles 4K only.
- Anchors, trills and split-jumptrills are detected by rule of thumb.
- Split rolls, pitch relevance, whether each note follows a sound, and how a spread
  steps between difficulties are not checked at all.
