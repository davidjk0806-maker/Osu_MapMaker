# mapmaker

Turns a song into an osu!mania chart (`.osz`) at the star rating you ask for.

## Set it up (once, on a new computer)

It needs Python 3 (it was built on 3.14). In this folder:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

That install is large, because Demucs (which splits a song into drums, voice and the
rest) brings PyTorch with it, and the first song made downloads the Demucs model into
`.cache/`. On macOS any usual audio file works as it is; elsewhere an `.m4a` needs
`ffmpeg` installed.

## Start it

Double-click **`MapMaker.command`**. A page opens in your browser; drop a song on it.
Closing the Terminal window it opened stops the app.

From a terminal instead:

```
.venv/bin/python -m mapmaker                      # the drag-and-drop page
.venv/bin/python -m mapmaker "song.mp3" --stars 4.5 --keys 4 --style stream
.venv/bin/python -m mapmaker "song.mp3" --style auto       # the style that suits the song
.venv/bin/python -m mapmaker --help               # every option
```

A new song takes about half a minute the first time. After that, other keys, styles
and ratings for the same song take a few seconds, because the listening is cached.

## What you can set

| Setting | Choices |
| --- | --- |
| Star rating | 1 to 7 on the page (any value from the command line). Tuned against osu!lazer's own formula. |
| Keys | 4K, 5K, 6K, 7K |
| Style | **Recommended** (the app picks, see below) or one of the seven below. Each is a single skill; a map uses one style throughout. |
| Long notes | In the LN style: the share of the notes that are short holds. In Coordination: 0% has no LNs, 50% is the style's own mix, towards 100% every phrase is written in LNs only. |
| More options | Creator, difficulty name, title/artist, pattern variation, BPM and offset overrides, half/double tempo, nudge, OD/HP, background image. (Setting BPM, offset or half/double tempo holds the map to one tempo.) |

### The styles

Each style was modelled on charts of that kind (they are in `research/examples`) and
checked against them in numbers. They are tuned for 4K; other key counts work but have
not been tuned.

| Style | What it is | Tempo it suits |
| --- | --- | --- |
| **Coordination** | A kind of LN chart that makes finger independence the point, with fewer notes than the other styles at the same rating. Something is held nearly all the time, a beat or two at a stretch, each hold in another column than the last and usually on the other hand, while the other fingers play single notes and light chords. About half the holds get a **reverse shield** (a tap in the hold's own column a quarter beat after it ends); a **shield** (a tap a quarter beat before it starts) is rare, and never part of a jump into a chord of holds or the end of a jack. Some phrases are **inverse**, where nearly every finger is down and it is the gaps one plays (none up to 3.5★, about one in eight at 4.5★, one in four at 5.5★), and some are written in LNs used as normal notes (about one in fifteen up to 4.5★, one in four at 5.5★). A column is hit twice running on 1/4 notes only where the rating can't be reached without it, never three times, and two rows a quarter beat apart never share two columns. | up to about 187 BPM; songs under about 130 BPM may stop short of 4.5★ |
| **Tech** | Row spacing keeps changing: 1/4 notes with 1/8 and 1/6 bursts woven in front of about half the rows. Real fast hits are used where the song has them; the rest of the bursts are invented, and the map says so. | 120-208 BPM |
| **Stream** | Steady streams on the song's 1/4 notes (its 1/8 notes on songs under about 106 BPM), never faster to reach a rating, and with no holes inside a run: a bar's run grows back from the next downbeat in one piece. The rating comes from chords: single notes and light jumpstream low down, then jumpstream, then handstream, quads from 5★. Each beat shares its notes out as the Stream examples do: mostly a jump or a hand on the beat and a staircase of single notes after it, with the jump now and then moved off the beat, and short jumptrills into a new phrase; each bar favours one of those layouts, so a phrase keeps its shape while its columns move on, and each row's columns follow on from the rows before as in those maps (staircases that turn, a jump at the foot of a stair or in it). No column is hit on two rows running (except next to a quad), nor shared by two chords half a beat apart, as in ranked streams, so there is no chordjack. That caps a stream at about 0.027★ per BPM: 4.5★ takes a song of about 167 BPM or more, 4★ about 148, 3.5★ about 130. A slower song stops short, at the lightest map within a tenth of a star of the most it can reach. | at 4.5★ 167-214 BPM (or half that); at 4★ from 148, at 3.5★ from 130 |
| **Speed** | Fast single notes, hands taking turns, few chords; long runs where the song is loud. In 4K the streams move the way Palette GAMMA's do: short rolls of three columns that turn back, and hand-to-hand switches, rather than full 1234 rolls. Where the song's fast notes are too fast to sustain, they come as one run per bar. | 114-156 or 228+ BPM |
| **Speedjack** | Stream-like runs of light chords in which a column is hit again on the very next row, two to four times running at 4.5★ (up to six from 6★). How fast those jacks come is set by the rating: 105-122 ms apart at 4.5★, 83-100 ms at 5.5★. The rows are on the song's 1/4 notes where those are about that fast, and in triplets (1/3 of a beat) where its 1/4 notes would be faster, with more hands to make up the rating; so 2016 (177 BPM) is in triplets up to about 5★ and on its 1/4 notes from 5.5★. | about 123-190 BPM at 4.5★ (1/4 notes up to 143, triplets above) |
| **Chordjack** | Chords about 130 ms apart whose columns repeat from row to row; quads at the top end. | 120-143 or 176-240 BPM |
| **LN** | Over half the notes are short holds (a quarter or half a beat), so fingers let go and press in quick succession; what is sung is held for as long as it sounds. (Long sustained notes were a style of their own, "Holds", until they were grouped here.) | any |

### Recommended

**Recommended** picks the style from the song itself. Its tempo comes first: each style's
range at the rating asked for (as in the table above), and a style fits best in the
middle of its range. Then what the instruments play and how loud the song is:

| Style | Picked when the song has |
| --- | --- |
| Stream | drums that keep going on back-to-back 1/4 notes through the loud parts (or, counting for less, a steady eighth-note beat), more so if it stays loud throughout |
| Speed | fast notes between the beats (or 1/4 notes already about 60 ms apart) |
| Tech | hits between the beats on more than one snap (1/8 and 1/6, say) |
| Chordjack | the same drum hit again on back-to-back rows about 130 ms apart, in a song mastered loud and hard |
| Speedjack | the same on back-to-back 1/4 notes, or a voice that sings one note again straight away (as in Mi Mi Mi), with something on most 1/4 notes |
| LN | a voice or lead that holds its notes, and quiet parts as well as loud ones |
| Coordination | held notes over a light rhythm; less often above 4.5★ |

LN and Coordination can chart any song, so a song in which nothing stands out gets one
of them: Coordination at slower tempos and ordinary ratings, LN otherwise. Once a song
has been mapped once (so it has been listened to), the page says under the styles which
one it would pick at the rating set and why, and marks that style "best fit". A map
made with Recommended says on its card which style it got and why.

It is a guess from what the app hears, so it is only as good as that. On the eleven
example songs whose mappers chose a style it picks the mapper's style for six and has
it second for two more (`research/tools/recommend_check.py`; `research/tools/recommend_library.py`
shows its picks for the songs in the library). The two Tech songs get Speedjack and LN,
because their fast notes are in the voice and melody, which the app does not hear as
notes; NoNamed's Speedjack Pack gets Stream, its drums running on 1/4 notes. Since
2026-10-16 Speedjack suits most tempos (by taking triplets on fast songs), so it is no
longer passed over for its tempo alone; Stream suits a tempo by the rating asked (a
stream with no jacks needs a faster song for a higher rating), and songs whose voice
holds its notes still mostly get LN or Coordination.

Parts of a song without drums (an opening, a break, the wind-down) are charted from
the melody, in every style, so that they are played along with the music: a note where
a note of the melody starts on a beat or half beat, 1/4 notes only inside a run, sung
notes where they begin, and any light drum hits as they are. This is only done where
the melody keeps time. What is heard in a held voice, a pad or a reverb tail often
falls anywhere between the beats; such a stretch gets its sung notes and nothing else,
rather than notes that follow nothing.

A song's loudest stretch is treated as its high point. Where some of a song's full
bars stand clearly above the rest in loudness (2 dB or more: a drop), those bars are
charted fuller than the style would otherwise have them (up to two and a half times,
and never fuller than a full chart), however few drum hits they have. If that leaves
them more than half a star harder than everything else, they are eased until they are
not: the high point of the chart, not all of it. Without this a half-time drop, which
has the fewest drum hits of the whole song, came out as its thinnest part.

When a song's tempo puts a style's rows outside the spacing it is meant for, the map
carries a note saying so and naming the styles that tempo suits. For Speedjack that
spacing depends on the rating: in the speedjack charts at hand the jacks are 119 ms
apart at 4.5★ and 86-90 ms apart only from about 5.5★. Until 2026-10-16 a fast song
made a Speedjack map whose jacks played much harder than its rating (2016 at 4.5★: 85
ms jacks, "insanely fast"); now its rows go into triplets instead, and the map says
so and from what rating the song's own 1/4 notes are used. Where even that leaves the
jacks faster than the rating (a song of about 150 BPM, whose triplets would be too slow
to reach it), the note says how much harder it plays, and, when that is a lot, how far
to slow it down in osu! (Half Time) and what rating to ask for at that speed.

The page shows the same guide before you make a map: under the style's description,
which tempos the style suits at the star rating chosen, and once a song has been
mapped once (so its tempo is known), whether it fits; styles it is too fast or too
slow for are marked on their buttons.

Each finished map shows the star rating it actually reached, its BPM and offset (or
its range of tempos and how many tempo sections it has), how much of it sits on real
sounds, how it plays (what most of it plays like as a share of its rating and how much
of it plays near the rating, its chords, its usual row spacing, its jacks), and any
warnings.

### What the star rating changes

"What the star rating changes", under the rating on the page, explains this for the
rating and style chosen. In short: osu! rates a map from its notes alone (how close
together they come, per column and across the keys, notes under a held LN counting a
little more), and its hardest ten seconds or so make up nearly all of the number, so it
can't tell a map that keeps its hardest density up all song from one that peaks in a
few bursts, nor a jack from a stream of the same density. To reach a rating the app
changes, in this order: how many of the song's sounds get a note (the strongest drum
hits and sung notes, then every beat and half beat, then 1/4 runs between them); where
(loud parts first, verses at about 60%, quiet parts following the melody); chord size;
and last, patterns (harsher ones only where nothing gentler gets there). Since
2026-10-16 the quieter choruses are also charted up to 30% lighter than the loudest
ones (`Style.contour`), so that the rating is reached where the song peaks and the rest
plays easier: ranked 4K Insane maps play at about 0.84 of their rating most of the way,
with about 30% of them near it, while the app's were at 0.91 and 56%. Ranked 4K maps
average about 2-5 notes a second at Easy, 6-9 at Normal, 8-14 at Hard, 12-19 at Insane
and 13-23 at Expert (the reference sets in `research/`, measured by
`research/tools/balance.py refs`). **Preview** plays the chart against
the song in the page; **Open in osu!** hands the `.osz` to the game.

## How it works

1. The song is split into drums, bass, vocals and the rest (Demucs).
2. Tempo and offset are measured from the first sample of each drum hit. If one tempo
   fits less than 70% of the song, the beat is followed through its changes instead
   (see below).
3. Drum hits, held vocal (or lead instrument) notes and bass notes are placed on the
   grid. Hits that fall between two grid lines are kept too, each on the snap it was
   played on (1/6, 1/8, 1/12, 1/16), for the styles that chart them.
4. Filler is woven between them, and the amount is tuned until a port of osu!'s star
   formula reads the target. Comfort limits (no fast jacks, no long one-hand runs) are
   loosened only when the target can't be reached otherwise.
5. Audio is re-packed as `.ogg` from the exact samples that were analysed, so the
   timing in the game matches the timing that was measured.

### How the notes are made

All of this is in `mapmaker/chart.py`. What it starts from (the grid, the drum hits,
the sung notes, which bars are chorus, verse or a quiet part) comes from `timing.py`
and `listen.py`.

1. **How many notes each grid line gets** (`plan`). Each bar gets a level from its
   section and from the density being tried. Real sounds always get a note. As the
   level rises, more of the empty lines fill, in a fixed order (`METERS`): beats and
   half beats first, then 1/4 runs that grow back from the snare and the downbeat. Chords
   grow on the strongest drum hits first. Stream goes its own way: `_rung` sets each
   row's chord size, `_figures` shares each beat's notes out as the Stream examples do,
   and `_apart` keeps neighbouring rows from needing the same columns.
2. **Which columns** (`Patterner`, run row by row from `build_chart`). Every
   combination of free columns of that size is tried. Some are ruled out
   (`comfortable`: a jack that runs too long, one hand playing line after line, a chord
   that reuses too many of the columns just pressed). The rest are scored (`cost`). Hitting a column
   again quickly, keeping one hand busy or repeating a chord costs more. Following the
   current motif (a roll, trill, zigzag, split, anchor or jack, picked anew every two
   bars) or the pitch of a sung note costs less. The cheapest wins, with a little
   randomness from the seed. In Speed the motif's next column, and in Stream the whole
   row, come from tables counted in real maps (`_FLOW`, `_STEPS`).
3. **Long notes**: sung notes held while they sound, short holds in LN
   (`_short_holds`), and Coordination's shields, reverse shields and inverse
   (`_independence`).
4. **Reaching the star rating** (`_tune`). `stars.py` is a port of osu!lazer's mania
   star rating. The whole chart is rebuilt at density 0.00, 0.01, 0.02 and so on until
   its rating crosses the target, then in steps of 0.001 around that point. If the
   gentlest comfort rules can't get there, they are loosened a rung at a time
   (`COMFORT`). If the bare song is already harder than asked, its hits are thinned out
   instead.

Each style is a `Style` in `STYLES`: a set of knobs for those steps (how fast chords
grow, the order runs fill in, comfort limits, motifs, long notes, the tempos it
suits). Its comments say which ranked maps it was modelled on and what changed after
each playtest.

### Songs that change tempo

When one tempo does not fit, the drum hits are followed three seconds at a time, and
the result is tidied into steady stretches: a step from one tempo to the next, or a
slide cut into short pieces. A stretch is given a whole number of BPM when that moves
none of its lines by more than 1.5 ms. Bars start where the tempo steps, if the steps
agree on that.

The song is then charted in a straightened copy of its time, in which every 1/4 line
is as long as where the song is fastest, and the notes are put back where the song
really has them. So every style and rule works as for any other song, and nothing
comes out closer together than the style allows. The star rating is that of the chart
in the song's own time.

The `.osu` gets a red line where each stretch starts (and a second one on the next bar
line if the tempo changed inside a bar, so that the game's bar lines stay with the
music), each followed by a green line that keeps the scroll speed the same all song
long. The page shows the range of tempos, and its preview draws the beat lines of
each stretch.

`research/tools/in_sync.py "song" [style]` charts a song and counts how many of its drum
hits have a note on them, half a minute at a time; `--one-tempo` does the same with
the song held to one tempo, to compare.

## Limits worth knowing

- **Tempo changes are followed from the drums.** Where a song has no drums for a
  while the beat is bridged across the gap, and a jump in the beat there (a part
  spliced in a little early or late) is spread over the gap instead of being found.
  A song is only followed when one tempo clearly does not fit; one that merely drifts
  a little gets a warning and one tempo. Songs in threes are not followed.
- **Every song has a ceiling** for each style. If a target can't be reached the map
  says how close it got. On slower songs LN and Tech go to a finer grid (1/8 notes)
  first; Stream and Coordination never do (since 2026-10-16), so on slower songs they
  may stop short: Coordination on songs under about 130 BPM, Stream (which never
  repeats a column on the next row) on songs under about 167 BPM at 4.5★ (Curse These
  Metal Hands, 124 BPM, reaches about 3.4★).
- **Fast melodies are not heard as notes.** Drum hits between the beats are (and about
  nine in ten of those found are ones a human mapper charted too), but the quick runs of
  a melody or a voice are not, so the Tech style invents most of its bursts.
- **A melody is followed on beats and half beats, by its note starts only.** In the
  drumless bars of the example songs about three in four of the notes charted this way
  are ones the mapper charted too, and about half of the mapper's notes are found
  (`research/tools/breaks.py`). Which pitch a note has is not heard, so columns follow the
  melody only where it is sung.
- **Songs in threes** (swing, shuffle, 6/8) are charted on a triplet grid. That path
  is newer and less tested than straight 4/4. Songs in 6/8 show the dotted-quarter
  BPM (a jig listed as 180 shows as 120; the notes are the same either way).
- **3/4 songs** get 4/4 bar lines; the notes are still on the beat.
- A chart well above what a song naturally holds is mostly filler. The "on real
  sounds" figure on each map says how much.

## Checking a change

The engine is covered by two sets of checks. Run both after changing anything in
`mapmaker/`:

```
.venv/bin/python tests/regress.py    # every baseline chart is still note-for-note the same
.venv/bin/python tests/engine.py     # mixed snaps, spacing rules, listening on known rhythms, tempo changes, breaks, the drop, Coordination, Stream, jack limits, Recommended
```

`regress.py` first charts the 2016 song in the style the app began with and compares
it with `tests/reference/d4vd - 2016 (CASNoob) [Tech].osu`, the chart that was played
and approved. That style is no longer offered (its name, Coordination, now stands for
the LN style above); the checks keep it as "original", with 47 recorded charts that
nothing about breaks or long notes touches, so that any change to them means the
engine underneath has changed. Its last few charts are of a made-up song whose tempo steps and slides, where
the time of every beat is known; `engine.py` checks those charts and the beat follower
against that truth. A change
that is meant to alter charts is accepted with `tests/regress.py --record <words>`.
The test songs are fetched once with `tests/fetch_songs.py`.

For a change to the page, `tests/browser/.venv/bin/python tests/shots.py [name] [--make style:stars]`
saves screenshots of it (desktop, options open, preview, phone width) to `tests/browser/shots`
and reports any script errors on the page; `--make` first makes a map of the newest
song, to see its card. The browser is set up as its docstring says and can be deleted
afterwards.
`tests/smoke.py` drives the page's API end to end.

## Where things live

- `library/` - the songs you dropped in and the maps made from them
- `.cache/` - separated stems, analysis, and the Demucs model (safe to delete; it is rebuilt)
- `mapmaker/` - the code: `timing`, `listen`, `chart`, `recommend`, `stars`, `pipeline`, `server`, `web/`
- `tests/` - the checks above, their songs, and their own cache
- `research/` - what the styles are modelled on, and the tools for studying it:
  - `Osu Songs For Reference/` - the example maps you supplied; `examples/` - the example charts, one file per chart, by style; `mine/` - your own charts of 2016; `reference/` - the examples unpacked beside their audio (made by `tools/compare.py`)
  - `tools/` - describe a chart in numbers (`measure.py`), draw it as text (`draw.py`), set one of the app's styles beside the mapper's chart of the same song (`against.py`, `compare.py`), compare the drumless bars (`breaks.py`), see how a chart uses its long notes (`noodles.py`), see what Recommended picks for the example songs (`recommend_check.py`) and the library (`recommend_library.py`), measure the app's maps of the library songs and every 4K reference difficulty in the numbers of the 2026-10-16 playtest: jacks into long notes, stream spacing, jack speed, chords, how flat a map is (`balance.py`), trace where a Coordination chart's shields come from (`shield_trace.py`), check a map against the ranking criteria (`criteria_check.py`), see how varied a chart's streams are: where its chords sit, its staircases, how often a beat repeats (`shapes.py`), print the table Stream draws its columns from (`steps.py`), and make maps for playtesting (`playtest_maps.py`, `playtest_library.py`)
  - `notes/` - how the star rating works (`star_rating_reference.md`), the osu! wiki pages it is drawn from (`osu-wiki/`), and `pattern_glossary-osumania.webp`, which names the patterns
  - `logs/` - what past runs of the tools printed; `cache/` - their own listening cache; `demos/` - a throwaway demo chart
- `playtest/` - maps made for trying each style out (`research/tools/playtest_maps.py`; the songs in the library with `research/tools/playtest_library.py`, e.g. `playtest/2026-10-16-stream/`)
- `backups/` - the engine as it was before each large change

### Not on GitHub

`.gitignore` keeps out the songs and everything made from them (`library/`,
`playtest/`, `backups/`, `.cache/`), other people's beatmaps and audio
(`research/Osu Songs For Reference/`, `examples/`, `reference/`, `demos/`), the copies
of the osu! wiki (read them at osu.ppy.sh/wiki) and the pattern glossary image, and
what the tools and tests cache or print. Most of the research tools read those
example maps, so they won't run from a copy of this repository. Of the checks, `tests/fetch_songs.py` downloads the test songs
(Creative Commons, from incompetech.com), but `regress.py` and `engine.py` also chart
the author's copy of d4vd's "2016", which isn't included.
