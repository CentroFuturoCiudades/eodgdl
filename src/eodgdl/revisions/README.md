# Hand decisions over the trip chains

`load_eod()` applies `chains.csv.gz`, the hand decisions, to the chain rules' output in one
pass (`eodgdl.review.apply_revisions`; `load_eod(revise_chains=False)` stops before it). Each
row is one decision, columns `household, person, trip, field, before, after, note, source`:

- `before` is the rules' value as a review sheet shows it — the shipped one on a row the rules
  dropped — and `after` the value decided: `HH:MM`, a level's own label, a zone id or `home`,
  whole minutes under `leg min` (the trip's travel minutes, spread over its legs in
  proportion), and `dropped` or `restored` under `status`.
- `note` is the reviewer's reason and `source` the review sheet that made the decision
  (`notebooks/<source>.csv`).
- There is one decision at most per trip and field, and each is made against the rules'
  output, never against another decision, so the table applies at once and in no particular
  order. The person's answer to where the day started (`orig. type`) sits on their first
  trip once the decisions are in.

## A review round

Export the chains pending a fix from the tables `load_eod()` returns (`eodgdl review export`:
by `problemas` code, by person with `--persons 8992/1,…`, or by `--screen`, below), edit the
sheet, check it (`eodgdl review verify`) and merge it (`eodgdl review freeze`, which runs
`eodgdl.review.freeze_round`). A new value replaces the field's decision and a value that is the
rules' own removes it; `dropped` adds a drop and takes the row's other decisions with it, and
`restored` removes a drop (or, on a row the rules dropped, is a decision of its own). A decision
the rules now give by themselves is removed. `freeze` writes nothing unless the merged decisions,
applied once to the rules' output, give exactly what the sheet's edits give on top of today's
tables. A round never stacks on another: an earlier decision is replaced or removed in place,
and git keeps the history. A round whose criterion is a rule rather than a call per chain keeps
the script that wrote its sheet under `scripts/revisions/`.

With no breaking code left, the codes no longer point at chains to fix; the screens do
(`eodgdl.review.screens`, and `verify` counts them before and after): a trip that leaves the
minute the previous one arrives (`zero_stay`), more than 14 hours at work (`long_workday`), a day
of more than 20 hours (`long_day`). A consistent chain can still be implausible, as round 8 and
round 10 found.

## When the rules change

Every decision is checked against the rules' output, so a change to the chain rules that moves a
value a decision was made against makes that decision **stale**: `load_eod()` fails, naming it,
rather than overwrite a value its reviewer never saw. The review tools load with
`load_stages(skip_stale=True)`, which sets the stale decisions aside with a warning, so
`eodgdl review export --stale` writes a sheet of the persons that hold them, each stale decision
written into its `new …` cell with why in the note. Keep the cell to decide it again against
today's rules, change it, or clear it to let the rules' value stand; `freeze` then replaces or
removes every stale decision the sheet shows, and refuses while a stale decision's trip is on no
sheet. `eodgdl.review.decision_outcomes` tells, decision by decision, whether it applies, holds
already, is moot or is stale; `tests/test_review.py` holds every committed decision to "applies".

## How the decisions are read

`eodgdl.review.apply_edits`, since the review of the chain fixes (2026-09-25):

- **Every decision is checked against the rules' output.** Its `before` must be what the
  rules give — the shipped value on a row the same decisions restore — or the load fails,
  naming every such decision (see *When the rules change*).
- **The origin type is one answer per person** ("¿En qué tipo de lugar inició su primer
  viaje?"): the decision on the person's first trip sets it on every row, one on any other
  trip is moot, and the sheet shows it on the first trip only.
- **A mode decision keeps the trip on its legs**: on a one-leg trip the leg takes the new
  mode, on a trip of several legs the new mode must be one of theirs, since the model build
  reads the main mode and the reweighting the legs.
- **A leg-minutes decision sets the trip's travel minutes**, spread over its legs in
  proportion to what each held, a minute each at least: the model's `Duration`.
- **`ajustes` is net.** Field by field, a value that is the rules' carries the rules' codes,
  one decided by hand carries `<field>:revision`, and one that is the shipped value again
  carries nothing; `fila:revision` marks a row the rules dropped and a decision restored.

## Review rounds

Each round merged with `eodgdl review freeze` since the passes were folded into the table.

| source | sheet | persons | decisions |
|---|---|---|---|
| `chain_review_8` | `notebooks/chain_review_8.csv` (one night shift that is a split shift; 2026-09-25) | 1 | 2 added |
| `chain_review_9` | `notebooks/chain_review_9.csv`, written by `scripts/revisions/chains_9.py` (leg minutes behind stays of no minutes and overlaps, the clock they let back, the overlaps left, the daycare escorts; 2026-09-26) | 177 | 425 added, 7 changed, 147 removed |
| `chain_review_10` | `notebooks/chain_review_10.csv` (evening returns for the split shifts pass 1 read as night shifts; 2026-09-26) | 10 | 2 added, 10 changed, 2 removed |

Round 8 (Claude, 2026-09-25): household 8992, person 1, read 21:00 to work, 02:00 home, 16:00
to work, 19:00 home, which the rules read as a night shift followed by a 22-hour day. It is a
split shift with lunch at home, 09:00–14:00 and 16:00–19:00: the first start was entered with
12 hours too many and the return with 12 too few, and the overnight reading kept the typo search
from trying either. None of the other 26 work trips to that AGEB starts at night (24 start
between 05:00 and 11:00), and the 02:00 return is the only one of 60 from there between 22:00
and 06:59.

Round 9 (Claude, 2026-09-26; the criteria are `scripts/revisions/chains_9.py`) follows a review
of the chain fixes and two decisions of the user's that day: TASHA cannot schedule a trip that
starts before the previous one arrives, so every overlap is breaking (`hora_traslapada` and
`hora_repetida` joined `chains.BREAKING_ISSUES`), and a stay of no minutes is better than an
overlap but worse than a stay, so leg-minute errors come before moved starts. Pass 1 had closed
868 overlaps by moving the start to the minute the previous trip arrives, 689 of them within the
old 15-minute tolerance, leaving stays of no minutes. Across the survey, trips that overlap the
next start are about as fast for their straight-line distance as any others, so the leg minutes
are not the usual culprit; but a motorised trip that reports twice the survey's typical minutes
for its mode and distance band (the median over the survey's trips, AGEB centroid to centroid,
30 or more of them) is slow for no reason the clock explains. On the persons the rules' output
flags (a `problemas` code on one of their trips; a stay of no minutes is no code, and the user
kept the survey the codes do not flag as recorded), 149 such trips stood behind a stay of no
minutes or an overlap; they and 222 equally slow trips of the same persons over the same routes
take the typical minutes (mostly 120 or 90 cut to 25–60), and 155 starts go back to the survey's
own value, or where a pass had moved them to the old arrival to the rules', now that they fit
(7425/3 and 8108/2 read exactly as recorded). The 4 overlaps left that the minutes do
not explain move to the arrival, the 46 'Guardería' trips of persons aged 12 or more that pass 1
had left become escorts, and 3 decisions the rules now give by themselves go (the typo search
reaches three rows it could not while a return that stopped short of home still counted as
reaching it). No breaking code is left, and the persons touched go from 175 stays of no minutes
to 19. Every decision in the table sits on a person the rules' output flags
(`tests/test_review.py`).

Round 10 (Claude, 2026-09-26; `scripts/revisions/chains_10.py` writes the sheet and each call's
evidence) rereads the day-plus-night-shift days pass 1 made. 61 diaries share the template "work,
home, work, home" whose first departure is recorded at night, after the next return: pass 1 read
that hour with 12 hours too many as the morning departure and, in 39 of them, the last return in
the evening; in 12 it took the last return to the small hours of the next day, a night shift
after the day's. Round 9's minutes gave 13603/4 its recorded evening back. In six others, and in
three six-trip diaries of the same shape after a morning walk (11846/1, 11942/2, 12047/2), the
first row's own hour fits after the second trip to work arrives — "left at 9, back at 9" — and
the workplaces agree: their other workers come home between 19:00 and 23:00 352 times, between
01:00 and 06:00 5 times. Their last return is that hour now, and 8763/1 leaves at 08:00 and goes
back to work at its recorded 17:02. 13583/1 reads a 1 keyed as a 2 in 22:00 and 23:30, a split
shift 08:30–12:00 and 13:30–15:00 ending at its recorded 15:00. Left as pass 1 read them, a
double shift being no error in itself: 16714/1, 11165/1 and 13088/3 (no same-day reading fits
the recorded hours), 7776/3 (a second job by bicycle), 8201/1 (its 90- and 120-minute trips to an
access point are what its zone reports, and no evening hour fits); and 1771/1, 11303/1 and
11303/2, whose template second tour at 22:00–23:30 cannot fit their minutes and keeps its stays
of no minutes.

## History: seven passes, folded into one table

Until 2026-09-25 the review was seven passes frozen one after another (`chains_<n>.csv.gz`),
each made against the tables the passes before it left, so later passes undid earlier ones:
12,344 frozen edits for the 5,262 decisions that stand. `scripts/revisions/squash_passes.py`
merged them in order into `chains.csv.gz`, exactly as a review round merges, and checked after
each pass that the table, applied once, gives what the passes gave one after another. The
passes are archived under `notebooks/revisions/`, beside their sheets; nothing loads them.

| pass | sheet | persons | frozen edits | decisions standing |
|---|---|---|---|---|
| 1 | `notebooks/chain_review.csv` (every person with a `problemas` code, exported 2026-09-08, returned 2026-09-24) | 4,435 | 11,669: 11,157 cells, 512 rows dropped | 4,858 |
| 2 | `notebooks/chain_review_2.csv` (the persons still carrying a breaking code after pass 1, plus 75 first-trip returns pass 1 had dropped; filled 2026-09-25) | 146 | 260: 149 cells, 34 rows dropped, 77 restored | 127 |
| 3 | `notebooks/chain_review_3.csv` (a second look at pass 2's judgment calls; filled 2026-09-25) | 67 | 93: 51 cells, 42 rows dropped | 71 |
| 4 | `notebooks/chain_review_4.csv` (the non-trips pass 1 made out of real trips; filled 2026-09-25) | 3 | 9 cells | 3 |
| 5 | `notebooks/chain_review_5.csv` (first trips whose recorded origin pass 1 replaced with home; filled 2026-09-25) | 52 | 124 cells | 64 |
| 6 | `notebooks/chain_review_6.csv` (activity trips that arrive home, `actividad_en_casa`; filled 2026-09-25) | 138 | 141 cells | 139 |
| 7 | `notebooks/chain_review_7.csv` (three kinds of pass-1 edit undone, written by `scripts/revisions/chains_7.py`; 2026-09-25) | 38 | 48: 25 cells, 23 rows restored | 0: every edit undoes a pass-1 decision |

The frozen edits that left no decision are the 6,357 origin types pass 1 typed per trip (moot:
the origin type is one answer per person), values a later pass put back, drop-and-restore
pairs, and edits whose value already held.

Pass 1 was frozen with 26 values normalized from the sheet: 21 starts written `HH:MM:00`,
one `22.13` (household 7791, person 4, trip 4, read 22:13), four `Regresar a casa`. It was
filled before the chain rules gave a moved origin its zone as well as its AGEB (d7b4fea);
`apply_edits` sets the zone from the AGEB, so the two agree either way.

Pass 2 was filled by Claude (the coding assistant) at the user's request, one decision per
chain with the reason in its note, and verified with `eodgdl review verify`: it leaves no
breaking code (`chains.BREAKING_ISSUES`) in the survey. Its decisions, by pattern:

- **Night shifts** (34 persons): the day opens with the morning return from the job and the
  last row is the next morning's return, outside the survey day and a repeat of the first
  trip; that row is dropped (the day ends at work, `fin_fuera_de_casa`) and the evening trips
  pass 1 had moved to the morning go back to their recorded times. Two of them had their
  morning return dropped by pass 1 as a return made from home; it is restored from the job.
- **Readings pass 1 broke** (9): a recorded or rules' time that made the chain consistent is
  put back (a day shift instead of a night one where the day starts and ends at home).
- **Leg minutes overrunning the next start** (2): the start moves the least that fits,
  leaving at most a tolerated overlap.
- **One-offs** (9): 12-hour-clock or leading-1 slips from the typo menu; where no menu
  reading fits, the first hour with the recorded minutes after the previous arrival; two
  returns that never reached home recoded as the activity at the place the day started.
- **Returns after midnight from an afternoon shift** (17): no edit. They are consistent and
  are marked `hora_nocturna` since `chains._remaining_issues` reads a return home by 06:00
  after a trip that started at or after noon as overnight (the typo search keeps 18:00).
- **The 75 first-trip returns from another AGEB** (see 01ef010) are restored with their
  origin type from the person's or household's other trips to that AGEB (14), else
  `Otros (especifique)`; the two recorded one row before their trip out are swapped with it.

Pass 3 (Claude, 2026-09-25) is a second look at pass 2's judgment calls. **Night shifts,
applied to every diary that shows one**: 89 persons open the day with the morning return
from a job and leave for it again at 19:00 or later. Pass 2 had applied its convention only
to the diaries still carrying a breaking code; 52 others kept a reading the times allow but
the day does not — pass 1's "AM/PM correction" that sent the evening work trip to the
morning (a night worker home from the factory at 07:15 back at work at 09:00), or a
typo-search repair that read the next morning's return as the same evening (a 26-minute
shift). The 42 with the full signature — the morning return comes from the very AGEB they
leave for at night, and their answer to where the day started is that place, not home —
get the pass-2 convention: the evening work trip at its recorded time, the next morning's
return dropped. The other 10 answered that the day started at home, which pass 1's day
reading agrees with; they are left as they are. **Origin types**: 25 of the restored rows
left as `Otros` take the place type that 60% or more of at least five trips to that AGEB by
people of the same occupation report (a student returning from an AGEB whose student trips
are 91% to a school is returning from school). The two day-shift readings of pass 2
(9245/4, 9417/4) were checked and stand: a household member works 08:00–18:00 at the same
AGEB, and none of 51 other work trips to the other factory AGEB starts after 18:00.

Pass 4 (Claude, 2026-09-25) looks at the 48 rows that became returns made from home through
pass-1 edits rather than through the rules. 45 are the return recorded twice: pass 1 recoded
the return from work or school (typed with the activity's motive, `actividad_en_casa`) as a
return, and the row after it — a 'Regresar a Casa' from home to home with the same mode and,
in most, the same leg minutes — is its duplicate; they stay non-trips. In the other three
pass 1 also moved the first trip's origin home and overwrote the person's day-start answer
(a workplace) with 'Su casa', turning the return from work into a non-trip; the recorded
origin and answer come back (14417/3's bicycle commute among them), and 14018/1's
home-to-home staff-bus ride at 21:00 is the trip to the night shift.

Pass 5 (Claude, 2026-09-25) revisits pass 1's rule for the day's first trip — "nobody spends
the night at a shop, so the day started at home" — which replaced the person's day-start
answer (a place) with 'Su casa' on 647 first trips and moved 642 origins to the home AGEB.
The survey's own evidence says pass 1 was mostly right: in 363 the recorded origin is a copy
of the destination (same AGEB and type), in 153 the type is copied and the recorded AGEB lies
near home (median 1.1 km, the dwelling coded one AGEB off), in 60 the AGEB is copied, and in
all three groups the trip's reported minutes fit a trip from home (87–92%). What pass 5 does:
- **Rotated records** (27): the first trip was recorded from a place to home and the next
  from home onwards, so the place is where the first trip went. Pass 1 moved the origin home
  and left the destination at home, a trip from home to home; the place becomes the first
  trip's destination and the next trip's origin (11 pass 1 had already rotated are left).
- **A separate start** (25): the recorded origin has its own AGEB and type, appears nowhere
  else in the day, and the minutes fit a trip from there better than from home or do not
  tell; the recorded origin and answer come back (the day starts away from home).
The other first trips keep pass 1's reading, as do those whose minutes favour home, whose
place recurs later in the day, or whose AGEB has no census geometry to measure.

Pass 6 (Claude, 2026-09-25) takes the 603 activity trips that end at a place typed 'Su casa'
(`actividad_en_casa`). The 141 that arrive at the household's AGEB from elsewhere — nearly
all right after an activity at their origin and followed by a trip that leaves home — are
the return home typed with the activity's motive, and are recoded 'Regresar a Casa' (the
model's purpose on arriving home is H, whatever the person does there). In 85 of them the
next row repeats that return from home with the same mode and minutes, the double-recorded
return of pass 4, and becomes a non-trip. Left as they are: 375 that go from the home AGEB
to the home AGEB — 372 recorded so by the survey, often a whole tour ("to work" and back,
home to home, 20–40 minutes by bus) whose place was never recorded, so the activity stays in
the home zone for the model — and 87 whose 'Su casa' lies in another AGEB, where only the
type label is odd and the model's purpose and zone are right.

Pass 7 (Claude, 2026-09-25, from a review of the chain fixes; the criteria are
`scripts/revisions/chains_7.py`) undoes three kinds of pass-1 edit:
- **Home-to-home tours** (13 persons, 23 rows): pass 1 dropped 15 activity trips from the
  home AGEB to the home AGEB and the 8 returns closing them as "no corresponde a un viaje
  real". Seven are commutes to work by light rail, bus or car of 20–75 minutes each way, and
  five persons (18/1, 31/7, 504/2, 633/2, 646/1) had lost every trip. Pass 6 keeps 375 trips of
  that shape — the place was never recorded, so the activity stays in the home zone for the
  model — and these come back as the rules left them.
- **Children's daycare trips** (23): pass 1 recoded 'Guardería' as 'Llevar o recoger a
  alguien' for students aged 6–11, the reading it gave the adults. A child in daycare attends
  it; the motive goes back, and the model reads it as school under 12.
- **Two modes** (2338/1/7, 3005/2/1): pass 1 changed the main mode of two one-leg trips with no
  evidence, away from the leg the reweighting reads; both go back to the survey's.
