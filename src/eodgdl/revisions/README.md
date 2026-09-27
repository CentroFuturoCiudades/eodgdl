# Hand decisions over the trip chains

Two tables hold the hand work on the trip chains, read at the two ends of the chain rules:

- `leg_minutes.csv.gz`, the **travel minutes** a review corrected, goes in **before** the rules
  (`eodgdl.chains._correct_minutes`, called by `clean_trip_chains`): the rules read the minutes,
  since a trip's arrival — the moment every later start is tested against — is its start plus its
  minutes. One row per trip, columns `household, person, trip, before, after, note, source`:
  `before` is the trip's reported minutes summed over its legs, `after` the minutes it takes
  instead, spread over its legs in proportion to what each reported, a minute each at least. The
  load fails if a trip's minutes are not `before`. The corrected trips carry `minutos:revision`
  in `ajustes`, and a review sheet shows their minutes, shipped → corrected, under `leg min`,
  which takes no new value.
- `chains.csv.gz`, the **hand decisions**, goes in **after** the rules: `load_eod()` applies it
  to their output in one pass (`eodgdl.review.apply_revisions`; `load_eod(revise_chains=False)`
  stops before it). Each row is one decision, columns `household, person, trip, field, before,
  after, note, source`, described below.

For the decisions:

- `before` is the rules' value as a review sheet shows it — the shipped one on a row the rules
  dropped — and `after` the value decided: `HH:MM`, a level's own label, a zone id or `home`,
  and `dropped` or `restored` under `status`.
- `note` is the reviewer's reason and `source` the review sheet that made the decision
  (`notebooks/<source>.csv`).
- There is one decision at most per trip and field, and each is made against the rules'
  output, never against another decision, so the table applies at once and in no particular
  order. The person's answer to where the day started (`orig. type`) is a decision about the
  person, trip `0`, whichever of their trips is first.

The hand decisions sit only on persons the rules' output marks — a `problemas` or an `ajustes`
code on one of their trips (`eodgdl.review.marked_persons`): the survey nothing marks keeps its
values (user, 2026-09-26).

A change to `leg_minutes.csv.gz` changes what the rules read, so it is a change to the rules:
take a snapshot first (below) and make it with a script under `scripts/revisions/`, as round 15's
writes it; no review sheet edits it.

## A review round

Export the chains pending a fix from the tables `load_eod()` returns (`eodgdl review export`:
by `problemas` code, by person with `--persons 8992/1,…`, by `--screen`, below, or by what moved
since a snapshot, `--since`), edit the sheet, check it (`eodgdl review verify`) and merge it
(`eodgdl review freeze`, which runs `eodgdl.review.freeze_round`). `verify` is `freeze
--dry-run`: the same checks and the same report, and nothing written, so a sheet that verifies
freezes. A new value replaces the field's decision and a value that is the rules' own removes
it; `dropped` adds a drop and takes the row's other decisions with it, and `restored` removes a
drop (or, on a row the rules dropped, is a decision of its own). A note typed where nothing else
changes gives the decisions already on that trip — and on the person's first trip, their answer —
a new reason, keeping their values and source. A decision the rules now give by themselves is
removed. `freeze` writes nothing unless the merged decisions, applied once to the rules' output,
give exactly what the sheet's edits give on top of today's tables. A round never stacks on
another: an earlier decision is replaced or removed in place, and git keeps the history. A round
whose criterion is a rule rather than a call per chain keeps the script that wrote its sheet
under `scripts/revisions/`, and a second run of it finds nothing.

With no breaking code left, the codes no longer point at chains to fix; the screens do
(`eodgdl.review.screens`; `verify` counts them on the rules' output, before the edits and after,
and names the persons a screen picks out that the rules' output does not): a trip that leaves the
minute the previous one arrives (`zero_stay`), one that leaves work so (`zero_work`), one that
leaves work or school within half an hour (`short_work`), more than 14 hours at work
(`long_workday`), a day of more than 20 hours (`long_day`), a first non-work trip before 05:00
(`early_start`), eight hours or more at an errand (`long_errand`). `screens` also counts a day's
`problems`, the measure a criterion round reads a day by (rounds 14 and 15); a day of more than 20
hours counts unless it ends with a night shift's return the next morning, before 10:00. A
consistent chain can still be implausible.

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
already, is moot, is stale, or names what no table can take; `tests/test_review.py` holds every
committed decision to "applies".

The stale check guards decisions, not agreements: where a round let the rules' value stand, no
decision holds it, and a rule change that moves it moves it silently. So before changing a rule
take a snapshot (`eodgdl review snapshot --out before.csv.gz`), and after it export the persons
whose values moved (`eodgdl review export --since before.csv.gz`) to look at them. Round 14 read
such a snapshot, `notebooks/revisions/snapshot_13.csv.gz`, taken before the rules of the fourth
review of the chains; round 15 reads it too.

## How the decisions are read

`eodgdl.review.apply_edits`, since the review of the chain fixes (2026-09-25):

- **Every decision is checked against the rules' output.** Its `before` must be what the
  rules give — the shipped value on a row the same decisions restore — or the load fails,
  naming every such decision (see *When the rules change*).
- **The origin type is one answer per person** ("¿En qué tipo de lugar inició su primer
  viaje?"): a decision gives it under trip `0` and sets it on every row the person has; a
  sheet shows it on the person's first trip and reads it there only (typed on any other trip
  it is moot), even when the same sheet drops that trip.
- **A mode decision keeps the trip on its legs**: on a one-leg trip the leg takes the new
  mode, on a trip of several legs the new mode must be one of theirs, since the model build
  reads the main mode and the reweighting the legs.
- **`ajustes` is net.** Field by field, a value that is the rules' carries the rules' codes,
  one decided by hand carries `<field>:revision`, and one that is the shipped value again
  carries nothing; `fila:revision` marks a row the rules dropped and a decision restored. The
  travel minutes carry `minutos:revision` from the rules' output, since they are corrected
  before the rules.

## Round 15: the table made again (2026-09-26)

The fifth review of the chains changed the rules under round 14's decisions (`eodgdl.chains`, the
fifth review in each rule's docstring):

- round 9's travel minutes are read **before** the rules (`leg_minutes.csv.gz`); round 14 had
  carried them as decisions applied after, so the typo search read the reported minutes and the
  hand then undid its readings start by start: household 7425, person 3, reports 120 minutes for
  four trips whose hours fit 60, and the search had read two of them two hours later;
- a day read past midnight must **hold together** after it: no trip to work or school after the
  night, and its last start less than 24 hours after its first (`_day_wraps`). Round 14 had
  kept 15 days that went to work again after the night, 11 of them running past 24 hours, and
  had lost round 8's split shift for household 8992, person 1 (21:00 →
  02:00 → 16:00 → 19:00, back to 09:00 → 14:00 → 16:00 → 19:00 now), because `screens` took
  any return from work on the next day, at any hour, for a night shift's; it no longer does;
- the typo search runs **once more** after the move to the arrival, where a contradiction of
  minutes had kept it from reading an hour elsewhere in the chain (`_read_start_times`: household
  834, person 1, an escort at 04:30 that is 16:30);
- two of round 14's criteria are **rules**: a first trip recorded from a place to home whose next
  trip leaves from home went from home to the place (`_rotated_first_trips`, 47 days: round 14's
  16, 22 of the fifth pass's rotations, and 8 days nobody had read), and a return left away from
  home goes home if it is the day's last trip to an AGEB nothing else touches, else to 'Otros'
  (`_short_returns`, 29 returns).

The rules then leave 246 persons with a breaking code, all of it `hora_invertida`, against 291.
Round 14's table (1,274 decisions) was retired whole, as round 14 had retired the table before it,
and round 15 (`scripts/revisions/chains_15.py`, which reads round 14's inputs,
`notebooks/revisions/decisions_13.csv.gz` and `snapshot_13.csv.gz`) made it again:

- **Leg minutes** (371 trips, 432 legs): every one of round 9's corrections, written as
  `leg_minutes.csv.gz` with round 9's reason.
- **Carried** (62): the decisions of passes 2–6 on any field but the start, where the rules'
  output still holds the value each was made against and not yet the value it sets — pass 5's
  rotations the rule does not read, pass 4's staff bus, pass 2's swaps and night shifts outside
  the rule's signature, pass 6's recodes. Pass 5's rotations are completed (origin and answer)
  where no rule starts the day at home.
- **Start times** (514): 518 days read again. Candidates:
  - the rules' reading, made again on the chain as the carried decisions leave it (9 starts);
  - the earlier tables' reading, the first pass's and its corrections, a suggestion;
  - where the rules' reading still overlaps, the fewest minutes moved in all.

  The day keeps the reading with no overlap and the fewest problems (`screens`, plus a departure
  moved into the small hours). On a tie it keeps the fewest starts changed, then the fewest
  minutes. 241 days keep the rules' reading, 211 take the earlier one and 66 the fewest minutes.

576 decisions, from round 14's 1,274 (of which 371 are the leg minutes, now read before the
rules). No breaking code is left, every decision applies and sits on a person the rules' output
marks, and a second run of the script finds nothing.

## After round 15: the long errands and the capture's return times (2026-09-26)

A review of the long errands and the heaped returns changed the rules once more, the same day:

- The typo search now also reads a return home that starts 12 hours or more after the arrival at a
  short errand, in a chain whose times hold. It uses the −12 h edit alone (`hora:-12h`).
- Two tolerated codes mark a return that repeats the trip before it and starts exactly 1h01 or
  5h01 after it: `hora_1h01` and `hora_5h01`, the capture's own start for a return. The values
  are kept.

The table did not change. A snapshot taken before the change and `review export --since` after it
found only the 123 chains the new reading takes, and none of them holds a decision. No decision
went stale, and round 15's script run on the new rules finds nothing: the 123 days join the days
it weighs, each keeps the rules' reading, and the rest read as before.
The codes change no value. They mark about 1,300 more persons, so the default `review export`
sheet grows; `--codes breaking` does not.

## History

Pass 1 (`notebooks/chain_review.csv`, every person with a `problemas` code, exported 2026-09-08
and returned 2026-09-24) was followed by passes 2–7 (2026-09-25), folded into one table the same
day, and rounds 8–13 merged into it by `eodgdl review freeze`:
- passes 2–3: night shifts, first-trip returns;
- pass 4: non-trips pass 1 made;
- pass 5: rotated first trips;
- pass 6: activities arriving home;
- pass 7: pass-1 edits undone;
- round 8: a split shift;
- round 9: leg minutes;
- round 10: evening returns;
- round 11: the rules of the third review;
- round 12: implausible days;
- round 13: notes.

Round 14 (2026-09-26, the fourth review of the chains) retired every earlier decision and made the
table again on rules that took over the first pass's templates; round 15 (the same day, the fifth
review) did the same on the rules that took over two of round 14's criteria. The sheets of passes
1–10, round 9's and 10's scripts and the frozen passes are in git (commit 9570b7c and earlier),
round 14's script and sheet in commit 42b0cbb; round 15's notes say which earlier decision each
carried decision keeps.
