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
  order. The person's answer to where the day started (`orig. type`) is a decision about the
  person, trip `0`, whichever of their trips is first.

The hand decisions sit only on persons the rules' output marks — a `problemas` or an `ajustes`
code on one of their trips (`eodgdl.review.marked_persons`): the survey nothing marks keeps its
values (user, 2026-09-26).

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
`problems`, the measure a criterion round reads a day by (round 14's). A consistent chain can
still be implausible.

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
review of the chains.

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
- **A leg-minutes decision sets the trip's travel minutes**, spread over its legs in
  proportion to what each held, a minute each at least: the model's `Duration`.
- **`ajustes` is net.** Field by field, a value that is the rules' carries the rules' codes,
  one decided by hand carries `<field>:revision`, and one that is the shipped value again
  carries nothing; `fila:revision` marks a row the rules dropped and a decision restored.

## Round 14: the table made again (2026-09-26)

The fourth review of the chains took the first hand pass as a suggestion only and turned its
templates into chain rules (`eodgdl.chains`, the fourth review in each rule's docstring):

- a return recorded ending in its own origin's AGEB goes home (`destino:copia`);
- an activity trip that arrives at 'Su casa' in the home zone from another zone is a return
  (`motivo:casa`, what passes 1 and 6 did by hand);
- a return that ends elsewhere takes the motive the person's trips give the place
  (`motivo:lugar`);
- a daycare trip from age 12 is an escort (`motivo:guarderia`);
- a night shift's next-morning return, the last row of a day that opens with the return from
  the job, is no trip of the day (`regreso_dia_siguiente`, the convention of passes 2 and 3);
- every trip starts where the previous one ended (`origen:anterior`);
- a first trip whose answer says home, or copies its destination's place type, starts at home
  (`origen:respuesta`, `origen:tipo_copia`);
- a day that opens with a return from elsewhere answers where it started
  (`tipo_origen:regreso`, passes 2 and 3);
- the hours get a second menu, a night window and a limit at work.

The rules then leave 291 persons with a breaking code, against 1,100 before. Every decision was
retired: the first pass's (2,642, its notes with them), and rounds 11–13's, which existed to carry
and mend it. `notebooks/revisions/decisions_13.csv.gz` keeps the table as rounds 1–13 left it (3,663
decisions), and `notebooks/revisions/snapshot_13.csv.gz` the tables they gave.

Round 14 (`scripts/revisions/chains_14.py`, which reads both files) made again what still says
something the rules do not, and is the table's one source:

- **Carried** (476): the decisions of passes 2–6 and round 9 on any field but the start, where
  the rules' output still holds the value each was made against and not yet the value it sets.
  Among them are round 9's 371 leg minutes, pass 5's rotated first trips, pass 2's night shifts
  outside the rule's signature, and pass 4's staff bus. Pass 6's recodes whose trip only the
  first pass had put at home, and every day-start answer and night-shift drop a rule now gives,
  are left. Pass 5's rotations are completed (27 days) where no rule starts the day at home: the
  origin and the answer that the first pass had set.
- **Rotated first trips** (16 days): a day that still says it started at home while its first
  trip, an activity, was recorded from a place to home and the next from home onwards takes pass
  5's reading. The first trip went from home to the place, and the next leaves from there.
- **Start times** (647): 689 days read again. Candidates:
  - the rules' reading, made again on the chain as the carried decisions leave it (176 starts);
  - the earlier tables' reading, the first pass's and its corrections, a suggestion;
  - where the rules' reading still overlaps, the fewest minutes moved in all.

  The day keeps the reading with no overlap and the fewest problems (`screens`, plus a departure
  moved into the small hours). On a tie it keeps the fewest starts changed, then the fewest
  minutes. 417 days keep the rules' reading, 205 take the earlier one and 67 the fewest minutes.
- **Returns that stop short** (29): the day's last trip to an AGEB that no other trip of the day
  touches ends at home; any other becomes 'Otros', the O the model already gave it.

1,274 decisions, from 3,663 in eleven sources. No breaking code is left, every decision applies
and sits on a person the rules' output marks, and a second run of the script finds nothing.

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

The sheets of passes 1–10, round 9's and 10's scripts and the frozen passes are in git (commit
9570b7c and earlier); round 14's notes say which earlier decision each carried decision keeps.
