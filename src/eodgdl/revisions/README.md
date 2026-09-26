# Hand revisions of the trip chains

`load_eod()` applies these files, in order, after the chain rules
(`eodgdl.review.apply_revisions`; `load_eod(revise_chains=False)` stops before them).
Each is one hand pass over a review sheet (`eodgdl review export`), frozen as the edits
`eodgdl.review.sheet_edits` recovers from it: one row per changed cell, columns
`household, person, trip, field, before, after, note`. `before` is what the sheet showed,
`after` the value applied, `note` the reviewer's reason (once per trip, on its first edit).
Every applied value leaves a `<field>:revision` code in `trips.ajustes`.

A later pass is exported from the tables with the earlier passes applied, so its `before`
is the earlier pass's `after`. Never edit a frozen pass: correct it with a new pass.

| file | sheet | persons | edits |
|---|---|---|---|
| `chains_1.csv.gz` | `notebooks/chain_review.csv` (every person with a `problemas` code, exported 2026-09-08, returned 2026-09-24) | 4,435 | 11,157 cells, 512 rows dropped |
| `chains_2.csv.gz` | `notebooks/chain_review_2.csv` (the persons still carrying a breaking code after pass 1, plus 75 first-trip returns pass 1 had dropped; filled 2026-09-25) | 146 | 149 cells, 34 rows dropped, 77 restored |
| `chains_3.csv.gz` | `notebooks/chain_review_3.csv` (a second look at pass 2's judgment calls; filled 2026-09-25) | 67 | 51 cells, 42 rows dropped |
| `chains_4.csv.gz` | `notebooks/chain_review_4.csv` (the non-trips pass 1 made out of real trips; filled 2026-09-25) | 3 | 9 cells |

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
