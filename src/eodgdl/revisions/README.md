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

Pass 1 was frozen with 26 values normalized from the sheet: 21 starts written `HH:MM:00`,
one `22.13` (household 7791, person 4, trip 4, read 22:13), four `Regresar a casa`. It was
filled before the chain rules gave a moved origin its zone as well as its AGEB (d7b4fea);
`apply_edits` sets the zone from the AGEB, so the two agree either way.
