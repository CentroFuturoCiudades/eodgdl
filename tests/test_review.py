"""eodgdl.review: the review sheet round trip — export, edit by hand, recover the edits, apply, verify, freeze."""
import gzip

import pandas as pd
import pytest

from eodgdl import EODTables, clean_trip_chains, review
from eodgdl._resources import chain_revisions
from eodgdl.chains import BREAKING_ISSUES, ISSUE_CODES, PERSON, TOLERATED_ISSUES, has_code, mark_issues, non_trips
from test_eod import AGEB, CAR, ELSEWHERE, SHOP, VIV, _trips

EDIT_COLUMNS = review.KEYS + ["field", "before", "after", "note"]


def _fixture():
    """Five persons: a repaired typo, an imputed return with its duplicate, a clean day, an inversion left, a day away."""
    return _trips([
        (1, 8, 0, "Trabajar"), (1, 5, 30, "Regresar a Casa"),                       # 05:30 → 17:30, a 12-hour entry
        (2, 8, 0, "Trabajar", "Oficina", None, 30, CAR),
        (2, None, None, None, None, AGEB, 30, CAR),                                 # the return, block lost
        (2, 18, 0, "Regresar a Casa", None, None, 5, CAR),                          # its home-to-home duplicate: dropped
        (3, 8, 0, "Compras (comida)"), (3, 9, 0, "Regresar a Casa"),               # clean
        (4, 8, 0, "Trabajar"), (4, 7, 0, "Regresar a Casa"),
        (4, 13, 0, "Compras (comida)"), (4, 14, 0, "Regresar a Casa"),              # 07:00: no reading within the cost cap
        (5, 8, 0, "Trabajar"),                                                      # never comes home
    ])


def _survey(trips):
    """The shipped and the cleaned tables around a fixture trip table; no legs, the traslado columns hold the minutes."""
    persons = trips.index.droplevel("folio_viaje").unique()
    hab = pd.DataFrame({
        "sexo_nacimiento": "Mujeres", "edad": 30, "ocupacion": "Empleado",
        "viajes_contados": trips.groupby(level=PERSON).size().reindex(persons).to_numpy(), "diario_repetido": False,
    }, index=persons)
    cleaned, _ = clean_trip_chains(trips, VIV)
    hab_clean = hab.copy()
    lost = trips.index.difference(cleaned.index).droplevel("folio_viaje").value_counts()
    hab_clean.loc[lost.index, "viajes_contados"] -= lost.to_numpy()
    return EODTables(VIV, hab, trips, None), EODTables(VIV, hab_clean, cleaned, None)


def test_the_sheet_is_the_browsers_table_and_reads_back_without_edits(tmp_path):
    shipped, cleaned = _survey(_fixture())
    rows = review.chain_rows(cleaned, shipped)
    sheet = review.chain_sheet(rows, cleaned.hab)
    assert list(sheet.columns) == review.COLUMNS and len(sheet) == len(shipped.trips)
    assert review.COLUMNS[7:11] == ["trip", "start", "new start", "motive"] and review.COLUMNS[-3:] == ["status", "new status", "note"]
    one = sheet.set_index(review.KEYS)
    # shipped → cleaned where the rules changed a value, the value alone elsewhere, HH:MM, home for the household's zone
    assert one.loc[(1, 1, 2), ["start", "ajustes", "problemas", "status"]].tolist() == ["05:30 → 17:30", "hora:+12h", "", "changed"]
    assert one.loc[(1, 2, 2), ["start", "motive", "dest. type", "origin", "destination"]].tolist() == [
        "— → 18:00", "— → Regresar a Casa", "— → Su casa", ELSEWHERE, "home"]
    assert one.loc[(1, 2, 2), ["mode", "leg min", "ajustes"]].tolist() == [CAR, 30, "hora:duplicado;motivo:duplicado"]
    # the origin type is the person's one answer, shown on the first trip; a dropped row keeps its shipped values, no codes
    assert one.loc[(1, 2), "orig. type"].tolist() == ["Su casa", "", "Su casa"]
    assert one.loc[(1, 2, 3), ["start", "motive", "origin", "ajustes", "problemas", "status"]].tolist() == [
        "18:00", "Regresar a Casa", "home", "", "", "dropped"]
    assert one.loc[(1, 4, 2), ["start", "problemas", "status"]].tolist() == ["07:00", "hora_invertida", ""]
    assert one.loc[(1, 5, 1), "problemas"] == "fin_fuera_de_casa"
    assert (one.home == AGEB).all() and (one.note == "").all()
    assert one.loc[(1, 3, 1), ["sex", "age", "occupation", "repeated"]].tolist() == ["Mujeres", "30", "Empleado", ""]
    # the persons pending a fix: any problemas code, or the codes given
    assert list(review.pending_persons(rows)) == [(1, 4), (1, 5)]
    assert list(review.pending_persons(rows, ["hora_invertida"])) == [(1, 4)]
    with pytest.raises(ValueError, match="not a problemas code"):
        review.pending_persons(rows, ["hora_invertida", "nope"])
    pending = review.chain_sheet(rows, cleaned.hab, review.pending_persons(rows))
    assert pending.person.tolist() == [4, 4, 4, 4, 5]
    empty = review.chain_sheet(rows, cleaned.hab, review.pending_persons(rows, ["regreso_sin_llegar"]))
    assert list(empty.columns) == review.COLUMNS and empty.empty            # nothing pending: an empty sheet, no error
    assert has_code(pd.Series([], dtype=str), "hora_invertida").dtype == bool
    # written and read back, a sheet is what it was: no edits, against itself or against the full sheet
    path = review.write_sheet(pending, tmp_path / "chain_review.csv")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")              # a BOM, so spreadsheets read UTF-8
    back = review.read_sheet(path)
    assert list(back.columns) == review.COLUMNS and back.trip.dtype.kind == "i"
    assert (back[[review.new_column(c) for c in review.EDITABLE]] == "").all().all()
    assert review.sheet_edits(back).empty
    with pytest.raises(ValueError, match="not a review sheet"):
        review.read_sheet(pd.DataFrame({"a": [1]}).to_csv(tmp_path / "other.csv", index=False) or tmp_path / "other.csv")


def test_edits_are_the_filled_new_columns():
    shipped, cleaned = _survey(_fixture())
    sheet = review.chain_sheet(review.chain_rows(cleaned, shipped), cleaned.hab)
    edited = sheet.copy()
    at = lambda hh, p, t: edited.index[(edited.household == hh) & (edited.person == p) & (edited.trip == t)][0]
    edited.loc[at(1, 4, 2), "new start"] = " 17:00 "                    # a new value beside the old one
    edited.loc[at(1, 1, 2), "new start"] = "18:30"                      # the before is what the sheet showed after the arrow
    edited.loc[at(1, 4, 2), "new motive"] = "Regresar a Casa"           # the same value: not an edit
    edited.loc[at(1, 3, 2), "new start"] = "9:00:00"                    # ...nor as a spreadsheet writes it,
    edited.loc[at(1, 3, 1), "new motive"] = "compras (comida)"          # in another case,
    edited.loc[at(1, 5, 1), "new origin"] = AGEB                        # or the home zone by its id
    edited.loc[at(1, 5, 1), "new destination"] = "home"
    edited.loc[at(1, 5, 1), "note"] = "went home"
    edited.loc[at(1, 2, 3), "new status"] = "restored"                  # a dropped row, brought back
    edited.loc[at(1, 3, 2), "new status"] = "Dropped"
    edited.loc[at(1, 3, 1), "new status"] = "restored"                  # a kept row: nothing to restore, not an edit
    edited.loc[at(1, 2, 2), "new status"] = "maybe"                     # recovered here, refused by apply_edits
    edited.loc[at(1, 3, 1), "note"] = "fine"                            # a note alone is an edit of its own
    edited.loc[at(1, 1, 1), "start"] = "09:00"                          # the original column is not read as an edit
    edited = edited[~((edited.household == 1) & (edited.person == 4) & (edited.trip == 4))]   # a missing row is nothing
    edits = review.sheet_edits(edited)
    assert list(edits.columns) == EDIT_COLUMNS
    assert [tuple(r) for r in edits.itertuples(index=False)] == [
        (1, 1, 2, "start", "17:30", "18:30", ""),
        (1, 2, 2, "status", "", "maybe", ""),
        (1, 2, 3, "status", "dropped", "restored", ""),
        (1, 3, 1, "note", "", "fine", "fine"),
        (1, 3, 2, "status", "", "Dropped", ""),
        (1, 4, 2, "start", "07:00", "17:00", ""),
        (1, 5, 1, "destination", ELSEWHERE, "home", "went home"),
    ]
    with pytest.raises(ValueError, match="repeats"):
        review.sheet_edits(pd.concat([sheet, sheet.iloc[[0]]]))
    with pytest.raises(ValueError, match="not a column the sheet reads back"):
        review.sheet_edits(sheet.assign(**{"new leg min": ""}))


def test_apply_edits_writes_the_fields_and_recomputes_both_columns():
    shipped, cleaned = _survey(_fixture())
    edits = pd.DataFrame([
        (1, 4, 2, "start", "07:00", "12:00", "a 12-hour entry"),        # the inverted return, put between work and the shop
        (1, 5, 1, "motive", "Trabajar", "Regresar a Casa", ""),         # the day away, made a return home...
        (1, 5, 1, "destination", ELSEWHERE, "home", ""),
        (1, 5, 1, "dest. type", SHOP, "Su casa", ""),
        (1, 3, 2, "status", "", "dropped", ""),                         # the clean day loses its return
        (1, 2, 3, "status", "dropped", "restored", ""),                 # the duplicate comes back
        (1, 3, 1, "note", "", "fine", "fine"),
        (1, 1, 2, "start", "17:30", "17:30", ""),                       # already holds: skipped, no code
    ], columns=EDIT_COLUMNS)
    revised = review.apply_edits(cleaned, edits, shipped, cleaned)
    t = revised.trips
    assert (t.loc[(1, 4, 2), "hora_inicio_h"], t.loc[(1, 4, 2), "hora_inicio_m"]) == (12, 0)
    assert t.loc[(1, 4, 2), "ajustes"] == "hora:revision" and (t.loc[(1, 4)].problemas == "").all()   # cleared
    assert t.loc[(1, 5, 1), ["motivo_viaje", "destino", "tipo_lugar_destino"]].tolist() == ["Regresar a Casa", AGEB, "Su casa"]
    assert set(t.loc[(1, 5, 1), "ajustes"].split(";")) == {"motivo:revision", "destino:revision", "tipo_destino:revision"}
    assert t.loc[(1, 5, 1), "problemas"] == "regreso_en_casa" and non_trips(t).tolist().count(True) == 2   # ...from home: made a non-trip
    assert (1, 3, 2) not in t.index and t.loc[(1, 3)].problemas.tolist() == ["fin_fuera_de_casa"]        # dropping the return leaves the day away
    assert t.loc[(1, 2, 3), ["ajustes", "problemas"]].tolist() == ["fila:revision", "regreso_en_casa"]    # restored, and still home to home
    assert revised.hab.viajes_contados.tolist() == [2, 3, 1, 4, 1] and cleaned.hab.viajes_contados.tolist() == [2, 2, 2, 4, 1]
    assert t.loc[(1, 1, 2), "ajustes"] == "hora:+12h"                       # the rules' work stays
    assert t.problemas.equals(mark_issues(t, VIV))
    # the edits are applied whole or not at all, and every fault is listed
    bad = pd.DataFrame([
        (1, 4, 2, "start", "07:00", "25:00", ""),
        (1, 4, 3, "leg min", "0", "99", ""),
        (1, 4, 3, "origin", "home", "0000000000000", ""),
        (1, 1, 9, "motive", "", "Trabajar", ""),
        (1, 3, 1, "motive", "Trabajar", "Estudiar", ""),               # the sheet showed a value the table does not hold
        (1, 3, 2, "status", "", "maybe", ""),
        (9, 9, 9, "status", "", "restored", ""),
    ], columns=EDIT_COLUMNS)
    with pytest.raises(ValueError) as err:
        review.apply_edits(cleaned, bad, shipped, cleaned)
    msg = str(err.value)
    assert msg.count("\n") == 7
    for text in ("not a time", "not a column", "not a zone", "no such trip in", "another version", "status must be",
                 "no such trip to restore"):
        assert text in msg
    same = review.apply_edits(cleaned, bad.iloc[:0], shipped, cleaned)
    assert same.trips.equals(cleaned.trips) and same.hab.equals(cleaned.hab)


def test_a_row_a_pass_dropped_comes_back_as_the_rules_left_it():
    shipped, rules = _survey(_fixture())
    drop = pd.DataFrame([(1, 1, 2, "status", "", "dropped", ""), (1, 3, 2, "status", "", "dropped", "")], columns=EDIT_COLUMNS)
    first = review.apply_edits(rules, drop, shipped, rules)
    assert (1, 1, 2) not in first.trips.index and first.hab.viajes_contados.tolist() == [1, 2, 1, 4, 1]
    back = pd.DataFrame([
        (1, 1, 2, "status", "dropped", "restored", ""),
        (1, 3, 2, "status", "dropped", "restored", ""),
        (1, 3, 2, "start", "09:00", "09:30", "restored and edited in one pass"),   # the sheet showed the dropped row as shipped
    ], columns=EDIT_COLUMNS)
    second = review.apply_edits(first, back, shipped, rules)
    t = second.trips
    # the rules' version: the repaired 17:30 with its code; a row back where the rules left it carries no code of its own
    assert (t.loc[(1, 1, 2), "hora_inicio_h"], t.loc[(1, 1, 2), "ajustes"]) == (17, "hora:+12h")
    assert (t.loc[(1, 3, 2), "hora_inicio_m"], t.loc[(1, 3, 2), "ajustes"]) == (30, "hora:revision")
    assert second.hab.viajes_contados.equals(rules.hab.viajes_contados)
    assert t.problemas.equals(mark_issues(t, VIV))
    # the passes in a row, as load_eod applies them: the rules' tables are where a restore comes from
    both = review.apply_revisions(rules, shipped, [("a", drop), ("b", back)])
    assert both.trips.equals(t)


def test_ajustes_says_what_the_values_owe_to_the_rules_and_to_the_hand():
    shipped, rules = _survey(_fixture())
    one = pd.DataFrame([(1, 1, 2, "start", "17:30", "18:30", ""), (1, 4, 2, "start", "07:00", "12:00", "")], columns=EDIT_COLUMNS)
    first = review.apply_edits(rules, one, shipped, rules)
    assert first.trips.loc[(1, 1, 2), "ajustes"] == "hora:revision"       # the hand's value supersedes the rule's reading
    two = pd.DataFrame([(1, 1, 2, "start", "18:30", "17:30", ""), (1, 4, 2, "start", "12:00", "07:00", "")], columns=EDIT_COLUMNS)
    undone = review.apply_edits(first, two, shipped, rules)
    assert undone.trips.equals(rules.trips)                               # undone, nothing of either pass is left on the rows
    three = pd.DataFrame([(1, 1, 2, "start", "17:30", "05:30", "")], columns=EDIT_COLUMNS)
    assert review.apply_edits(rules, three, shipped, rules).trips.loc[(1, 1, 2), "ajustes"] == ""   # the survey's own value


def test_the_origin_type_is_one_answer_per_person():
    shipped, rules = _survey(_fixture())
    edits = pd.DataFrame([
        (1, 4, 1, "orig. type", "Su casa", "Otra vivienda", "slept at a relative's"),   # the first trip: the answer
        (1, 4, 3, "orig. type", "", "Oficina", "per trip"),                              # a later trip: moot
        (1, 2, 3, "status", "dropped", "restored", ""),                                  # restored with the person's answer...
        (1, 2, 1, "orig. type", "Su casa", "Otra vivienda", ""),                         # ...which the same edits change
    ], columns=EDIT_COLUMNS)
    v = review.verify_edits(rules, shipped, edits, rules)
    t = v.tables.trips
    assert (t.loc[(1, 4)].tipo_lugar_origen == "Otra vivienda").all() and (t.loc[(1, 2)].tipo_lugar_origen == "Otra vivienda").all()
    assert has_code(t.loc[(1, 4)].ajustes, "tipo_origen:revision").all()   # every row holds the changed answer
    assert has_code(t.loc[(1, 4)].problemas, "inicio_fuera_de_casa").tolist() == [True, False, False, False]
    assert v.summary["edits_moot"] == 1 and t.groupby(level=PERSON).tipo_lugar_origen.nunique().max() == 1
    assert v.sheet.set_index(review.KEYS).loc[(1, 4), "orig. type"].tolist() == ["Su casa → Otra vivienda", "", "", ""]


def test_verify_edits_says_what_the_edits_cleared_left_and_made():
    shipped, cleaned = _survey(_fixture())
    edits = pd.DataFrame([
        (1, 4, 2, "start", "07:00", "12:00", "a 12-hour entry"),
        (1, 5, 1, "motive", "Trabajar", "Regresar a Casa", ""),
        (1, 5, 1, "destination", ELSEWHERE, "home", ""),
    ], columns=EDIT_COLUMNS)
    v = review.verify_edits(cleaned, shipped, edits, cleaned)
    assert v.persons.values.tolist() == [
        [1, 4, 1, 0, 0, 1, "hora_invertida", "", True],
        [1, 5, 2, 0, 0, 0, "fin_fuera_de_casa", "regreso_en_casa", False],
    ]
    assert v.summary == {"persons": 2, "cells": 3, "rows_dropped": 0, "rows_restored": 0, "edits_already_holding": 0,
                         "edits_moot": 0, "defects_before": 2, "defects_after": 1, "persons_cleared": 1,
                         "persons_with_new_defects": 1}
    assert v.sheet.person.tolist() == [4, 4, 4, 4, 5]
    after = v.sheet.set_index(review.KEYS)
    assert after.loc[(1, 4, 2), ["start", "ajustes", "problemas", "status"]].tolist() == ["07:00 → 12:00", "hora:revision", "", "changed"]
    assert after.loc[(1, 5, 1), ["motive", "destination", "problemas"]].tolist() == [
        "Trabajar → Regresar a Casa", f"{ELSEWHERE} → home", "regreso_en_casa"]
    assert v.tables.trips.problemas.equals(mark_issues(v.tables.trips, VIV))


def test_a_spreadsheet_s_habits_are_not_edits_that_fail(tmp_path):
    shipped, cleaned = _survey(_fixture())
    motives = sorted(set(cleaned.trips.motivo_viaje.dropna()))
    cleaned.trips["motivo_viaje"] = cleaned.trips.motivo_viaje.astype(pd.CategoricalDtype(motives))   # levels, as load_eod's
    # a time with :00 seconds, a label in another case, notes under "comments"
    edits = pd.DataFrame([
        (1, 4, 2, "start", "7:00", "12:00:00", ""),
        (1, 5, 1, "motive", "Trabajar", "regresar a casa", ""),
    ], columns=EDIT_COLUMNS)
    once = review.apply_edits(cleaned, edits, shipped, cleaned)
    t = once.trips
    assert (t.loc[(1, 4, 2), "hora_inicio_h"], t.loc[(1, 4, 2), "hora_inicio_m"]) == (12, 0)
    assert t.loc[(1, 5, 1), "motivo_viaje"] == "Regresar a Casa"                # the level, not the text typed
    with pytest.raises(ValueError, match="not a time"):
        review.apply_edits(cleaned, edits.assign(after=["12:00:30", "Trabajar"]), shipped, cleaned)
    # a second pass must start from what the first left, not from what the first found
    with pytest.raises(ValueError, match="another version"):
        review.apply_edits(once, edits.assign(after=["12:30", "Trabajar"]), shipped, cleaned)
    twice = review.apply_edits(once, edits.assign(before=["12:00", "Regresar a Casa"], after=["12:30", "Trabajar"]), shipped, cleaned)
    assert twice.trips.loc[(1, 4, 2), "ajustes"] == "hora:revision" and twice.trips.loc[(1, 4, 2), "hora_inicio_m"] == 30
    assert twice.trips.loc[(1, 5, 1), "ajustes"] == ""                         # the survey's motive again: nothing to say
    sheet = review.chain_sheet(review.chain_rows(cleaned, shipped), cleaned.hab).rename(columns={"note": "comments"})
    sheet.loc[0, "comments"] = "seen"
    back = review.read_sheet(review.write_sheet(sheet, tmp_path / "s.csv"))
    assert "note" in back and review.sheet_edits(back).note.tolist() == ["seen"]
    # the issue groups split the codes, and select persons by name
    assert set(BREAKING_ISSUES) | set(TOLERATED_ISSUES) == set(ISSUE_CODES) and not set(BREAKING_ISSUES) & set(TOLERATED_ISSUES)
    rows = review.chain_rows(cleaned, shipped)
    assert list(review.pending_persons(rows, ["breaking"])) == [(1, 4)]          # hora_invertida
    assert list(review.pending_persons(rows, ["tolerated"])) == [(1, 5)]         # fin_fuera_de_casa


def test_freeze_keeps_what_changes_in_canonical_form(tmp_path):
    shipped, cleaned = _survey(_fixture())
    motives = sorted(set(cleaned.trips.motivo_viaje.dropna()))
    cleaned.trips["motivo_viaje"] = cleaned.trips.motivo_viaje.astype(pd.CategoricalDtype(motives))   # levels, as load_eod's
    sheet = review.chain_sheet(review.chain_rows(cleaned, shipped), cleaned.hab)
    at = lambda hh, p, t: sheet.index[(sheet.household == hh) & (sheet.person == p) & (sheet.trip == t)][0]
    sheet.loc[at(1, 4, 2), ["new start", "new motive", "note"]] = ["9:05:00", "compras (comida)", "one note per trip"]
    sheet.loc[at(1, 3, 2), ["new status", "new start", "note"]] = ["dropped", "10:00", "the field of a dropped row is moot"]
    sheet.loc[at(1, 5, 1), "note"] = "a note alone"
    edits = review.sheet_edits(sheet)
    frozen = review.freeze_edits(edits, cleaned, shipped, cleaned)
    assert [tuple(r) for r in frozen.itertuples(index=False)] == [
        (1, 3, 2, "status", "", "dropped", "the field of a dropped row is moot"),
        (1, 4, 2, "start", "07:00", "09:05", "one note per trip"),
        (1, 4, 2, "motive", "Regresar a Casa", "Compras (comida)", ""),
    ]
    path = review.write_pass(frozen, tmp_path / "chains_9.csv.gz")
    assert path.read_bytes() == review.write_pass(frozen, tmp_path / "again.csv.gz").read_bytes()   # reproducible bytes
    back = pd.read_csv(gzip.open(path), dtype=str, keep_default_na=False).astype({k: int for k in review.KEYS})
    assert back.equals(frozen)
    assert review.apply_edits(cleaned, back, shipped, cleaned).trips.equals(review.apply_edits(cleaned, edits, shipped, cleaned).trips)
    with pytest.raises(ValueError, match="not a time"):
        review.freeze_edits(edits.assign(after=edits.after.replace("9:05:00", "9.05")), cleaned, shipped, cleaned)


def test_a_mode_edit_keeps_the_trip_on_its_legs(stages):
    shipped, rules, revised = stages
    t, legs = revised.trips, revised.legs
    n = legs.groupby(level=[0, 1, 2]).size().reindex(t.index)
    one = t.index[(n == 1).to_numpy() & (t.modo_principal == "A PIE").to_numpy()][0]
    several = t.index[(n > 1).to_numpy()][0]
    other = [m for m in legs.loc[several].traslado_medio.astype(str) if m != str(t.at[several, "modo_principal"])][0]
    edits = pd.DataFrame([
        (*one, "mode", "A PIE", "Bicicleta", "a one-leg trip: the leg follows"),
        (*several, "mode", str(t.at[several, "modo_principal"]), other, "another of its legs"),
    ], columns=EDIT_COLUMNS)
    out = review.apply_edits(revised, edits, shipped, rules)
    assert out.legs.loc[one].traslado_medio.tolist() == ["Bicicleta"] and out.trips.at[one, "ajustes"].endswith("modo:revision")
    assert out.legs.loc[several].equals(legs.loc[several]) and out.trips.at[several, "modo_principal"] == other
    absent = next(m for m in legs.traslado_medio.cat.categories if m not in set(legs.loc[several].traslado_medio.astype(str)))
    with pytest.raises(ValueError, match="none of the trip's leg modes"):
        review.apply_edits(revised, edits.iloc[1:].assign(after=absent), shipped, rules)


def test_the_pending_sheet_round_trips_on_the_survey(stages, tmp_path):
    shipped, cleaned = stages.shipped, stages.rules    # what pass 1 was exported from
    assert mark_issues(cleaned.trips, cleaned.viv, cleaned.legs).equals(cleaned.trips.problemas)
    rows = review.chain_rows(cleaned, shipped)
    assert len(rows) == 154_662 and int(rows.dropped.sum()) == 38
    pending = review.pending_persons(rows)
    sheet = review.chain_sheet(rows, cleaned.hab, pending)
    assert (len(pending), len(sheet)) == (4_511, 14_771)
    assert sheet.status.value_counts().to_dict() == {"": 14_293, "changed": 476, "dropped": 2}
    back = review.read_sheet(review.write_sheet(sheet, tmp_path / "chain_review.csv"))
    assert set(back.destination.str.len()) <= {4, 9, 13}                # zone ids as strings, home as home
    sel = rows[rows.index.droplevel("folio_viaje").isin(pending)]
    changed = int(((sel.start_shipped != sel.start_clean) & ~sel.dropped).sum())
    assert (back.motive == "Guardería").any() and back.start.str.contains("→").sum() == changed   # accents and arrows intact
    assert review.sheet_edits(back).empty
    same = review.apply_edits(cleaned, review.sheet_edits(back), shipped, cleaned)
    assert same.trips.equals(cleaned.trips) and same.legs.equals(cleaned.legs) and same.hab.equals(cleaned.hab)


def test_an_edited_trip_end_takes_the_zone_coded_for_its_ageb(stages):
    shipped, rules, cleaned = stages
    t, viv = cleaned.trips, cleaned.viv
    home = viv.ageb.astype(str).reindex(t.index.get_level_values("folio_vivienda")).to_numpy()
    away = t[(t.origen.astype(str).to_numpy() != home) & (t.destino.astype(str).to_numpy() != home)].head(50)
    other = t.destino.astype(str).loc[lambda s: s != away.destino.astype(str).iloc[0]].iloc[0]
    edits = pd.DataFrame(
        [(*k, "origin", str(away.at[k, "origen"]), "home", "") for k in away.index]
        + [(*away.index[0], "destination", str(away.destino.iloc[0]), other, "")],
        columns=EDIT_COLUMNS,
    )
    revised = review.apply_edits(cleaned, edits, shipped, rules).trips.loc[away.index]
    assert (revised.zona_origen.astype(str).to_numpy()
            == viv.centralidad.astype(str).reindex(away.index.get_level_values("folio_vivienda")).to_numpy()).all()
    coded = t.zona_destino.astype(str)[t.destino.astype(str) == other].iloc[0]
    assert revised.zona_destino.astype(str).iloc[0] == coded
    ends = pd.concat([pd.DataFrame({"ageb": revised[p].astype(str), "zone": revised[z].astype(str)})
                      for p, z in (("origen", "zona_origen"), ("destino", "zona_destino"))])
    assert (ends.groupby("ageb").zone.nunique() == 1).all()


def test_load_eod_applies_the_frozen_hand_passes(stages):
    shipped, rules, revised = stages
    t = revised.trips
    passes = chain_revisions()
    assert [(name, len(edits)) for name, edits in passes] == [
        ("chains_1.csv.gz", 11_669), ("chains_2.csv.gz", 260), ("chains_3.csv.gz", 93), ("chains_4.csv.gz", 9),
        ("chains_5.csv.gz", 124), ("chains_6.csv.gz", 141), ("chains_7.csv.gz", 48)]
    # a row's last status wins: pass 1 drops 512, pass 2 restores 77 of them and drops 34 more, pass 3 drops 42,
    # pass 7 restores 23
    status = pd.concat([e[e.field == "status"] for _, e in passes]).drop_duplicates(review.KEYS, keep="last")
    gone = pd.MultiIndex.from_frame(status.loc[status.after == "dropped", review.KEYS])
    back = pd.MultiIndex.from_frame(status.loc[status.after == "restored", review.KEYS])
    assert (len(gone), len(back)) == (488, 100)
    assert not t.index.isin(gone).any() and back.isin(t.index).all()
    assert len(t) == len(rules.trips) - 488 == 154_136 and revised.legs.index.droplevel("folio_traslado").isin(t.index).all()
    assert (revised.hab.viajes_contados == t.groupby(level=PERSON).size().reindex(revised.hab.index).fillna(0)).all()
    assert review.sheet_edits(review.chain_sheet(review.chain_rows(revised, shipped))).empty
    # the last pass again changes nothing: its values hold, its rows are gone or back
    again = review.apply_edits(revised, passes[-1][1], shipped, rules)
    assert again.trips.equals(t) and again.hab.equals(revised.hab) and again.legs.equals(revised.legs)
    assert t.problemas.equals(mark_issues(t, revised.viv, revised.legs))
    assert has_code(t.ajustes, "origen:revision").sum() == 1_365 and has_code(t.ajustes, "fila:revision").sum() == 0
    assert non_trips(t).sum() == 134
    # what is left: no breaking chain, the tolerated codes left alone
    assert {c: int(has_code(t.problemas, c).sum()) for c in BREAKING_ISSUES} == dict.fromkeys(BREAKING_ISSUES, 0)
    assert len(review.pending_persons(review.chain_rows(revised, shipped), ["breaking"])) == 0
    assert int(has_code(t.problemas, "hora_nocturna").sum()) == 160 and int(has_code(t.problemas, "hora_2301").sum()) == 90
    # the zone follows the AGEB on every revised trip end
    ends = pd.concat([pd.DataFrame({"ageb": t[p].astype(str), "zone": t[z].astype(str)})
                      for p, z in (("origen", "zona_origen"), ("destino", "zona_destino"))])
    assert (ends.groupby("ageb").zone.nunique() == 1).all()
    # the day-start answer is one per person, and every main mode is one of its trip's legs, the survey's legs
    assert t.groupby(level=PERSON).tipo_lugar_origen.nunique().max() == 1
    modes = revised.legs.traslado_medio.astype(str).groupby(level=[0, 1, 2]).agg(set).reindex(t.index)
    assert all(m in s for m, s in zip(t.modo_principal.astype(str), modes))
    assert revised.legs.traslado_medio.equals(shipped.legs.traslado_medio.reindex(revised.legs.index))
