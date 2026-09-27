"""eodgdl.review: the review sheet round trip — export, edit by hand, recover the edits, apply, verify, freeze."""
import gzip

import pandas as pd
import pytest

from eodgdl import EODStages, EODTables, clean_trip_chains, review
from eodgdl._resources import chain_decisions, leg_minutes
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
    assert one.loc[(1, 2, 2), ["mode", "leg min", "ajustes"]].tolist() == [CAR, "30", "hora:duplicado;motivo:duplicado"]
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
        review.sheet_edits(sheet.assign(**{"new ajustes": ""}))


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
        (1, 4, 3, "ajustes", "", "hora:+12h", ""),
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


def _merge(decisions, edits, tables, shipped, rules, source):
    """A review round: the sheet's edits against the tables it was exported from, merged into the decisions."""
    return review.merge_decisions(decisions, review.freeze_edits(edits, tables, shipped, rules), rules, shipped, source)


def _same(a, b):
    return a.trips.equals(b.trips) and a.hab.equals(b.hab)


def test_a_round_merges_into_the_decisions_that_apply_in_one_pass():
    shipped, rules = _survey(_fixture())
    one = pd.DataFrame([
        (1, 1, 2, "status", "", "dropped", "a duplicate"),
        (1, 3, 2, "status", "", "dropped", "not a trip"),
        (1, 4, 2, "start", "07:00", "12:00", "a 12-hour entry"),
    ], columns=EDIT_COLUMNS)
    d1 = _merge(review.decision_frame([]), one, rules, shipped, rules, "round 1")
    t1 = review.apply_revisions(rules, shipped, d1)
    assert _same(t1, review.apply_edits(rules, one, shipped, rules))
    # the next round is exported from the tables the decisions give, and merges into the same decisions
    two = pd.DataFrame([
        (1, 1, 2, "status", "dropped", "restored", "a real trip after all"),
        (1, 3, 2, "status", "dropped", "restored", "a real trip after all"),
        (1, 3, 2, "start", "09:00", "09:30", ""),          # restored and edited in one round: the sheet showed it as shipped
        (1, 4, 2, "start", "12:00", "07:00", "the rules had it right"),
        (1, 5, 1, "status", "", "dropped", "not a trip"),
    ], columns=EDIT_COLUMNS)
    d2 = _merge(d1, two, t1, shipped, rules, "round 2")
    # a restore removes the drop, the rules' own value removes its decision: nothing is reverted by a later pass
    assert [tuple(r) for r in d2.itertuples(index=False)] == [
        (1, 3, 2, "start", "09:00", "09:30", "a real trip after all", "round 2"),
        (1, 5, 1, "status", "", "dropped", "not a trip", "round 2"),
    ]
    assert review.decision_changes(d1, d2) == {"added": 2, "changed": 0, "removed": 3, "notes": 0}
    once = review.apply_revisions(rules, shipped, d2)
    assert _same(once, review.apply_edits(t1, two, shipped, rules))
    # a row back comes as the rules left it: the repaired 17:30 with its code
    assert (once.trips.loc[(1, 1, 2), "hora_inicio_h"], once.trips.loc[(1, 1, 2), "ajustes"]) == (17, "hora:+12h")
    # applied again on the tables they give, the decisions all hold
    assert _same(review.apply_edits(once, d2, shipped, rules), once)


def test_an_answer_is_a_decision_about_the_person():
    shipped, rules = _survey(_fixture())
    one = pd.DataFrame([(1, 4, 1, "orig. type", "Su casa", "Otra vivienda", "slept at a relative's")], columns=EDIT_COLUMNS)
    d1 = _merge(review.decision_frame([]), one, rules, shipped, rules, "round 1")
    # typed on the first trip, kept under trip 0: the person
    assert d1[["trip", "field", "before", "after"]].values.tolist() == [[review.PERSON_TRIP, "orig. type", "Su casa", "Otra vivienda"]]
    t1 = review.apply_revisions(rules, shipped, d1)
    two = pd.DataFrame([(1, 4, 1, "status", "", "dropped", "not a trip")], columns=EDIT_COLUMNS)
    d2 = _merge(d1, two, t1, shipped, rules, "round 2")
    # a drop of the first trip leaves the answer where it is, with its note and round
    assert d2[d2.field == "orig. type"][["trip", "after", "note", "source"]].values.tolist() == [
        [review.PERSON_TRIP, "Otra vivienda", "slept at a relative's", "round 1"]]
    once = review.apply_revisions(rules, shipped, d2)
    assert (once.trips.loc[(1, 4)].tipo_lugar_origen == "Otra vivienda").all()
    assert _same(once, review.apply_edits(t1, two, shipped, rules))
    # the answer set back to the survey's, on the row the sheet now shows it on, removes the decision
    three = pd.DataFrame([(1, 4, 2, "orig. type", "Otra vivienda", "Su casa", "")], columns=EDIT_COLUMNS)
    assert (_merge(d2, three, once, shipped, rules, "round 3").field != "orig. type").all()
    # typed on a first trip the same sheet drops, the answer still applies: it is the person's
    four = pd.DataFrame([
        (1, 4, 1, "status", "", "dropped", "not a trip"),
        (1, 4, 1, "orig. type", "Su casa", "Otra vivienda", "slept at a relative's"),
    ], columns=EDIT_COLUMNS)
    both = review.apply_edits(rules, four, shipped, rules)
    assert (1, 4, 1) not in both.trips.index and (both.trips.loc[(1, 4)].tipo_lugar_origen == "Otra vivienda").all()
    assert _same(review.apply_revisions(rules, shipped, _merge(review.decision_frame([]), four, rules, shipped, rules, "r")), both)
    # a person every trip of whom the edits drop has no answer to give: moot
    gone = pd.DataFrame([(1, 5, 1, "status", "", "dropped", ""), (1, 5, 1, "orig. type", "Su casa", "Oficina", "")],
                        columns=EDIT_COLUMNS)
    assert review.verify_edits(rules, shipped, gone, rules).summary["edits_moot"] == 1


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
    screened = {f"{name}_{when}": 0 for name in ["zero_stays", *review.SCREENS] for when in ("rules", "before", "after")}
    screened.update(short_work_rules=1, short_work_before=1)   # the overlap left at work, which the edit closes
    assert v.summary == {"persons": 2, "cells": 3, "rows_dropped": 0, "rows_restored": 0, "edits_already_holding": 0,
                         "edits_moot": 0, "defects_before": 2, "defects_after": 1, "persons_cleared": 1,
                         "persons_with_new_defects": 1, **screened,
                         "persons_with_new_screens": 0, "persons_with_screens_not_in_rules": 0}
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
    # a time with :00 seconds, a label in another case
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
    sheet = review.chain_sheet(review.chain_rows(cleaned, shipped), cleaned.hab)
    sheet.loc[0, "note"] = "seen"
    back = review.read_sheet(review.write_sheet(sheet, tmp_path / "s.csv"))
    assert review.sheet_edits(back).note.tolist() == ["seen"]
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
    decisions = review.merge_decisions(review.decision_frame([]), frozen, cleaned, shipped, "round 1")
    path = review.write_decisions(decisions, tmp_path / "chains.csv.gz")
    assert path.read_bytes() == review.write_decisions(decisions, tmp_path / "again.csv.gz").read_bytes()   # reproducible bytes
    back = pd.read_csv(gzip.open(path), dtype=str, keep_default_na=False).astype({k: int for k in review.KEYS})
    assert back.equals(decisions) and (back.note != "").all()                  # every decision keeps its trip's note
    assert review.apply_revisions(cleaned, shipped, back).trips.equals(review.apply_edits(cleaned, edits, shipped, cleaned).trips)
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
    assert (len(pending), len(sheet)) == (3_468, 11_430)
    assert sheet.status.value_counts().to_dict() == {"": 10_587, "changed": 842, "dropped": 1}
    back = review.read_sheet(review.write_sheet(sheet, tmp_path / "chain_review.csv"))
    ends = back.destination.str.split(" → ")                           # a return that copied its origin goes home
    assert set(ends.str[-1].str.len()) <= {4, 9, 13} and set(ends.str[0].str.len()) <= {4, 9, 13}   # ids as strings
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


def test_load_eod_applies_the_hand_decisions(stages):
    shipped, rules, revised = stages
    t = revised.trips
    decisions = chain_decisions()
    # one decision per trip and field, every one with its reason and the review sheet it came from
    assert len(decisions) == 576 and not decisions.duplicated(review.KEYS + ["field"]).any()
    assert decisions.field.value_counts().to_dict() == {
        "start": 514, "origin": 16, "motive": 13, "destination": 10, "status": 10, "orig. type": 7, "dest. type": 6}
    # one round made them all, on the rules of the fifth review of the chains (scripts/revisions/chains_15.py)
    assert set(decisions.source) == {"chain_review_15"}
    # the day-start answer is a decision about the person: trip 0
    assert (decisions.loc[decisions.field == "orig. type", "trip"] == review.PERSON_TRIP).all()
    # the hand review touches only persons the rules' output marks: a problemas or an ajustes code on one of their trips
    who = pd.MultiIndex.from_frame(decisions[review.KEYS[:2]])
    assert who.isin(review.marked_persons(rules)).all()
    # every decision changes the rules' output: none stale, none that the rules already give or that cannot take effect
    assert (review.decision_outcomes(rules, shipped, decisions).outcome == "apply").all()
    assert (decisions.note != "").all() and decisions.source.str.fullmatch(r"chain_review(_\d+)?").all()
    gone = pd.MultiIndex.from_frame(decisions.loc[decisions.after == "dropped", review.KEYS])
    assert len(gone) == 10 and not t.index.isin(gone).any() and not (decisions.after == "restored").any()
    assert len(t) == len(rules.trips) - 10 == 154_614 and revised.legs.index.droplevel("folio_traslado").isin(t.index).all()
    assert (revised.hab.viajes_contados == t.groupby(level=PERSON).size().reindex(revised.hab.index).fillna(0)).all()
    assert review.sheet_edits(review.chain_sheet(review.chain_rows(revised, shipped))).empty
    # applied again on the tables they give, the decisions all hold: nothing changes
    again = review.apply_edits(revised, decisions, shipped, rules)
    assert again.trips.equals(t) and again.hab.equals(revised.hab) and again.legs.equals(revised.legs)
    assert t.problemas.equals(mark_issues(t, revised.viv, revised.legs))
    assert has_code(t.ajustes, "origen:revision").sum() == 16 and has_code(t.ajustes, "fila:revision").sum() == 0
    assert has_code(t.ajustes, "origen:copia").sum() == 955 and has_code(t.ajustes, "hora:llegada").sum() == 654
    # the legs whose minutes a review corrected before the rules, and no other, differ from the survey's
    minutes = revised.legs.traslado_min.groupby(level=[0, 1, 2]).sum()
    differs = minutes != shipped.legs.traslado_min.groupby(level=[0, 1, 2]).sum().reindex(minutes.index)
    assert set(differs[differs].index) == set(t.index[has_code(t.ajustes, "minutos:revision").to_numpy()]) and differs.sum() == 371
    assert non_trips(t).sum() == 554 + 75
    # what is left: no breaking chain, the tolerated codes left alone
    assert {c: int(has_code(t.problemas, c).sum()) for c in BREAKING_ISSUES} == dict.fromkeys(BREAKING_ISSUES, 0)
    assert len(review.pending_persons(review.chain_rows(revised, shipped), ["breaking"])) == 0
    assert int(has_code(t.problemas, "hora_nocturna").sum()) == 216 and int(has_code(t.problemas, "hora_2301").sum()) == 92
    assert int(has_code(t.problemas, "hora_madrugada").sum()) == 60
    assert int(has_code(t.problemas, "hora_1h01").sum()) == 718 and int(has_code(t.problemas, "hora_5h01").sum()) == 613
    # the zone follows the AGEB on every revised trip end
    ends = pd.concat([pd.DataFrame({"ageb": t[p].astype(str), "zone": t[z].astype(str)})
                      for p, z in (("origen", "zona_origen"), ("destino", "zona_destino"))])
    assert (ends.groupby("ageb").zone.nunique() == 1).all()
    # the day-start answer is one per person, and every main mode is one of its trip's legs, the survey's legs
    assert t.groupby(level=PERSON).tipo_lugar_origen.nunique().max() == 1
    modes = revised.legs.traslado_medio.astype(str).groupby(level=[0, 1, 2]).agg(set).reindex(t.index)
    assert all(m in s for m, s in zip(t.modo_principal.astype(str), modes))
    assert revised.legs.traslado_medio.equals(shipped.legs.traslado_medio.reindex(revised.legs.index))


def test_leg_minutes_show_on_the_sheet_and_take_no_edit(stages, tmp_path):
    # the rules read the travel minutes, so a review corrects them before the rules (eodgdl/revisions/leg_minutes.csv.gz)
    # and the sheet only shows them, shipped → corrected
    shipped, rules, revised = stages
    table = leg_minutes()
    assert len(table) == 371 and set(table.source) == {"chain_review_9"} and table.note.str.startswith("Novena").all()
    keys = pd.MultiIndex.from_frame(table[review.KEYS])
    minutes = rules.legs.traslado_min.groupby(level=[0, 1, 2]).sum()
    assert (minutes.reindex(keys).to_numpy() == table.after.to_numpy()).all()
    reported = shipped.legs.traslado_min.groupby(level=[0, 1, 2]).sum()
    assert (reported.reindex(keys).to_numpy() == table.before.to_numpy()).all()
    rows = review.chain_rows(revised, shipped)
    sheet = review.chain_sheet(rows, revised.hab, pd.MultiIndex.from_tuples([keys[0][:2]], names=PERSON))
    assert "leg min" in sheet.columns and review.new_column("leg min") not in sheet.columns
    shown = sheet.set_index(review.KEYS).loc[keys[0], "leg min"]
    assert shown == f"{table.before.iloc[0]} → {table.after.iloc[0]}"
    with pytest.raises(ValueError, match="not a column the sheet reads back"):
        review.sheet_edits(sheet.assign(**{"new leg min": ""}))
    # a decision on the minutes is one no table can take
    edits = pd.DataFrame([(*keys[0], "leg min", str(table.after.iloc[0]), "10", "")], columns=EDIT_COLUMNS)
    with pytest.raises(ValueError, match="not a column that takes a new value"):
        review.apply_edits(revised, edits, shipped, rules)
    # written again, the table is the same bytes
    path = review.write_leg_minutes(table, tmp_path / "leg_minutes.csv.gz")
    assert path.read_bytes() == review.leg_minutes_path().read_bytes()


def test_a_stale_decision_is_set_aside_and_decided_again_in_a_round():
    shipped, rules = _survey(_fixture())
    one = pd.DataFrame([
        (1, 4, 2, "start", "07:00", "12:00", "a 12-hour entry"),
        (1, 1, 2, "status", "", "dropped", "not a trip"),
    ], columns=EDIT_COLUMNS)
    decisions = _merge(review.decision_frame([]), one, rules, shipped, rules, "round 1")
    # a change to the rules moves the value the first decision was made against
    moved = rules.trips.copy()
    moved.loc[(1, 4, 2), "hora_inicio_m"] = 30
    changed = EODTables(rules.viv, rules.hab, moved, rules.legs)
    stale = review.stale_decisions(changed, shipped, decisions)
    assert stale[["trip", "field"]].values.tolist() == [[2, "start"]] and "07:30" in stale.problem.iloc[0]
    with pytest.raises(ValueError, match="review export --stale"):
        review.apply_revisions(changed, shipped, decisions)
    revised = review.apply_revisions(changed, shipped, decisions, skip_stale=True)
    assert (1, 1, 2) not in revised.trips.index and revised.trips.at[(1, 4, 2), "hora_inicio_m"] == 30
    stages = EODStages(shipped, changed, revised)
    rows = review.chain_rows(revised, shipped)
    sheet = review.prefill_stale(review.chain_sheet(rows, revised.hab, pd.MultiIndex.from_tuples([(1, 4)])), stale)
    row = sheet.set_index(review.KEYS).loc[(1, 4, 2)]
    assert row["new start"] == "12:00" and row["note"].startswith("Decisión vencida (round 1)")
    # kept, the decision is made again against today's rules
    edits = review.sheet_edits(sheet)
    done = review.freeze_round(edits, stages, "round 2", decisions=decisions, shown=review.sheet_keys(sheet))
    assert done.decisions[["person", "trip", "field", "before", "after", "source"]].values.tolist() == [
        [1, 2, "status", "", "dropped", "round 1"], [4, 2, "start", "07:30", "12:00", "round 2"]]
    assert len(done.stale) == 1 and _same(review.apply_revisions(changed, shipped, done.decisions),
                                          review.apply_edits(revised, done.frozen, shipped, changed))
    # cleared, the rules' value stands and the decision goes
    cleared = review.sheet_edits(sheet.assign(**{"new start": ""}))
    gone = review.freeze_round(cleared, stages, "round 2", decisions=decisions, shown=review.sheet_keys(sheet))
    assert gone.decisions.field.tolist() == ["status"] and gone.frozen.empty
    # a stale decision the sheet does not show is refused
    with pytest.raises(ValueError, match="not on this sheet"):
        review.freeze_round(cleared, stages, "round 2", decisions=decisions, shown=set())


def test_a_round_removes_the_decisions_the_rules_now_give():
    shipped, rules = _survey(_fixture())
    one = pd.DataFrame([(1, 4, 2, "start", "07:00", "12:00", "a 12-hour entry")], columns=EDIT_COLUMNS)
    decisions = _merge(review.decision_frame([]), one, rules, shipped, rules, "round 1")
    moved = rules.trips.copy()
    moved.loc[(1, 4, 2), "hora_inicio_h"] = 12          # a rule now reads the hour the decision set
    changed = EODTables(rules.viv, rules.hab, moved, rules.legs)
    assert review.decision_outcomes(changed, shipped, decisions).outcome.tolist() == ["holds"]
    revised = review.apply_revisions(changed, shipped, decisions)
    two = pd.DataFrame([(1, 5, 1, "status", "", "dropped", "not a trip")], columns=EDIT_COLUMNS)
    done = review.freeze_round(two, EODStages(shipped, changed, revised), "round 2", decisions=decisions)
    assert done.decisions[["trip", "field"]].values.tolist() == [[1, "status"]] and len(done.dead) == 1
    assert done.changes == {"added": 1, "changed": 0, "removed": 1, "notes": 0} and done.verified.summary["rows_dropped"] == 1
    with pytest.raises(ValueError, match="nothing to freeze"):
        review.freeze_round(two.iloc[:0], EODStages(shipped, rules, rules), "round 3", decisions=review.decision_frame([]))


def test_screens_measure_the_stays_and_the_day():
    trips = _trips([
        (1, 8, 0, "Trabajar", None, None, 30), (1, 8, 30, "Regresar a Casa", None, None, 30),      # leaves as it arrives
        (2, 6, 0, "Trabajar", None, None, 30), (2, 21, 30, "Regresar a Casa", None, None, 30),     # 15 hours at work
        (3, 4, 0, "Compras (comida)", None, None, 30), (3, 23, 45, "Regresar a Casa", None, None, 30),  # over 20 hours
    ])
    hab = pd.DataFrame(index=trips.index.droplevel("folio_viaje").unique())
    s = review.screens(EODTables(VIV, hab, trips, None))
    assert s.zero_stays.tolist() == [1, 0, 0] and s.work_minutes.tolist() == [0, 900, 0]
    assert s.zero_work_stays.tolist() == [1, 0, 0]                        # the stay of no minutes is at work
    assert list(review.SCREENS) == ["zero_stay", "zero_work", "short_work", "long_workday", "long_day", "early_start",
                                    "long_errand"]
    assert s[list(review.SCREENS)].values.tolist() == [
        [True, True, True, False, False, False, False],       # no time at work: a short stay too
        [False, False, False, True, False, False, False],
        [False, False, False, False, True, True, True]]       # shopping at 04:00 for 19 hours: early, long and all day
    assert s.problems.tolist() == [1, 0, 3]                   # what a review round reads a day by
    assert list(review.screened_persons(EODTables(VIV, hab, trips, None), ["long_day"])) == [(1, 3)]
    with pytest.raises(ValueError, match="not a screen"):
        review.screened_persons(EODTables(VIV, hab, trips, None), ["short_day"])


def test_a_sheet_read_back_drops_blank_rows_and_a_write_drops_the_cached_decisions(tmp_path):
    shipped, cleaned = _survey(_fixture())
    sheet = review.chain_sheet(review.chain_rows(cleaned, shipped), cleaned.hab)
    path = review.write_sheet(sheet, tmp_path / "s.csv")
    with open(path, "a", encoding="utf-8") as f:
        f.write("," * (len(sheet.columns) - 1) + "\n")                   # the empty row a spreadsheet leaves
    assert len(review.read_sheet(path)) == len(sheet)
    chain_decisions()
    review.write_decisions(review.decision_frame([]), tmp_path / "chains.csv.gz")
    assert chain_decisions.cache_info().currsize == 0


def test_a_note_alone_gives_the_trip_s_decisions_a_new_reason():
    shipped, rules = _survey(_fixture())
    one = pd.DataFrame([
        (1, 4, 2, "start", "07:00", "12:00", "a 12-hour entry"),
        (1, 4, 1, "orig. type", "Su casa", "Otra vivienda", "slept at a relative's"),
    ], columns=EDIT_COLUMNS)
    decisions = _merge(review.decision_frame([]), one, rules, shipped, rules, "round 1")
    revised = review.apply_revisions(rules, shipped, decisions)
    sheet = review.chain_sheet(review.chain_rows(revised, shipped), revised.hab, pd.MultiIndex.from_tuples([(1, 4)]))
    at = {k: i for i, k in enumerate(zip(sheet.household, sheet.person, sheet.trip))}
    sheet.loc[at[(1, 4, 2)], "note"] = "read on a 12-hour clock: noon"
    sheet.loc[at[(1, 4, 1)], "note"] = "the day began at a relative's"      # the row that shows the answer
    sheet.loc[at[(1, 4, 3)], "note"] = "no decision here: nothing to say"
    edits = review.sheet_edits(sheet)
    done = review.freeze_round(edits, EODStages(shipped, rules, revised), "round 2", decisions=decisions,
                               shown=review.sheet_keys(sheet))
    assert done.changes == {"added": 0, "changed": 0, "removed": 0, "notes": 2} and done.frozen.empty
    got = done.decisions.set_index(["trip", "field"])
    assert got.loc[(2, "start"), ["after", "note", "source"]].tolist() == ["12:00", "read on a 12-hour clock: noon", "round 1"]
    assert got.loc[(review.PERSON_TRIP, "orig. type"), "note"] == "the day began at a relative's"
    assert _same(review.apply_revisions(rules, shipped, done.decisions), revised)


def test_a_decision_no_table_can_take_is_not_called_stale():
    shipped, rules = _survey(_fixture())
    good = _merge(review.decision_frame([]), pd.DataFrame([(1, 4, 2, "start", "07:00", "12:00", "noon")],
                                                          columns=EDIT_COLUMNS), rules, shipped, rules, "round 1")
    bad = good.assign(after="noon")
    assert review.decision_outcomes(rules, shipped, bad).outcome.tolist() == ["problem"]
    assert review.stale_decisions(rules, shipped, bad).empty
    with pytest.raises(ValueError, match="outside `eodgdl review freeze`"):
        review.apply_revisions(rules, shipped, bad)
    with pytest.raises(ValueError, match="review export --stale"):
        review.apply_revisions(rules, shipped, good.assign(before="08:00"))
    with pytest.raises(ValueError, match="no table can take"):
        review.freeze_round(pd.DataFrame(columns=EDIT_COLUMNS), EODStages(shipped, rules, rules), "r", decisions=bad)


def test_a_snapshot_finds_the_persons_a_change_moved(tmp_path):
    shipped, rules = _survey(_fixture())
    before = review.read_snapshot(review.write_snapshot(review.snapshot(rules, shipped), tmp_path / "s.csv.gz"))
    assert before.loc[(1, 1, 2), ["start", "status"]].tolist() == ["17:30", ""] and before.loc[(1, 2, 3), "status"] == "dropped"
    assert review.changed_persons(before, review.snapshot(rules, shipped)).empty
    # a rule change that moves a value no decision holds is seen only here
    moved = rules.trips.copy()
    moved.loc[(1, 3, 2), "hora_inicio_m"] = 5
    after = review.snapshot(EODTables(rules.viv, rules.hab, moved, rules.legs), shipped)
    assert list(review.changed_persons(before, after)) == [(1, 3)]
