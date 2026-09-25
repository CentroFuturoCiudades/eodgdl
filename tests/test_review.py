"""eodgdl.review: the review sheet round trip — export, edit by hand, recover the edits, apply, verify."""
from pathlib import Path

import pandas as pd
import pytest

from eodgdl import EODTables, clean_trip_chains, load_eod, review
from eodgdl._resources import chain_revisions
from eodgdl.chains import BREAKING_ISSUES, ISSUE_CODES, PERSON, TOLERATED_ISSUES, has_code, mark_issues, non_trips
from test_eod import AGEB, CAR, ELSEWHERE, SHOP, VIV, _trips

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HAS_DATA = (DATA_DIR / "IMEPLAN_Base_Viajes_Master.csv").exists()
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
    assert one.loc[(1, 2, 2), ["start", "motive", "dest. type", "orig. type", "origin", "destination"]].tolist() == [
        "— → 18:00", "— → Regresar a Casa", "— → Su casa", "Su casa", ELSEWHERE, "home"]
    assert one.loc[(1, 2, 2), ["mode", "leg min", "ajustes"]].tolist() == [CAR, 30, "hora:duplicado;motivo:duplicado"]
    # a dropped row keeps its shipped values, with no codes
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


def test_apply_edits_writes_the_fields_with_a_code_and_recomputes_problemas():
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
    revised = review.apply_edits(cleaned, edits, shipped)
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
        (1, 3, 2, "status", "", "maybe", ""),
        (9, 9, 9, "status", "", "restored", ""),
    ], columns=EDIT_COLUMNS)
    with pytest.raises(ValueError) as err:
        review.apply_edits(cleaned, bad, shipped)
    msg = str(err.value)
    assert msg.count("\n") == 6
    for text in ("not a time", "not a column", "not a zone", "no such trip in", "status must be", "no such trip to restore"):
        assert text in msg
    same = review.apply_edits(cleaned, bad.iloc[:0], shipped)
    assert same.trips.equals(cleaned.trips) and same.hab.equals(cleaned.hab)


def test_a_row_a_pass_dropped_comes_back_as_the_rules_left_it():
    shipped, rules = _survey(_fixture())
    drop = pd.DataFrame([(1, 1, 2, "status", "", "dropped", ""), (1, 3, 2, "status", "", "dropped", "")], columns=EDIT_COLUMNS)
    first = review.apply_edits(rules, drop, shipped, rules)
    assert (1, 1, 2) not in first.trips.index and first.hab.viajes_contados.tolist() == [1, 2, 1, 4, 1]
    back = pd.DataFrame([
        (1, 1, 2, "status", "dropped", "restored", ""),
        (1, 3, 2, "status", "dropped", "restored", ""),
        (1, 3, 2, "start", "09:00", "09:30", "restored and edited in one pass"),
    ], columns=EDIT_COLUMNS)
    second = review.apply_edits(first, back, shipped, rules)
    t = second.trips
    # the rules' version: the repaired 17:30, its code kept, the restore added
    assert (t.loc[(1, 1, 2), "hora_inicio_h"], t.loc[(1, 1, 2), "ajustes"]) == (17, "hora:+12h;fila:revision")
    assert (t.loc[(1, 3, 2), "hora_inicio_m"], t.loc[(1, 3, 2), "ajustes"]) == (30, "fila:revision;hora:revision")
    assert second.hab.viajes_contados.equals(rules.hab.viajes_contados)
    assert t.problemas.equals(mark_issues(t, VIV))
    # without the rules' tables the row comes back as shipped
    assert review.apply_edits(first, back.iloc[:1], shipped).trips.loc[(1, 1, 2), "hora_inicio_h"] == 5
    # the passes in a row, as load_eod applies them: the rules' tables are where a restore comes from
    both = review.apply_revisions(rules, shipped, [("a", drop), ("b", back)])
    assert both.trips.equals(t)

def test_verify_edits_says_what_the_edits_cleared_left_and_made():
    shipped, cleaned = _survey(_fixture())
    edits = pd.DataFrame([
        (1, 4, 2, "start", "07:00", "12:00", "a 12-hour entry"),
        (1, 5, 1, "motive", "Trabajar", "Regresar a Casa", ""),
        (1, 5, 1, "destination", ELSEWHERE, "home", ""),
    ], columns=EDIT_COLUMNS)
    v = review.verify_edits(cleaned, shipped, edits)
    assert v.persons.values.tolist() == [
        [1, 4, 1, 0, 0, 1, "hora_invertida", "", True],
        [1, 5, 2, 0, 0, 0, "fin_fuera_de_casa", "regreso_en_casa", False],
    ]
    assert v.summary == {"persons": 2, "cells": 3, "rows_dropped": 0, "rows_restored": 0, "defects_before": 2,
                         "defects_after": 1, "persons_cleared": 1, "persons_with_new_defects": 1}
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
        (1, 4, 2, "start", "07:00", "12:00:00", ""),
        (1, 5, 1, "motive", "Trabajar", "regresar a casa", ""),
    ], columns=EDIT_COLUMNS)
    once = review.apply_edits(cleaned, edits, shipped)
    t = once.trips
    assert (t.loc[(1, 4, 2), "hora_inicio_h"], t.loc[(1, 4, 2), "hora_inicio_m"]) == (12, 0)
    assert t.loc[(1, 5, 1), "motivo_viaje"] == "Regresar a Casa"                # the level, not the text typed
    with pytest.raises(ValueError, match="not a time"):
        review.apply_edits(cleaned, edits.assign(after=["12:00:30", "Trabajar"]), shipped)
    # a second pass over a field leaves its code once
    twice = review.apply_edits(once, edits.assign(after=["12:30", "Trabajar"]), shipped)
    assert twice.trips.loc[(1, 4, 2), "ajustes"] == "hora:revision" and twice.trips.loc[(1, 4, 2), "hora_inicio_m"] == 30
    sheet = review.chain_sheet(review.chain_rows(cleaned, shipped), cleaned.hab).rename(columns={"note": "comments"})
    sheet.loc[0, "comments"] = "seen"
    back = review.read_sheet(review.write_sheet(sheet, tmp_path / "s.csv"))
    assert "note" in back and review.sheet_edits(back).note.tolist() == ["seen"]
    # the issue groups split the codes, and select persons by name
    assert set(BREAKING_ISSUES) | set(TOLERATED_ISSUES) == set(ISSUE_CODES) and not set(BREAKING_ISSUES) & set(TOLERATED_ISSUES)
    rows = review.chain_rows(cleaned, shipped)
    assert list(review.pending_persons(rows, ["breaking"])) == [(1, 4)]          # hora_invertida
    assert list(review.pending_persons(rows, ["tolerated"])) == [(1, 5)]         # fin_fuera_de_casa


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_the_pending_sheet_round_trips_on_the_survey(tmp_path):
    shipped, cleaned = load_eod(DATA_DIR, clean_chains=False), load_eod(DATA_DIR, revise_chains=False)   # what pass 1 was exported from
    assert mark_issues(cleaned.trips, cleaned.viv, cleaned.legs).equals(cleaned.trips.problemas)
    rows = review.chain_rows(cleaned, shipped)
    assert len(rows) == 154_662 and int(rows.dropped.sum()) == 38
    pending = review.pending_persons(rows)
    sheet = review.chain_sheet(rows, cleaned.hab, pending)
    assert (len(pending), len(sheet)) == (4_435, 14_557)
    assert sheet.status.value_counts().to_dict() == {"": 14_080, "changed": 475, "dropped": 2}
    back = review.read_sheet(review.write_sheet(sheet, tmp_path / "chain_review.csv"))
    assert set(back.destination.str.len()) <= {4, 9, 13}                # zone ids as strings, home as home
    sel = rows[rows.index.droplevel("folio_viaje").isin(pending)]
    changed = int(((sel.start_shipped != sel.start_clean) & ~sel.dropped).sum())
    assert (back.motive == "Guardería").any() and back.start.str.contains("→").sum() == changed   # accents and arrows intact
    assert review.sheet_edits(back).empty
    same = review.apply_edits(cleaned, review.sheet_edits(back), shipped)
    assert same.trips.equals(cleaned.trips) and same.legs.equals(cleaned.legs) and same.hab.equals(cleaned.hab)


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_an_edited_trip_end_takes_the_zone_coded_for_its_ageb():
    shipped, cleaned = load_eod(DATA_DIR, clean_chains=False), load_eod(DATA_DIR)
    t, viv = cleaned.trips, cleaned.viv
    home = viv.ageb.astype(str).reindex(t.index.get_level_values("folio_vivienda")).to_numpy()
    away = t[(t.origen.astype(str).to_numpy() != home) & (t.destino.astype(str).to_numpy() != home)].head(50)
    other = t.destino.astype(str).loc[lambda s: s != away.destino.astype(str).iloc[0]].iloc[0]
    edits = pd.DataFrame(
        [(*k, "origin", "", "home", "") for k in away.index]
        + [(*away.index[0], "destination", "", other, "")],
        columns=EDIT_COLUMNS,
    )
    revised = review.apply_edits(cleaned, edits, shipped).trips.loc[away.index]
    assert (revised.zona_origen.astype(str).to_numpy()
            == viv.centralidad.astype(str).reindex(away.index.get_level_values("folio_vivienda")).to_numpy()).all()
    coded = t.zona_destino.astype(str)[t.destino.astype(str) == other].iloc[0]
    assert revised.zona_destino.astype(str).iloc[0] == coded
    ends = pd.concat([pd.DataFrame({"ageb": revised[p].astype(str), "zone": revised[z].astype(str)})
                      for p, z in (("origen", "zona_origen"), ("destino", "zona_destino"))])
    assert (ends.groupby("ageb").zone.nunique() == 1).all()



@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_load_eod_applies_the_frozen_hand_passes():
    shipped = load_eod(DATA_DIR, clean_chains=False)
    rules, revised = load_eod(DATA_DIR, revise_chains=False), load_eod(DATA_DIR)
    t = revised.trips
    (name, edits), = chain_revisions()
    assert name == "chains_1.csv.gz" and len(edits) == 11_669
    # every frozen value holds, every row dropped by hand is gone with its legs, the counts follow
    dropped = edits[edits.field == "status"]
    assert (dropped.after == "dropped").all() and len(dropped) == 512
    assert not t.index.isin(pd.MultiIndex.from_frame(dropped[review.KEYS])).any()
    assert len(t) == len(rules.trips) - 512 == 154_112 and revised.legs.index.droplevel("folio_traslado").isin(t.index).all()
    assert (revised.hab.viajes_contados == t.groupby(level=PERSON).size().reindex(revised.hab.index).fillna(0)).all()
    assert review.sheet_edits(review.chain_sheet(review.chain_rows(revised, shipped))).empty
    again = review.apply_revisions(revised, shipped)
    assert again.trips.equals(t)                                            # the values already hold: a pass is idempotent
    assert t.problemas.equals(mark_issues(t, revised.viv, revised.legs))
    assert has_code(t.ajustes, "origen:revision").sum() == 1_361 and non_trips(t).sum() == 51
    # what is left: few breaking chains, the tolerated codes left alone
    breaking = {c: int(has_code(t.problemas, c).sum()) for c in BREAKING_ISSUES}
    assert breaking == {"hora_invertida": 68, "hora_anterior": 1, "origen_discontinuo": 1,
                        "regreso_sin_llegar": 2, "inicio_zona_ajena": 0}
    assert len(review.pending_persons(review.chain_rows(revised, shipped), ["breaking"])) == 71
    # the zone follows the AGEB on every revised trip end
    ends = pd.concat([pd.DataFrame({"ageb": t[p].astype(str), "zone": t[z].astype(str)})
                      for p, z in (("origen", "zona_origen"), ("destino", "zona_destino"))])
    assert (ends.groupby("ageb").zone.nunique() == 1).all()
