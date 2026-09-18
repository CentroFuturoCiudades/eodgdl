"""eodgdl.review: the review sheet round trip — export, edit by hand, recover the edits, apply, verify."""
from pathlib import Path

import pandas as pd
import pytest

from eodgdl import EODTables, clean_trip_chains, load_eod, review
from eodgdl.eod import PERSON, mark_issues, non_trips
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


@pytest.mark.skipif(not HAS_DATA, reason="in-repo data/ not present")
def test_the_pending_sheet_round_trips_on_the_survey(tmp_path):
    shipped, cleaned = load_eod(DATA_DIR, clean_chains=False), load_eod(DATA_DIR)
    assert mark_issues(cleaned.trips, cleaned.viv, cleaned.legs).equals(cleaned.trips.problemas)
    rows = review.chain_rows(cleaned, shipped)
    assert len(rows) == 154_662 and int(rows.dropped.sum()) == 38
    pending = review.pending_persons(rows)
    sheet = review.chain_sheet(rows, cleaned.hab, pending)
    assert (len(pending), len(sheet)) == (4_435, 14_557)
    assert sheet.status.value_counts().to_dict() == {"": 14_081, "changed": 474, "dropped": 2}
    back = review.read_sheet(review.write_sheet(sheet, tmp_path / "chain_review.csv"))
    assert set(back.destination.str.len()) <= {4, 9, 13}                # zone ids as strings, home as home
    sel = rows[rows.index.droplevel("folio_viaje").isin(pending)]
    changed = int(((sel.start_shipped != sel.start_clean) & ~sel.dropped).sum())
    assert (back.motive == "Guardería").any() and back.start.str.contains("→").sum() == changed   # accents and arrows intact
    assert review.sheet_edits(back).empty
    same = review.apply_edits(cleaned, review.sheet_edits(back), shipped)
    assert same.trips.equals(cleaned.trips) and same.legs.equals(cleaned.legs) and same.hab.equals(cleaned.hab)
