#!/usr/bin/env python
"""Export the cleaned EOD 2023 tables as a standalone, code-free dataset, in Spanish.

Writes the four tables `load_eod()` returns as UTF-8 CSVs, one YAML data dictionary
beside each, and a README, so the directory can be handed to someone who will never run
this package. Everything the dataset says — dictionary keys included — is in Spanish,
the language of the survey and of its columns; only this script is in English.

    uv run python scripts/export_dataset.py                       # -> output/eod2023_clean/
    uv run python scripts/export_dataset.py --data data --out /tmp/eod

Deliberately not part of the package: it is a one-shot packaging step, not part of the
survey → model pipeline, and nothing in `eodgdl` imports it.

Where each column's description comes from: the survey's own glossaries (the three
`IMEPLAN_Base_*_Glosario.csv` files), keyed by the raw Spanish header the rename map
maps to the snake_case name; the columns the loader adds, and the few whose header alone
says too little, are written out in DESCRIPTIONS below, and NOTES carries what a reader
has to know about a column the cleaning changed. Every count quoted in the prose is
computed from the exported tables (COUNTS), so none of it can go stale.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

# --- the tables -----------------------------------------------------------------------

# name on disk → the EODTables field, the survey level its columns were renamed from,
# the source file, and the prose that heads its dictionary.
TABLES = {
    "viviendas": dict(
        attr="viv",
        level="viviendas",
        source="IMEPLAN_Base_Viviendas_Master.csv",
        title="Viviendas (hogares)",
        description=(
            "Una fila por vivienda encuestada: dónde está, qué vehículos tiene y cómo "
            "califican sus ocupantes la movilidad de su zona."
        ),
    ),
    "habitantes": dict(
        attr="hab",
        level="habitantes",
        source="IMEPLAN_Base_Habitantes_Master.csv",
        title="Personas",
        description=(
            "Una fila por persona de una vivienda encuestada: datos demográficos, "
            "ocupación, discapacidad, hábitos de viaje en fin de semana y cuántos viajes "
            "reportó para el día de referencia."
        ),
    ),
    "viajes": dict(
        attr="trips",
        level="viajes",
        source="IMEPLAN_Base_Viajes_Master.csv",
        title="Viajes",
        description=(
            "Una fila por viaje que una persona reportó para el día de referencia, en el "
            "orden de la cadena: zona de origen y de destino, hora de inicio, motivo, medio "
            "principal, acompañamiento y estacionamiento. Las cadenas de viaje están "
            "depuradas; el README describe qué se corrigió y qué quedó sin resolver."
        ),
    ),
    "traslados": dict(
        attr="legs",
        level="viajes",
        source="IMEPLAN_Base_Viajes_Master.csv",
        title="Traslados (etapas de cada viaje)",
        description=(
            "Una fila por traslado de un viaje, desdoblada de las columnas "
            "`traslado1..5_*` de la tabla de viajes: el medio utilizado, su duración y su "
            "pago. Un viaje se compone de uno a cinco traslados; `n_traslados` los cuenta "
            "en la tabla de viajes."
        ),
    ),
}

# --- the words -------------------------------------------------------------------------

# The YAML the dataset carries is Spanish down to its keys.
K = {
    "table": "tabla", "title": "título", "file": "archivo", "description": "descripción",
    "key": "clave_primaria", "rows": "n_filas", "cols": "n_columnas",
    "source_file": "archivo_origen", "provenance": "procedencia", "columns": "columnas",
    "name": "nombre", "type": "tipo", "source_column": "columna_original", "unit": "unidad",
    "nullable": "admite_faltantes", "n_missing": "n_faltantes", "ordered": "ordenada",
    "categories": "categorías", "min": "mínimo", "max": "máximo",
    "n_unique": "n_valores_distintos", "note": "nota",
}
# Columns `load_eod` adds that this dataset does not carry: the per-row audit marks. What
# they recorded is reported in the README as counts instead.
DROP = {"habitantes": ("diario_repetido",), "viajes": ("ajustes", "problemas")}

TYPES = {
    "categorical": "categoría", "date": "fecha", "boolean": "booleano",
    "integer": "entero", "number": "número", "string": "texto",
}
UNITS = {
    "edad": "años",
    "hora_inicio_h": "hora del día, 0–23",
    "hora_inicio_m": "minutos después de la hora, 0–59",
    "traslado_min": "minutos",
    "traslado_pago": "pesos",
    "estacionamiento_pago_total": "pesos",
    "ponderador": "factor de expansión (unidades que representa la fila)",
}

# The residual defects the cleaning could not resolve, one short label each, ordered by
# how many rows carry them in the README. The per-row marks are not part of this dataset
# (`problemas` is not exported), so the README reports them as counts.
DEFECTOS_ES = {
    "inicio_fuera_de_casa": "el primer viaje del día no sale de 'Su casa'",
    "hora_invertida": "empieza antes de que hubiera podido llegar el viaje anterior, por más de "
                      "15 minutos y sin ser un viaje nocturno",
    "actividad_en_casa": "un motivo de actividad con tipo de lugar de destino 'Su casa': trabajo "
                         "desde casa, o un regreso a casa etiquetado al revés",
    "inicio_zona_ajena": "el primer viaje del día sale de 'Su casa' pero no de la zona de la vivienda",
    "hora_traslapada": "empieza antes de la llegada reportada del viaje anterior, por 15 minutos "
                       "o menos: redondeo, no un error de hora",
    "fin_fuera_de_casa": "el último viaje del día no es un regreso que llegue a la zona de la vivienda",
    "regreso_en_casa": "un 'Regresar a Casa' hecho estando ya en casa: no es un viaje",
    "motivo_guarderia": "motivo 'Guardería', en su mayoría personas adultas que llevan o recogen "
                        "a un menor",
    "regreso_sin_llegar": "un 'Regresar a Casa' que no llegó a la zona de la vivienda y cuyo tipo "
                          "de lugar de destino no permite recodificarlo",
    "hora_repetida": "empieza en el mismo minuto que el viaje anterior",
    "hora_nocturna": "empieza de madrugada después de un viaje de la tarde o la noche: se lee "
                     "como viaje nocturno y se deja igual",
    "origen_discontinuo": "no empieza en la zona donde terminó el viaje anterior",
    "hora_anterior": "empieza antes que el viaje anterior pero dentro de la tolerancia de 15 "
                     "minutos respecto de su llegada: ruido de minutos",
    "tipo_destino_dudoso": "un 'Regresar a Casa' que sí llegó a la zona de la vivienda pero "
                           "reporta un tipo de lugar de destino que no es una casa",
}

_FECHA = (
    "Fecha de la entrevista. El diario que recoge es el del día de referencia anterior "
    "—registrado como día de la semana en `dia_semana_viajes`, casi siempre el día hábil "
    "previo, pero no siempre—, así que el día de los viajes se lee de esa columna y no de esta."
)
_ID_AREA = (
    "con el identificador de área de la propia encuesta: una clave AGEB (CVEGEO) de 13 "
    "caracteres o una clave de localidad de 9, 154 de las cuales terminan en letra. Es texto, "
    "nunca un número."
)


def descriptions(c: dict[str, int]) -> dict[tuple[str, str], str]:
    """Hand-written definitions, keyed by (table, column).

    The columns the loader adds or reshapes, which have no glossary entry, and the few
    whose original header alone does not say what the column holds. ``c`` is COUNTS.
    """
    return {
        ("traslados", "folio_traslado"): (
            "Número de traslado dentro del viaje, de 1 a 5, en el orden en que el viaje los "
            "reportó (la N del bloque `traslado N` del archivo original)."
        ),
        ("traslados", "traslado_medio"): (
            "Medio utilizado en este traslado (encabezado original: 'Traslados que utilizó "
            "para su viaje: | Traslado N | Medio')."
        ),
        ("traslados", "traslado_min"): (
            "Minutos que duró este traslado (encabezado original: 'Traslados que utilizó para "
            "su viaje: | Traslado N | Minutos')."
        ),
        ("traslados", "traslado_pago"): (
            "Pago hecho en este traslado, en pesos (encabezado original: 'Traslados que "
            "utilizó para su viaje: | Traslado N | Pago (pesos)')."
        ),
        ("viviendas", "fecha"): _FECHA,
        ("habitantes", "fecha"): _FECHA,
        ("viviendas", "municipio"): "Municipio en el que se ubica la vivienda.",
        ("habitantes", "viajes_contados"): (
            "Número de viajes que la persona reportó para el día de referencia; 0 para quien "
            "no salió de casa."
        ),
        ("viajes", "origen"): (
            f"Dónde empezó el viaje, {_ID_AREA} `zona_origen` es ese mismo lugar expresado "
            "como una de las 71 zonas EOD."
        ),
        ("viajes", "destino"): (
            f"Dónde terminó el viaje, {_ID_AREA} `zona_destino` es ese mismo lugar expresado "
            "como una de las 71 zonas EOD."
        ),
        ("viajes", "n_traslados"): (
            "Número de traslados que componen el viaje, de 1 a 5; es igual al número de filas "
            "de este viaje en la tabla de traslados."
        ),
        ("viajes", "modo_principal"): (
            "Medio principal del viaje: el resumen que la propia encuesta hace de sus traslados."
        ),
    }


_PONDERADOR = (
    "Las tres tablas traen tres factores de expansión DISTINTOS: son tres etapas de "
    "calibración anidadas, no un mismo peso repetido. Expanda viviendas con el factor de la "
    "vivienda, personas con el de la persona y viajes con el de viaje; expandir los viajes "
    "con el factor de la persona sobrestima el viaje en autobús alrededor de 77%."
)


def notes(c: dict[str, int]) -> dict[tuple[str, str], str]:
    """What a reader has to know about a column beyond its definition. ``c`` is COUNTS."""
    return {
        ("viviendas", "ponderador"): _PONDERADOR,
        ("habitantes", "ponderador"): _PONDERADOR,
        ("viajes", "ponderador"): _PONDERADOR,
        ("viviendas", "ageb"): (
            "Texto, nunca un número: claves AGEB (CVEGEO) de 13 caracteres y claves de "
            "localidad de 9, 154 de las cuales terminan en letra. Lea la columna como texto."
        ),
        ("habitantes", "viajes_contados"): (
            "Es igual al número de filas que la persona tiene en la tabla de viajes. Se le "
            f"restó uno a las personas a las que se les eliminó un regreso duplicado "
            f"({c['eliminadas']} en total), para que siga contando sus filas."
        ),
        ("viajes", "folio_viaje"): (
            "Se conserva tal como viene en el archivo original, con un hueco donde estaba cada "
            f"uno de los {c['eliminadas']} regresos duplicados que se eliminaron, así que no "
            "siempre va de 1 a n. El orden de las filas es la cadena del día: ordene por esta "
            "columna dentro de cada persona para recuperarla."
        ),
        ("viajes", "hora_inicio_h"): (
            f"Reparada donde la hora recibida no podía ser correcta: {c['horas_reparadas']:,} "
            "horas se releyeron con el menor número de erratas que permite que cada viaje "
            f"empiece después de que llegó el anterior, y se imputó la de {c['sin_hora']} "
            f"viajes que no traían hora. Aun así quedan {c['hora_invertida']} filas que "
            "empiezan antes de que llegara el viaje anterior."
        ),
        ("viajes", "hora_inicio_m"): (
            "Se reparó e imputó junto con la hora; vea `hora_inicio_h`."
        ),
        ("viajes", "motivo_viaje"): (
            f"Recodificado en {c['recodificados']} viajes 'Regresar a Casa' que no llegaron a "
            "la zona de la vivienda (a partir de su tipo de lugar de destino) e imputado en "
            f"los {c['sin_motivo']} que no traían ninguno: {c['motivo_vec']} por votación de "
            f"sus 30 viajes con hora más parecidos y {c['motivo_dup']} a partir del regreso "
            "casa–casa que los duplicaba y que después se eliminó. Ojo: los "
            f"{c['no_viajes']} regresos a casa hechos estando ya en casa se conservan en la "
            "tabla y no son viajes; el README dice cómo identificarlos."
        ),
        ("viajes", "tipo_lugar_destino"): (
            "Se recodificó e imputó junto con el motivo; vea `motivo_viaje`."
        ),
        ("viajes", "origen"): (
            f"Fijado en la zona de la vivienda en {c['origen_casa']} viajes que siguen a un "
            "regreso a casa que sí llegó. Quedan "
            f"{c['origen_discontinuo']} viajes que aun así no empiezan donde terminó el anterior."
        ),
    }


# --- helpers ---------------------------------------------------------------------------


def read_glossaries(data_dir: Path) -> dict[str, dict[str, str]]:
    """Raw Spanish header → its glossary description, per survey level.

    The glossaries are ISO-8859-1, headerless, and their first column matches the rename
    map's keys exactly once stripped; most descriptions are empty, in which case the
    header itself (the question as asked) is the definition.
    """
    files = {
        "viviendas": "IMEPLAN_Base_Viviendas_Glosario.csv",
        "habitantes": "IMEPLAN_Base_Habitantes_Glosario.csv",
        "viajes": "IMEPLAN_Base_Viajes_Glosario.csv",
    }
    out = {}
    for level, name in files.items():
        g = pd.read_csv(
            data_dir / name, encoding="ISO-8859-1", header=None, names=["col", "desc"]
        )
        out[level] = {
            str(c).strip(): ("" if pd.isna(d) else str(d).strip())
            for c, d in zip(g["col"], g["desc"])
        }
    return out


def logical_type(s: pd.Series) -> str:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "number"
    return "string"


def column_entry(s: pd.Series, name: str, source_header: str, gloss: str,
                 description: str, note: str) -> dict:
    """One column's dictionary entry, keyed and typed in Spanish."""
    kind = logical_type(s)
    entry = {K["name"]: name, K["type"]: TYPES[kind]}
    entry[K["description"]] = description or gloss or source_header
    if source_header:
        entry[K["source_column"]] = source_header
    if name in UNITS:
        entry[K["unit"]] = UNITS[name]

    # an empty string is a blank cell in the CSV like any other missing value, so the
    # count a reader can reproduce counts it: `ajustes` and `problemas` are empty on
    # every row the cleaning left alone.
    blank = s.isna() | (s.astype("string").str.len().eq(0) if kind == "string" else False)
    n_missing = int(blank.sum())
    entry[K["nullable"]] = bool(n_missing)
    entry[K["n_missing"]] = n_missing

    if kind == "categorical":
        entry[K["ordered"]] = bool(s.cat.ordered)
        entry[K["categories"]] = [str(c) for c in s.cat.categories]
    elif kind in ("integer", "number"):
        valid = s.dropna()
        if len(valid):
            lo, hi = valid.min(), valid.max()
            entry[K["min"]] = int(lo) if kind == "integer" else float(lo)
            entry[K["max"]] = int(hi) if kind == "integer" else float(hi)
    elif kind == "date":
        valid = s.dropna()
        if len(valid):
            entry[K["min"]] = str(valid.min().date())
            entry[K["max"]] = str(valid.max().date())
    elif kind == "string":
        entry[K["n_unique"]] = int(s[~blank].nunique())

    if note:
        entry[K["note"]] = note
    return entry


def code_counts(s: pd.Series) -> dict[str, int]:
    """How often each `;`-joined code appears in ``s``."""
    codes = s[s.astype(str) != ""].astype(str).str.split(";").explode()
    return codes.value_counts().to_dict()


def counts(tables, shipped_trips: int) -> dict[str, int]:
    """Every figure the prose quotes, read back off the tables themselves."""
    fix = code_counts(tables.trips["ajustes"])
    issue = code_counts(tables.trips["problemas"])
    hours = [c for c in fix if c.startswith("hora:") and c[5] in "+-"]
    edited = tables.trips.index[
        tables.trips["ajustes"].str.contains("hora:[+-]", regex=True)
    ].droplevel("folio_viaje")
    return {
        "eliminadas": shipped_trips - len(tables.trips),
        "hora_dup": fix.get("hora:duplicado", 0),
        "hora_vec": fix.get("hora:vecinos", 0),
        "sin_hora": fix.get("hora:duplicado", 0) + fix.get("hora:vecinos", 0),
        "motivo_dup": fix.get("motivo:duplicado", 0),
        "motivo_vec": fix.get("motivo:vecinos", 0),
        "sin_motivo": fix.get("motivo:duplicado", 0) + fix.get("motivo:vecinos", 0),
        "recodificados": fix.get("motivo:tipo_destino", 0),
        "origen_casa": fix.get("origen:casa", 0),
        "horas_reparadas": sum(fix[c] for c in hours),
        "cadenas_reparadas": int(edited.nunique()),
        "no_viajes": issue.get("regreso_en_casa", 0),
        "hora_invertida": issue.get("hora_invertida", 0),
        "origen_discontinuo": issue.get("origen_discontinuo", 0),
        "filas_con_problema": int((tables.trips["problemas"] != "").sum()),
        "diarios_repetidos": int(tables.hab["diario_repetido"].sum()),
        "viviendas_varias_fechas": int(
            (tables.hab.reset_index().groupby("folio_vivienda")["fecha"].nunique() > 1).sum()
        ),
        "viviendas": len(tables.viv),
        "personas": len(tables.hab),
        "viajes": len(tables.trips),
        "traslados": len(tables.legs),
    }


def to_csv_frame(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """The table as it is written: keys as ordinary columns, dates as plain dates,
    without the audit columns of DROP."""
    out = df.reset_index().drop(columns=list(DROP.get(table, ())))
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out


def provenance() -> dict:
    from eodgdl import __version__ as version  # noqa: PLC0415  (kept out of import time)

    rev = "desconocido"
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"version_eodgdl": version, "commit_eodgdl": rev, "generado": str(date.today())}


# --- the export --------------------------------------------------------------------------


def export(data_dir: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    os.environ["EODGDL_DATA_DIR"] = str(data_dir)
    from eodgdl._resources import imeplan_rename_map  # noqa: PLC0415  (after the env var)
    from eodgdl.data._catalog import VIAJES_CSV  # noqa: PLC0415
    from eodgdl.eod import load_eod  # noqa: PLC0415

    tables = load_eod(data_dir)   # `counts` reads the audit columns before DROP removes them
    # the trip table as shipped, to say how many rows the cleaning removed
    shipped = len(pd.read_csv(data_dir / VIAJES_CSV, encoding="ISO-8859-1", usecols=[0]))
    c = counts(tables, shipped)
    issues = code_counts(tables.trips["problemas"])
    desc, note = descriptions(c), notes(c)

    glossaries = read_glossaries(data_dir)
    rename = imeplan_rename_map()
    # snake_case name → the raw Spanish header it was renamed from, per level.
    origin = {lvl: {v: k for k, v in rename[lvl].items()} for lvl in rename}
    prov = provenance()

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, spec in TABLES.items():
        df = getattr(tables, spec["attr"])
        frame = to_csv_frame(df, name)
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
        written[name] = frame

        level = spec["level"]
        columns = []
        for col in frame.columns:
            header = origin[level].get(col, "")
            # the table's own dtype, not the CSV's: index columns only exist on `frame`
            series = df[col] if col in df.columns else frame[col]
            columns.append(column_entry(
                series, col, header, glossaries[level].get(header, ""),
                desc.get((name, col), ""), note.get((name, col), ""),
            ))

        doc = {
            K["table"]: name,
            K["title"]: spec["title"],
            K["file"]: f"{name}.csv",
            K["description"]: spec["description"],
            K["key"]: list(df.index.names),
            K["rows"]: int(len(df)),
            K["cols"]: int(len(frame.columns)),
            K["source_file"]: spec["source"],
            K["provenance"]: prov,
            K["columns"]: columns,
        }
        with (out_dir / f"{name}.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False, width=98)

    (out_dir / "README.md").write_text(readme(c, issues, prov, written), encoding="utf-8")
    return written


def verify(out_dir: Path) -> None:
    """Re-read the dataset the way a consumer will and check the dictionaries hold.

    Nothing here may import eodgdl: the point is that the files stand on their own.
    """
    tables = {}
    for name in TABLES:
        doc = yaml.safe_load((out_dir / f"{name}.yaml").read_text(encoding="utf-8"))
        cols = {c[K["name"]]: c for c in doc[K["columns"]]}
        text = [n for n, c in cols.items() if c[K["type"]] == TYPES["string"]]
        df = pd.read_csv(out_dir / f"{name}.csv", dtype=dict.fromkeys(text, str))
        tables[name] = df

        assert list(df.columns) == list(cols), f"{name}: columns differ from its dictionary"
        assert len(df) == doc[K["rows"]], f"{name}: row count differs from its dictionary"
        assert not df.duplicated(doc[K["key"]]).any(), f"{name}: primary key repeats"
        for col, spec in cols.items():
            where, kind = f"{name}.{col}", spec[K["type"]]
            assert int(df[col].isna().sum()) == spec[K["n_missing"]], f"{where}: n_faltantes"
            if kind == TYPES["categorical"]:
                unknown = set(df[col].dropna().unique()) - set(spec[K["categories"]])
                assert not unknown, f"{where}: values outside its categories: {unknown}"
            elif kind in (TYPES["integer"], TYPES["number"]):
                seen = pd.to_numeric(df[col]).dropna()
                assert (seen.min(), seen.max()) == (spec[K["min"]], spec[K["max"]]), \
                    f"{where}: range"
            elif kind == TYPES["date"]:
                seen = pd.to_datetime(df[col])
                assert str(seen.min().date()) == spec[K["min"]], f"{where}: range"
                assert str(seen.max().date()) == spec[K["max"]], f"{where}: range"
            elif kind == TYPES["string"]:
                assert df[col].dropna().nunique() == spec[K["n_unique"]], f"{where}: n_unique"

    viv, hab, trips, legs = (tables[n] for n in TABLES)
    person = ["folio_vivienda", "folio_habitante"]
    trip = person + ["folio_viaje"]
    assert hab.folio_vivienda.isin(viv.folio_vivienda).all(), "a person has no dwelling"
    assert trips.set_index(person).index.isin(hab.set_index(person).index).all(), \
        "a trip has no person"
    assert legs.set_index(trip).index.isin(trips.set_index(trip).index).all(), \
        "a leg has no trip"
    counted = hab.set_index(person).viajes_contados
    assert (counted == trips.groupby(person).size().reindex(counted.index, fill_value=0)).all(), \
        "viajes_contados does not match the trip rows"
    assert legs.groupby(trip).size().sum() == trips.n_traslados.sum(), \
        "n_traslados does not match the leg rows"


def readme(c: dict[str, int], issue: dict[str, int], prov: dict,
           written: dict[str, pd.DataFrame]) -> str:
    defectos = "\n".join(
        f"| {issue[code]:,} | {label} |"
        for code, label in sorted(DEFECTOS_ES.items(), key=lambda kv: -issue.get(kv[0], 0))
        if issue.get(code)
    )
    filas = "\n".join(
        f"| `{name}.csv` | `{name}.yaml` | {TABLES[name]['title']} | "
        f"{len(df):,} | {len(df.columns)} |"
        for name, df in written.items()
    )
    return f"""# Encuesta Origen-Destino 2023, Guadalajara — tablas depuradas

La encuesta origen-destino en hogares del Área Metropolitana de Guadalajara levantada por
IMEPLAN (**EOD 2023**), leída, validada, reorganizada en cuatro tablas ligadas y con sus
cadenas de viaje depuradas. Todas las columnas conservan el vocabulario de la encuesta: las
etiquetas son las suyas, no se recodificaron.

| datos | diccionario | nivel | filas | columnas |
|---|---|---|---|---|
{filas}

Cada `.yaml` es el diccionario de datos del `.csv` que está a su lado: cada columna con su
tipo, su definición, el encabezado original del que proviene, sus valores válidos y una nota
donde la depuración la modificó.

## Cómo se ligan las tablas

```
viviendas   folio_vivienda
  habitantes  folio_vivienda + folio_habitante
    viajes      folio_vivienda + folio_habitante + folio_viaje
      traslados   folio_vivienda + folio_habitante + folio_viaje + folio_traslado
```

Una columna que es constante dentro de la vivienda (municipio, AGEB, centralidad) se conserva
solo en `viviendas`, y una constante dentro de la persona solo en `habitantes`: para
recuperarlas hay que unir con el nivel de arriba, no están repetidas en las tablas de abajo.
`fecha` es la excepción, y está tanto en `viviendas` como en `habitantes`, porque
{c['viviendas_varias_fechas']:,} viviendas se entrevistaron en más de un día.

## Cómo leer los archivos

- UTF-8, separados por coma, punto decimal, un solo renglón de encabezado, campo vacío =
  dato faltante.
- **Las claves de área y de zona son texto, nunca números** (`ageb`, `origen`, `destino`):
  claves AGEB (CVEGEO) de 13 caracteres y claves de localidad de 9, 154 de las cuales terminan
  en letra. Léalas como texto o una hoja de cálculo las va a estropear.
- `fecha` viene como `AAAA-MM-DD` y es la fecha de la *entrevista*. Los viajes reportados son
  los del día de referencia anterior; ese día está en `dia_semana_viajes`, en `habitantes`
  (un día hábil, casi siempre el previo a la entrevista, pero no siempre).
- Los valores booleanos se escriben `True` / `False`.

## Factores de expansión

`viviendas`, `habitantes` y `viajes` traen cada una su propio `ponderador`, y los tres **no**
son el mismo peso: son tres etapas de calibración anidadas. Expanda viviendas con el factor
de la vivienda, personas con el de la persona y viajes con el de viaje. Expandir los viajes
con el factor de la persona sobrestima el viaje en autobús alrededor de 77%.

Los factores reconcilian con los totales publicados sobre la encuesta tal como se recibió;
estas tablas están cortas el peso de los {c['eliminadas']} viajes duplicados que se
eliminaron (vea el paso 8).

## Del archivo original a estas tablas

Todo lo que se hizo, en orden. Fuera del paso 8 no se eliminó ninguna fila de ninguna tabla,
y ninguna etiqueta se recodificó salvo donde se indica.

1. **Lectura.** Se leen los tres archivos originales (`IMEPLAN_Base_Viviendas_Master.csv`,
   `…_Habitantes_…`, `…_Viajes_…`), codificados en ISO-8859-1, con una fila por vivienda,
   por persona y por viaje respectivamente.
2. **Renombrado de columnas.** Cada encabezado original —que es la pregunta completa del
   cuestionario— se renombra a un nombre corto en minúsculas, sin acentos ni espacios. El
   encabezado original se conserva en `columna_original` dentro de cada diccionario.
3. **Faltantes y fechas.** Los textos `N/D`, `ND` y la cadena vacía se convierten en dato
   faltante (en la entrega 2023 ninguno de esos textos aparece: todo faltante es un campo
   vacío, y la mayoría son saltos del cuestionario —una pregunta que no se hizo— y no
   respuestas desconocidas). `fecha` se convierte de `%d-%b-%y` a fecha calendario.
4. **Correcciones a mano.** Tres celdas de medio de traslado traían el código suelto `17`,
   que ningún glosario documenta. Se etiquetaron `Transporte informal` a partir de un
   registro gemelo: la vivienda 879, del mismo AGEB y la misma fecha, repite viaje por viaje
   y traslado por traslado los diarios de esas dos personas y etiqueta así esos mismos
   traslados. Es la única corrección manual sobre los datos.
5. **Validación y tipado.** Cada tabla se valida contra un esquema estricto que fija el tipo
   de cada columna (entero, entero con faltantes, fecha, texto) y los niveles válidos de cada
   variable categórica: municipios, las 71 centralidades, los 20 medios de transporte, etc.
   Cualquier columna nueva o cualquier valor fuera de los niveles detiene la carga. Los
   niveles son los que cada diccionario lista en `categorías`, en su orden.
6. **Eliminación de columnas repetidas entre niveles.** Las columnas cuyo valor es constante
   dentro de la vivienda (municipio, AGEB, centralidad, …) se eliminan de `habitantes` y de
   `viajes`, y las constantes dentro de la persona se eliminan de `viajes`, después de
   verificar que efectivamente coinciden. Por eso hay que unir hacia arriba para recuperarlas.
7. **Detección de diarios repetidos.** Sobre la encuesta tal como se recibió, antes de tocar
   las cadenas, se identifican {c['diarios_repetidos']:,} personas cuyo diario completo es
   también el diario de una persona de otra vivienda del mismo AGEB y la misma fecha, con
   todas las horas de inicio a menos de diez minutos. Son copias con las horas recorridas, no
   viajes compartidos. No se eliminó ninguna —los dos miembros de cada par son igual de
   creíbles—, así que **los totales expandidos cuentan esos diarios dos veces**. Esta entrega
   no incluye la marca persona por persona.
8. **Depuración de las cadenas de viaje.** El orden de las filas es la cadena del día. Las
   reglas se aplican en este orden:
   1. *Imputación de lo que se perdió del cuestionario*: a {c['sin_hora']} viajes les falta la
      hora de inicio, y a esos mismos más otros pocos —{c['sin_motivo']} en total— les falta
      el motivo: se perdió el bloque del cuestionario donde iban. En vez de descartarlos se
      imputaron: {c['motivo_dup']} tomaron su respuesta del regreso casa–casa que los
      duplicaba ({c['hora_dup']} de ellos también la hora), y el resto por votación de sus 30
      viajes con hora más parecidos (mismo medio, sexo, ocupación, zonas, edad, …), con la
      hora contada hacia atrás desde el viaje siguiente.
   2. *Recodificación de regresos*: {c['recodificados']} viajes con motivo 'Regresar a Casa'
      cuya zona de destino no es la de la vivienda no fueron a casa; su motivo se recodificó
      a partir del tipo de lugar de destino que reportan.
   3. *Identificación de no-viajes*: {c['no_viajes']} regresos a casa hechos estando ya en
      casa se **conservan** en la tabla, pero son registros, no viajes, y conviene excluirlos
      al contar viajes; la sección siguiente dice cómo reconocerlos.
   4. *Origen después de un regreso*: {c['origen_casa']} viajes que siguen a un regreso a casa
      que sí llegó empiezan en la zona de la vivienda.
   5. *Reparación de horas mal anotadas*: {c['horas_reparadas']:,} horas de inicio, en
      {c['cadenas_reparadas']:,} cadenas, se releyeron con el menor número de erratas
      posibles (reloj de 12 horas, un 1 de más o de menos al principio) que permite que cada
      viaje empiece después de que llegó el anterior. Las horas imputadas nunca se editan, y
      donde no hay una relectura única no se toca nada.
   6. *Diagnóstico de lo que queda mal*: cada defecto que las reglas no resolvieron —horas,
      anclas del día, continuidad de zonas y motivos— se identifica fila por fila, con una
      tolerancia de 15 minutos al juzgar las horas. La sección siguiente los resume;
      {c['filas_con_problema']:,} de los {c['viajes']:,} viajes traen alguno.

   Lo único que se elimina son los {c['eliminadas']} regresos casa–casa que duplican a un
   regreso imputado, una vez que su hora y su motivo quedaron en la fila que repetían. A esas
   personas se les restó uno en `viajes_contados`, y `folio_viaje` conserva su numeración
   original con un hueco donde estaba la fila eliminada.
9. **Desdoblado de los traslados.** Las quince columnas `traslado1..5_{{medio,min,pago}}` de la
   tabla de viajes se convierten en la tabla `traslados`, con una fila por traslado
   efectivamente reportado, y se eliminan de `viajes`.
10. **Orden.** Las cuatro tablas quedan ordenadas por sus claves.

## Qué queda sin resolver

Las reglas anteriores no arreglan todo. Estos son los defectos que quedan, contados sobre las
{c['viajes']:,} filas de `viajes.csv` (una fila puede tener más de uno, y en total
{c['filas_con_problema']:,} traen alguno):

| filas | defecto |
|---:|---|
{defectos}

Dos de ellos conviene tenerlos presentes al usar los datos:

- Los **{c['no_viajes']} regresos a casa hechos estando ya en casa** se conservan en la tabla
  pero no son viajes, y el modelo de demanda los deja fuera. Se reproducen recorriendo la
  cadena de cada persona en el orden de `folio_viaje`: la persona empieza el día en casa si el
  primer viaje sale de `tipo_lugar_origen` = 'Su casa', queda en casa después de cada
  `motivo_viaje` = 'Regresar a Casa' y sale de casa con cualquier otro motivo; un 'Regresar a
  Casa' hecho mientras está en casa es uno de estos registros.
- Las **{c['hora_invertida']} filas con la hora invertida** empiezan antes de que hubiera
  podido llegar el viaje anterior. El orden de las filas es la cadena real; la hora es el dato
  ruidoso. No se reordenaron los viajes por hora: hacerlo rompe la continuidad de zonas y
  convierte en primer viaje del día un 'Regresar a Casa' en más de mil casos.

Nada se corrigió en silencio: todo lo que cambió está en la sección anterior, con sus conteos.

## Procedencia

Fuente: **Encuesta Origen-Destino 2023**, IMEPLAN (Instituto Metropolitano de Planeación del
Área Metropolitana de Guadalajara).
<https://www.imeplan.mx/plataformas-de-informacion/visualizador-eod>

Generado a partir de los archivos públicos `IMEPLAN_Base_*_Master.csv` con `eodgdl`
{prov['version_eodgdl']} (commit `{prov['commit_eodgdl']}`) el {prov['generado']}.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"),
                    help="directory holding the IMEPLAN CSVs (default: data)")
    ap.add_argument("--out", type=Path, default=Path("output/eod2023_clean"),
                    help="directory to write the dataset into (default: output/eod2023_clean)")
    args = ap.parse_args()

    out = args.out.resolve()
    written = export(args.data.resolve(), out)
    verify(out)
    for name, df in written.items():
        size = (out / f"{name}.csv").stat().st_size / 1e6
        print(f"{name:12} {len(df):>7,} rows  {len(df.columns):>3} cols  {size:6.1f} MB")
    print(f"\nwrote {out}"
          "\nchecked: every dictionary matches its file, keys are unique and the levels join")


if __name__ == "__main__":
    main()
