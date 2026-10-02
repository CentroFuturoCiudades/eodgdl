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
    "regreso_dia_siguiente": "el regreso de la mañana siguiente de un turno nocturno, anotado como último "
                             "viaje del día: no es un viaje del día de la encuesta",
    "hora_traslapada": "empieza antes de la llegada reportada del viaje anterior, por 15 minutos "
                       "o menos: redondeo, no un error de hora",
    "fin_fuera_de_casa": "el último viaje del día no es un regreso que llegue a la zona de la vivienda",
    "regreso_en_casa": "un 'Regresar a Casa' hecho estando ya en casa: no es un viaje",
    "motivo_guarderia": "motivo 'Guardería' de un menor de 12 años, que asiste a la guardería",
    "regreso_sin_llegar": "un 'Regresar a Casa' que no llegó a la zona de la vivienda: las reglas leen "
                          "todos, así que lo dejó una edición a mano",
    "hora_repetida": "empieza en el mismo minuto que el viaje anterior",
    "hora_nocturna": "empieza de madrugada después de un viaje de la tarde o la noche: se lee "
                     "como viaje nocturno y se deja igual",
    "origen_discontinuo": "no empieza en la zona donde terminó el viaje anterior",
    "hora_anterior": "empieza antes que el viaje anterior pero dentro de la tolerancia de 15 "
                     "minutos respecto de su llegada: ruido de minutos",
    "tipo_destino_dudoso": "un 'Regresar a Casa' que sí llegó a la zona de la vivienda pero "
                           "reporta un tipo de lugar de destino que no es una casa",
    "hora_2301": "empieza a las 23:01 tal como se reportó, el minuto en que la encuesta amontona "
                 "los regresos tardíos: casi todos fueron a esa hora o después, quizá pasada la "
                 "medianoche, y unos pocos son la hora que la captura ponía a un regreso (1h01 o "
                 "5h01 después del viaje de ida)",
    "hora_1h01": "empieza 1h01 después del viaje anterior, que repite en sentido contrario (las zonas "
                 "al revés, el mismo modo principal y los mismos minutos), las dos horas tal como se "
                 "reportaron: una hora que llenó la captura, casi todas en las salidas a la tienda de "
                 "las tardes de abril de 2023; no se sabe cuándo fue el regreso, y la hora se deja igual",
    "hora_5h01": "empieza 5h01 después del viaje anterior, que repite en sentido contrario (las zonas "
                 "al revés, el mismo modo principal y los mismos minutos), las dos horas tal como se "
                 "reportaron: la hora que la captura ponía por omisión a un regreso; no se sabe cuándo "
                 "fue el regreso, y la hora se deja igual",
    "hora_madrugada": "el primer viaje del día, a algo que no es el trabajo, empieza antes de las "
                      "05:00: un mercado o un acompañamiento temprano, o una hora de reloj de 12 horas "
                      "que las reglas no pudieron releer",
    "hora_acompanante_12h": "empieza 12 horas antes o después de la hora que otro miembro de la vivienda "
                            "reporta para el mismo viaje, hecho juntos, las dos horas tal como se "
                            "reportaron: una de las dos se anotó en reloj de 12 horas y nada dice cuál",
}

_FECHA = (
    "Fecha de la entrevista. El diario que recoge es el del día de referencia anterior "
    "—registrado como día de la semana en `dia_semana_viajes`, casi siempre el día hábil "
    "previo, pero no siempre—, así que el día de los viajes se lee de esa columna y no de esta."
)
_ID_AREA = (
    "con el identificador de área de la propia encuesta: una clave AGEB urbana (CVEGEO) de 13 "
    "caracteres, una clave de AGEB rural de 9 —no una clave de localidad, aunque algunas "
    "coinciden como texto— o uno de los siete puntos de acceso `99999000x`; 154 claves terminan "
    "en letra. Es texto, nunca un número."
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
            f"su viaje: | Traslado N | Minutos'). Se corrigieron en los traslados de {c['minutos_a_mano']} "
            "viajes motorizados que reportaban el doble de lo típico para su distancia, y se recortaron "
            f"en los de {c['regresos_tope']} regresos a casa que llegaban después de la siguiente salida "
            "de la persona (vea el README)."
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
            "Texto, nunca un número: claves AGEB urbanas (CVEGEO) de 13 caracteres y claves de "
            "AGEB rural de 9 (no son claves de localidad), 154 de las cuales terminan en letra. "
            "Lea la columna como texto."
        ),
        ("habitantes", "viajes_contados"): (
            "Es igual al número de filas que la persona tiene en la tabla de viajes: se ajustó "
            f"en las personas que perdieron filas en la depuración ({c['duplicados']} regresos "
            f"duplicados y {c['descartadas']} filas que descartó la revisión a mano), para que "
            "siga contando sus filas."
        ),
        ("viajes", "folio_viaje"): (
            "Se conserva tal como viene en el archivo original, con un hueco donde estaba cada "
            f"una de las {c['eliminadas']} filas que se eliminaron (vea el README), así que no "
            "siempre va de 1 a n. El orden de las filas es la cadena del día: ordene por esta "
            "columna dentro de cada persona para recuperarla."
        ),
        ("viajes", "hora_inicio_h"): (
            f"Reparada donde la hora recibida no podía ser correcta: {c['horas_reparadas']:,} "
            "horas se releyeron con el menor número de erratas que permite que cada viaje "
            f"empiece después de que llegó el anterior, se imputó la de {c['sin_hora']} "
            f"viajes que no traían hora y {c['horas_a_mano']:,} se fijaron en la revisión a mano. "
            f"Quedan {c['hora_invertida']} filas que empiezan más de 15 minutos antes de que "
            f"llegara el viaje anterior, fuera de las {c['hora_nocturna']} que cruzan la medianoche."
        ),
        ("viajes", "hora_inicio_m"): (
            "Se reparó e imputó junto con la hora; vea `hora_inicio_h`."
        ),
        ("viajes", "motivo_viaje"): (
            f"Recodificado en {c['recodificados'] + c['regresos_lugar'] + c['regresos_otros']} viajes "
            "'Regresar a Casa' que no llegaron a la zona de la vivienda (a partir de su tipo de lugar "
            "de destino, de lo que la persona hace en ese lugar, u 'Otros' donde nada lo dice), en "
            f"{c['llegadas_casa']} viajes de actividad que "
            f"llegan a 'Su casa' en la zona de la vivienda (son el regreso) y en {c['guarderia']} "
            "viajes 'Guardería' de personas de 12 años o más (llevan o recogen a un menor), e "
            f"imputado en los {c['sin_motivo']} que no traían ninguno: {c['motivo_vec']} por "
            f"votación de sus 30 viajes con hora más parecidos y {c['motivo_dup']} a partir del "
            "regreso casa–casa que los duplicaba y que después se eliminó; la revisión a mano fijó "
            f"{c['motivos_a_mano']} más. Ojo: los {c['no_viajes']} regresos a casa hechos estando "
            f"ya en casa y los {c['turnos_nocturnos']} regresos de la mañana siguiente de un turno "
            "nocturno se conservan en la tabla y no son viajes; el README dice cómo identificarlos."
        ),
        ("viajes", "tipo_lugar_destino"): (
            "Se recodificó e imputó junto con el motivo; vea `motivo_viaje`. Desde las entrevistas del 3 de abril "
            f"de 2023, {c['abril_cc']:,} viajes de compras del tercero del día en adelante traen 'Centro cultural o "
            "área recreativa', un cambio de la captura que se dejó como vino (vea el README)."
        ),
        ("viajes", "origen"): (
            f"Fijado donde terminó el viaje anterior en {c['origen_casa'] + c['origen_anterior']} "
            f"viajes ({c['origen_casa']} en la zona de la vivienda, después de un regreso a casa que "
            f"sí llegó), en la zona de la vivienda en {c['primeros_casa']:,} primeros viajes del día "
            f"anotados saliendo de otra zona y en {c['primeros_rotados']} anotados al revés, de un lugar "
            "a casa (van de casa a ese lugar), y a mano en "
            f"{c['origenes_a_mano']:,}. Quedan {c['origen_discontinuo']} viajes que no empiezan "
            "donde terminó el anterior."
        ),
        ("viajes", "tipo_lugar_origen"): (
            "Una sola respuesta por persona, repetida en todas sus filas: dónde empezó su primer "
            "viaje del día. No es el tipo de lugar donde empieza cada viaje; el de un viaje que no "
            "es el primero es el `tipo_lugar_destino` del viaje anterior. Se fijó en 'Su casa' donde "
            "el primer viaje empieza en casa (vea `origen`), en el tipo de lugar de donde regresa en "
            f"las {c['respuestas_regreso']} personas cuyo día empieza con un regreso desde otra zona, "
            f"y la revisión a mano la cambió en {c['tipos_origen_a_mano']:,} personas."
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


def counts(tables, rules, shipped_trips: int, decisions: int) -> dict[str, int]:
    """Every figure the prose quotes, read back off the tables themselves.

    ``rules`` is the chain rules' output before the hand revision (``load_stages().rules``):
    what each rule did is counted there, since on the revised tables a rule's code stays
    only where its value still stands, and the duplicates the rules drop are told apart from
    the rows the revision drops.
    """
    fix = code_counts(rules.trips["ajustes"])            # what the rules did
    by_hand = code_counts(tables.trips["ajustes"])       # what the revised values owe to the hand
    issue = code_counts(tables.trips["problemas"])
    hours = [c for c in fix if c.startswith("hora:") and c[5] in "+-"]
    edited = rules.trips.index[
        rules.trips["ajustes"].str.contains("hora:[+-]", regex=True)
    ].droplevel("folio_viaje")
    revised = tables.trips["ajustes"]
    return {
        "eliminadas": shipped_trips - len(tables.trips),
        "duplicados": shipped_trips - len(rules.trips),
        "descartadas": len(rules.trips.index.difference(tables.trips.index)),
        "decisiones": decisions,
        "filas_a_mano": int(revised.str.contains(":revision", regex=False).sum()),
        "horas_a_mano": by_hand.get("hora:revision", 0),
        "motivos_a_mano": by_hand.get("motivo:revision", 0),
        "origenes_a_mano": by_hand.get("origen:revision", 0),
        "minutos_a_mano": by_hand.get("minutos:revision", 0),
        "destinos_a_mano": by_hand.get("destino:revision", 0),
        "tipos_origen_a_mano": int(
            tables.trips.index[revised.str.contains("tipo_origen:revision", regex=False)]
            .droplevel("folio_viaje").nunique()
        ),
        "hora_nocturna": issue.get("hora_nocturna", 0),
        "hora_dup": fix.get("hora:duplicado", 0),
        "hora_vec": fix.get("hora:vecinos", 0),
        "sin_hora": fix.get("hora:duplicado", 0) + fix.get("hora:vecinos", 0),
        "motivo_dup": fix.get("motivo:duplicado", 0),
        "motivo_vec": fix.get("motivo:vecinos", 0),
        "sin_motivo": fix.get("motivo:duplicado", 0) + fix.get("motivo:vecinos", 0),
        "recodificados": fix.get("motivo:tipo_destino", 0),
        "destino_copia": fix.get("destino:copia", 0),
        "llegadas_casa": fix.get("motivo:casa", 0),
        "regresos_lugar": fix.get("motivo:lugar", 0),
        "guarderia": fix.get("motivo:guarderia", 0),
        "origen_casa": fix.get("origen:casa", 0),
        "origen_anterior": fix.get("origen:anterior", 0),
        "primeros_casa": fix.get("origen:copia", 0) + fix.get("origen:respuesta", 0) + fix.get("origen:tipo_copia", 0),
        "primeros_rotados": fix.get("origen:rotado", 0),
        "regresos_casa": fix.get("destino:casa", 0),
        "regresos_otros": fix.get("motivo:otros", 0),
        "respuestas_regreso": int(
            rules.trips.index[rules.trips["ajustes"].str.contains("tipo_origen:regreso", regex=False)]
            .droplevel("folio_viaje").nunique()
        ),
        "horas_llegada": fix.get("hora:llegada", 0),
        "regresos_tope": fix.get("minutos:regreso", 0),
        "horas_reparadas": sum(fix[c] for c in hours),
        "cadenas_reparadas": int(edited.nunique()),
        "no_viajes_reglas": code_counts(rules.trips["problemas"]).get("regreso_en_casa", 0),
        "no_viajes": issue.get("regreso_en_casa", 0),
        "turnos_nocturnos": issue.get("regreso_dia_siguiente", 0),
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


def april(shipped) -> dict[str, int]:
    """The shopping trips typed 'Centro cultural o área recreativa', on the survey as shipped, by each person's own
    interview date: from the third trip of the day on, from 3 April 2023 and before, and on the first two trips from
    that day (``reports/loading.qmd``, *April's later trips*: the capture of the later trips changed that day)."""
    trips = shipped.trips
    person = trips.index.droplevel("folio_viaje")
    after = (shipped.hab.fecha.dt.tz_localize(None).reindex(person) >= pd.Timestamp("2023-04-03")).to_numpy()
    later = trips.groupby(level=["folio_vivienda", "folio_habitante"]).cumcount().to_numpy() >= 2
    shopping = trips.motivo_viaje.isin(["Compras (comida)", "Compras (bienes, productos y servicios)"]).to_numpy()
    cc = (trips.tipo_lugar_destino == "Centro cultural o área recreativa").to_numpy()
    return {
        "abril_cc": int((shopping & later & after & cc).sum()), "abril_compras": int((shopping & later & after).sum()),
        "antes_cc": int((shopping & later & ~after & cc).sum()), "antes_compras": int((shopping & later & ~after).sum()),
        "primeros_cc": int((shopping & ~later & after & cc).sum()),
        "primeros_compras": int((shopping & ~later & after).sum()),
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
    from eodgdl.data import VIAJES_CSV  # noqa: PLC0415
    from eodgdl._resources import chain_decisions  # noqa: PLC0415
    from eodgdl.eod import load_stages  # noqa: PLC0415

    stages = load_stages(data_dir)
    tables = stages.revised   # `counts` reads the audit columns before DROP removes them
    # the trip table as shipped, to say how many rows the cleaning removed
    shipped = len(pd.read_csv(data_dir / VIAJES_CSV, encoding="ISO-8859-1", usecols=[0]))
    c = counts(tables, stages.rules, shipped, len(chain_decisions())) | april(stages.shipped)
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
  claves AGEB urbanas (CVEGEO) de 13 caracteres, claves de AGEB rural de 9 (no son claves de
  localidad) y, en `origen` y `destino`, siete puntos de acceso `99999000x`; 154 claves
  terminan en letra. Léalas como texto o una hoja de cálculo las va a estropear.
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
estas tablas están cortas el peso de las {c['eliminadas']} filas de viaje que se eliminaron
(vea el paso 8).

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
   traslados. Es la única corrección manual antes de depurar las cadenas; los minutos de
   traslado que corrigió la revisión se describen en el paso 8.
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
8. **Depuración de las cadenas de viaje.** El orden de las filas es la cadena del día. Antes
   de las reglas, los minutos de traslado de {c['minutos_a_mano']} viajes se corrigen: son
   motorizados que reportaban el doble de lo que la encuesta suele reportar para su medio a su
   distancia en línea recta y chocaban con la salida siguiente, y toman los minutos típicos,
   repartidos entre sus traslados en proporción. Van antes porque las reglas leen los minutos:
   la llegada de un viaje, contra la que se juzga cada salida siguiente, es su hora de inicio
   más sus minutos. Las reglas se aplican después en este orden:
   1. *Imputación de lo que se perdió del cuestionario*: a {c['sin_hora']} viajes les falta la
      hora de inicio, y a esos mismos más otros pocos —{c['sin_motivo']} en total— les falta
      el motivo: se perdió el bloque del cuestionario donde iban. En vez de descartarlos se
      imputaron: {c['motivo_dup']} tomaron su respuesta del regreso casa–casa que los
      duplicaba ({c['hora_dup']} de ellos también la hora), y el resto por votación de sus 30
      viajes con hora más parecidos (mismo medio, sexo, ocupación, zonas, edad, …), con la
      hora contada hacia atrás desde el viaje siguiente.
   2. *Motivos y destinos*: {c['recodificados']} viajes con motivo 'Regresar a Casa' cuya zona
      de destino no es la de la vivienda no fueron a casa, y su motivo se recodificó a partir
      del tipo de lugar de destino que reportan; {c['destino_copia']} que se anotaron terminando
      en la misma AGEB de donde salen sí fueron a casa (el destino es una copia del origen);
      {c['llegadas_casa']} viajes con motivo de actividad que llegan a 'Su casa' en la zona de la
      vivienda desde otra zona son el regreso a casa; y {c['guarderia']} viajes 'Guardería' de
      personas de 12 años o más son de quien lleva o recoge a un menor.
   3. *Identificación de no-viajes*: {c['no_viajes_reglas']} regresos a casa hechos estando ya
      en casa se marcan como no-viajes, igual que {c['turnos_nocturnos']} regresos de la mañana
      siguiente de un turno nocturno: el último viaje de un día que empieza con el regreso
      desde el trabajo y termina con la salida a él por la tarde o la noche. Se **conservan**
      en la tabla, pero son registros, no viajes, y conviene excluirlos al contar viajes; la
      sección siguiente dice cómo reconocerlos. Los demás regresos que terminan fuera de casa
      —{c['regresos_lugar']}— toman el motivo que la persona tiene en ese lugar en sus otros
      viajes, o el del lugar donde empezó su día.
   4. *Orígenes*: cada viaje empieza donde terminó el anterior —{c['origen_casa']} en la zona
      de la vivienda, después de un regreso a casa que sí llegó, y {c['origen_anterior']} en el
      destino del viaje anterior—, y {c['primeros_casa']:,} primeros viajes del día anotados
      saliendo de otra zona empiezan en casa: el origen copia la AGEB del destino, la persona
      respondió que el día empezó en su casa, o la respuesta copia el tipo de lugar del destino.
      {c['primeros_rotados']} primeros viajes anotados de un lugar a casa, cuyo viaje siguiente
      sale de casa, se leen al revés: van de casa a ese lugar, y el siguiente sale de ahí.
      Las {c['respuestas_regreso']} personas cuyo día empieza con un regreso a casa desde otra
      zona y que respondieron 'Su casa' toman como respuesta el tipo de lugar de donde regresan.
      Los regresos a casa que siguen fuera de casa y que nada lee terminan en casa si son el
      último viaje del día a una AGEB que ningún otro viaje toca ({c['regresos_casa']}), y si no
      toman 'Otros' como motivo y tipo de lugar ({c['regresos_otros']}).
   5. *Reparación de horas mal anotadas*: antes de releer una hora, {c['regresos_tope']}
      regresos a casa que llegaban después de la siguiente salida de la persona —las dos horas
      tal como se reportaron y en orden— toman los minutos hasta esa salida, siempre que les
      quede al menos la mitad de lo reportado: el regreso tardó menos de lo que se anotó, y las
      horas de la encuesta se sostienen. Después, {c['horas_reparadas']:,} horas de inicio, en
      {c['cadenas_reparadas']:,} cadenas, se releyeron con el menor número de erratas
      posibles (reloj de 12 horas en los dos sentidos, un 1 de más o de menos al principio) que
      permite que cada viaje empiece después de que llegó el anterior, deja al menos media hora
      en el trabajo o la escuela y no deja a nadie medio día en un mandado corto. En un día cuyas
      horas están en orden, un regreso a casa 12 horas
      o más después de llegar a un mandado corto —compras, el médico, un trámite, un pago, llevar o
      recoger a alguien— se lee 12 horas antes: una hora de la mañana anotada del lado de la tarde
      del reloj de 12 horas. Un regreso a casa antes de las 10:00 después de salir al
      trabajo a las 18:00 o más tarde es de la mañana siguiente: un turno nocturno, siempre que
      después no se vuelva al trabajo o a la escuela y el día no llegue a 24 horas. Las horas
      imputadas nunca se editan, y donde no hay una relectura única no se toca nada. Después,
      {c['horas_llegada']} salidas que caían antes de la llegada del viaje anterior —por 15
      minutos o menos, o por hasta una hora donde ninguna relectura cabe y el viaje anterior no
      va al trabajo ni a la escuela— se recorren a esa llegada, y las cadenas que eso despeja se
      releen una vez más.
   6. *Revisión a mano*: lo que las reglas dejaron se revisó con un criterio escrito, persona
      por persona —contando lo que registró otro miembro de la vivienda que hizo el mismo viaje—,
      y la revisión quedó en {c['decisiones']:,} decisiones, cada una con su motivo. Fijó {c['horas_a_mano']:,} horas de inicio, {c['motivos_a_mano']} motivos,
      {c['origenes_a_mano']:,} orígenes y {c['destinos_a_mano']} destinos, cambió en
      {c['tipos_origen_a_mano']:,} personas la respuesta a dónde empezó su día y descartó
      {c['descartadas']} filas —en los turnos nocturnos que la regla no alcanza, el regreso de la
      mañana siguiente, que cae fuera del día de la encuesta—; en total {c['filas_a_mano']:,}
      filas guardan un valor fijado a mano, contando los minutos corregidos. Donde dos miembros
      de la vivienda reportaron un viaje hecho juntos con 12 horas de diferencia, la revisión
      toma la hora de uno de los dos. Después de
      ella ningún viaje empieza donde no terminó el anterior, y ninguno empieza antes de que
      hubiera podido llegar el anterior (su hora de inicio más sus minutos de traslado) salvo los
      que cruzan la medianoche: el modelo de demanda no admite traslapes.
   7. *Diagnóstico de lo que queda*: cada defecto que queda —horas, anclas del día y motivos—
      se identifica fila por fila, con una tolerancia de 15 minutos al juzgar las horas. La
      sección siguiente los resume; {c['filas_con_problema']:,} de los {c['viajes']:,} viajes
      traen alguno.

   Se eliminan {c['eliminadas']} filas: los {c['duplicados']} regresos casa–casa que duplican
   a un regreso imputado, una vez que su hora y su motivo quedaron en la fila que repetían, y
   las {c['descartadas']} que descartó la revisión a mano. A esas personas se les ajustó
   `viajes_contados`, y `folio_viaje` conserva su numeración original con un hueco donde
   estaba cada fila eliminada.
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

- Los **{c['no_viajes']} regresos a casa hechos estando ya en casa** y los
  **{c['turnos_nocturnos']} regresos de la mañana siguiente de un turno nocturno** se conservan
  en la tabla pero no son viajes, y el modelo de demanda los deja fuera. Los primeros se
  reproducen recorriendo la cadena de cada persona en el orden de `folio_viaje`: la persona
  empieza el día en casa si `tipo_lugar_origen` es 'Su casa' y su primer viaje sale de la AGEB
  de la vivienda (`origen` igual al `ageb` de `viviendas`), queda en casa después de cada
  `motivo_viaje` = 'Regresar a Casa' que llega a la AGEB de la vivienda y sale de casa con
  cualquier otro viaje; un 'Regresar a Casa' hecho mientras está en casa es uno de estos
  registros. Los segundos, entre los viajes que quedan: el último viaje del día es un 'Regresar
  a Casa' desde la AGEB a la que fue el viaje anterior, un 'Trabajar' que salió a las 18:00 o
  más tarde, y el primer viaje del día es un 'Regresar a Casa' desde esa misma AGEB.
- Las **{c['hora_nocturna']} filas nocturnas** empiezan de madrugada, antes que el viaje
  anterior: el día cruza la medianoche y la hora es la reportada. El orden de las filas es la
  cadena real; la hora es el dato ruidoso. No se reordenaron los viajes por hora: hacerlo rompe
  la continuidad de zonas y convierte en primer viaje del día un 'Regresar a Casa' en más de
  mil casos.

Además, una marca de la captura que ninguna regla corrige:

- **La captura de abril de 2023.** Desde las entrevistas del 3 de abril, el tercer viaje del día y
  los siguientes se capturaron de otra manera: {c['abril_cc']:,} de los {c['abril_compras']:,} viajes
  de compras en esas posiciones ({c['abril_cc'] / c['abril_compras']:.0%}) traen como tipo de lugar
  'Centro cultural o área recreativa', contra {c['antes_cc'] / c['antes_compras']:.1%} antes y
  {c['primeros_cc'] / c['primeros_compras']:.1%} en los dos primeros viajes. En esas posiciones casi
  desaparecen la recreación y 'Otros', y ahí están casi todos los regresos a 1h01 (`hora_1h01`) y
  los terceros viajes sin hora ni motivo. Puede estar mal el tipo de lugar o el motivo, y los datos
  no dicen cuál, así que no se recodificó nada: al comparar tipos de lugar, compras o recreación por
  fecha o posición, separe lo anterior al 3 de abril.

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
