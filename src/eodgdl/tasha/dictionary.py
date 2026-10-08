"""The data dictionary of the three tables, generated from the contract (``model_schema.yaml``: types, codes, ranges,
what each means) and the mapping (``mappings.yaml``: the survey's columns and answers behind each code, how a column
is derived, its status and caveats), so it cannot drift from what the build writes. ``tasha build`` writes it beside
the tables (``od_dictionary.md``) with each code's row count in them; ``eodgdl tasha dictionary`` writes it alone.
:func:`codes` is the short version, one small table per file (``od_<table>_codes.csv``): each column and what its codes
mean.

    from eodgdl import tasha

    text = tasha.dictionary()                                  # the definitions
    text = tasha.dictionary(od, expansion="design", draw=0)    # ... plus what the built tables hold
    tasha.codes("trips")                                       # column, code, meaning
"""
from __future__ import annotations

import re

import pandas as pd

from eodgdl.tasha._schema import imputed_lookups, load_mappings, load_schema, tables

# mappings.yaml's `status`, in words (its header comment says the same)
STATUS = {
    "ok": "from the survey's answers",
    "imputed": "some or all values drawn by an imputation model (one completed dataset of the eodgdl.impute pipeline)",
    "assumed": "no survey question: filled by a rule",
    "not_surveyed": "no survey question and no rule: a constant",
    "pending": "builds, but a decision is pending before the column is used",
}
EXPANSION = {"design": "each table's own survey weight (`ponderador`), `expansion: design`",
             "tmg": "TMG.SurveyReweight's household weight on every table, `expansion: tmg`"}
GRAIN = {"households": "dwelling", "people": "person (aged 6+: the survey interviews no younger)", "trips": "trip"}


def _text(value) -> str:
    """A YAML prose value as Markdown: folded lines joined, its blank-line paragraphs kept."""
    return "\n\n".join(" ".join(part.split()) for part in str(value).strip().split("\n") if part.strip())


def _cell(value) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def _names(source) -> str:
    return ", ".join(f"`{name}`" for name in ([source] if isinstance(source, str) else source))


def _sources(entry: dict) -> dict[str, list[str]]:
    """{code: what in the survey or the imputation gives it}: the answers of each lookup (the survey's, the
    override's, the imputed levels), the default and the constant."""
    found: dict[str, list[str]] = {}

    def add(values, label=""):
        by_code: dict[str, list[str]] = {}
        for answer, code in (values or {}).items():
            by_code.setdefault(str(code), []).append(str(answer))
        for code, answers in by_code.items():
            found.setdefault(code, []).append(label + "; ".join(answers))

    if entry.get("source") is not None:
        add(entry.get("values"))
    override = entry.get("override") or {}
    if override.get("source") is not None:
        add(override.get("values"), f"{override['source']} = ")
    for label, _, column, values in imputed_lookups(entry):
        add(values, f"drawn {column} = ")
    if "default" in entry:
        found.setdefault(str(entry["default"]), []).append("anything else (default)")
    if "constant" in entry:
        found.setdefault(str(entry["constant"]), []).append("every row (constant)")
    return found


def _observed(series: pd.Series, spec: dict, zones: dict) -> str:
    """What a built column holds, in a line (a coded column's counts go in its table instead)."""
    if spec.get("role") == "zone":
        sentinel = next(iter(spec.get("sentinel") or zones.get("sentinel") or {}), None)
        line = f"{series.nunique():,} distinct TAZ"
        if sentinel is not None and spec.get("sentinel"):
            line += f"; {int((series.astype(str) == str(sentinel)).sum()):,} rows at {sentinel}"
        return line
    if pd.api.types.is_numeric_dtype(series):
        line = f"min {series.min():,.6g}, median {series.median():,.6g}, max {series.max():,.6g}"
        return line + (f"; sum {series.sum():,.0f}" if spec.get("role") == "weight" else "")
    return f"{series.nunique():,} distinct values"


def _column(table: str, column: str, spec: dict, entry: dict, zones: dict, built: pd.DataFrame | None, expansion) -> list[str]:
    lines = [f'<a id="{table}-{column.lower()}"></a>', "", f"### `{column}`", "", f"{spec.get('dtype', '')} · {'required' if spec.get('required') else 'optional'} · {spec.get('role', '')}"
             + (f" · status: **{entry['status']}**" if entry.get("status", "ok") != "ok" else ""), ""]
    if spec.get("description"):
        lines += [_text(spec["description"]), ""]
    if spec.get("role") == "zone":
        lines += ["A TAZ: see [Zones](#zones).", ""]
    facts = []
    if "format" in spec:
        facts.append(f"**Format:** {spec['format']}")
    if "range" in spec:
        low, high = spec["range"]
        facts.append(f"**Range:** {'' if low is None else low} to {'no limit' if high is None else high}")
    if spec.get("sentinel"):
        facts.append("**Sentinel:** " + "; ".join(f"`{key}` {_cell(value)}" for key, value in spec["sentinel"].items()))
    if entry.get("source") is not None:
        facts.append(f"**Survey columns:** {_names(entry['source'])}")
    lookups = imputed_lookups(entry)
    if lookups:
        facts.append("**Imputed:** " + "; ".join(f"pipeline `{pipeline}`, column `{name}`" + (" (override)" if label != "imputed" else "")
                                                 for label, pipeline, name, _ in lookups))
    if entry.get("derivation"):
        facts.append(f"**Derivation:** {_text(entry['derivation'])}")
    override = entry.get("override") or {}
    if override.get("when"):
        facts.append(f"**Override:** where {_cell(override['when'])}")
    held = [EXPANSION[expansion]] if column == "ExpansionFactor" and expansion else []
    if built is not None and column in built and not spec.get("domain"):
        held.append(_observed(built[column], spec, zones))
    if held:
        facts.append(f"**In these tables:** {'; '.join(held)}")
    if not spec.get("domain") and entry.get("values") and entry.get("source") is not None:
        facts.append("**Lookup:** " + ", ".join(f"{_cell(answer)} → {code}" for answer, code in entry["values"].items()))
    lines += [f"- {fact}" for fact in facts] + ([""] if facts else [])

    if spec.get("domain"):
        sources = _sources(entry)
        counts = built[column].astype(str).value_counts() if built is not None and column in built else None
        header = "| Code | Meaning | From |" + (" Rows |" if counts is not None else "")
        lines += [header, "|---|---|---|" + ("---:|" if counts is not None else "")]
        for code, meaning in spec["domain"].items():
            row = f"| `{code}` | {_cell(meaning)} | {_cell(' · '.join(sources.get(str(code), [])))} |"
            lines.append(row + (f" {int(counts.get(str(code), 0)):,} |" if counts is not None else ""))
        lines.append("")
    if spec.get("notes"):
        lines += [f"*Contract.* {_text(spec['notes'])}", ""]
    if entry.get("note"):
        lines += [f"*Survey.* {_text(entry['note'])}", ""]
    return lines


def dictionary(od=None, expansion: str | None = None, draw: int | None = None) -> str:
    """The data dictionary as Markdown: every table and column of the contract with what the mapping says of it.
    ``od`` (:class:`~eodgdl.tasha.ODTables`, or ``{table: frame}``): the built tables, whose rows per code (and each
    other column's range) it adds; ``expansion`` and ``draw``: the weight and the completed dataset they were built
    with, stated where they apply."""
    schema, mappings = load_schema(), load_mappings()
    zones = schema.get("zones", {})
    built = dict(zip(tables(), od)) if od is not None and not isinstance(od, dict) else (od or {})
    meta = mappings.get("meta", {})
    lines = ["# TASHA tables: data dictionary", "",
             "Generated by eodgdl (`eodgdl tasha dictionary`, or written by `eodgdl tasha build` beside the tables) from "
             "`src/eodgdl/tasha/model_schema.yaml`, the contract the travel-demand model reads, and "
             "`src/eodgdl/tasha/mappings.yaml`, how the survey fills it. Edit those, not this file.", "",
             f"Survey: {meta.get('source', '')}. Three tables in a TASHA / GTAModel style: one row per dwelling, per person "
             "and per trip, joined on `HouseholdId` (and `PersonNumber`)."]
    if built:
        lines += ["", "These tables: " + ", ".join(f"{name} {len(frame):,} rows" for name, frame in built.items()) + "."
                  + (f" Imputed columns from completed dataset {draw} of the pipeline `tasha`." if draw is not None else "")
                  + (f" `ExpansionFactor`: {EXPANSION[expansion]}." if expansion else "")]
    lines += ["", "## Zones", "", f"Every zone column holds the **{zones.get('name', 'TAZ')}** ({zones.get('dtype', 'int')}). {_text(zones.get('description', ''))}",
              "", _text(zones.get("notes", "")), "", "## Status", "",
              "A column's `status` says where its values come from:", ""]
    lines += [f"- **{status}**: {meaning}" for status, meaning in STATUS.items()] + [""]
    for table in tables():
        spec_table = schema["tables"][table]
        entries = mappings.get(table, {})
        frame = built.get(table)
        lines += [f"## `{spec_table['file']}`", "",
                  f"One row per {GRAIN.get(table, table)}; key {', '.join(f'`{key}`' for key in spec_table['key'])}."
                  + (f" {len(frame):,} rows." if frame is not None else ""), ""]
        if spec_table.get("notes"):
            lines += [_text(spec_table["notes"]), ""]
        lines += ["| Column | Type | Required | Status | What |", "|---|---|---|---|---|"]
        for column, spec in spec_table["columns"].items():
            entry = entries.get(column, {})
            what = _label(spec)
            lines.append(f"| [`{column}`](#{table}-{column.lower()}) | {spec.get('dtype', '')} | {'yes' if spec.get('required') else 'no'} | "
                         f"{entry.get('status', 'ok')} | {_cell(what)} |")
        lines.append("")
        for column, spec in spec_table["columns"].items():
            lines += _column(table, column, spec, entries.get(column, {}), zones, frame, expansion)
    return "\n".join(lines).rstrip() + "\n"



def _label(spec: dict) -> str:
    """A column's one-line meaning: its ``label``, else its description's first sentence."""
    if spec.get("label"):
        return " ".join(str(spec["label"]).split())
    return re.split(r"(?<=\.)\s", " ".join(str(spec.get("description") or "").split()), maxsplit=1)[0]


def codes(table: str) -> pd.DataFrame:
    """One table's columns and what their codes mean, from the contract: a row per column (``code`` empty: what the
    column is, its format or that it holds a TAZ), then a row per code of a coded column, and per sentinel of a zone
    column. Columns ``column``, ``code``, ``meaning``."""
    rows = []
    for column, spec in load_schema()["tables"][table]["columns"].items():
        meaning = _label(spec)
        if spec.get("role") == "zone":
            meaning = f"{meaning.rstrip('.')} (TAZ)."
        if spec.get("format"):
            meaning += f" Format: {spec['format']}."
        rows.append((column, "", meaning))
        for code, text in {**(spec.get("sentinel") or {}), **(spec.get("domain") or {})}.items():
            rows.append((column, str(code), " ".join(str(text).split())))
    return pd.DataFrame(rows, columns=["column", "code", "meaning"])
