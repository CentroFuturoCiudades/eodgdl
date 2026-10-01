"""AMAI's socioeconomic level (NSE) as derive steps of the chain ``nse``: the points of the six components, AMAI's cut
into levels, and the rank calibration against AMAI's NSE distribution by AGEB. Ported from informal-jobs-model's
notebook 08. Every table of points, cut and file layout is in the chain YAML (``impute/chains/nse.yaml``)."""

import functools

import numpy as np
import pandas as pd

from . import register_derive


def _task_points(values, task):
    """AMAI points of a task's value per completion: its class's score, or (expected propagation) the value itself."""
    from ..spec import load_task

    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float).to_numpy()
    scores = load_task(task).scores
    points = values.astype(str).map(scores)
    assert points.notna().all(), f"{task}: classes without AMAI points: {sorted(set(values.astype(str)) - set(scores))}"
    return points.astype(float).to_numpy()


@register_derive("amai_puntos")
def amai_points(completions, base, config):
    """``amai_puntos``: the points of each task's value (``tasks``) plus those of the dwelling's own answers
    (``observed``: column -> {answer: points}); missing where an observed answer is missing, so the level cut from them
    is too and the chain stops, naming the dwellings (no EOD dwelling lacks one). An answer the completions carry (a
    pipeline's given column: the workers aged 14+ with the drawn ones) is read from the completion, else from the
    source."""
    total = np.zeros(len(completions))
    for task in config["tasks"]:
        total += _task_points(completions[task], task)
    rows = base.iloc[completions["row"].to_numpy()]
    for column, points in config["observed"].items():
        answer = pd.Series(completions[column].to_numpy() if column in completions else rows[column].to_numpy()).astype("string")
        mapped = answer.map({str(k): float(v) for k, v in points.items()})
        unmapped = set(answer.dropna()) - set(map(str, points)) - {config.get("missing_label", "no_especificado")}
        assert not unmapped, f"{column}: answers without AMAI points: {sorted(unmapped)}"
        total += mapped.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    return pd.DataFrame({config.get("column", "amai_puntos"): total})


@register_derive("nse_nivel")
def nse_level(completions, base, config):
    """``nse``: AMAI's level of the points (left-closed ``edges``, ``labels``)."""
    edges = [float(edge) for edge in config["edges"]]
    levels = pd.cut(completions[config["from"]].astype(float), bins=edges, labels=config["labels"], right=False)
    return pd.DataFrame({config.get("column", "nse"): levels.astype("string").to_numpy()})


@functools.cache
def read_amai_ageb(file, state, municipalities, columns):
    """AMAI's NSE by AGEB file (``file`` in the data catalog, ``columns`` its column names) for the ``state``'s
    ``municipalities``, with ``clave_ageb``, the 13-character AGEB key; "N/D" counts are missing. Read once per
    process (the file is 21 MB)."""
    from eodgdl.data import resolve

    table = pd.read_excel(resolve(file))
    table.columns = list(columns)
    table = table.iloc[1:].replace("N/D", pd.NA)                    # the first row is a second header
    table = table[table["nombre_entidad"].eq(state) & table["nombre_municipio"].isin(municipalities)].copy()
    table["clave_ageb"] = ("14" + pd.to_numeric(table["municipio"]).astype(int).astype(str).str.zfill(3)
                           + pd.to_numeric(table["localidad"]).astype(int).astype(str).str.zfill(4) + table["ageb"].astype(str).str.zfill(4))
    return table


def _amai_file(config):
    return read_amai_ageb(config["file"], config["state"], tuple(config["municipalities"]), tuple(config["file_columns"]))


@functools.cache
def load_nse_ageb(file, state, municipalities, columns, counts):
    """AMAI's NSE distribution by AGEB (:func:`read_amai_ageb`): one row per 13-character AGEB key with the share of
    dwellings in each level (``counts``: level -> count column), rows with an incomplete distribution dropped."""
    table = read_amai_ageb(file, state, municipalities, columns)
    shares = pd.DataFrame({level: pd.to_numeric(table[column], errors="coerce") for level, column in dict(counts).items()}).div(
        pd.to_numeric(table["num_viviendas"], errors="coerce"), axis=0)
    shares.index = table["clave_ageb"].to_numpy()
    shares = shares.dropna()
    shares = shares[shares.sum(axis=1) > 0]
    return shares.div(shares.sum(axis=1), axis=0)


def nse_ageb_shares(config):
    """:func:`load_nse_ageb` for a calibration step's configuration (its ``file``, ``state``, ``municipalities``,
    ``file_columns``, ``counts`` and the ``nse_calibrado`` levels)."""
    levels = list(config["levels"]["nse_calibrado"])
    return load_nse_ageb(config["file"], config["state"], tuple(config["municipalities"]), tuple(config["file_columns"]),
                         tuple((level, config["counts"][level]) for level in levels))


def ageb_dwellings(config):
    """Dwellings per AGEB in AMAI's file (the weight of each AGEB's distribution), for a calibration step's
    configuration."""
    table = _amai_file(config)
    return pd.Series(pd.to_numeric(table["num_viviendas"], errors="coerce").to_numpy(), index=table["clave_ageb"].to_numpy())


@register_derive("nse_ageb_calibrado")
def nse_ageb_calibrated(completions, base, config):
    """``nse_calibrado``: within each completion (a draw, or the plug-in completion) and each AGEB with AMAI's
    distribution, the dwellings ranked by their points (weighted mid-rank with the dwelling weight) take the level of
    AMAI's cumulative distribution at their rank; elsewhere, and where the points are missing, the uncalibrated level.
    The household ordering by points is kept; the AGEB's mix is AMAI's. Dwellings with equal points are ranked in a
    random order drawn anew in each completion (``seed``), so where a tie straddles a level, which of them take the
    lower one does not follow the survey's row order. It did up to 0.3.0: 5.6% of the draws sat in such a tie, and the
    first of a tie in the survey's order averaged level 2.58 against 3.44 for the others (0 = E ... 6 = A/B)."""
    if config.get("propagation") == "enumerate":
        raise ValueError("the rank calibration needs complete datasets (propagation draws or expected), not scenarios")
    levels = list(config["levels"]["nse_calibrado"])
    shares = nse_ageb_shares(config)
    cumulative = shares[levels].cumsum(axis=1)
    cumulative[levels[-1]] = 1.0
    rows = base.iloc[completions["row"].to_numpy()]
    frame = pd.DataFrame({"completion": completions["completion"].to_numpy(), "ageb": rows[config["ageb"]].astype(str).to_numpy(),
                          "points": completions[config["from"]].astype(float).to_numpy(), "weight": rows[config["weight"]].astype(float).to_numpy(),
                          "level": completions[config["level"]].astype("string").to_numpy()})
    calibrated = frame["level"].copy()
    ranked = frame[frame["ageb"].isin(cumulative.index) & frame["points"].notna()]
    ranked = ranked.assign(tie=np.random.default_rng(config["seed"]).random(len(ranked))).sort_values(["completion", "ageb", "points", "tie"])
    group = [ranked["completion"], ranked["ageb"]]
    cumulative_weight = ranked["weight"].groupby(group).cumsum()
    rank = ((cumulative_weight - ranked["weight"] / 2) / ranked["weight"].groupby(group).transform("sum")).to_numpy()
    bounds = cumulative.loc[ranked["ageb"]].to_numpy()
    index = np.clip((bounds <= rank[:, None]).sum(axis=1), 0, len(levels) - 1)     # searchsorted(side="right") per row
    calibrated.loc[ranked.index] = np.asarray(levels, dtype=object)[index]
    return pd.DataFrame({config.get("column", "nse_calibrado"): calibrated.to_numpy()})
