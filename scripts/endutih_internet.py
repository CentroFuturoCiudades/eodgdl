"""Derive data/ENDUTIH_internet_AMG.csv: the share of dwellings with internet, 2020 and 2023.

INEGI's ENDUTIH (Encuesta Nacional sobre Disponibilidad y Uso de Tecnologías de la
Información en los Hogares) open data carry no city identifier, only the state (ENT), the
urban/rural domain and the locality-size class (TLOC). The domain used for the Guadalajara
metropolitan area is **Jalisco, localities of 100,000 inhabitants or more** (ENT 14,
TLOC 1), which the metro dominates. A dwelling counts as connected when any of its
households answers P4_4 = 1 ("¿Disponen de conexión a internet en el hogar?"), weighted by
FAC_VIV; the standard error is the linearized one for a ratio mean with strata EST_DIS and
primary units UPM_DIS (with-replacement approximation). The fixed-connection share
(P4_5 = 1 or 3) is kept for reference: it moves the same way, so mobile-only connections
do not explain any gap with the census.

    uv run python scripts/endutih_internet.py [--out data/ENDUTIH_internet_AMG.csv]

Downloads ~20 MB from INEGI into a temporary directory.
"""
from __future__ import annotations

import argparse
import io
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = {
    2020: ("https://www.inegi.org.mx/contenidos/programas/dutih/2020/datosabiertos/"
           "conjunto_de_datos_endutih_2020_csv.zip", "hogar_anual_2020", "vivienda_anual_2020"),
    2023: ("https://www.inegi.org.mx/contenidos/programas/endutih/2023/datosabiertos/"
           "conjunto_de_datos_endutih_2023_csv.zip", "hogares_anual_2023", "viviendas_anual_2023"),
}
DOMAINS = {
    "Jalisco, localities 100k+": lambda d: (d.ENT == 14) & (d.TLOC == 1),
    "Jalisco": lambda d: d.ENT == 14,
    "Mexico, localities 100k+": lambda d: d.TLOC == 1,
}
KEY = ["UPM", "VIV_SEL"]


def _read(zf, stem):
    name = next(n for n in zf.namelist() if n.endswith(f"tr_endutih_{stem}.csv"))
    return pd.read_csv(io.BytesIO(zf.read(name)), dtype=str, encoding="latin1")


def dwellings(year, cache):
    url, households, viviendas = SOURCES[year]
    path = Path(cache) / f"endutih_{year}.zip"
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    with zipfile.ZipFile(path) as zf:
        h, v = _read(zf, households), _read(zf, viviendas)
    h["internet"] = h.P4_4 == "1"
    h["fixed"] = h.internet & h.P4_5.isin(["1", "3"])
    d = v.merge(h.groupby(KEY)[["internet", "fixed"]].any().reset_index(), on=KEY)
    return d.assign(ENT=d.ENT.astype(int), TLOC=d.TLOC.astype(int), w=d.FAC_VIV.astype(float))


def estimate(d, column):
    """Weighted share and its linearized standard error (strata EST_DIS, PSUs UPM_DIS)."""
    p = float(np.average(d[column], weights=d.w))
    z = d.w * (d[column] - p) / d.w.sum()
    psu = z.groupby([d.EST_DIS, d.UPM_DIS]).sum()
    var = 0.0
    for _, g in psu.groupby(level=0):
        if len(g) > 1:
            var += len(g) / (len(g) - 1) * float(((g - g.mean()) ** 2).sum())
    return p, var ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/ENDUTIH_internet_AMG.csv")
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as cache:
        for year in SOURCES:
            d = dwellings(year, cache)
            for domain, mask in DOMAINS.items():
                sub = d[mask(d)]
                share, se = estimate(sub, "internet")
                fixed, _ = estimate(sub, "fixed")
                rows.append({"year": year, "domain": domain, "share": round(share, 5),
                             "se": round(se, 5), "fixed_share": round(fixed, 5),
                             "n_dwellings": len(sub), "n_psu": sub.UPM_DIS.nunique()})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
