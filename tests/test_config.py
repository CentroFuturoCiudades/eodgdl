"""The run parameters (eodgdl.config): config/config.yaml, checked, and the run's weight reaching the EOD sources."""
import pytest

from eodgdl import config
from eodgdl.impute import sources
from eodgdl.impute.sources import get_source, source_weight, weight_file

EOD_SOURCES = ("eod.workers", "eod.persons", "eod.labour", "eod.dwellings")


def test_the_clone_s_config_holds_every_parameter():
    assert config.path() == config.REPO / "config" / "config.yaml"
    values = config.load()
    assert set(values) == set(config.PARAMETERS) and values["weight"] in ("design", "tmg") and values["expansion"] in ("design", "tmg")
    # the delivery's imputations train on the design weight: the reweighting reaches the TASHA tables only through
    # their ExpansionFactor (expansion: tmg once TMG's weight is in)
    assert values["weight"] == "design" and config.expansion() == values["expansion"]
    assert values["bootstrap"] >= 0 and values["draws"] > 0 and values["draw"] >= 0


@pytest.mark.parametrize(("changes", "message"), [({"bootstrap": "50"}, "bootstrap: '50' is not int"),
                                                  ({"draw": True}, "draw: True is not int"),
                                                  ({"weights": "design"}, "weights: not a run parameter")])
def test_a_config_with_a_typo_fails(run_config, changes, message):
    run_config(**changes)
    with pytest.raises(ValueError, match=message):
        config.load()


def test_the_run_s_weight_chooses_what_the_eod_sources_read(run_config):
    run_config(weight="design")
    for name in EOD_SOURCES:
        source = get_source(name)
        assert source.weight == "ponderador" and weight_file(source.config) is None and sources.weight_versions(source.config) == {}
    run_config(weight="tmg")
    for name in EOD_SOURCES:
        source = get_source(name)
        assert source.weight == "peso" and weight_file(source.config) == {"file": "EOD_peso_hogar_TMG.csv", "column": "peso", "key": "folio_vivienda"}
    # ENOE's and ENIGH's tasks train on their surveys' own weights, whatever the run's
    assert get_source("enoe.workers").weight == "survey_weight" and get_source("enigh.households").weight == "factor"


def test_a_run_weight_no_source_names_fails(run_config):
    run_config(weight="raked")
    with pytest.raises(ValueError, match="the run's weight 'raked'"):
        source_weight(get_source("eod.persons").config)
