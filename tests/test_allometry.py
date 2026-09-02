"""Real, hand-computed regression checks for src/biomass/allometry.py's two cited models
(Chave et al. 2014, Jucker et al. 2017) -- see that module's docstring for the source equations.
"""
import math

import pytest

from src.biomass import allometry


def test_chave_2014_known_value():
    # dbh=30cm, height=20m, wsg=0.5 -> 0.0673 * (0.5*900*20)^0.976, computed independently
    est = allometry.estimate_agb_from_dbh(dbh_cm=30.0, height_m=20.0, species=None)
    # species=None triggers the regional fallback WSG (0.412), not 0.5 -- recompute with that
    expected = 0.0673 * (allometry.REGIONAL_MEAN_WSG_G_CM3 * 30.0**2 * 20.0) ** 0.976
    assert est.agb_kg == pytest.approx(expected, rel=1e-9)
    assert est.model == "chave_2014"
    assert est.wsg.source == "regional_fallback"


def test_chave_2014_species_lookup_uses_real_wsg():
    est = allometry.estimate_agb_from_dbh(dbh_cm=30.0, height_m=20.0, species="Pinus ponderosa")
    expected = 0.0673 * (0.38 * 30.0**2 * 20.0) ** 0.976
    assert est.agb_kg == pytest.approx(expected, rel=1e-9)
    assert est.wsg.source == "species_lookup"
    assert est.wsg.value_g_cm3 == 0.38
    # known species -> the model's own reported CV only, no WSG-substitution penalty
    assert est.relative_uncertainty == pytest.approx(allometry.CHAVE_2014_MEAN_CV)


def test_chave_2014_unknown_species_widens_uncertainty():
    known = allometry.estimate_agb_from_dbh(30.0, 20.0, species="Pinus ponderosa")
    unknown = allometry.estimate_agb_from_dbh(30.0, 20.0, species=None)
    assert unknown.relative_uncertainty > known.relative_uncertainty
    expected_cv = math.sqrt(allometry.CHAVE_2014_MEAN_CV**2 + allometry.REGIONAL_WSG_CV**2)
    assert unknown.relative_uncertainty == pytest.approx(expected_cv)


def test_chave_2014_bounds_bracket_point_estimate():
    est = allometry.estimate_agb_from_dbh(30.0, 20.0, species="Quercus kelloggii")
    assert est.agb_kg_lower < est.agb_kg < est.agb_kg_upper
    assert est.domain_caveat  # non-empty -- the tropical-fit-vs-temperate-site mismatch must
    # always be surfaced, never silently dropped


def test_jucker_2017_gymnosperm_known_value():
    est = allometry.estimate_agb_from_crown(height_m=15.0, crown_diameter_m=6.0, species="Pinus ponderosa")
    expected = 0.109 * (15.0 * 6.0) ** 1.790 * math.exp(0.204**2 / 2)
    assert est.agb_kg == pytest.approx(expected, rel=1e-6)
    assert est.model == "jucker_2017"


def test_jucker_2017_angiosperm_differs_from_gymnosperm():
    conifer = allometry.estimate_agb_from_crown(15.0, 6.0, species="Pinus ponderosa")
    oak = allometry.estimate_agb_from_crown(15.0, 6.0, species="Quercus kelloggii")
    assert conifer.agb_kg != pytest.approx(oak.agb_kg)
    expected_oak = 0.016 * (15.0 * 6.0) ** 2.013 * math.exp(0.204**2 / 2)
    assert oak.agb_kg == pytest.approx(expected_oak, rel=1e-6)


def test_jucker_2017_unknown_species_defaults_to_regional_functional_type():
    default = allometry.estimate_agb_from_crown(15.0, 6.0, species=None)
    conifer = allometry.estimate_agb_from_crown(15.0, 6.0, species="Pinus ponderosa")
    assert default.agb_kg == pytest.approx(conifer.agb_kg)  # regional default is gymnosperm --
    # 4 of the 5 species in this project's table are conifers


def test_jucker_2017_cv_matches_lognormal_identity():
    assert allometry.JUCKER_2017_CV == pytest.approx(math.sqrt(math.exp(0.204**2) - 1))


def test_resolve_wood_specific_gravity_species_vs_fallback():
    known = allometry.resolve_wood_specific_gravity("Abies concolor")
    assert known.value_g_cm3 == 0.37
    assert known.source == "species_lookup"

    unknown = allometry.resolve_wood_specific_gravity("Sequoiadendron giganteum")  # not in table
    assert unknown.value_g_cm3 == allometry.REGIONAL_MEAN_WSG_G_CM3
    assert unknown.source == "regional_fallback"


def test_taller_or_wider_tree_never_produces_lower_agb():
    baseline = allometry.estimate_agb_from_crown(10.0, 4.0, species="Pinus ponderosa")
    taller = allometry.estimate_agb_from_crown(15.0, 4.0, species="Pinus ponderosa")
    wider = allometry.estimate_agb_from_crown(10.0, 6.0, species="Pinus ponderosa")
    assert taller.agb_kg > baseline.agb_kg
    assert wider.agb_kg > baseline.agb_kg
