"""Real, hand-computed checks for src/biomass/carbon.py -- see that module's docstring for the
cited IPCC (2006) carbon-fraction table and the CO2:C molar mass ratio.
"""
import pytest

from src.biomass import allometry, carbon


def test_co2_per_carbon_ratio_is_molar_mass_stoichiometry():
    assert carbon.CO2_PER_CARBON == pytest.approx(44.01 / 12.011)
    assert carbon.CO2_PER_CARBON == pytest.approx(3.664, abs=1e-3)


def test_carbon_kg_from_agb_kg_default_fraction():
    assert carbon.carbon_kg_from_agb_kg(100.0) == pytest.approx(47.0)


def test_co2e_kg_from_carbon_kg_known_value():
    assert carbon.co2e_kg_from_carbon_kg(47.0) == pytest.approx(47.0 * 44.01 / 12.011)


def test_estimate_co2e_default_carbon_fraction():
    agb = allometry.estimate_agb_from_dbh(30.0, 20.0, species="Quercus kelloggii")
    est = carbon.estimate_co2e(agb, functional_type=None)
    assert est.carbon_fraction_used == carbon.DEFAULT_CARBON_FRACTION
    expected = agb.agb_kg * carbon.DEFAULT_CARBON_FRACTION * carbon.CO2_PER_CARBON
    assert est.co2e_kg == pytest.approx(expected)


def test_estimate_co2e_functional_type_override_changes_carbon_fraction():
    agb = allometry.estimate_agb_from_crown(15.0, 6.0, species="Pinus ponderosa")
    conifer_est = carbon.estimate_co2e(agb, functional_type=allometry.GYMNOSPERM)
    broadleaf_est = carbon.estimate_co2e(agb, functional_type=allometry.ANGIOSPERM)
    assert conifer_est.carbon_fraction_used == 0.51
    assert broadleaf_est.carbon_fraction_used == 0.48
    assert conifer_est.co2e_kg > broadleaf_est.co2e_kg  # same AGB, higher CF -> more carbon


def test_estimate_co2e_bounds_track_agb_bounds():
    agb = allometry.estimate_agb_from_crown(15.0, 6.0, species="Pinus ponderosa")
    est = carbon.estimate_co2e(agb)
    assert est.co2e_kg_lower < est.co2e_kg < est.co2e_kg_upper
    # linear transform -- relative spread must be preserved exactly, not just bracket the point
    assert est.co2e_kg_lower / est.co2e_kg == pytest.approx(agb.agb_kg_lower / agb.agb_kg)
    assert est.co2e_kg_upper / est.co2e_kg == pytest.approx(agb.agb_kg_upper / agb.agb_kg)
