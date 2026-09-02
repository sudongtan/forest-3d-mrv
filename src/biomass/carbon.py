"""Converts above-ground biomass (AGB, kg) to carbon and CO2-equivalent -- the last step before
`scene_state.json`'s `trees[].co2e_kg`.

Both constants below are cited, per CLAUDE.md's guardrail against uncited magic numbers.
"""
from dataclasses import dataclass

from src.biomass.allometry import ANGIOSPERM, GYMNOSPERM, AGBEstimate

# Carbon fraction of dry biomass (CF), IPCC (2006) Guidelines for National Greenhouse Gas
# Inventories, Volume 4, Chapter 4, Table 4.3 "Carbon fraction of aboveground forest biomass"
# (verified against the published table text). Two rows apply to this project's temperate site:
#   - Temperate and Boreal, "All": 0.47 (range 0.47-0.49) -- used as the default here since most
#     of this project's trees aren't yet resolved to species (see allometry.py's WSG fallback).
#   - Temperate and Boreal, species-group-specific (Lamlom & Savidge, 2003): conifers 0.51
#     (0.47-0.55), broad-leaved 0.48 (0.46-0.50) -- available as an override once a tree's
#     functional type is known, reusing the same GYMNOSPERM/ANGIOSPERM grouping as allometry.py.
DEFAULT_CARBON_FRACTION = 0.47
CARBON_FRACTION_BY_FUNCTIONAL_TYPE = {
    GYMNOSPERM: 0.51,
    ANGIOSPERM: 0.48,
}

# CO2:C molar mass ratio -- standard stoichiometry (CO2 = 44.01 g/mol, C = 12.011 g/mol), the
# same conversion IPCC's own guidelines use throughout Volume 4 (commonly rounded to 44/12).
CO2_PER_CARBON = 44.01 / 12.011


@dataclass(frozen=True)
class CarbonEstimate:
    co2e_kg: float
    co2e_kg_lower: float
    co2e_kg_upper: float
    carbon_fraction_used: float


def carbon_kg_from_agb_kg(agb_kg: float, carbon_fraction: float = DEFAULT_CARBON_FRACTION) -> float:
    return agb_kg * carbon_fraction


def co2e_kg_from_carbon_kg(carbon_kg: float) -> float:
    return carbon_kg * CO2_PER_CARBON


def estimate_co2e(agb: AGBEstimate, functional_type: str | None = None) -> CarbonEstimate:
    """AGB (with its own propagated uncertainty from allometry.py) -> CO2e, with the same
    relative bounds carried through -- CF and the CO2:C ratio are both fixed multipliers, so no
    additional error-combination step is needed here (unlike allometry.py's WSG-substitution
    case, which combines independent multiplicative sources).
    """
    carbon_fraction = (
        CARBON_FRACTION_BY_FUNCTIONAL_TYPE.get(functional_type, DEFAULT_CARBON_FRACTION)
        if functional_type is not None
        else DEFAULT_CARBON_FRACTION
    )

    def to_co2e(agb_kg: float) -> float:
        return co2e_kg_from_carbon_kg(carbon_kg_from_agb_kg(agb_kg, carbon_fraction))

    return CarbonEstimate(
        co2e_kg=to_co2e(agb.agb_kg),
        co2e_kg_lower=to_co2e(agb.agb_kg_lower),
        co2e_kg_upper=to_co2e(agb.agb_kg_upper),
        carbon_fraction_used=carbon_fraction,
    )
