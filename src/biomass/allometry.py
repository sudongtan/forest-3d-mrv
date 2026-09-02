"""Published allometric equations converting tree measurements to above-ground biomass (AGB, kg).

Two independent, real, cited models are implemented, because they need different inputs and this
project's actual pipeline can only supply one of them:

- **Chave et al. (2014)** needs trunk diameter at breast height (DBH) -- a ground-survey
  measurement (available from Open Forest Observatory's ground-reference plots), not something a
  nadir/oblique drone photo can see (the trunk is occluded by the canopy from above). Use this
  path when a real DBH measurement is available.
- **Jucker et al. (2017)** needs only tree height and crown diameter -- both of which
  `geometry/canopy_height.py` already produces from the drone/SfM pipeline. **This is the model
  this project's own drone-derived trees actually use**, precisely because it doesn't require an
  input the pipeline can't measure.

Per CLAUDE.md's guardrail, every constant here is cited, not a magic number, and every estimate
carries an explicit uncertainty range derived from that same source's own reported error --
never a bare point estimate.
"""
import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------------------------
# Species table: wood specific gravity (for Chave 2014) + functional type (for Jucker 2017),
# one source of truth for both models. Covers the 5 dominant species at this project's Phase 2
# validation site (Sierra Nevada mixed conifer/oak, OFO mission_000001) -- extend as new sites
# are added, not as a general/complete species database.
#
# Wood specific gravity (green-volume basis -- oven-dry weight / green volume, matching Chave
# 2014's own definition of rho) source: Miles, P.D.; Smith, W.B. 2009. "Specific Gravity and
# Other Properties of Wood and Bark for 156 Tree Species Found in North America." Res. Note
# NRS-38. USDA Forest Service, Northern Research Station. Table 1A, "Specific gravity ...
# Green volume basis" column (that table's own values trace to Jenkins et al. 2004).
# ---------------------------------------------------------------------------------------------
GYMNOSPERM = "gymnosperm"  # conifer -- Jucker et al. 2017's species=1
ANGIOSPERM = "angiosperm"  # broadleaf -- Jucker et al. 2017's species=2


@dataclass(frozen=True)
class SpeciesProperties:
    common_name: str
    wood_specific_gravity_g_cm3: float
    functional_type: str  # GYMNOSPERM | ANGIOSPERM


SPECIES_TABLE: dict[str, SpeciesProperties] = {
    "Abies concolor": SpeciesProperties("white fir", 0.37, GYMNOSPERM),
    "Calocedrus decurrens": SpeciesProperties("incense-cedar", 0.35, GYMNOSPERM),
    "Pinus ponderosa": SpeciesProperties("ponderosa pine", 0.38, GYMNOSPERM),
    "Pseudotsuga menziesii": SpeciesProperties("Douglas-fir", 0.45, GYMNOSPERM),
    "Quercus kelloggii": SpeciesProperties("California black oak", 0.51, ANGIOSPERM),
}

# Regional fallback WSG: the unweighted mean of the 5 species above, *not* a basal-area-weighted
# community survey mean (that would need real stand-composition data this project doesn't have)
# and *not* a global/tropical mean (which would be inappropriate for a temperate conifer-
# dominated site -- conifers run notably lower WSG than the tropical hardwoods that dominate
# Chave 2014's own dataset). A deliberately narrow, honestly-labeled stand-in, not a general
# constant -- revisit if this pipeline is ever run on a genuinely different forest type.
_WSG_VALUES = [s.wood_specific_gravity_g_cm3 for s in SPECIES_TABLE.values()]
REGIONAL_MEAN_WSG_G_CM3 = sum(_WSG_VALUES) / len(_WSG_VALUES)
# CV of WSG *within this same regional species table* -- used as a proxy for the extra
# uncertainty introduced by not knowing which of these species a given tree actually is. Derived
# from our own cited data, not a separate external estimate; see `resolve_wood_specific_gravity`.
_wsg_mean = REGIONAL_MEAN_WSG_G_CM3
REGIONAL_WSG_CV = (sum((v - _wsg_mean) ** 2 for v in _WSG_VALUES) / len(_WSG_VALUES)) ** 0.5 / _wsg_mean

REGIONAL_MEAN_FUNCTIONAL_TYPE = GYMNOSPERM  # 4 of 5 species in the table are conifers; used only
# when species is completely unknown and Jucker's model is being applied (functional type
# changes that model's coefficients materially -- see estimate_agb_from_crown).


@dataclass(frozen=True)
class WSGEstimate:
    value_g_cm3: float
    source: str  # "species_lookup" | "regional_fallback"
    species: str | None


def resolve_wood_specific_gravity(species: str | None) -> WSGEstimate:
    """Species -> wood specific gravity, with an explicit, uncertainty-propagating fallback --
    per CLAUDE.md: "fall back to a regional/community mean WSG ... propagate the resulting extra
    uncertainty into the biomass error bar rather than silently picking one species' value."
    """
    if species is not None and species in SPECIES_TABLE:
        return WSGEstimate(SPECIES_TABLE[species].wood_specific_gravity_g_cm3, "species_lookup", species)
    return WSGEstimate(REGIONAL_MEAN_WSG_G_CM3, "regional_fallback", species)


@dataclass(frozen=True)
class AGBEstimate:
    agb_kg: float
    agb_kg_lower: float
    agb_kg_upper: float
    relative_uncertainty: float  # the (approximately symmetric) CV used for the bounds above
    model: str  # "chave_2014" | "jucker_2017"
    wsg: WSGEstimate | None  # only set for the Chave path
    domain_caveat: str


# ---------------------------------------------------------------------------------------------
# Chave et al. (2014), Global Change Biology 20(10), 3177-3190, doi:10.1111/gcb.12629, Model 4
# (Eq. 4, verified against the published PDF): AGB_est = 0.0673 * (rho * D^2 * H)^0.976, D in cm,
# H in m, rho (WSG) in g/cm^3, AGB in kg. This coefficient already includes the paper's own
# log-normal bias correction (its Eq. 2), so 0.0673/0.976 are used directly as point-estimate
# coefficients, not re-corrected here.
#
# Uncertainty: the paper reports a mean tree-level coefficient of variation, CV(j), of 56.5%
# across its 58 validation sites for this exact model (Fig. 3b text) -- "the typical relative
# error that should be expected in the estimate of a single tree" (paper's own words). Used
# directly as this model's baseline relative uncertainty.
#
# Domain: fit on n=4004 harvested trees from 58 **tropical and subtropical** sites. This
# project's Phase 2 validation site (Sierra Nevada mixed conifer/oak, temperate) is outside that
# domain. Per CLAUDE.md's guardrail ("don't imply it's more universal than the source paper
# claims"), this is stated on every estimate, not silently assumed away -- the true error from
# this domain mismatch is real but not quantified by the paper's own stats above.
# ---------------------------------------------------------------------------------------------
CHAVE_2014_ALPHA = 0.0673
CHAVE_2014_BETA = 0.976
CHAVE_2014_MEAN_CV = 0.565
CHAVE_2014_DOMAIN_CAVEAT = (
    "Chave et al. (2014)'s pantropical model was fit on tropical/subtropical trees; applying it "
    "to a temperate conifer/oak site (this project's actual validation plot) is outside its "
    "fitted domain, and the resulting extra error is real but not captured by the 56.5% CV above."
)


def estimate_agb_from_dbh(dbh_cm: float, height_m: float, species: str | None = None) -> AGBEstimate:
    """Chave et al. (2014) Model 4. Requires a real DBH measurement (e.g. an OFO ground-reference
    plot) -- this project's drone/SfM pipeline cannot supply one; use `estimate_agb_from_crown`
    for pipeline-derived trees instead.
    """
    wsg = resolve_wood_specific_gravity(species)
    agb_kg = CHAVE_2014_ALPHA * (wsg.value_g_cm3 * dbh_cm**2 * height_m) ** CHAVE_2014_BETA

    relative_uncertainty = CHAVE_2014_MEAN_CV
    if wsg.source == "regional_fallback":
        # Combine two independent multiplicative error sources in quadrature (root-sum-of-
        # squares): the model's own reported per-tree CV, and the spread of WSG across this
        # project's own species table (REGIONAL_WSG_CV) as a proxy for "how wrong the mean WSG
        # could be for the real, unknown species". An approximation, not a rigorously propagated
        # error model -- documented as such, not asserted as more precise than it is.
        relative_uncertainty = (CHAVE_2014_MEAN_CV**2 + REGIONAL_WSG_CV**2) ** 0.5

    return AGBEstimate(
        agb_kg=agb_kg,
        agb_kg_lower=agb_kg * (1 - relative_uncertainty),
        agb_kg_upper=agb_kg * (1 + relative_uncertainty),
        relative_uncertainty=relative_uncertainty,
        model="chave_2014",
        wsg=wsg,
        domain_caveat=CHAVE_2014_DOMAIN_CAVEAT,
    )


# ---------------------------------------------------------------------------------------------
# Jucker et al. (2017), Global Change Biology 23(1), 177-190, doi:10.1111/gcb.13388. General
# (non-biome-specific) height x crown-diameter model, coefficients confirmed against the
# `itcSegment` R package's `agb()` implementation (Dalponte, CRAN), which cites this same paper:
#
#   AGB_est = (0.016 + a) * (H * CD)^(2.013 + b) * exp(0.204^2 / 2)
#
# with (a, b) = (0.093, -0.223) for gymnosperms, (0, 0) for angiosperms. H and CD (crown
# diameter) in meters, AGB in kg. The exp(sigma^2/2) term is the same log-normal bias correction
# convention Chave (a co-author on this paper too) uses in Chave (2014) Eq. 2 -- sigma=0.204 is
# this model's own fitted log-scale residual error.
#
# Uncertainty: converting the fitted log-scale sigma to a linear coefficient of variation via the
# standard log-normal identity CV = sqrt(exp(sigma^2) - 1) gives CV ~= 20.6%.
#
# Why this is the model this project's own drone-derived trees use: it needs only height and
# crown diameter, both of which `geometry/canopy_height.py` produces directly -- no DBH.
# ---------------------------------------------------------------------------------------------
_JUCKER_COEFFS = {
    GYMNOSPERM: (0.093, -0.223),
    ANGIOSPERM: (0.0, 0.0),
}
_JUCKER_SIGMA = 0.204
JUCKER_2017_CV = math.sqrt(math.exp(_JUCKER_SIGMA**2) - 1)
JUCKER_2017_DOMAIN_CAVEAT = (
    "Jucker et al. (2017)'s general model is fit globally across biomes and is not tropical-"
    "specific, so it doesn't carry the same domain mismatch as Chave (2014) for this project's "
    "temperate site. It does, however, inherit this project's own crown_diameter_m coarseness "
    "(see geometry/canopy_height.py's TreeHeightEstimate docstring) as an unquantified extra "
    "error source on top of the 20.6% CV below."
)


def estimate_agb_from_crown(
    height_m: float, crown_diameter_m: float, species: str | None = None
) -> AGBEstimate:
    """Jucker et al. (2017)'s general height x crown-diameter model -- the AGB path for trees
    coming out of this project's own drone/SfM pipeline (`geometry/canopy_height.py`), which
    cannot measure DBH. Functional type (conifer vs. broadleaf) is resolved from `species` via
    `SPECIES_TABLE` when known, else defaults to `REGIONAL_MEAN_FUNCTIONAL_TYPE`.
    """
    functional_type = REGIONAL_MEAN_FUNCTIONAL_TYPE
    if species is not None and species in SPECIES_TABLE:
        functional_type = SPECIES_TABLE[species].functional_type

    a, b = _JUCKER_COEFFS[functional_type]
    agb_kg = (0.016 + a) * (height_m * crown_diameter_m) ** (2.013 + b) * math.exp(_JUCKER_SIGMA**2 / 2)

    return AGBEstimate(
        agb_kg=agb_kg,
        agb_kg_lower=agb_kg * (1 - JUCKER_2017_CV),
        agb_kg_upper=agb_kg * (1 + JUCKER_2017_CV),
        relative_uncertainty=JUCKER_2017_CV,
        model="jucker_2017",
        wsg=None,
        domain_caveat=JUCKER_2017_DOMAIN_CAVEAT,
    )
