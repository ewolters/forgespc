"""forgespc is the capability-dialect witness. Generic engine-citizenship is
asserted by the shared conformance kit; the exact-dialect drift guard, the
bulk catalog-registration coverage, and the domain normalizations stay here."""

from forgecore import CAPABILITY, ROLE_SPEC_LIMIT, ChartSpec, Dataset, Series, result_registry
from forgecore.testing import assert_result_conforms, assert_solver_conforms

from forgespc.models import ProcessCapability


def _cap() -> ProcessCapability:
    return ProcessCapability(
        cp=1.33, cpk=1.0, cpu=1.0, cpl=1.66,
        pp=1.3, ppk=0.95, ppu=0.95, ppl=1.6,
        sigma_within=0.5, sigma_overall=0.52, sigma_level=3.0,
        dpmo=2700.0, yield_percent=99.73,
        usl=53.0, lsl=47.0, mean=50.0, n_samples=100,
        interpretation="marginal",
    )


def _ds() -> Dataset:
    return Dataset(
        series=[Series(name="d", values=[50.1, 49.9, 50.2, 49.8, 50.0, 50.3, 49.7, 50.1])],
        meta={"usl": 53.0, "lsl": 47.0},
    )


def test_process_capability_is_an_engine_citizen():
    assert_result_conforms(_cap())


def test_capability_solver_is_an_engine_citizen():
    assert_solver_conforms("capability_from_dataset", _ds())


def test_all_forgespc_solver_results_register_in_the_catalog():
    import forgespc.advanced as adv
    import forgespc.bayesian as bay
    import forgespc.conformal as con
    import forgespc.gage as gage
    from forgespc import models

    expected = {
        "CUSUMResult": adv.CUSUMResult,
        "EWMAResult": adv.EWMAResult,
        "MEWMAResult": adv.MEWMAResult,
        "GeneralizedVarianceResult": adv.GeneralizedVarianceResult,
        "BayesianCapabilityResult": bay.BayesianCapabilityResult,
        "BayesianChangepointResult": bay.BayesianChangepointResult,
        "BayesianControlResult": bay.BayesianControlResult,
        "ConformalControlResult": con.ConformalControlResult,
        "EntropySPCResult": con.EntropySPCResult,
        "GageRRResult": gage.GageRRResult,
        "ControlChartResult": models.ControlChartResult,
        "StatisticalSummary": models.StatisticalSummary,
    }
    reg = result_registry()
    for name, cls in expected.items():
        assert reg.get(name) is cls, name


def test_capability_accessor_speaks_the_full_capability_dialect():
    # Drift guard: renaming/dropping a dialect field fails the build.
    assert set(_cap().capability().keys()) == CAPABILITY


def test_capability_accessor_normalizes_sigma_level_to_sigma():
    cap = _cap().capability()
    assert cap["sigma"] == 3.0
    assert cap["cpk"] == 1.0
    assert cap["usl"] == 53.0


def test_to_render_emits_a_chartspec_with_both_spec_limits():
    spec = _cap().to_render()
    assert isinstance(spec, ChartSpec)
    spec_limits = [r for r in spec.reference_lines if r.role == ROLE_SPEC_LIMIT]
    assert len(spec_limits) == 2  # USL + LSL, theme-neutral
