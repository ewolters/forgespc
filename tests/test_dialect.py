"""forgespc is the capability-dialect witness: ProcessCapability must speak
the dialect and conform to the engine Result protocol (forgerender)."""

from forgerender import CAPABILITY, ROLE_SPEC_LIMIT, ChartSpec, Result

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


def test_process_capability_conforms_to_result_protocol():
    assert isinstance(_cap(), Result)


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
