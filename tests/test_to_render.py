from forgespc.models import ControlChartResult, ControlLimits


def _result():
    return ControlChartResult(
        chart_type="I-MR",
        data_points=[9.8, 10.1, 9.9, 10.5, 9.7],
        limits=ControlLimits(ucl=10.6, cl=10.0, lcl=9.4, usl=11.0, lsl=9.0),
        out_of_control=[{"index": 3, "value": 10.5, "reason": "beyond_limits"}],
        run_violations=[{"rule": "run_8", "indices": [1, 2], "description": "8 in a row"}],
        in_control=False,
        summary="1 OOC point",
    )


def test_to_render_returns_chartspec_with_neutral_colors_and_roles():
    spec = _result().to_render()
    d = spec.to_dict()

    assert d["chart_type"] == "control_chart"
    assert len(d["traces"]) == 1
    assert d["traces"][0]["role"] == "data"

    roles = {r["role"] for r in d["reference_lines"]}
    assert {"control_limit", "centerline", "spec_limit"} <= roles

    assert d["traces"][0]["color"] == ""
    assert all(r["color"] == "" for r in d["reference_lines"])

    marker_roles = {m["role"] for m in d["markers"]}
    assert "out_of_control" in marker_roles


def test_to_render_does_not_import_forgeviz():
    # Run in a CLEAN interpreter so sys.modules pollution from other tests in
    # the full-suite run can't give a false pass. Proves that even calling
    # to_render() never reaches into the renderer.
    import subprocess
    import sys
    code = (
        "import sys; from forgespc.models import ControlChartResult, ControlLimits; "
        "ControlChartResult(chart_type='I', data_points=[1.0, 2.0], "
        "limits=ControlLimits(ucl=3.0, cl=2.0, lcl=1.0), out_of_control=[], "
        "run_violations=[], in_control=True, summary='').to_render(); "
        "assert 'forgeviz' not in sys.modules, 'forgeviz was imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
