from forgespc.advanced import (
    CUSUMResult,
    EWMAResult,
    GeneralizedVarianceResult,
    MEWMAResult,
)
from forgespc.capability import calculate_capability
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


def test_views_composes_the_imr_pair_from_secondary_chart():
    mr = ControlChartResult(
        chart_type="MR",
        data_points=[0.3, 0.2, 0.6, 0.8],
        limits=ControlLimits(ucl=1.2, cl=0.4, lcl=0.0),
        out_of_control=[],
        run_violations=[],
        in_control=True,
        summary="MR in control",
    )
    primary = _result()
    primary.secondary_chart = mr

    views = primary.views()

    assert [v.subtitle for v in views] == ["I-MR", "MR"]
    # each chart is the result's own self-portrait, drawn from its own fields
    assert views[1].to_dict()["reference_lines"][0]["value"] == 1.2


def test_views_is_single_chart_without_secondary():
    views = _result().views()
    assert [v.subtitle for v in views] == ["I-MR"]


def test_capability_views_compose_histogram_and_probability_plot_from_own_data():
    cap = calculate_capability(
        [5.1, 5.0, 5.2, 4.9, 5.1, 5.3, 5.0, 5.1, 4.8, 5.2], usl=5.5, lsl=4.5
    )
    views = cap.views()

    assert [v.chart_type for v in views] == ["capability_histogram", "probability_plot"]
    hist = views[0].to_dict()
    assert {r["role"] for r in hist["reference_lines"]} >= {"spec_limit", "centerline"}
    assert all(t["color"] == "" for t in hist["traces"])  # theme-neutral
    prob = views[1].to_dict()
    assert len(prob["traces"][0]["y"]) == 10  # every sample point plotted


def test_capability_views_fall_back_to_fitted_curve_without_data():
    cap = calculate_capability(
        [5.1, 5.0, 5.2, 4.9, 5.1, 5.3, 5.0, 5.1, 4.8, 5.2], usl=5.5, lsl=4.5
    )
    cap.data = None
    assert [v.chart_type for v in cap.views()] == ["capability"]


def _cusum():
    return CUSUMResult(
        cusum_pos=[0.0, 1.2, 2.4, 3.6, 5.5],
        cusum_neg=[0.0, 0.5, 0.0, 0.3, 0.0],
        signals_up=[4],
        signals_down=[1],
        target=10.0,
        sigma=2.0,
        k=0.5,
        h=5.0,
        n=5,
        in_control=False,
    )


def test_cusum_to_render_carries_both_sides_in_one_chart():
    d = _cusum().to_render().to_dict()

    assert d["chart_type"] == "control_chart"
    assert d["subtitle"] == "CUSUM"
    assert [t["name"] for t in d["traces"]] == ["CUSUM+", "CUSUM-"]
    assert d["traces"][0]["y"] == [0.0, 1.2, 2.4, 3.6, 5.5]
    assert d["traces"][1]["y"] == [0.0, 0.5, 0.0, 0.3, 0.0]
    assert all(t["role"] == "data" and t["color"] == "" for t in d["traces"])


def test_cusum_to_render_decision_interval_is_h_in_z_units():
    # CUSUM is computed on z = (x - target)/sigma and signals where cusum > h,
    # so the decision interval on the chart IS h — never h * sigma.
    d = _cusum().to_render().to_dict()
    ref = {r["label"]: r for r in d["reference_lines"]}
    assert ref["UCL"]["value"] == 5.0
    assert ref["UCL"]["role"] == "control_limit"
    assert ref["CL"]["value"] == 0.0
    assert ref["CL"]["role"] == "centerline"


def test_cusum_to_render_marks_signals_from_both_sides():
    d = _cusum().to_render().to_dict()
    assert len(d["markers"]) == 1
    assert d["markers"][0]["indices"] == [1, 4]
    assert d["markers"][0]["role"] == "out_of_control"


def test_ewma_to_render_matches_chart_result_conversion():
    r = EWMAResult(
        ewma_values=[10.0, 10.2, 9.8, 10.1, 11.3],
        ucl=[10.5, 10.8, 10.9, 11.0, 11.0],
        lcl=[9.5, 9.2, 9.1, 9.0, 9.0],
        ucl_steady=11.0,
        lcl_steady=9.0,
        target=10.0,
        sigma=1.0,
        lambda_param=0.2,
        L=3.0,
        n=5,
        out_of_control_indices=[4],
        in_control=False,
    )
    d = r.to_render().to_dict()

    assert d["chart_type"] == "control_chart"
    assert d["subtitle"] == "EWMA"
    assert d["traces"][0]["y"] == [10.0, 10.2, 9.8, 10.1, 11.3]
    ref = {x["label"]: x["value"] for x in d["reference_lines"]}
    assert ref["UCL"] == 11.0
    assert ref["CL"] == 10.0
    assert ref["LCL"] == 9.0
    assert d["markers"][0]["indices"] == [4]


def test_mewma_to_render_plots_t2_against_chi2_limit():
    r = MEWMAResult(
        t2_values=[1.0, 2.5, 14.2],
        ucl=11.8,
        target=[10.0, 5.0],
        lambda_param=0.2,
        n_vars=2,
        n=3,
        out_of_control_indices=[2],
        in_control=False,
    )
    d = r.to_render().to_dict()

    assert d["chart_type"] == "control_chart"
    assert d["subtitle"] == "MEWMA"
    assert d["traces"][0]["y"] == [1.0, 2.5, 14.2]
    ref = {x["label"]: x["value"] for x in d["reference_lines"]}
    assert ref["UCL"] == 11.8


def test_generalized_variance_to_render():
    r = GeneralizedVarianceResult(
        gv_values=[0.5, 0.7, 2.4],
        ucl=2.0,
        cl=0.8,
        lcl=0.1,
        n_vars=2,
        subgroup_size=5,
        out_of_control_indices=[2],
        in_control=False,
    )
    d = r.to_render().to_dict()

    assert d["chart_type"] == "control_chart"
    assert d["subtitle"] == "|S|"
    assert d["traces"][0]["y"] == [0.5, 0.7, 2.4]
    ref = {x["label"]: x["value"] for x in d["reference_lines"]}
    assert ref["UCL"] == 2.0
    assert ref["CL"] == 0.8


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
