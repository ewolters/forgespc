"""Dataset adapter: forgecore.Dataset in, untouched capability solver runs."""

from forgecore import Dataset, Series

from forgespc.capability import calculate_capability, capability_from_dataset

DATA = [5.1, 5.0, 5.2, 4.9, 5.1, 5.3, 5.0, 5.1, 4.8, 5.2]


def _ds():
    return Dataset(
        series=[Series(name="diameter", values=DATA, unit="mm")],
        meta={"usl": 5.5, "lsl": 4.5},
    )


def test_capability_from_dataset_runs_the_untouched_solver():
    direct = calculate_capability(DATA, usl=5.5, lsl=4.5)
    adapted = capability_from_dataset(_ds())
    assert adapted.cpk == direct.cpk
    assert adapted.usl == 5.5


def test_result_carries_its_data():
    # §5b: the result owns every field its views need
    assert capability_from_dataset(_ds()).data == DATA
    assert calculate_capability(DATA, usl=5.5, lsl=4.5).data == DATA


def test_named_series_selects_among_several():
    ds = _ds()
    ds.series.append(Series(name="noise", values=[1.0, 2.0, 1.5]))
    assert capability_from_dataset(ds, series="diameter").data == DATA
