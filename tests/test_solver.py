"""capability_from_dataset is registered as an engine subsystem: the forgecore
kernel discovers it on a Dataset world (solvers_for), drives it (run), and
renders its result (render). Importing forgespc.capability fires the @solver
registration."""

from forgecore import Dataset, Series
from forgecore import render, run, solver_registry, solvers_for

from forgespc.capability import capability_from_dataset

DATA = [5.1, 5.0, 5.2, 4.9, 5.1, 5.3, 5.0, 5.1, 4.8, 5.2]


def _ds() -> Dataset:
    return Dataset(
        series=[Series(name="diameter", values=DATA, unit="mm")],
        meta={"usl": 5.5, "lsl": 4.5},
    )


def test_capability_solver_is_registered():
    info = solver_registry()["capability_from_dataset"]
    assert info.consumes == frozenset({"dataset"})
    assert "ProcessCapability" in info.produces
    assert info.dialects == frozenset({"capability"})
    assert info.package == "forgespc"


def test_solvers_for_a_dataset_find_the_capability_solver():
    names = [info.name for info in solvers_for(_ds())]
    assert "capability_from_dataset" in names


def test_kernel_run_drives_the_registered_solver():
    ds = _ds()
    via_kernel = run("capability_from_dataset", ds)
    direct = capability_from_dataset(ds)
    assert via_kernel.cpk == direct.cpk
    assert via_kernel.usl == direct.usl


def test_kernel_renders_the_result_into_chartspecs():
    specs = render(run("capability_from_dataset", _ds()))
    assert specs
    assert all(hasattr(spec, "to_dict") for spec in specs)


def test_importing_the_package_registers_the_solver():
    # A fresh interpreter: importing the package alone must fill the registry
    # (ENGINE.md: "the registries fill when the caller imports solver packages").
    import subprocess
    import sys
    code = ("import forgespc; from forgecore import solver_registry; "
            "assert 'capability_from_dataset' in solver_registry()")
    subprocess.run([sys.executable, "-c", code], check=True)
