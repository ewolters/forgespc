# ForgeSPC

Statistical Process Control engine for manufacturing. Pure computation — no web framework, no database.

## What This Is

Standalone Python package extracted from SVEND's SPC engine. Consolidates two codebases (`agents_api/spc.py` + `agents_api/analysis/spc/`) into one package.

## Architecture

```
forgespc/
├── constants.py     # Chart constants (A2, D3, D4, d2, c4 for subgroups 2-10)
├── models.py        # Dataclasses: ControlLimits, ControlChartResult, ProcessCapability, etc.
├── charts.py        # Control charts: I-MR, X-bar/R, X-bar/S, p, c, u
├── capability.py    # Cp/Cpk/Pp/Ppk, sigma level, DPMO, yield
├── rules.py         # Nelson rules (2-8), Western Electric rules
├── gage.py          # Gage R&R (crossed), Hotelling T² (requires numpy)
├── calibration.py   # Self-calibration service with golden reference files
└── golden/          # Calibrated reference values shipped with package
```

## Usage

```python
from forgespc.charts import individuals_moving_range_chart
from forgespc.capability import calculate_capability
from forgespc.calibration import calibrate

result = individuals_moving_range_chart(data)
cap = calculate_capability(data, usl=53.0, lsl=47.0)
report = calibrate()  # Self-test against golden references
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Dependencies

- Core: stdlib only (math, statistics, dataclasses)
- Gage R&R / Hotelling: numpy (optional `[advanced]`)
- No Django, no web framework, no I/O

## License

MIT
