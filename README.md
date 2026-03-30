# ForgeSPC

Statistical Process Control engine for manufacturing. Pure computation library -- no web framework, no database, no I/O.

## Install

```bash
pip install forgespc
```

## Quick Start

```python
from forgespc.charts import individuals_moving_range_chart, xbar_r_chart
from forgespc.capability import calculate_capability
from forgespc.rules import check_nelson_rules

result = individuals_moving_range_chart([25.01, 24.99, 25.03, 25.00, 24.98])
cap = calculate_capability([25.01, 24.99, 25.03, 25.00], usl=25.05, lsl=24.95)
violations = check_nelson_rules(result)
```

### Advanced (requires numpy)

```python
from forgespc.gage import gage_rr_crossed
from forgespc.bayesian import bayesian_capability

grr = gage_rr_crossed(measurements, parts, operators)
bcap = bayesian_capability(data, usl=53.0, lsl=47.0)
```

## Modules

| Module | Contents |
|---|---|
| `constants` | Control chart constants (A2, D3, D4, d2, c4) for subgroups 2-10 |
| `models` | ControlLimits, ControlChartResult, ProcessCapability, StatisticalSummary |
| `charts` | I-MR, X-bar/R, p, c, u, np charts |
| `capability` | Cp, Cpk, Pp, Ppk, sigma level, DPMO, yield |
| `rules` | Nelson rules (2-8), Western Electric rules |
| `advanced` | CUSUM, EWMA, X-bar/S charts |
| `gage` | Gage R&R (crossed/nested), attribute agreement, Hotelling T-squared |
| `bayesian` | Bayesian capability, changepoint detection, Bayesian control chart |
| `conformal` | Conformal prediction control, entropy SPC |
| `calibration` | Self-calibration against golden reference files |

## Dependencies

- **Core**: Python stdlib only (math, statistics, dataclasses)
- **Advanced** (gage, bayesian, conformal): numpy, scipy

## License

MIT
