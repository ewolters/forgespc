"""Nelson rules and Western Electric rules for control chart analysis."""

from __future__ import annotations


def check_nelson_rules(
    data: list[float],
    center: float,
    sigma: float,
) -> list[dict]:
    """
    Check Nelson Rules (complete set of 8 rules) for control charts.

    Nelson Rules:
    1. One point beyond 3 sigma - detected as out-of-control point
    2. Nine points in a row on same side of center
    3. Six points in a row steadily increasing or decreasing
    4. Fourteen points in a row alternating up and down
    5. Two of three consecutive points beyond 2 sigma (same side)
    6. Four of five consecutive points beyond 1 sigma (same side)
    7. Fifteen consecutive points within 1 sigma of center (stratification)
    8. Eight points in a row beyond 1 sigma (either side, mixture)
    """
    violations = []
    n = len(data)

    if n < 2 or sigma <= 0:
        return violations

    # Calculate z-scores for zone classification
    zones = [(x - center) / sigma for x in data]

    # Rule 2: Nine points in a row on same side of center
    if n >= 9:
        for i in range(8, n):
            window = zones[i - 8 : i + 1]
            if all(z > 0 for z in window):
                violations.append(
                    {
                        "rule": 2,
                        "indices": list(range(i - 8, i + 1)),
                        "description": "9 consecutive points above center line",
                    }
                )
            elif all(z < 0 for z in window):
                violations.append(
                    {
                        "rule": 2,
                        "indices": list(range(i - 8, i + 1)),
                        "description": "9 consecutive points below center line",
                    }
                )

    # Rule 3: Six points in a row steadily increasing or decreasing
    if n >= 6:
        for i in range(5, n):
            window = data[i - 5 : i + 1]
            increasing = all(window[j] < window[j + 1] for j in range(5))
            decreasing = all(window[j] > window[j + 1] for j in range(5))
            if increasing:
                violations.append(
                    {
                        "rule": 3,
                        "indices": list(range(i - 5, i + 1)),
                        "description": "6 consecutive points steadily increasing",
                    }
                )
            elif decreasing:
                violations.append(
                    {
                        "rule": 3,
                        "indices": list(range(i - 5, i + 1)),
                        "description": "6 consecutive points steadily decreasing",
                    }
                )

    # Rule 4: Fourteen points in a row alternating up and down
    if n >= 14:
        for i in range(13, n):
            window = data[i - 13 : i + 1]
            alternating = True
            for j in range(13):
                if j % 2 == 0:
                    if window[j] >= window[j + 1]:
                        alternating = False
                        break
                else:
                    if window[j] <= window[j + 1]:
                        alternating = False
                        break
            if not alternating:
                # Check opposite pattern
                alternating = True
                for j in range(13):
                    if j % 2 == 0:
                        if window[j] <= window[j + 1]:
                            alternating = False
                            break
                    else:
                        if window[j] >= window[j + 1]:
                            alternating = False
                            break
            if alternating:
                violations.append(
                    {
                        "rule": 4,
                        "indices": list(range(i - 13, i + 1)),
                        "description": "14 consecutive points alternating up and down",
                    }
                )

    # Rule 5: Two of three consecutive points beyond 2 sigma (same side)
    if n >= 3:
        for i in range(2, n):
            window = zones[i - 2 : i + 1]
            above_2 = sum(1 for z in window if z > 2)
            below_2 = sum(1 for z in window if z < -2)
            if above_2 >= 2:
                violations.append(
                    {
                        "rule": 5,
                        "indices": list(range(i - 2, i + 1)),
                        "description": "2 of 3 points beyond +2 sigma",
                    }
                )
            if below_2 >= 2:
                violations.append(
                    {
                        "rule": 5,
                        "indices": list(range(i - 2, i + 1)),
                        "description": "2 of 3 points beyond -2 sigma",
                    }
                )

    # Rule 6: Four of five consecutive points beyond 1 sigma (same side)
    if n >= 5:
        for i in range(4, n):
            window = zones[i - 4 : i + 1]
            above_1 = sum(1 for z in window if z > 1)
            below_1 = sum(1 for z in window if z < -1)
            if above_1 >= 4:
                violations.append(
                    {
                        "rule": 6,
                        "indices": list(range(i - 4, i + 1)),
                        "description": "4 of 5 points beyond +1 sigma",
                    }
                )
            if below_1 >= 4:
                violations.append(
                    {
                        "rule": 6,
                        "indices": list(range(i - 4, i + 1)),
                        "description": "4 of 5 points beyond -1 sigma",
                    }
                )

    # Rule 7: Fifteen consecutive points within 1 sigma of center (stratification)
    if n >= 15:
        for i in range(14, n):
            window = zones[i - 14 : i + 1]
            if all(-1 < z < 1 for z in window):
                violations.append(
                    {
                        "rule": 7,
                        "indices": list(range(i - 14, i + 1)),
                        "description": "15 consecutive points within +/- 1 sigma (stratification)",
                    }
                )

    # Rule 8: Eight points in a row beyond 1 sigma (either side, mixture)
    if n >= 8:
        for i in range(7, n):
            window = zones[i - 7 : i + 1]
            if all(abs(z) > 1 for z in window):
                # Check that it's a mixture (not all on one side)
                above = sum(1 for z in window if z > 1)
                below = sum(1 for z in window if z < -1)
                if above > 0 and below > 0:
                    violations.append(
                        {
                            "rule": 8,
                            "indices": list(range(i - 7, i + 1)),
                            "description": "8 consecutive points beyond +/- 1 sigma (mixture)",
                        }
                    )

    # Deduplicate violations (same rule, overlapping indices)
    seen = set()
    unique_violations = []
    for v in violations:
        key = (v["rule"], tuple(v["indices"]))
        if key not in seen:
            seen.add(key)
            unique_violations.append(v)

    return unique_violations


# Alias for backwards compatibility
def check_western_electric_rules(
    data: list[float],
    center: float,
    sigma: float,
) -> list[dict]:
    """Alias for check_nelson_rules for backwards compatibility."""
    return check_nelson_rules(data, center, sigma)


# =============================================================================
# Process Capability
# =============================================================================


