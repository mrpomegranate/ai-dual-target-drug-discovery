
"""Conservative IC50 parsing for BindingDB observations."""

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

Activity = Literal["active", "inactive", "indeterminate"]
Relation = Literal["=", "<", "<=", ">", ">="]

_IC50_PATTERN = re.compile(
    r"^\s*(?P<relation><=|>=|<|>|=)?\s*"
    r"(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


@dataclass(frozen=True)
class ParsedIC50:
    raw: str
    relation: Relation
    value_nm: float
    pic50: float | None
    activity: Activity


def parse_ic50(
    raw: str | None,
    threshold_nm: float = 1000.0,
) -> ParsedIC50 | None:
    """Classify IC50 using a strict IC50 < threshold rule.

    Empty values return None. Invalid formats raise ValueError.
    Censored measurements never receive exact pIC50 values.
    """
    if raw is None or not raw.strip():
        return None

    if not math.isfinite(threshold_nm) or threshold_nm <= 0:
        raise ValueError("threshold_nm must be positive and finite")

    normalized = raw.replace("≤", "<=").replace("≥", ">=")
    match = _IC50_PATTERN.fullmatch(normalized)

    if match is None:
        raise ValueError(f"Unsupported IC50 format: {raw!r}")

    relation: Relation = match.group("relation") or "="

    try:
        number = Decimal(match.group("number"))

        if not number.is_finite() or number <= 0:
            raise ValueError(f"IC50 must be positive: {raw!r}")

        value_nm = float(number)

        if not math.isfinite(value_nm) or value_nm <= 0:
            raise ValueError(f"IC50 outside float range: {raw!r}")

    except (InvalidOperation, OverflowError) as exc:
        raise ValueError(f"Invalid IC50 number: {raw!r}") from exc

    cutoff = Decimal(str(threshold_nm))

    if relation == "=":
        activity: Activity = (
            "active" if number < cutoff else "inactive"
        )
    elif relation == "<":
        activity = (
            "active" if number <= cutoff else "indeterminate"
        )
    elif relation == "<=":
        activity = (
            "active" if number < cutoff else "indeterminate"
        )
    elif relation in (">", ">="):
        activity = (
            "inactive" if number >= cutoff else "indeterminate"
        )
    else:
        raise AssertionError("Unexpected relation")

    pic50 = (
        9.0 - math.log10(value_nm)
        if relation == "="
        else None
    )

    return ParsedIC50(
        raw=raw,
        relation=relation,
        value_nm=value_nm,
        pic50=pic50,
        activity=activity,
    )
