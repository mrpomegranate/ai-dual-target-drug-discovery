
import pytest
from dual_target.ic50 import parse_ic50


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("800", "active"),
        ("1000", "inactive"),
        ("<1000", "active"),
        ("<=1000", "indeterminate"),
        (">1000", "inactive"),
        (">999", "indeterminate"),
        ("<2000", "indeterminate"),
        ("<=999", "active"),
        (">=1000", "inactive"),
        ("1e3", "inactive"),
        ("≤ 1000", "indeterminate"),
        ("≥ 1000", "inactive"),
    ],
)
def test_activity_threshold(raw, expected):
    result = parse_ic50(raw)
    assert result is not None
    assert result.activity == expected


def test_exact_pic50():
    result = parse_ic50("100")
    assert result is not None
    assert result.pic50 == pytest.approx(7.0)


def test_censored_has_no_exact_pic50():
    result = parse_ic50("> 1000")
    assert result is not None
    assert result.relation == ">"
    assert result.pic50 is None


def test_missing():
    assert parse_ic50("") is None
    assert parse_ic50(None) is None


@pytest.mark.parametrize(
    "raw",
    ["0", "-5", "~500", "N/A", "1,000", "100;200", "1e999"],
)
def test_invalid(raw):
    with pytest.raises(ValueError):
        parse_ic50(raw)


def test_real_preview_values():
    observations = [800, 220, 160, 378, 114, 14]

    for nm in observations:
        result = parse_ic50(str(nm))
        assert result is not None
        assert result.activity == "active"
        assert result.pic50 is not None
