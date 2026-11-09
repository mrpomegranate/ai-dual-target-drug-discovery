import pytest
from dual_target.identity_review import classify_identity, review_priority


@pytest.mark.parametrize(
    ("source", "computed", "expected"),
    [
        ("CGWOMXNQGSNOHL-UHFFFAOYSA-N", "CGWOMXNQGSNOHL-UHFFFAOYSA-N", "exact_match"),
        (
            "WZACGJSPDNLOEJ-UHFFFAOYSA-N",
            "WZACGJSPDNLOEJ-ROJLCIKYSA-N",
            "source_unspecified_stereochemical_block",
        ),
        (
            "NHZUIHJUVWDGGV-LIJFRPJRSA-N",
            "NHZUIHJUVWDGGV-ROJLCIKYSA-N",
            "same_connectivity_other_inchi_layer_difference",
        ),
        (
            "FVDRAHRSWLZRPW-UHFFFAOYSA-N",
            "QDAVLQQAHAKDBT-FYYLOGMGSA-N",
            "different_connectivity_block",
        ),
        ("invalid", "CGWOMXNQGSNOHL-UHFFFAOYSA-N", "invalid_inchikey_format"),
        ("CGWOMXNQGSNOHL-UHFFFAOYSA-N", "malformed", "invalid_inchikey_format"),
    ],
)
def test_identity_classes(source, computed, expected):
    assert classify_identity("parsed", source, computed) == expected


def test_unparseable_structure():
    assert classify_identity("rdkit_parse_failure", "source", "") == "rdkit_unavailable"


@pytest.mark.parametrize(
    ("category", "fragments", "priority"),
    [
        ("exact_match", 1, "routine"),
        ("exact_match", 2, "medium"),
        ("rdkit_unavailable", None, "high"),
        ("different_connectivity_block", 1, "high"),
        ("source_unspecified_stereochemical_block", 1, "review_by_policy"),
        ("same_connectivity_other_inchi_layer_difference", 1, "medium"),
        ("invalid_inchikey_format", 1, "high"),
    ],
)
def test_triage_priority(category, fragments, priority):
    assert review_priority(category, fragments) == priority
