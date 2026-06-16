from pathlib import Path

import pytest

from app.services.fuzzy_matcher import (
    DEFAULT_MATCH_THRESHOLD,
    EmptyVendorMasterError,
    FuzzyMatchResult,
    VendorMasterNotFoundError,
    VendorMasterRowError,
    VendorMasterSchemaError,
    VendorRecord,
    find_best_vendor_match,
    load_vendor_master,
    load_vendor_master_for_run,
    normalize_company_name,
)


def write_csv(
    path: Path,
    content: str,
) -> None:
    """
    Create a temporary CSV file for a unit test.

    parent.mkdir ensures nested folders such as input_snapshot
    are created before the file is written.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
    )


def sample_vendor_records() -> list[VendorRecord]:
    """
    Return structured vendors used by fuzzy-matching tests.

    Both vendors are approved so exact and near-exact matches can
    verify that the accepted record retains its vendor-master status.
    """
    return [
        VendorRecord(
            vendor_id="V001",
            vendor_name="Acme Services GmbH",
            country="Germany",
            status="approved",
            risk_level="low",
        ),
        VendorRecord(
            vendor_id="V002",
            vendor_name="Global Data Ltd",
            country="United Kingdom",
            status="approved",
            risk_level="medium",
        ),
    ]


def test_load_vendor_master_returns_structured_records(
    tmp_path: Path,
) -> None:
    """
    A valid vendor master must be loaded into VendorRecord objects.
    """
    csv_path = tmp_path / "vendor_master.csv"

    write_csv(
        csv_path,
        (
            "vendor_id,vendor_name,country,status,risk_level\n"
            "V001,Acme Services GmbH,Germany,approved,low\n"
            "V002,Global Data Ltd,United Kingdom,new,medium\n"
        ),
    )

    records = load_vendor_master(csv_path)

    # Confirm that both CSV rows were loaded.
    assert len(records) == 2

    # Confirm raw dictionaries were converted into structured models.
    assert isinstance(records[0], VendorRecord)
    assert isinstance(records[1], VendorRecord)

    # Confirm the first vendor's values.
    assert records[0].vendor_id == "V001"
    assert records[0].vendor_name == "Acme Services GmbH"
    assert records[0].country == "Germany"
    assert records[0].status == "approved"
    assert records[0].risk_level == "low"

    # Confirm the second vendor was preserved in CSV order.
    assert records[1].vendor_id == "V002"


def test_load_vendor_master_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """
    A missing vendor_master.csv must raise a clear exception.
    """
    missing_path = tmp_path / "vendor_master.csv"

    with pytest.raises(
        VendorMasterNotFoundError,
        match="vendor_master.csv was not found",
    ):
        load_vendor_master(missing_path)


def test_load_vendor_master_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    """
    Every required CSV column must be present.

    This file intentionally omits risk_level.
    """
    csv_path = tmp_path / "vendor_master.csv"

    write_csv(
        csv_path,
        (
            "vendor_id,vendor_name,country,status\n"
            "V001,Acme Services GmbH,Germany,approved\n"
        ),
    )

    with pytest.raises(
        VendorMasterSchemaError,
        match="risk_level",
    ):
        load_vendor_master(csv_path)


def test_load_vendor_master_rejects_empty_csv(
    tmp_path: Path,
) -> None:
    """
    A CSV containing only the header has no usable vendor records.
    """
    csv_path = tmp_path / "vendor_master.csv"

    write_csv(
        csv_path,
        ("vendor_id,vendor_name,country,status,risk_level\n"),
    )

    with pytest.raises(
        EmptyVendorMasterError,
        match="contains no vendor records",
    ):
        load_vendor_master(csv_path)


def test_load_vendor_master_rejects_invalid_row(
    tmp_path: Path,
) -> None:
    """
    Required values must not be blank.

    This row intentionally contains an empty vendor_name.
    """
    csv_path = tmp_path / "vendor_master.csv"

    write_csv(
        csv_path,
        (
            "vendor_id,vendor_name,country,status,risk_level\n"
            "V001,,Germany,approved,low\n"
        ),
    )

    with pytest.raises(
        VendorMasterRowError,
        match="row 2",
    ):
        load_vendor_master(csv_path)


def test_run_snapshot_vendor_master_is_preferred(
    tmp_path: Path,
) -> None:
    """
    Agent C must prefer the immutable run snapshot over the current
    policies file when both are available.
    """
    run_dir = tmp_path / "runs" / "run_001"

    snapshot_path = run_dir / "input_snapshot" / "vendor_master.csv"

    policy_path = tmp_path / "policies" / "vendor_master.csv"

    write_csv(
        snapshot_path,
        (
            "vendor_id,vendor_name,country,status,risk_level\n"
            "V-SNAPSHOT,Snapshot Vendor,Germany,approved,low\n"
        ),
    )

    write_csv(
        policy_path,
        (
            "vendor_id,vendor_name,country,status,risk_level\n"
            "V-POLICY,Policy Vendor,France,new,medium\n"
        ),
    )

    records = load_vendor_master_for_run(
        run_dir,
        policy_path=policy_path,
    )

    # The snapshot record must be selected, not the policy record.
    assert len(records) == 1
    assert records[0].vendor_id == "V-SNAPSHOT"
    assert records[0].vendor_name == "Snapshot Vendor"


def test_policy_vendor_master_is_used_as_fallback(
    tmp_path: Path,
) -> None:
    """
    The policies copy is used when the run snapshot is unavailable.
    """
    run_dir = tmp_path / "runs" / "run_001"

    policy_path = tmp_path / "policies" / "vendor_master.csv"

    write_csv(
        policy_path,
        (
            "vendor_id,vendor_name,country,status,risk_level\n"
            "V001,Acme Services GmbH,Germany,approved,low\n"
        ),
    )

    records = load_vendor_master_for_run(
        run_dir,
        policy_path=policy_path,
    )

    assert len(records) == 1
    assert records[0].vendor_id == "V001"


def test_normalize_company_name_handles_casing() -> None:
    """
    Company-name capitalization must not affect matching.
    """
    first_name = normalize_company_name("Acme Services Gmbh")

    second_name = normalize_company_name("ACME Services GmbH")

    third_name = normalize_company_name("acme services gmbh")

    assert first_name == "acme services gmbh"
    assert first_name == second_name
    assert second_name == third_name


def test_normalize_company_name_handles_spacing() -> None:
    """
    Leading, trailing, and duplicated whitespace must be removed.
    """
    normalized_name = normalize_company_name("   Acme    Services     GmbH   ")

    assert normalized_name == "acme services gmbh"


def test_normalize_company_name_handles_punctuation() -> None:
    """
    Common punctuation differences must not change the normalized name.
    """
    name_without_punctuation = normalize_company_name("Acme Services GmbH")

    name_with_period = normalize_company_name("Acme Services GmbH.")

    name_with_commas = normalize_company_name("Acme, Services, GmbH")

    assert name_without_punctuation == "acme services gmbh"
    assert name_with_period == name_without_punctuation
    assert name_with_commas == name_without_punctuation


def test_normalize_company_name_handles_suffix_variations() -> None:
    """
    Equivalent common legal suffix variations must be canonicalized.
    """
    full_suffix = normalize_company_name("Acme Holdings Limited")

    abbreviated_suffix = normalize_company_name("Acme Holdings Ltd.")

    dotted_suffix = normalize_company_name("Example L.L.C.")

    plain_suffix = normalize_company_name("Example LLC")

    assert full_suffix == "acme holdings ltd"
    assert abbreviated_suffix == "acme holdings ltd"
    assert full_suffix == abbreviated_suffix

    assert dotted_suffix == "example llc"
    assert dotted_suffix == plain_suffix


def test_normalize_company_name_preserves_meaningful_words() -> None:
    """
    Normalization must preserve the company's meaningful name words.
    """
    normalized_name = normalize_company_name("Acme Global Technology Services GmbH")

    assert normalized_name == ("acme global technology services gmbh")

    assert "global" in normalized_name
    assert "technology" in normalized_name
    assert "services" in normalized_name


def test_normalize_company_name_is_deterministic() -> None:
    """
    The same input must always produce exactly the same output.
    """
    company_name = "  ACME, Services GmbH.  "

    first_result = normalize_company_name(company_name)
    second_result = normalize_company_name(company_name)
    third_result = normalize_company_name(company_name)

    assert first_result == "acme services gmbh"
    assert first_result == second_result
    assert second_result == third_result


def test_find_best_vendor_match_accepts_exact_match() -> None:
    """
    An identical normalized company name must produce score 100
    and return the approved official vendor.
    """
    result = find_best_vendor_match(
        "Acme Services GmbH",
        sample_vendor_records(),
    )

    assert isinstance(result, FuzzyMatchResult)
    assert result.match_score == 100
    assert result.threshold == DEFAULT_MATCH_THRESHOLD
    assert result.match_status == "matched"

    assert result.best_candidate.vendor_id == "V001"
    assert result.matched_vendor is not None
    assert result.matched_vendor.vendor_id == "V001"
    assert result.matched_vendor.status == "approved"


def test_find_best_vendor_match_accepts_typo_match() -> None:
    """
    A small spelling mistake should still exceed the default threshold.

    "Servces" intentionally omits the second letter "i".
    """
    result = find_best_vendor_match(
        "Acme Servces GmbH",
        sample_vendor_records(),
    )

    assert 85 <= result.match_score < 100
    assert result.match_status == "matched"

    assert result.matched_vendor is not None
    assert result.matched_vendor.vendor_id == "V001"
    assert result.matched_vendor.status == "approved"


def test_find_best_vendor_match_handles_casing_mismatch() -> None:
    """
    Case differences disappear during deterministic normalization.
    """
    result = find_best_vendor_match(
        "ACME SERVICES GMBH",
        sample_vendor_records(),
    )

    assert result.normalized_input_name == "acme services gmbh"
    assert result.match_score == 100
    assert result.match_status == "matched"

    assert result.matched_vendor is not None
    assert result.matched_vendor.vendor_id == "V001"


def test_find_best_vendor_match_rejects_weak_match() -> None:
    """
    A partially similar name below threshold must remain unknown.
    """
    result = find_best_vendor_match(
        "Acme Solutions GmbH",
        sample_vendor_records(),
    )

    # Acme Services remains the closest candidate,
    # but its score is too weak to accept.
    assert result.best_candidate.vendor_id == "V001"
    assert result.match_score < DEFAULT_MATCH_THRESHOLD

    assert result.match_status == "unknown"
    assert result.matched_vendor is None


def test_find_best_vendor_match_returns_unknown_for_no_match() -> None:
    """
    A materially unrelated company name must not be linked to a vendor.
    """
    result = find_best_vendor_match(
        "Zebra Quantum Mining PLC",
        sample_vendor_records(),
    )

    # The service still records the numerically closest candidate
    # for auditability, but does not accept it.
    assert result.best_candidate is not None
    assert result.match_score < DEFAULT_MATCH_THRESHOLD

    assert result.match_status == "unknown"
    assert result.matched_vendor is None


def test_find_best_vendor_match_uses_configurable_threshold() -> None:
    """
    The same score may be accepted or rejected depending on the
    configured corporate threshold.

    A score equal to the threshold must be accepted.
    """
    vendors = sample_vendor_records()
    counterparty_name = "Acme Servces GmbH"

    # First discover the deterministic score for the typo.
    default_result = find_best_vendor_match(
        counterparty_name,
        vendors,
    )

    # Equal-to-threshold must count as matched.
    equal_threshold_result = find_best_vendor_match(
        counterparty_name,
        vendors,
        threshold=default_result.match_score,
    )

    # One point above the score must reject the same candidate.
    stricter_result = find_best_vendor_match(
        counterparty_name,
        vendors,
        threshold=default_result.match_score + 1,
    )

    assert equal_threshold_result.match_status == "matched"
    assert equal_threshold_result.matched_vendor is not None

    assert stricter_result.match_status == "unknown"
    assert stricter_result.matched_vendor is None


def test_find_best_vendor_match_is_deterministic() -> None:
    """
    Repeated matching with identical input and vendor data must produce
    exactly the same structured result.
    """
    vendors = sample_vendor_records()

    first_result = find_best_vendor_match(
        "Acme Servces GmbH",
        vendors,
    )

    second_result = find_best_vendor_match(
        "Acme Servces GmbH",
        vendors,
    )

    third_result = find_best_vendor_match(
        "Acme Servces GmbH",
        vendors,
    )

    assert first_result.model_dump() == second_result.model_dump()
    assert second_result.model_dump() == third_result.model_dump()
