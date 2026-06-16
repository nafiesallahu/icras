from pathlib import Path

import pytest

from app.services.fuzzy_matcher import (
    EmptyVendorMasterError,
    VendorMasterNotFoundError,
    VendorMasterRowError,
    VendorMasterSchemaError,
    VendorRecord,
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
