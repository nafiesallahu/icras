import json
from pathlib import Path
import pytest

from app.agents.counterparty_agent import (
    COUNTERPARTY_NOT_FOUND_FLAG,
    COUNTERPARTY_MISMATCH_FLAG,
    DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD,
    HIGH_RISK_COUNTERPARTY_FLAG,
    MANUAL_REVIEW_REQUIRED_FLAG,
    NEW_COUNTERPARTY_FLAG,
    classify_counterparty_status,
    compare_counterparty_names,
    detect_counterparty_mismatch_from_run,
    read_counterparty_names,
    NORMALIZED_COUNTERPARTY_FILENAME,
    write_normalized_counterparty,
    CounterpartyInputError,
    main,
    run_counterparty_agent,
)
from app.schemas.finding import (
    FindingRiskLevel,
    Severity,
    UnifiedFinding,
)
from app.schemas.normalized_counterparty import NormalizedCounterparty
from app.services.fuzzy_matcher import (
    FuzzyMatchResult,
    VendorRecord,
    normalize_company_name,
)


def write_json(
    path: Path,
    payload: dict,
) -> None:
    """
    Write a temporary JSON artifact for an Agent C unit test.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    content: str,
) -> None:
    """
    Write a temporary vendor-master CSV for Agent C tests.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
    )


def create_complete_agent_c_run(
    tmp_path: Path,
    *,
    include_evidence_index: bool = True,
) -> Path:
    """
    Create a complete temporary run directory for Agent C.
    """

    run_dir = tmp_path / "runs" / "run_001"

    write_json(
        run_dir / "context_packet.json",
        {
            "run_id": "run_001",
            "counterparty_name_from_manifest": ("Acme Services GmbH"),
        },
    )

    write_json(
        run_dir / "extracted_contract.json",
        {"parties": {"counterparty_name": ("Acme Services Gmbh")}},
    )

    write_csv(
        (run_dir / "input_snapshot" / "vendor_master.csv"),
        (
            "vendor_id,vendor_name,country,"
            "status,risk_level\n"
            "V001,Acme Services GmbH,Germany,"
            "approved,low\n"
        ),
    )

    if include_evidence_index:
        write_json(
            run_dir / "evidence_index.json",
            {
                "documents": [],
                "evidence_items": [
                    {
                        "evidence_id": "EV-001",
                        "text_excerpt": ("Acme Services GmbH"),
                    }
                ],
            },
        )

    write_json(
        run_dir / "metrics.json",
        {
            "run_id": "run_001",
            "agents": {"agent_a": {"status": "completed"}},
        },
    )

    (run_dir / "audit_log.md").write_text(
        "# ICRAS Audit Log\n",
        encoding="utf-8",
    )

    return run_dir


def create_match_result(
    vendor: VendorRecord,
    *,
    input_name: str,
    match_score: int,
    accepted: bool,
) -> FuzzyMatchResult:
    """
    Create a fuzzy-match result for classification tests.

    For unknown cases, the vendor remains the best_candidate,
    but matched_vendor is None because the threshold was not met.
    """

    return FuzzyMatchResult(
        input_counterparty_name=input_name,
        normalized_input_name=normalize_company_name(input_name),
        best_candidate=vendor,
        matched_vendor=vendor if accepted else None,
        match_score=match_score,
        threshold=85,
        match_status="matched" if accepted else "unknown",
    )


def test_matching_counterparty_names_do_not_create_finding() -> None:
    """
    Identical names must not produce a mismatch finding.
    """

    result = compare_counterparty_names(
        "Acme Services GmbH",
        "Acme Services GmbH",
    )

    assert result.similarity_score == 100

    assert result.mismatch_threshold == DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD

    assert result.is_mismatch is False
    assert result.finding is None


def test_minor_counterparty_variations_do_not_create_finding() -> None:
    """
    Casing, surrounding spaces, and punctuation differences
    must not cause a false mismatch.
    """

    result = compare_counterparty_names(
        "  ACME Services GmbH.  ",
        "Acme Services Gmbh",
    )

    # Both names normalize to "acme services gmbh".
    assert result.similarity_score == 100
    assert result.is_mismatch is False
    assert result.finding is None


def test_material_counterparty_mismatch_creates_finding() -> None:
    """
    Materially different company names must produce
    a counterparty_mismatch finding.
    """

    result = compare_counterparty_names(
        "Acme Services GmbH",
        "Global Data Ltd",
    )

    assert result.similarity_score < (DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD)

    assert result.is_mismatch is True
    assert result.finding is not None
    assert isinstance(result.finding, UnifiedFinding)

    # Verify the finding follows the current unified structure.
    assert result.finding.finding_id == "CP-MISMATCH-001"
    assert result.finding.field == "counterparty"
    assert result.finding.severity == Severity.HIGH

    assert "counterparty_mismatch" in result.finding.message

    assert result.finding.policy_rule == "counterparty.names_must_match"

    assert result.finding.expected_value == "Acme Services GmbH"

    assert result.finding.actual_value == "Global Data Ltd"

    assert result.finding.recommendation == (
        "Manually review the contract and manifest "
        "counterparty names before approval."
    )


def test_mismatch_severity_can_be_configured_as_medium() -> None:
    """
    Configuration may classify the material mismatch as medium
    instead of high.
    """

    result = compare_counterparty_names(
        "Acme Services GmbH",
        "Global Data Ltd",
        mismatch_severity=Severity.MEDIUM,
    )

    assert result.is_mismatch is True
    assert result.finding is not None
    assert result.finding.severity == Severity.MEDIUM


def test_agent_reads_counterparty_names_from_run_files(
    tmp_path: Path,
) -> None:
    """
    Agent C must read the manifest counterparty from context_packet.json
    and the extracted counterparty from extracted_contract.json.
    """

    run_dir = tmp_path / "runs" / "run_001"

    write_json(
        run_dir / "context_packet.json",
        {"counterparty_name_from_manifest": ("Acme Services GmbH")},
    )

    write_json(
        run_dir / "extracted_contract.json",
        {"parties": {"counterparty_name": "Global Data Ltd"}},
    )

    (
        manifest_counterparty_name,
        extracted_counterparty_name,
    ) = read_counterparty_names(run_dir)

    assert manifest_counterparty_name == "Acme Services GmbH"

    assert extracted_counterparty_name == "Global Data Ltd"

    # Confirm the values read from disk are used by
    # the actual mismatch detector.
    result = detect_counterparty_mismatch_from_run(run_dir)

    assert result.manifest_counterparty_name == ("Acme Services GmbH")

    assert result.extracted_counterparty_name == ("Global Data Ltd")

    assert result.is_mismatch is True
    assert result.finding is not None


def test_approved_vendor_returns_approved_status() -> None:
    """
    An accepted vendor with status approved must be classified
    as approved and retain its official vendor information.
    """

    vendor = VendorRecord(
        vendor_id="V001",
        vendor_name="Acme Services GmbH",
        country="Germany",
        status="approved",
        risk_level="low",
    )

    match_result = create_match_result(
        vendor,
        input_name="Acme Services Gmbh",
        match_score=97,
        accepted=True,
    )

    result = classify_counterparty_status(
        match_result,
        manifest_counterparty_name="Acme Services GmbH",
        extracted_counterparty_name="Acme Services GmbH",
    )

    assert isinstance(result, NormalizedCounterparty)

    assert result.matched_vendor_id == "V001"
    assert result.matched_vendor_name == "Acme Services GmbH"
    assert result.match_score == 97

    assert result.status == "approved"
    assert result.risk_level == "low"
    assert result.flags == []


def test_new_vendor_returns_new_status_and_flag() -> None:
    """
    An accepted vendor with status new must be classified as new
    and include the new_counterparty flag.
    """

    vendor = VendorRecord(
        vendor_id="V002",
        vendor_name="New Technology Ltd",
        country="United Kingdom",
        status="new",
        risk_level="medium",
    )

    match_result = create_match_result(
        vendor,
        input_name="New Technology Limited",
        match_score=100,
        accepted=True,
    )

    result = classify_counterparty_status(match_result)

    assert result.matched_vendor_id == "V002"
    assert result.matched_vendor_name == "New Technology Ltd"
    assert result.match_score == 100

    assert result.status == "new"
    assert result.risk_level == "medium"

    assert NEW_COUNTERPARTY_FLAG in result.flags
    assert MANUAL_REVIEW_REQUIRED_FLAG in result.flags


def test_unknown_counterparty_returns_null_vendor_fields() -> None:
    """
    A best candidate below the matching threshold must not be
    treated as an accepted vendor.
    """

    closest_candidate = VendorRecord(
        vendor_id="V001",
        vendor_name="Acme Services GmbH",
        country="Germany",
        status="approved",
        risk_level="low",
    )

    match_result = create_match_result(
        closest_candidate,
        input_name="Zebra Quantum Mining PLC",
        match_score=28,
        accepted=False,
    )

    result = classify_counterparty_status(match_result)

    # The low-scoring candidate must not be exposed as a match.
    assert result.matched_vendor_id is None
    assert result.matched_vendor_name is None

    # The similarity score is still retained for audit.
    assert result.match_score == 28

    assert result.status == "unknown"
    assert result.risk_level == "high"

    assert COUNTERPARTY_NOT_FOUND_FLAG in result.flags
    assert MANUAL_REVIEW_REQUIRED_FLAG in result.flags

    # The finding must be included in NormalizedCounterparty.
    assert len(result.findings) == 1

    finding = result.findings[0]

    assert finding.finding_id == "CP-UNKNOWN-001"
    assert finding.source_agent == "agent_c"
    assert finding.category == "counterparty"
    assert finding.severity == Severity.HIGH
    assert finding.risk_level == FindingRiskLevel.HIGH
    assert finding.score == 80
    assert finding.finding_type == "unknown_counterparty"
    assert finding.clause_id is None
    assert finding.policy_rule == "vendor_master.match_required"
    assert finding.expected is not None
    assert finding.actual is not None
    assert finding.recommendation is not None
    assert finding.evidence_ids == []
    assert finding.requires_human_review is True


def test_high_risk_vendor_returns_high_risk_status_and_flag() -> None:
    """
    A matched vendor with risk_level high must override its
    approved/new status and be classified as high_risk.
    """

    vendor = VendorRecord(
        vendor_id="V003",
        vendor_name="Unknown Offshore LLC",
        country="HighRiskCountryX",
        status="new",
        risk_level="high",
    )

    match_result = create_match_result(
        vendor,
        input_name="Unknown Offshore LLC",
        match_score=100,
        accepted=True,
    )

    result = classify_counterparty_status(match_result)

    assert result.matched_vendor_id == "V003"
    assert result.matched_vendor_name == "Unknown Offshore LLC"

    # High risk takes priority over the vendor's "new" status.
    assert result.status == "high_risk"
    assert result.risk_level == "high"

    assert HIGH_RISK_COUNTERPARTY_FLAG in result.flags
    assert MANUAL_REVIEW_REQUIRED_FLAG in result.flags
    assert COUNTERPARTY_NOT_FOUND_FLAG not in result.flags

    # The finding must be included in NormalizedCounterparty.
    assert len(result.findings) == 1

    finding = result.findings[0]

    assert finding.finding_id == "CP-HIGH-RISK-001"
    assert finding.source_agent == "agent_c"
    assert finding.category == "counterparty"
    assert finding.severity == Severity.HIGH
    assert finding.risk_level == FindingRiskLevel.HIGH
    assert finding.score == 90
    assert finding.finding_type == "high_risk_counterparty"
    assert finding.clause_id is None
    assert finding.policy_rule == "vendor_master.high_risk_review"
    assert finding.expected is not None
    assert finding.actual is not None
    assert finding.recommendation is not None
    assert finding.evidence_ids == []
    assert finding.requires_human_review is True


def test_high_risk_vendor_status_triggers_high_risk_classification() -> None:
    """
    A blocked or sanctioned vendor must be high risk even when
    its risk_level field is not explicitly high.
    """

    vendor = VendorRecord(
        vendor_id="V004",
        vendor_name="Blocked Vendor Ltd",
        country="Germany",
        status="blocked",
        risk_level="medium",
    )

    match_result = create_match_result(
        vendor,
        input_name="Blocked Vendor Ltd",
        match_score=100,
        accepted=True,
    )

    result = classify_counterparty_status(match_result)

    assert result.status == "high_risk"
    assert result.risk_level == "high"
    assert HIGH_RISK_COUNTERPARTY_FLAG in result.flags


def test_counterparty_mismatch_flag_and_finding_are_in_normalized_output() -> None:
    """
    A material manifest-versus-contract mismatch must be preserved
    as a finding and exposed through the counterparty_mismatch flag.
    """

    comparison = compare_counterparty_names(
        "Acme Services GmbH",
        "Global Data Ltd",
    )

    assert comparison.finding is not None

    matched_vendor = VendorRecord(
        vendor_id="V002",
        vendor_name="Global Data Ltd",
        country="United Kingdom",
        status="approved",
        risk_level="low",
    )

    match_result = create_match_result(
        matched_vendor,
        input_name="Global Data Ltd",
        match_score=100,
        accepted=True,
    )

    result = classify_counterparty_status(
        match_result,
        manifest_counterparty_name="Acme Services GmbH",
        extracted_counterparty_name="Global Data Ltd",
        findings=[comparison.finding],
    )

    assert COUNTERPARTY_MISMATCH_FLAG in result.flags
    assert MANUAL_REVIEW_REQUIRED_FLAG in result.flags

    mismatch_findings = [
        finding
        for finding in result.findings
        if finding.finding_type == "counterparty_mismatch"
    ]

    assert len(mismatch_findings) == 1

    finding = mismatch_findings[0]

    assert finding.source_agent == "agent_c"
    assert finding.category == "counterparty"
    assert finding.severity == Severity.HIGH
    assert finding.risk_level == FindingRiskLevel.HIGH
    assert finding.score == 85
    assert finding.requires_human_review is True


def test_agent_c_finding_contains_all_required_unified_fields() -> None:
    """
    Every Agent C finding serialized into normalized_counterparty.json
    must contain the required unified finding keys.
    """

    vendor = VendorRecord(
        vendor_id="V003",
        vendor_name="Risk Vendor LLC",
        country="HighRiskCountryX",
        status="approved",
        risk_level="high",
    )

    match_result = create_match_result(
        vendor,
        input_name="Risk Vendor LLC",
        match_score=100,
        accepted=True,
    )

    result = classify_counterparty_status(match_result)

    serialized = result.model_dump(mode="json")

    assert len(serialized["findings"]) == 1

    finding_data = serialized["findings"][0]

    required_fields = {
        "finding_id",
        "source_agent",
        "category",
        "severity",
        "risk_level",
        "score",
        "finding_type",
        "clause_id",
        "policy_rule",
        "expected",
        "actual",
        "recommendation",
        "evidence_ids",
        "requires_human_review",
    }

    assert required_fields.issubset(finding_data.keys())

    assert finding_data["source_agent"] == "agent_c"
    assert finding_data["category"] == "counterparty"


def test_findings_are_written_to_normalized_counterparty_json(
    tmp_path: Path,
) -> None:
    """
    Agent C findings must be physically included in the final
    normalized_counterparty.json artifact.
    """

    vendor = VendorRecord(
        vendor_id="V003",
        vendor_name="Risk Vendor LLC",
        country="HighRiskCountryX",
        status="approved",
        risk_level="high",
    )

    match_result = create_match_result(
        vendor,
        input_name="Risk Vendor LLC",
        match_score=100,
        accepted=True,
    )

    result = classify_counterparty_status(match_result)

    output_path = write_normalized_counterparty(
        tmp_path,
        result,
    )

    assert output_path.name == "normalized_counterparty.json"
    assert output_path.is_file()

    written_data = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(written_data["findings"]) == 1

    finding_data = written_data["findings"][0]

    assert finding_data["finding_type"] == ("high_risk_counterparty")
    assert finding_data["source_agent"] == "agent_c"
    assert finding_data["category"] == "counterparty"
    assert finding_data["requires_human_review"] is True


def test_run_counterparty_agent_executes_full_flow(
    tmp_path: Path,
) -> None:
    """
    Agent C must read all inputs, resolve the counterparty,
    write its output, and update audit and metrics files.
    """

    run_dir = create_complete_agent_c_run(tmp_path)

    result = run_counterparty_agent(run_dir)

    assert isinstance(
        result,
        NormalizedCounterparty,
    )

    assert result.status == "approved"
    assert result.matched_vendor_id == "V001"
    assert result.matched_vendor_name == ("Acme Services GmbH")
    assert result.match_score == 100

    output_path = run_dir / "normalized_counterparty.json"

    assert output_path.is_file()

    # Validate the exact JSON written to disk.
    written_result = NormalizedCounterparty.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )

    assert written_result == result

    audit_content = (run_dir / "audit_log.md").read_text(encoding="utf-8")

    assert "Agent C — Step 3: Counterparty Resolution" in audit_content

    assert "counterparty_resolution: completed" in audit_content
    assert "input_name: Acme Services Gmbh" in audit_content
    assert "manifest_name: Acme Services GmbH" in audit_content
    assert "extracted_name: Acme Services Gmbh" in audit_content
    assert "matched_vendor_id: V001" in audit_content
    assert "matched_vendor: Acme Services GmbH" in audit_content
    assert "match_score: 100" in audit_content
    assert "status: approved" in audit_content
    assert "flags: none" in audit_content
    assert "human_review_required: false" in audit_content

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    # Existing Agent A metrics must remain.
    assert metrics["agents"]["agent_a"]["status"] == "completed"

    agent_c_metrics = metrics["agents"]["agent_c"]

    assert agent_c_metrics["status"] == "completed"

    assert agent_c_metrics["counterparty_status"] == "approved"

    assert agent_c_metrics["evidence_index_available"] is True

    assert agent_c_metrics["evidence_item_count"] == 1

    assert agent_c_metrics["counterparty_resolution"] == "completed"

    assert agent_c_metrics["match_score"] == 100
    assert agent_c_metrics["flags_count"] == 0

    assert agent_c_metrics["human_review_required"] is False


def test_run_counterparty_agent_allows_missing_evidence_index(
    tmp_path: Path,
) -> None:
    """
    Agent C must continue when evidence_index.json is absent.
    """

    run_dir = create_complete_agent_c_run(
        tmp_path,
        include_evidence_index=False,
    )

    result = run_counterparty_agent(run_dir)

    assert result.status == "approved"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    agent_c_metrics = metrics["agents"]["agent_c"]

    assert agent_c_metrics["evidence_index_available"] is False

    assert agent_c_metrics["evidence_item_count"] == 0


def test_run_counterparty_agent_fails_clearly_for_missing_input(
    tmp_path: Path,
) -> None:
    """
    Missing extracted_contract.json must cause a clear failure
    and update audit_log.md and metrics.json.
    """

    run_dir = tmp_path / "runs" / "run_missing_input"

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # context_packet.json exists.
    write_json(
        run_dir / "context_packet.json",
        {"counterparty_name_from_manifest": ("Acme Services GmbH")},
    )

    # vendor_master.csv exists.
    write_csv(
        run_dir / "input_snapshot" / "vendor_master.csv",
        (
            "vendor_id,vendor_name,country,status,risk_level\n"
            "V001,Acme Services GmbH,Germany,approved,low\n"
        ),
    )

    # extracted_contract.json is intentionally missing.
    with pytest.raises(
        CounterpartyInputError,
        match="extracted_contract.json was not found",
    ):
        run_counterparty_agent(run_dir)

    # Verify the failure was written to audit_log.md.
    audit_content = (run_dir / "audit_log.md").read_text(encoding="utf-8")

    assert "Agent C — Step 3: Counterparty Resolution" in audit_content
    assert "counterparty_resolution: failed" in audit_content
    assert "status: failed" in audit_content
    assert "human_review_required: true" in audit_content
    assert "CounterpartyInputError" in audit_content
    assert "extracted_contract.json" in audit_content

    # Verify the failure was written to metrics.json.
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    agent_c_metrics = metrics["agents"]["agent_c"]

    assert agent_c_metrics["status"] == "failed"

    assert agent_c_metrics["counterparty_resolution"] == "failed"

    assert agent_c_metrics["match_score"] is None
    assert agent_c_metrics["flags_count"] == 0

    assert agent_c_metrics["human_review_required"] is True

    assert agent_c_metrics["error_status"] == "CounterpartyInputError"

    assert "extracted_contract.json" in agent_c_metrics["error_message"]

    assert metrics["status"] == "failed_at_agent_c"


def test_counterparty_agent_main_runs_with_directory_path(
    tmp_path: Path,
) -> None:
    run_dir = create_complete_agent_c_run(tmp_path)

    exit_code = main([str(run_dir)])

    assert exit_code == 0

    assert (run_dir / NORMALIZED_COUNTERPARTY_FILENAME).is_file()


def test_audit_and_metrics_record_human_review_for_high_risk_vendor(
    tmp_path: Path,
) -> None:
    """
    High-risk counterparties must record that human review
    is required in both audit_log.md and metrics.json.
    """

    run_dir = create_complete_agent_c_run(tmp_path)

    # Replace the normal low-risk vendor with a high-risk vendor.
    write_csv(
        run_dir / "input_snapshot" / "vendor_master.csv",
        (
            "vendor_id,vendor_name,country,status,risk_level\n"
            "V001,Acme Services GmbH,Germany,approved,high\n"
        ),
    )

    result = run_counterparty_agent(run_dir)

    assert result.status == "high_risk"

    assert MANUAL_REVIEW_REQUIRED_FLAG in result.flags

    audit_content = (run_dir / "audit_log.md").read_text(encoding="utf-8")

    assert "status: high_risk" in audit_content
    assert "flags: high_risk_counterparty, " "manual_review_required" in audit_content
    assert "human_review_required: true" in audit_content

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    agent_c_metrics = metrics["agents"]["agent_c"]

    assert agent_c_metrics["counterparty_resolution"] == "completed"
    assert agent_c_metrics["counterparty_status"] == "high_risk"
    assert agent_c_metrics["flags_count"] == 2
    assert agent_c_metrics["human_review_required"] is True


def test_counterparty_agent_output_is_deterministic(
    tmp_path: Path,
) -> None:
    """
    Identical Agent C inputs must produce identical normalized
    counterparty results and identical JSON output.
    """

    # Create two separate run directories containing
    # exactly the same synthetic input data.
    first_run_dir = create_complete_agent_c_run(tmp_path / "first")

    second_run_dir = create_complete_agent_c_run(tmp_path / "second")

    # Execute Agent C independently for both runs.
    first_result = run_counterparty_agent(first_run_dir)

    second_result = run_counterparty_agent(second_run_dir)

    # The validated Pydantic results must be identical.
    assert first_result == second_result

    first_output = (first_run_dir / NORMALIZED_COUNTERPARTY_FILENAME).read_text(
        encoding="utf-8"
    )

    second_output = (second_run_dir / NORMALIZED_COUNTERPARTY_FILENAME).read_text(
        encoding="utf-8"
    )

    # The actual JSON files must also be byte-for-byte identical.
    assert first_output == second_output

    # Confirm both outputs still validate against the schema.
    first_validated = NormalizedCounterparty.model_validate_json(first_output)

    second_validated = NormalizedCounterparty.model_validate_json(second_output)

    assert first_validated == second_validated
