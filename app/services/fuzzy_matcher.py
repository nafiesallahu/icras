from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

REQUIRED_VENDOR_COLUMNS = (
    "vendor_id",
    "vendor_name",
    "country",
    "status",
    "risk_level",
)

# Common legal suffixes that may be written in different but equivalent forms.
#
# The mapping is intentionally conservative:
# - it only changes the final word of a company name;
# - it does not remove the legal suffix;
# - it does not remove other meaningful company-name words.
#
# Examples:
# "Acme Limited" -> "acme ltd"
# "Acme Ltd."    -> "acme ltd"
LEGAL_SUFFIX_ALIASES: dict[str, str] = {
    "limited": "ltd",
    "ltd": "ltd",
    "incorporated": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "company": "co",
    "co": "co",
    "llc": "llc",
    "plc": "plc",
    "gmbh": "gmbh",
}


# Periods and apostrophes are removed instead of replaced with spaces.
#
# This allows abbreviations such as:
# "L.L.C." -> "llc"
# "S.A."   -> "sa"
# "O'Reilly" -> "oreilly"
PUNCTUATION_TO_REMOVE = {
    ".",
    "'",
    "’",
    "`",
}

# Locate the root directory of the repository.
#
# fuzzy_matcher.py is located at:
# app/services/fuzzy_matcher.py
#
# parents[0] = services
# parents[1] = app
# parents[2] = repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# This is the default vendor master maintained under the policy directory.
DEFAULT_VENDOR_MASTER_PATH = PROJECT_ROOT / "policies" / "vendor_master.csv"


# Agent A copies this file into each run's input_snapshot directory.
VENDOR_MASTER_FILENAME = "vendor_master.csv"


# These columns must exist in every valid vendor_master.csv file.
#
# A tuple is used so the order is deterministic when creating
# structured VendorRecord objects.
REQUIRED_VENDOR_COLUMNS = (
    "vendor_id",
    "vendor_name",
    "country",
    "status",
    "risk_level",
)


def normalize_company_name(name: str) -> str:
    """
    Normalize a company name before fuzzy matching.

    The normalization process is deterministic and performs these steps:

    1. validates that the input is a non-empty string;
    2. applies Unicode compatibility normalization;
    3. converts text to lowercase using casefold();
    4. removes surrounding whitespace;
    5. converts ampersands to the word "and";
    6. removes periods and apostrophes;
    7. converts other punctuation into spaces;
    8. collapses duplicated whitespace;
    9. canonicalizes common legal suffix variations.

    Meaningful company-name words are preserved.
    """

    # Reject non-string values with a clear error.
    if not isinstance(name, str):
        raise TypeError("Company name must be provided as a string.")

    # NFKC converts visually equivalent Unicode forms into one
    # consistent representation.
    #
    # casefold() is stronger and more Unicode-aware than lower().
    normalized_name = unicodedata.normalize(
        "NFKC",
        name,
    ).casefold()

    # Remove spaces at the beginning and end.
    normalized_name = normalized_name.strip()

    if not normalized_name:
        raise ValueError("Company name must not be empty.")

    normalized_characters: list[str] = []

    for character in normalized_name:
        # Treat "&" as the word "and".
        #
        # This allows:
        # "Acme & Sons Ltd"
        # "Acme and Sons Limited"
        #
        # to normalize to the same value.
        if character == "&":
            normalized_characters.append(" and ")
            continue

        # Remove periods and apostrophes without inserting spaces.
        #
        # Example:
        # "L.L.C." becomes "llc", not "l l c".
        if character in PUNCTUATION_TO_REMOVE:
            continue

        # Unicode punctuation categories begin with "P".
        #
        # Commas, hyphens, slashes, brackets, and similar punctuation
        # are converted to spaces so adjacent meaningful words remain
        # separate.
        if unicodedata.category(character).startswith("P"):
            normalized_characters.append(" ")
            continue

        # Preserve letters, numbers, and existing whitespace.
        normalized_characters.append(character)

    normalized_name = "".join(normalized_characters)

    # split() without an argument removes leading/trailing whitespace
    # and collapses any number of spaces, tabs, or line breaks.
    normalized_name = " ".join(normalized_name.split())

    if not normalized_name:
        raise ValueError("Company name contains no usable characters.")

    # Split the normalized name into words.
    name_tokens = normalized_name.split()

    # Only canonicalize the final word because legal entity suffixes
    # normally appear at the end of company names.
    #
    # This avoids modifying the same word when it is part of the
    # company's meaningful name elsewhere.
    final_token = name_tokens[-1]

    name_tokens[-1] = LEGAL_SUFFIX_ALIASES.get(
        final_token,
        final_token,
    )

    return " ".join(name_tokens)


class VendorMasterError(Exception):
    """
    Base exception for all vendor-master loading problems.

    Agent C or the orchestrator can catch this exception and then:
    - fail the current run;
    - write the error to audit_log.md;
    - update metrics.json with an error status.
    """


class VendorMasterNotFoundError(VendorMasterError):
    """Raised when vendor_master.csv cannot be found."""


class VendorMasterSchemaError(VendorMasterError):
    """Raised when required CSV columns are missing or malformed."""


class EmptyVendorMasterError(VendorMasterError):
    """Raised when vendor_master.csv contains no vendor records."""


class VendorMasterRowError(VendorMasterError):
    """Raised when an individual vendor row contains invalid data."""


class VendorRecord(BaseModel):
    """
    Structured internal representation of one vendor-master row.

    The CSV loader converts every valid row into this model before
    the fuzzy-matching logic uses it.
    """

    model_config = ConfigDict(
        # Prevent undeclared fields from entering internal records.
        extra="forbid",
        # Vendor master records must not be reassigned after validation.
        frozen=True,
        # Require the correct primitive data types.
        strict=True,
        # Remove accidental whitespace surrounding CSV values.
        str_strip_whitespace=True,
    )

    # Unique official vendor identifier.
    vendor_id: str = Field(
        ...,
        min_length=1,
        description="Official vendor identifier.",
    )

    # Official vendor name used during fuzzy matching.
    vendor_name: str = Field(
        ...,
        min_length=1,
        description="Official vendor name.",
    )

    # Country registered in the vendor master.
    country: str = Field(
        ...,
        min_length=1,
        description="Vendor country.",
    )

    # Business status from the vendor master.
    #
    # Examples:
    # - approved
    # - new
    status: str = Field(
        ...,
        min_length=1,
        description="Vendor approval status.",
    )

    # Risk classification from the vendor master.
    #
    # Examples:
    # - low
    # - medium
    # - high
    risk_level: str = Field(
        ...,
        min_length=1,
        description="Vendor risk level.",
    )


def resolve_vendor_master_path(
    *,
    run_dir: str | Path | None = None,
    policy_path: str | Path | None = None,
) -> Path:
    """
    Find the vendor master file Agent C should use.

    Search order:

    1. runs/<run_id>/input_snapshot/vendor_master.csv
    2. policies/vendor_master.csv

    The run snapshot is preferred because it preserves the exact policy
    file used during that specific execution. This supports deterministic
    re-runs and auditability.
    """

    candidate_paths: list[Path] = []

    if run_dir is not None:
        # Build the expected path to Agent A's copied snapshot.
        snapshot_path = Path(run_dir) / "input_snapshot" / VENDOR_MASTER_FILENAME
        candidate_paths.append(snapshot_path)

    # Use a provided policy path during tests or configuration.
    # Otherwise, use the repository's default policies file.
    resolved_policy_path = (
        Path(policy_path) if policy_path is not None else DEFAULT_VENDOR_MASTER_PATH
    )
    candidate_paths.append(resolved_policy_path)

    # Return the first valid file according to the search order.
    for candidate_path in candidate_paths:
        if candidate_path.is_file():
            return candidate_path

    # Include every checked location in the error message.
    # This makes debugging and audit logging much clearer.
    checked_locations = ", ".join(str(path) for path in candidate_paths)

    raise VendorMasterNotFoundError(
        "vendor_master.csv was not found. " f"Checked locations: {checked_locations}"
    )


def load_vendor_master(
    csv_path: str | Path,
) -> list[VendorRecord]:
    """
    Load and validate vendor records from vendor_master.csv.

    The function:

    1. verifies that the file exists;
    2. reads the CSV header;
    3. verifies all required columns;
    4. validates every non-empty row;
    5. returns structured VendorRecord objects.

    The original CSV row order is preserved so matching behavior remains
    deterministic when two vendors receive the same similarity score.
    """

    vendor_master_path = Path(csv_path)

    # Fail immediately with a clear service-specific exception.
    if not vendor_master_path.is_file():
        raise VendorMasterNotFoundError(
            "vendor_master.csv was not found at: " f"{vendor_master_path}"
        )

    records: list[VendorRecord] = []

    try:
        # utf-8-sig supports standard UTF-8 CSV files and files that
        # contain a UTF-8 byte-order mark.
        with vendor_master_path.open(
            mode="r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            # strict=True makes malformed CSV syntax raise csv.Error.
            reader = csv.DictReader(
                csv_file,
                strict=True,
            )

            # fieldnames is None when the file is completely blank.
            if reader.fieldnames is None:
                raise EmptyVendorMasterError(
                    "vendor_master.csv is empty and has no header."
                )

            # Normalize spaces surrounding header names.
            normalized_headers = {
                header.strip() for header in reader.fieldnames if header is not None
            }

            required_column_set = set(REQUIRED_VENDOR_COLUMNS)

            # Calculate which required columns are absent.
            missing_columns = sorted(required_column_set - normalized_headers)

            if missing_columns:
                raise VendorMasterSchemaError(
                    "vendor_master.csv is missing required columns: "
                    + ", ".join(missing_columns)
                )

            # CSV row 1 is the header, so data starts at row 2.
            for row_number, raw_row in enumerate(
                reader,
                start=2,
            ):
                # DictReader uses a None key if a row contains more
                # values than the header defines.
                if None in raw_row:
                    raise VendorMasterRowError(
                        "Malformed vendor_master.csv row "
                        f"{row_number}: more values than columns."
                    )

                # Normalize header names and remove surrounding
                # whitespace from string values.
                normalized_row = {
                    key.strip(): (value.strip() if value is not None else "")
                    for key, value in raw_row.items()
                    if key is not None
                }

                # Ignore completely blank lines between vendor records.
                if not any(normalized_row.values()):
                    continue

                # Only pass the five required fields into VendorRecord.
                #
                # The CSV may contain additional optional columns,
                # but they do not affect this internal record yet.
                vendor_data = {
                    column: normalized_row.get(column, "")
                    for column in REQUIRED_VENDOR_COLUMNS
                }

                try:
                    # Convert the raw CSV dictionary into a validated,
                    # immutable internal vendor record.
                    record = VendorRecord(**vendor_data)
                except ValidationError as exc:
                    # Convert the Pydantic error into a clear
                    # service-level exception for the orchestrator.
                    error_details = "; ".join(
                        (
                            ".".join(
                                str(location_part) for location_part in error["loc"]
                            )
                            + ": "
                            + error["msg"]
                        )
                        for error in exc.errors()
                    )

                    raise VendorMasterRowError(
                        "Invalid vendor_master.csv data at row "
                        f"{row_number}: {error_details}"
                    ) from exc

                records.append(record)

    except UnicodeDecodeError as exc:
        raise VendorMasterSchemaError(
            "vendor_master.csv must use valid UTF-8 encoding."
        ) from exc

    except csv.Error as exc:
        raise VendorMasterSchemaError(
            "vendor_master.csv contains malformed CSV syntax: " f"{exc}"
        ) from exc

    except OSError as exc:
        raise VendorMasterError(
            "vendor_master.csv could not be read: "
            f"{vendor_master_path}. Error: {exc}"
        ) from exc

    # A file containing only its header is not a usable vendor master.
    if not records:
        raise EmptyVendorMasterError("vendor_master.csv contains no vendor records.")

    return records


def load_vendor_master_for_run(
    run_dir: str | Path | None = None,
    *,
    policy_path: str | Path | None = None,
) -> list[VendorRecord]:
    """
    Resolve and load the correct vendor master for an Agent C execution.

    If a run directory is supplied, the input_snapshot copy is preferred.
    If it does not exist, the service falls back to the policies copy.
    """

    vendor_master_path = resolve_vendor_master_path(
        run_dir=run_dir,
        policy_path=policy_path,
    )

    return load_vendor_master(vendor_master_path)
