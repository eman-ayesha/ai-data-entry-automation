"""
Data Cleaning & Standardization Pipeline
----------------------------------------
Portfolio project: AI-Assisted Data Entry & Spreadsheet Automation

Purpose:
    Read a CSV file, clean and standardize common client/financial record
    fields, validate important data, remove duplicates, handle missing
    values, and save a clean CSV plus a validation/audit report.

Usage:
    python data_cleaning_pipeline.py input.csv
    python data_cleaning_pipeline.py input.csv cleaned_data.csv

The script is designed to work with common column names. It skips cleaning
rules for columns that are not present in the input file.

Dependencies:
    pandas
    numpy
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Common column-name variations -> one standardized internal name.
COLUMN_ALIASES = {
    "name": "Name",
    "full name": "Name",
    "client name": "Name",
    "customer name": "Name",
    "phone": "Phone",
    "phone number": "Phone",
    "mobile": "Phone",
    "mobile number": "Phone",
    "email": "Email",
    "email address": "Email",
    "city": "City",
    "area": "Area",
    "address": "Address",
    "status": "Status",
    "account type": "Account Type",
    "amount": "Amount",
    "date": "Date",
    "transaction date": "Date",
}


def normalize_column_name(column: str) -> str:
    """Normalize a column label for alias matching."""
    return re.sub(r"\s+", " ", str(column).strip().lower())


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column labels and standardize common aliases."""
    renamed = {}
    for column in df.columns:
        key = normalize_column_name(column)
        renamed[column] = COLUMN_ALIASES.get(key, str(column).strip())

    df = df.rename(columns=renamed)

    # Make duplicate column labels unique instead of silently overwriting data.
    seen = {}
    unique_columns = []
    for column in df.columns:
        count = seen.get(column, 0)
        unique_columns.append(column if count == 0 else f"{column}_{count + 1}")
        seen[column] = count + 1

    df.columns = unique_columns
    return df


def clean_text_column(series: pd.Series) -> pd.Series:
    """Trim whitespace and collapse repeated spaces."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def standardize_names(df: pd.DataFrame) -> None:
    if "Name" in df.columns:
        df["Name"] = clean_text_column(df["Name"]).str.title()


def standardize_locations(df: pd.DataFrame) -> None:
    for column in ("City", "Area"):
        if column in df.columns:
            df[column] = clean_text_column(df[column]).str.title()


def standardize_email(df: pd.DataFrame) -> None:
    if "Email" in df.columns:
        df["Email"] = clean_text_column(df["Email"]).str.lower()


def standardize_phone(df: pd.DataFrame) -> None:
    """
    Standardize Pakistani-style 03XXXXXXXXX phone numbers when possible.
    Other phone formats are preserved as cleaned text rather than guessed.
    """
    if "Phone" not in df.columns:
        return

    raw = clean_text_column(df["Phone"])
    digits = raw.str.replace(r"\D", "", regex=True)

    # 03001234567 -> 0300-1234567
    pakistani_mask = digits.str.fullmatch(r"03\d{9}", na=False)
    standardized = raw.copy()
    standardized.loc[pakistani_mask] = (
        digits.loc[pakistani_mask].str.slice(0, 4)
        + "-"
        + digits.loc[pakistani_mask].str.slice(4)
    )

    df["Phone"] = standardized


def standardize_status(df: pd.DataFrame) -> None:
    if "Status" not in df.columns:
        return

    df["Status"] = clean_text_column(df["Status"]).str.title()

    status_map = {
        "Active": "Active",
        "Inactive": "Inactive",
        "Pending": "Pending",
        "Complete": "Completed",
        "Completed": "Completed",
        "Complete ": "Completed",
    }
    df["Status"] = df["Status"].replace(status_map)


def standardize_amount(df: pd.DataFrame) -> None:
    if "Amount" not in df.columns:
        return

    # Convert values such as "$1,250.50" to numeric 1250.50.
    cleaned = (
        df["Amount"]
        .astype("string")
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace("", pd.NA)
    )
    df["Amount"] = pd.to_numeric(cleaned, errors="coerce")


def standardize_dates(df: pd.DataFrame) -> None:
    if "Date" not in df.columns:
        return

    parsed = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = parsed.dt.strftime("%Y-%m-%d")
    df.loc[parsed.isna(), "Date"] = pd.NA


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill text fields with 'Unknown' and numeric Amount with the median.
    Missing dates are left blank because inventing a date would compromise
    data integrity.
    """
    text_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        # Keep Date blank when it could not be parsed.
        if column != "Date":
            df[column] = df[column].fillna("Unknown")

    if "Amount" in df.columns:
        median_amount = df["Amount"].median()
        if pd.notna(median_amount):
            df["Amount"] = df["Amount"].fillna(median_amount)

    return df


def validate_records(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Add a ValidationStatus column and flag obvious data-quality problems.
    The original record is retained so questionable data can be reviewed.
    """
    issues = pd.Series("", index=df.index, dtype="string")

    if "Name" in df.columns:
        invalid_name = df["Name"].isin(["", "Unknown", pd.NA])
        issues.loc[invalid_name.fillna(False)] += "Missing name; "

    if "Email" in df.columns:
        email = df["Email"].astype("string")
        invalid_email = (
            (email != "Unknown")
            & ~email.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False)
        )
        issues.loc[invalid_email] += "Invalid email; "

    if "Phone" in df.columns:
        phone = df["Phone"].astype("string")
        invalid_phone = (
            (phone != "Unknown")
            & ~phone.str.match(r"^03\d{2}-\d{7}$", na=False)
        )
        issues.loc[invalid_phone] += "Review phone format; "

    if "Amount" in df.columns:
        invalid_amount = df["Amount"].notna() & (df["Amount"] < 0)
        issues.loc[invalid_amount] += "Negative amount; "

    df["ValidationStatus"] = np.where(
        issues.str.len().eq(0),
        "Valid",
        "Review: " + issues.str.rstrip("; "),
    )

    report = {
        "records_after_cleaning": int(len(df)),
        "valid_records": int((df["ValidationStatus"] == "Valid").sum()),
        "records_needing_review": int(
            (df["ValidationStatus"] != "Valid").sum()
        ),
    }

    return df, report


def clean_dataset(input_path: Path, output_path: Path) -> dict:
    """Run the complete cleaning pipeline."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() != ".csv":
        raise ValueError("Input file must be a CSV file.")

    df = pd.read_csv(input_path)
    original_rows = len(df)

    df = standardize_column_names(df)

    # Treat whitespace-only cells as missing.
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Remove exact duplicate records.
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Standardization steps.
    standardize_names(df)
    standardize_locations(df)
    standardize_email(df)
    standardize_phone(df)
    standardize_status(df)
    standardize_amount(df)
    standardize_dates(df)

    df = handle_missing_values(df)

    # Validate after cleaning.
    df, validation = validate_records(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_rows": original_rows,
        "duplicates_removed": duplicate_count,
        "rows_saved": len(df),
        "columns": len(df.columns),
        **validation,
    }

    return report


def print_report(report: dict) -> None:
    print("\n" + "=" * 55)
    print("DATA CLEANING & VALIDATION REPORT")
    print("=" * 55)

    for key, value in report.items():
        label = key.replace("_", " ").title()
        print(f"{label}: {value}")

    print("=" * 55)
    print("Cleaning completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and standardize a CSV dataset."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        help="Path for the cleaned CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.with_name(f"{input_path.stem}_cleaned.csv")
    )

    try:
        report = clean_dataset(input_path, output_path)
        print_report(report)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"\nERROR: {exc}")


if __name__ == "__main__":
    main()
