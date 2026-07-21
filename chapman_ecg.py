"""
explore_dataset.py
Step 1: Explore the Chapman-Shaoxing ECG Dataset

Run:
python explore_dataset.py
"""

import random
from pathlib import Path
from collections import Counter

import pandas as pd
import wfdb

# ==========================================================
# CHANGE THIS PATH TO YOUR DATASET LOCATION
# ==========================================================
DATA_PATH = Path(
    r"C:\Users\anany\Documents\major project\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
)

WFDB_RECORDS_DIR = DATA_PATH / "WFDBRecords"
CONDITIONS_CSV = DATA_PATH / "ConditionNames_SNOMED-CT.csv"
RECORDS_FILE = DATA_PATH / "RECORDS"


# ==========================================================
# Verify dataset
# ==========================================================
def check_paths():
    print("=" * 70)
    print("STEP 1 : VERIFYING DATASET")
    print("=" * 70)

    paths = [
        ("Dataset", DATA_PATH),
        ("WFDBRecords", WFDB_RECORDS_DIR),
        ("ConditionNames_SNOMED-CT.csv", CONDITIONS_CSV),
        ("RECORDS", RECORDS_FILE),
    ]

    for name, path in paths:
        if path.exists():
            print(f"✓ {name}")
        else:
            print(f"✗ {name} NOT FOUND")

    if not WFDB_RECORDS_DIR.exists():
        raise FileNotFoundError(
            "\nWFDBRecords folder not found.\nPlease check DATA_PATH."
        )


# ==========================================================
# Read RECORDS file
# ==========================================================
# ==========================================================
# Find all ECG records
# ==========================================================
def get_record_list():

    print("\n" + "=" * 70)
    print("STEP 2 : SCANNING FOR ECG RECORDS")
    print("=" * 70)

    record_paths = []

    # Search recursively for every .hea file
    for hea_file in WFDB_RECORDS_DIR.rglob("*.hea"):
        record_paths.append(hea_file.with_suffix(""))

    record_paths = sorted(record_paths)

    print(f"Found {len(record_paths)} ECG records.")

    if len(record_paths) == 0:
        raise FileNotFoundError(
            "No .hea files were found inside WFDBRecords."
        )

    return record_paths

# ==========================================================
# Read one sample record
# ==========================================================
def inspect_record(record_paths):

    print("\n" + "=" * 70)
    print("STEP 2 : SAMPLE RECORD")
    print("=" * 70)

    sample = random.choice(record_paths)
    print("\nSelected ECG Record:")
    print(sample)

    record = wfdb.rdrecord(str(sample))

    print("\nRecord Name :", record.record_name)
    print("Sampling Frequency :", record.fs)
    print("Signal Length :", record.sig_len)
    print("Number of Leads :", record.n_sig)
    print("Lead Names :", record.sig_name)
    print("Signal Shape :", record.p_signal.shape)

    print("\nComments:")
    for c in record.comments:
        print("  ", c)

    return record


# ==========================================================
# Check Patient ID
# ==========================================================
def check_patient_id(record):

    print("\n" + "=" * 70)
    print("STEP 3 : PATIENT ID CHECK")
    print("=" * 70)

    found = False

    for c in record.comments:

        if (
            "patient" in c.lower()
            or "subject" in c.lower()
            or "id" in c.lower()
        ):
            found = True
            print(c)

    if not found:
        print("No explicit patient ID found.")
        print("Need to verify whether each record corresponds to one patient.")


# ==========================================================
# Load SNOMED Mapping
# ==========================================================
def load_conditions():

    print("\n" + "=" * 70)
    print("STEP 4 : CONDITION TABLE")
    print("=" * 70)

    df = pd.read_csv(CONDITIONS_CSV)

    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ==========================================================
# Extract Diagnosis Codes
# ==========================================================
def extract_dx(comments):

    for c in comments:

        text = c.strip().replace("#", "")

        if text.lower().startswith("dx"):

            codes = text.split(":")[1]

            return [x.strip() for x in codes.split(",")]

    return []


# ==========================================================
# Distribution
# ==========================================================
def class_distribution(record_paths, condition_df, max_records=500):

    print("\n" + "=" * 70)
    print("STEP 5 : CLASS DISTRIBUTION")
    print("=" * 70)

    print(f"Scanning first {max_records} records...\n")

    code_to_name = {}

    if (
        "Snomed_CT" in condition_df.columns
        and "Full Name" in condition_df.columns
    ):
        code_to_name = dict(
            zip(
                condition_df["Snomed_CT"].astype(str),
                condition_df["Full Name"],
            )
        )

    counter = Counter()

    for path in record_paths[:max_records]:

        hea = str(path) + ".hea"

        try:

            with open(hea, "r") as f:
                lines = f.readlines()

            comments = [
                line.strip().replace("#", "")
                for line in lines
                if line.startswith("#")
            ]

            dx = extract_dx(comments)

            for code in dx:
                counter[code] += 1

        except:
            pass

    print("Unique Diagnosis Codes :", len(counter))

    print("\nTop 15 Diagnoses\n")

    for code, count in counter.most_common(15):

        name = code_to_name.get(code, "UNKNOWN")

        print(f"{code:<15} {count:<8} {name}")


# ==========================================================
# Main
# ==========================================================
def main():

    check_paths()

    record_paths = get_record_list()

    print("\nTotal Records :", len(record_paths))

    record = inspect_record(record_paths)

    check_patient_id(record)

    conditions = load_conditions()

    class_distribution(
        record_paths,
        conditions,
        max_records=500,
    )


if __name__ == "__main__":
    main()