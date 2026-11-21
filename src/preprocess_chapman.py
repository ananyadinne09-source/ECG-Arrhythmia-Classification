"""
preprocess_chapman.py
Step 2: Preprocessing pipeline for the Chapman-Shaoxing 12-lead ECG dataset.

Reads raw WFDB records (.hea + .mat), extracts signal + primary diagnosis,
normalizes signals, encodes labels, and saves them as .npy/.pkl arrays
ready for PyTorch Dataset/DataLoader consumption (Step 3).
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION — EDIT THESE PATHS FOR YOUR LOCAL MACHINE
# ------------------------------------------------------------------
DATA_PATH = Path(
    r"C:\Users\anany\Documents\major project\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
)

WFDB_RECORDS_DIR = DATA_PATH / "WFDBRecords"
CONDITIONS_CSV = DATA_PATH / "ConditionNames_SNOMED-CT.csv"
RECORDS_FILE = DATA_PATH / "RECORDS"

OUTPUT_DIR = Path(r"C:\Users\anany\Documents\major project\processed")
TARGET_LEN = 5000       # 10 seconds at 500 Hz
NUM_LEADS = 12


# ------------------------------------------------------------------
# 1. Load SNOMED-CT code -> diagnosis name mapping
# ------------------------------------------------------------------
def load_condition_mapping(csv_path):
    """
    Reads ConditionNames_SNOMED-CT.csv and returns a dict:
        { "164889003": "Atrial fibrillation", ... }
    Codes are stored as strings to match how they appear in .hea files.
    """
    df = pd.read_csv(csv_path)
    code_to_name = dict(zip(df["Snomed_CT"].astype(str), df["Full Name"]))
    return code_to_name


# ------------------------------------------------------------------
# 2. Collect all record paths (no extension)
# ------------------------------------------------------------------
def get_record_paths(data_path, records_file):
    """
    Scan the dataset recursively for every ECG record (.hea file).
    Returns record paths WITHOUT the .hea extension.
    """

    print("\nScanning dataset for ECG records...")

    wfdb_dir = data_path / "WFDBRecords"

    record_paths = []

    for hea_file in wfdb_dir.rglob("*.hea"):
        record_paths.append(hea_file.with_suffix(""))

    record_paths = sorted(record_paths)

    print(f"Found {len(record_paths)} ECG records.")

    return record_paths


# ------------------------------------------------------------------
# 3. Parse Dx code string out of header comments
# ------------------------------------------------------------------
def parse_header_fields(comments):
    """
    Extracts Age, Sex, and raw Dx string from a record's header comments.
    Header comment lines look like:
        'Age: 57'
        'Sex: Male'
        'Dx: 164889003,59118001'
    Returns a dict with keys: age, sex, dx_raw (dx_raw may be None).
    """
    fields = {"age": None, "sex": None, "dx_raw": None}
    for line in comments:
        clean = line.strip()
        if clean.lower().startswith("age"):
            fields["age"] = clean.split(":", 1)[1].strip()
        elif clean.lower().startswith("sex"):
            fields["sex"] = clean.split(":", 1)[1].strip()
        elif clean.lower().startswith("dx"):
            fields["dx_raw"] = clean.split(":", 1)[1].strip()
    return fields


# ------------------------------------------------------------------
# 4. Get only the PRIMARY diagnosis (first Dx code)
# ------------------------------------------------------------------
def get_primary_diagnosis(dx_raw, code_to_name):
    """
    Takes the raw Dx string (comma-separated SNOMED codes), keeps only
    the FIRST code as the primary diagnosis, and maps it to a name.
    Returns None if dx_raw is missing or the code is not in the CSV
    (i.e. unknown diagnosis -> caller should skip this record).
    """
    if not dx_raw:
        return None
    primary_code = dx_raw.split(",")[0].strip()
    return code_to_name.get(primary_code, None)


# ------------------------------------------------------------------
# 5. Fix signal length to a fixed target (pad or truncate)
# ------------------------------------------------------------------
def fix_signal_length(signal, target_len=TARGET_LEN):
    """
    Ensures signal has shape (target_len, n_leads).
    - If longer: truncate to the first target_len samples.
    - If shorter: zero-pad at the end.
    This dataset has minor variability in recording length, so a fixed
    length is enforced here to allow batching into a single tensor.
    """
    current_len = signal.shape[0]
    if current_len == target_len:
        return signal
    elif current_len > target_len:
        return signal[:target_len, :]
    else:
        pad_width = target_len - current_len
        pad = np.zeros((pad_width, signal.shape[1]))
        return np.vstack([signal, pad])


# ------------------------------------------------------------------
# 6. Z-score normalize each lead independently
# ------------------------------------------------------------------
def zscore_normalize(signal):

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)

    std[std < 1e-8] = 1.0

    signal = (signal - mean) / std

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    return signal

# ------------------------------------------------------------------
# 7. Main processing loop over all records
# ------------------------------------------------------------------
def process_all_records(record_paths, code_to_name):
    """
    Iterates over every record, reads signal + header, filters out
    records with unknown/missing primary diagnosis, fixes length,
    normalizes, and collects results.

    Returns:
        signals: list of np.ndarray, each shape (12, TARGET_LEN)
        labels:  list of str (diagnosis names)
    """
    signals = []
    labels = []
    skipped_unknown_dx = 0
    skipped_errors = 0

    for path in tqdm(record_paths, desc="Processing records"):
        try:
            record = wfdb.rdrecord(str(path))
        except Exception:
            skipped_errors += 1
            continue

        fields = parse_header_fields(record.comments)
        primary_dx_name = get_primary_diagnosis(fields["dx_raw"], code_to_name)

        if primary_dx_name is None:
            skipped_unknown_dx += 1
            continue

        sig = record.p_signal

        # Check raw signal
        if np.isnan(sig).any() or np.isinf(sig).any():
            print(f"Skipping corrupted RAW record: {path}")
            skipped_errors += 1
            continue

        if sig.shape[1] != NUM_LEADS:
            skipped_errors += 1
            continue

        sig = fix_signal_length(sig, TARGET_LEN)
        sig = zscore_normalize(sig)

        # Check normalized signal
        if np.isnan(sig).any() or np.isinf(sig).any():
            print(f"Skipping corrupted NORMALIZED record: {path}")
            skipped_errors += 1
            continue

        sig = sig.T

        signals.append(sig.astype(np.float32))
        labels.append(primary_dx_name)

    print(f"\nSkipped (unknown/missing Dx): {skipped_unknown_dx}")
    print(f"Skipped (read/shape errors):  {skipped_errors}")

    return signals, labels


# ------------------------------------------------------------------
# 8. Orchestration
# ------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SNOMED-CT condition mapping...")
    code_to_name = load_condition_mapping(CONDITIONS_CSV)

    print("Collecting record paths...")
    record_paths = get_record_paths(DATA_PATH, RECORDS_FILE)
    print(f"Total records found: {len(record_paths)}")

    signals, labels = process_all_records(record_paths, code_to_name)

    # Stack into final arrays
    X = np.stack(signals, axis=0)          # (N, 12, 5000)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)  # (N,)

    # Save outputs
    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    with open(OUTPUT_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # ------------------------------------------------------------------
    # Summary printout
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Number of records kept : {X.shape[0]}")
    print(f"Signal shape (X)       : {X.shape}")
    print(f"Labels shape (y)       : {y.shape}")
    print(f"Number of classes      : {len(label_encoder.classes_)}")

    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        class_name = label_encoder.inverse_transform([u])[0]
        print(f"   {class_name:<40} {c:>6}")

    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
