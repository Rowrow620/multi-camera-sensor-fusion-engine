#!/usr/bin/env python3
"""

This script reads metadata and labels from CSV files, then organizes the cropped images into class folders for training a CNN. 
It also includes a mode to generate a template CSV for labeling.
"""

from pathlib import Path
import argparse
import csv
import shutil
from collections import Counter


def build_parser():
    p = argparse.ArgumentParser(
        description="Create labeled class folders from stereo crop outputs."
    )
    p.add_argument("--raw-dir", type=Path, default=Path("crops/raw"), help="Directory with crop_XXXX.png and crops_metadata.csv")
    p.add_argument("--metadata-csv", type=Path, default=None, help="Path to crops_metadata.csv (defaults to <raw-dir>/crops_metadata.csv)")
    p.add_argument("--labels-csv", type=Path, default=None, help="Path to crop_labels.csv (defaults to <raw-dir>/crop_labels.csv)")
    p.add_argument("--output-dir", type=Path, default=Path("crops/labeled"), help="Output dir for class folders")
    p.add_argument("--mode", choices=("copy", "move"), default="copy", help="Copy or move files into class folders")
    p.add_argument("--init-template", action="store_true", help="Generate crop_labels.csv from metadata and exit")
    return p


def resolve_paths(args):
    raw_dir = args.raw_dir.resolve()
    metadata_csv = args.metadata_csv.resolve() if args.metadata_csv else (raw_dir / "crops_metadata.csv")
    labels_csv = args.labels_csv.resolve() if args.labels_csv else (raw_dir / "crop_labels.csv")
    return raw_dir, metadata_csv, labels_csv


def write_template(metadata_csv: Path, labels_csv: Path) -> None:
    """Create a `crop_labels.csv` with empty labels for every crop in metadata."""
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    with metadata_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "crop_file" not in (reader.fieldnames or []):
            raise ValueError("Metadata CSV must contain a 'crop_file' column")
        rows = [r for r in reader]

    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    with labels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_file", "label"])
        for r in rows:
            writer.writerow([r.get("crop_file", ""), ""])

    print(f"Wrote label template: {labels_csv}")
    print("Fill in the 'label' column, then rerun without --init-template.")


def normalize_label(raw: str) -> str:
    if raw is None:
        return ""
    label = raw.strip()
    if not label:
        return ""
    # Normalize whitespace and disallow path separators
    label = label.replace(" ", "_")
    if "/" in label or "\\" in label:
        raise ValueError(f"Invalid label '{raw}': path separators are not allowed")
    return label


def materialize_dataset(raw_dir: Path, labels_csv: Path, output_dir: Path, mode: str = "copy") -> None:
    """Read `crop_labels.csv` and copy/move files into `output_dir/<label>/` folders."""
    if not labels_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {labels_csv}. Run with --init-template first.")

    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"crop_file", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Label CSV must contain 'crop_file' and 'label' columns")
        rows = [r for r in reader]

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    skipped = 0

    for r in rows:
        crop = (r.get("crop_file") or "").strip()
        raw_label = r.get("label") or ""
        label = normalize_label(raw_label)
        if not crop:
            continue
        if not label:
            skipped += 1
            continue

        src = raw_dir / crop
        if not src.exists():
            print(f"WARN: Missing file, skipping: {src}")
            continue

        dst_dir = output_dir / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        counts[label] += 1

    print(f"Output dataset directory: {output_dir}")
    print(f"Skipped unlabeled rows: {skipped}")
    if not counts:
        print("No labeled crops were processed.")
        return
    print("Class counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    raw_dir, metadata_csv, labels_csv = resolve_paths(args)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw crop directory not found: {raw_dir}")

    if args.init_template:
        write_template(metadata_csv, labels_csv)
        return

    materialize_dataset(raw_dir, labels_csv, args.output_dir, args.mode)


if __name__ == "__main__":
    main()
