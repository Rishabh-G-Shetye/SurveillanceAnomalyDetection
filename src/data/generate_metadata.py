"""
Generate metadata.csv from the local UCSD dataset.

Parses the canonical MATLAB ground-truth files (UCSDped1.m and UCSDped2.m)
to extract frame-level ground-truth annotations for all 36 test sequences in
Ped1 and 12 in Ped2. Also tracks pixel-level mask folder existence.

Usage:
    python -m src.data.generate_metadata
"""

import re
from pathlib import Path
import pandas as pd

FRAME_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
GT_SUFFIXES = {".bmp"}
IGNORE_NAMES = {".DS_Store", "._.DS_Store", "._README.txt"}


def count_files(dir_path: Path, suffixes: set) -> int:
    """Count files in dir_path matching given suffixes, ignoring OS files."""
    if not dir_path.exists():
        return 0
    return sum(
        1 for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes and p.name not in IGNORE_NAMES
    )


def parse_matlab_gt(m_path: Path) -> dict[str, list[int]]:
    """
    Parse canonical UCSD .m ground-truth annotation files.
    
    Lines typically look like:
        TestVideoFile{end+1}.gt_frame = [60:152];
        TestVideoFile{end+1}.gt_frame = [5:90, 140:200];
    
    Returns:
        dict mapping 'Test001', 'Test002', etc. to list of 1-indexed anomalous frame numbers.
    """
    if not m_path.exists():
        return {}

    gt_map = {}
    content = m_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in content.splitlines() if "gt_frame" in line]

    for idx, line in enumerate(lines, start=1):
        seq_name = f"Test{idx:03d}"
        m = re.search(r"\[(.*?)\]", line)
        frames = set()
        if m and m.group(1).strip():
            raw_ranges = m.group(1).split(",")
            for part in raw_ranges:
                part = part.strip()
                if not part:
                    continue
                if ":" in part:
                    s, e = map(int, part.split(":"))
                    frames.update(range(s, e + 1))
                else:
                    frames.add(int(part))
        gt_map[seq_name] = sorted(frames)

    return gt_map


def build_inventory(ped_path: Path, ped_name: str) -> list[dict]:
    """Walk Train/Test folders of one Ped dataset and attach canonical ground truth."""
    rows = []
    m_filename = f"UCSD{ped_name.lower()}.m"
    # Canonical GT is inside Test/ directory
    matlab_gt = parse_matlab_gt(ped_path / "Test" / m_filename)
    if not matlab_gt:
        # Fallback to ped_path root
        matlab_gt = parse_matlab_gt(ped_path / m_filename)

    for split in ("Train", "Test"):
        split_path = ped_path / split
        if not split_path.exists():
            continue

        for seq_dir in sorted(split_path.iterdir()):
            if not seq_dir.is_dir() or seq_dir.name in IGNORE_NAMES:
                continue
            if seq_dir.name.endswith("_gt"):
                continue

            num_frames = count_files(seq_dir, FRAME_SUFFIXES)
            gt_dir = split_path / f"{seq_dir.name}_gt"
            has_pixel_mask = gt_dir.exists()

            if split == "Test":
                anom_frames = matlab_gt.get(seq_dir.name, [])
                has_gt = len(anom_frames) > 0 or (seq_dir.name in matlab_gt)
                num_gt_frames = len(anom_frames)
            else:
                anom_frames = []
                has_gt = False
                num_gt_frames = 0

            rows.append({
                "dataset": ped_name,
                "split": split,
                "sequence": seq_dir.name,
                "num_frames": num_frames,
                "has_gt": has_gt,
                "num_anom_frames": num_gt_frames,
                "has_pixel_mask": has_pixel_mask,
                "num_pixel_masks": count_files(gt_dir, GT_SUFFIXES) if has_pixel_mask else 0,
            })

    return rows


def generate_metadata(
    data_root: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Scan the UCSD dataset and produce a metadata CSV with canonical ground-truth info.
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    if data_root is None:
        data_root = project_root / "data" / "raw" / "UCSD_Anomaly_Dataset"
    data_root = Path(data_root)

    if output_path is None:
        output_path = project_root / "data" / "metadata.csv"
    output_path = Path(output_path)

    ped1_path = data_root / "UCSDped1"
    ped2_path = data_root / "UCSDped2"

    if not ped1_path.exists() and not ped2_path.exists():
        raise FileNotFoundError(
            f"Neither UCSDped1 nor UCSDped2 found in {data_root}.\n"
            f"Please extract UCSD_Anomaly_Dataset.zip into data/raw/"
        )

    inventory = []
    if ped1_path.exists():
        inventory.extend(build_inventory(ped1_path, "Ped1"))
    if ped2_path.exists():
        inventory.extend(build_inventory(ped2_path, "Ped2"))

    df = pd.DataFrame(inventory)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Metadata saved to: {output_path}")
    print(f"  Total sequences: {len(df)}")
    ped1_test = df[(df["dataset"] == "Ped1") & (df["split"] == "Test")]
    ped2_test = df[(df["dataset"] == "Ped2") & (df["split"] == "Test")]
    print(f"  Ped1 Test sequences with GT: {ped1_test['has_gt'].sum()} / {len(ped1_test)}")
    print(f"  Ped2 Test sequences with GT: {ped2_test['has_gt'].sum()} / {len(ped2_test)}")

    return df


if __name__ == "__main__":
    generate_metadata()
