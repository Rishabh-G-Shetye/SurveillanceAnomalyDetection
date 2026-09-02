"""
UCSD Anomaly Dataset -- windowed clip Dataset for PyTorch.

Loads short temporal windows ("clips") of consecutive frames from the
UCSD Ped1/Ped2 sequences, resizes/normalizes them, and returns clip-level
and frame-level anomaly labels for the Test split. Canonical ground truth
is loaded from UCSDped1.m / UCSDped2.m so all test sequences (36 in Ped1,
12 in Ped2) are properly labeled and evaluated.

Windows containing an unreadable (corrupted) frame are skipped.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

FRAME_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
DATASET_DIR_MAP = {"Ped1": "UCSDped1", "Ped2": "UCSDped2"}


@dataclass
class PreprocessConfig:
    """Per-dataset preprocessing configuration."""
    target_size: tuple[int, int] = (128, 128)
    window_length: int = 8
    stride: int = 4
    mean: float = 0.5
    std: float = 0.5
    horizontal_flip_prob: float = 0.0  # Optional flip during training


def build_transform(config: PreprocessConfig, is_train: bool = False) -> transforms.Compose:
    """Resize + tensor + normalize transform for a single grayscale frame."""
    transform_list = [
        transforms.Resize(config.target_size),
    ]
    if is_train and config.horizontal_flip_prob > 0:
        transform_list.append(transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob))
    transform_list.extend([
        transforms.ToTensor(),  # (1, H, W), values in [0, 1]
        transforms.Normalize(mean=[config.mean], std=[config.std]),
    ])
    return transforms.Compose(transform_list)


def parse_matlab_gt(m_path: Path) -> dict[str, set[int]]:
    """Parse canonical UCSD .m ground-truth annotation files into sets of frame numbers."""
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
        gt_map[seq_name] = frames

    return gt_map


def _is_readable(path: Path) -> bool:
    """Return False if an image file is missing, truncated, or undecodable."""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except (OSError, IOError):
        return False


class UCSDClipDataset(Dataset):
    """
    Windowed clip dataset for one UCSD sub-dataset (Ped1 or Ped2) and split.

    Each item is a clip of `config.window_length` consecutive frames from a
    single sequence, shaped (T, 1, H, W). Any window containing a corrupted
    frame is excluded at construction time.

    Test split: Uses canonical ground truth from UCSDped1.m / UCSDped2.m.
    `clip_label` = 1 if any frame in the window is anomalous;
    `frame_labels` gives the per-frame binary label, shape (T,).
    Train split (no ground truth): both are all-zero placeholders.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_root: Path,
        dataset_name: str,
        split: str,
        config: PreprocessConfig,
        is_train: bool | None = None,
    ):
        self.data_root = Path(data_root)
        self.config = config
        self.split = split
        self.dataset_name = dataset_name
        self.is_train = (split == "Train") if is_train is None else is_train
        self.transform = build_transform(config, is_train=self.is_train)

        # Load canonical MATLAB ground truth for test sequences
        ped_dir = DATASET_DIR_MAP[dataset_name]
        split_path = self.data_root / ped_dir / split
        m_filename = f"UCSD{dataset_name.lower()}.m"
        m_path = split_path / m_filename
        if not m_path.exists():
            m_path = self.data_root / ped_dir / m_filename
        self.matlab_gt = parse_matlab_gt(m_path) if split == "Test" else {}

        # Filter sequence names
        seq_names = metadata_df.loc[
            (metadata_df["dataset"] == dataset_name) & (metadata_df["split"] == split),
            "sequence",
        ].tolist()

        self.windows = []  # list of (seq_name, frame_paths)
        self.skipped_count = 0

        for seq_name in seq_names:
            seq_dir = split_path / seq_name
            if not seq_dir.exists():
                continue

            frame_paths = sorted(
                p for p in seq_dir.iterdir()
                if p.is_file() and p.suffix.lower() in FRAME_SUFFIXES
            )

            # Check each frame once
            readable = {p: _is_readable(p) for p in frame_paths}

            n = len(frame_paths)
            for start in range(0, n - config.window_length + 1, config.stride):
                fp_window = frame_paths[start:start + config.window_length]

                if not all(readable[p] for p in fp_window):
                    self.skipped_count += 1
                    continue

                self.windows.append((seq_name, fp_window))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        seq_name, frame_paths = self.windows[idx]

        frames = [self.transform(Image.open(p).convert("L")) for p in frame_paths]
        clip = torch.stack(frames, dim=0)  # (T, 1, H, W)

        frame_indices = np.array([int(p.stem) for p in frame_paths], dtype=np.int64)

        if self.split == "Test":
            anom_set = self.matlab_gt.get(seq_name, set())
            frame_labels = np.array(
                [1 if f_idx in anom_set else 0 for f_idx in frame_indices],
                dtype=np.int64,
            )
        else:
            frame_labels = np.zeros(len(frame_paths), dtype=np.int64)

        clip_label = int(frame_labels.any())

        return {
            "clip": clip,
            "clip_label": clip_label,
            "frame_labels": torch.from_numpy(frame_labels),
            "seq_name": seq_name,
            "frame_indices": torch.from_numpy(frame_indices),
        }
