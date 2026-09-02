"""
Evaluation metrics for Video Anomaly Detection.

Implements:
- AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
- EER (Equal Error Rate)
- Optimal thresholding (F1-maximizing / Youden's J statistic)
- Precision, Recall, F1-Score
- Frame-level evaluation across all test videos
- Event-level evaluation (temporal localization via 1D IoU segment overlap)
- Per-sequence min-max score normalization and temporal smoothing (standard UCSD benchmark protocol)
- Computational efficiency benchmarking (ms/frame and FPS)
"""

import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate: threshold where false positive rate equals false negative rate."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


def evaluate_scores(labels: np.ndarray, scores: np.ndarray, threshold: float = None) -> dict:
    """
    Compute comprehensive metrics given binary ground-truth labels and anomaly scores.
    If threshold is None, finds the optimal threshold that maximizes F1 score.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    # Handle edge case of single-class labels in small tests
    if len(np.unique(labels)) < 2:
        return {
            "auc_roc": 1.0 if (labels == (scores >= 0.5)).all() else 0.5,
            "eer": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "threshold_used": 0.5,
        }

    auc = roc_auc_score(labels, scores)
    eer = compute_eer(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find optimal threshold by maximizing F1 on candidate thresholds
    if threshold is None:
        # Sample candidate thresholds from percentiles
        candidate_threshs = np.unique(
            np.percentile(scores, np.linspace(5, 95, 50))
        )
        best_f1 = -1.0
        best_thresh = float(np.median(scores))

        for t in candidate_threshs:
            preds = (scores >= t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)
        threshold = best_thresh

    preds = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    return {
        "auc_roc": float(auc),
        "eer": float(eer),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold_used": float(threshold),
    }


def _to_segments(binary_arr: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True in a 1D boolean array -> [(start, end), ...]."""
    segments, start = [], None
    for i, v in enumerate(binary_arr):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(binary_arr)))
    return segments


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """1D temporal Intersection over Union between two intervals."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def event_level_accuracy(
    frame_labels: np.ndarray,
    frame_scores: np.ndarray,
    threshold: float,
    iou_thresh: float = 0.1,
) -> dict:
    """
    Event-level evaluation: groups contiguous ground-truth anomalous frames into
    events, and contiguous above-threshold predictions into predicted events.
    An event is detected if 1D IoU >= iou_thresh.
    """
    gt_segments = _to_segments(np.asarray(frame_labels).astype(bool))
    pred_segments = _to_segments(np.asarray(frame_scores) >= threshold)

    matched_gt = set()
    tp = 0
    for p in pred_segments:
        for i, g in enumerate(gt_segments):
            if i not in matched_gt and _iou(p, g) >= iou_thresh:
                tp += 1
                matched_gt.add(i)
                break

    precision = tp / len(pred_segments) if pred_segments else 0.0
    recall = tp / len(gt_segments) if gt_segments else (1.0 if not pred_segments else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "num_gt_events": len(gt_segments),
        "num_pred_events": len(pred_segments),
    }


def smooth_scores(scores: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Applies 1D Gaussian filter along temporal dimension to suppress high-frequency frame noise."""
    if len(scores) < 3 or sigma <= 0:
        return scores
    return gaussian_filter1d(scores, sigma=sigma, mode="nearest")


def frame_and_event_level_eval(
    model: torch.nn.Module,
    dataset,
    device: str | None = None,
    batch_size: int = 32,
    normalize_per_seq: bool = True,
    smooth: bool = True,
    iou_thresh: float = 0.1,
) -> dict:
    """
    High-performance batched frame-level and event-level evaluation.
    
    1. Runs model.per_frame_anomaly_score in batched DataLoader.
    2. Maps scores back to individual frames, averaging overlapping windows.
    3. Normalizes scores per video sequence (standard UCSD benchmark protocol).
    4. Computes pooled frame-level AUC-ROC, EER, F1, and sequence-averaged event metrics.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    seq_frame_scores = defaultdict(lambda: defaultdict(list))
    seq_frame_labels = defaultdict(dict)

    is_prediction_model = (model.__class__.__name__ == "FramePredictionNet")

    with torch.no_grad():
        for batch in loader:
            clips = batch["clip"].to(device)
            # frame_scores shape: (B, T) for reconstruction, (B,) for prediction
            scores = model.per_frame_anomaly_score(clips).cpu().numpy()
            frame_indices = batch["frame_indices"].numpy()  # (B, T)
            labels = batch["frame_labels"].numpy()          # (B, T)
            seq_names = batch["seq_name"]                   # list of strings, length B

            B = len(seq_names)
            for b in range(B):
                s_name = seq_names[b]
                if is_prediction_model:
                    # Prediction score belongs strictly to the predicted frame (last in window)
                    f_idx = int(frame_indices[b, -1])
                    seq_frame_scores[s_name][f_idx].append(float(scores[b]))
                    seq_frame_labels[s_name][f_idx] = int(labels[b, -1])
                else:
                    # Reconstruction errors map frame-by-frame
                    T = frame_indices.shape[1]
                    for t in range(T):
                        f_idx = int(frame_indices[b, t])
                        seq_frame_scores[s_name][f_idx].append(float(scores[b, t]))
                        seq_frame_labels[s_name][f_idx] = int(labels[b, t])

    # Aggregate per sequence
    all_scores, all_labels = [], []
    event_precisions, event_recalls, event_f1s = [], [], []

    for seq_name in sorted(seq_frame_scores.keys()):
        frame_map = seq_frame_scores[seq_name]
        f_idxs = sorted(frame_map.keys())

        # Average overlapping window scores for each frame
        raw_s = np.array([np.mean(frame_map[idx]) for idx in f_idxs], dtype=float)
        f_labels = np.array([seq_frame_labels[seq_name][idx] for idx in f_idxs], dtype=int)

        # Standard benchmark per-sequence min-max scaling
        if normalize_per_seq:
            s_min, s_max = raw_s.min(), raw_s.max()
            norm_s = (raw_s - s_min) / (s_max - s_min + 1e-8)
        else:
            norm_s = raw_s

        # Temporal smoothing
        if smooth:
            final_s = smooth_scores(norm_s, sigma=1.5)
        else:
            final_s = norm_s

        all_scores.append(final_s)
        all_labels.append(f_labels)

        # Sequence-level event accuracy
        seq_threshold = float(np.median(final_s))
        ev = event_level_accuracy(f_labels, final_s, seq_threshold, iou_thresh=iou_thresh)
        event_precisions.append(ev["event_precision"])
        event_recalls.append(ev["event_recall"])
        event_f1s.append(ev["event_f1"])

    pooled_scores = np.concatenate(all_scores)
    pooled_labels = np.concatenate(all_labels)

    frame_results = evaluate_scores(pooled_labels, pooled_scores)

    return {
        **{f"frame_{k}": v for k, v in frame_results.items()},
        "event_precision_avg": float(np.mean(event_precisions)),
        "event_recall_avg": float(np.mean(event_recalls)),
        "event_f1_avg": float(np.mean(event_f1s)),
        "total_evaluated_frames": len(pooled_labels),
        "total_anomalous_frames": int(pooled_labels.sum()),
        "total_evaluated_sequences": len(seq_frame_scores),
        "pooled_scores": pooled_scores,
        "pooled_labels": pooled_labels,
    }


def measure_inference_time(
    model: torch.nn.Module,
    sample_clip: torch.Tensor,
    device: str | None = None,
    n_runs: int = 50,
) -> dict:
    """
    Benchmark inference latency per clip and per frame, and compute FPS.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    clip = sample_clip.unsqueeze(0).to(device)

    with torch.no_grad():
        # Warmup
        for _ in range(5):
            model.per_frame_anomaly_score(clip)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            model.per_frame_anomaly_score(clip)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    ms_per_clip = (elapsed / n_runs) * 1000
    T = clip.shape[1]
    ms_per_frame = ms_per_clip / T
    fps = 1000.0 / ms_per_frame if ms_per_frame > 0 else 0.0

    return {
        "ms_per_clip": float(ms_per_clip),
        "ms_per_frame": float(ms_per_frame),
        "fps": float(fps),
    }
