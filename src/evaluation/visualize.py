"""
Shared visualization utilities for comparing anomaly-detection models.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve


def plot_roc_curves(model_curves: dict):
    """
    model_curves: {"ModelName": (labels, scores), ...}
    Plots ROC curve for each model with its respective AUC score in the legend.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    fig = plt.figure(figsize=(7, 5.5))
    for name, data in model_curves.items():
        if isinstance(data, (tuple, list)):
            y_true, y_score = np.asarray(data[0]), np.asarray(data[1])
        else:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, linewidth=1.8, label=f"{name} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Random Chance (0.50)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curves Comparison (Canonical UCSD Ped1)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_score_timeline(model, dataset, sequence_name: str, device=None):
    """Anomaly score across every frame of one test sequence with canonical GT shading."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Collect all frame paths for the given sequence
    anom_set = dataset.matlab_gt.get(sequence_name, set())

    # Filter windows for this sequence
    matching_windows = [
        (s_name, fps) for (s_name, fps) in dataset.windows if s_name == sequence_name
    ]
    if not matching_windows:
        print(f"Warning: No windows found for sequence {sequence_name}")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, f"No data for {sequence_name}", ha="center", va="center")
        return fig

    from collections import defaultdict
    frame_scores_map = defaultdict(list)
    is_prediction = (model.__class__.__name__ == "FramePredictionNet")

    with torch.no_grad():
        for s_name, frame_paths in matching_windows:
            frames = [dataset.transform(Image.open(p).convert("L")) for p in frame_paths]
            clip = torch.stack(frames).unsqueeze(0).to(device)
            scores = model.per_frame_anomaly_score(clip).cpu().numpy()

            if is_prediction:
                last_idx = int(frame_paths[-1].stem)
                frame_scores_map[last_idx].append(float(scores[0]))
            else:
                for t, p in enumerate(frame_paths):
                    f_idx = int(p.stem)
                    frame_scores_map[f_idx].append(float(scores[0, t]))

    frame_indices = sorted(frame_scores_map.keys())
    scores = np.array([np.mean(frame_scores_map[idx]) for idx in frame_indices])

    # Min-max normalization for visual clarity
    if len(scores) > 0 and (scores.max() - scores.min()) > 1e-8:
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        norm_scores = scores

    is_anom = np.array([idx in anom_set for idx in frame_indices])

    fig = plt.figure(figsize=(10, 3.5))
    plt.plot(frame_indices, norm_scores, color="steelblue", linewidth=1.8, label="Anomaly score")
    plt.fill_between(
        frame_indices, 0, 1,
        where=is_anom, color="red", alpha=0.25, label="Ground-truth anomaly"
    )
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Nominal threshold (0.5)")
    plt.xlabel("Frame index")
    plt.ylabel("Normalized anomaly score")
    plt.ylim(-0.05, 1.05)
    plt.title(f"Anomaly Score Timeline -- {sequence_name} ({model.__class__.__name__})")
    plt.legend(loc="upper left")
    plt.tight_layout()
    return fig


def plot_metric_comparison(results_by_model: dict, metrics=("frame_auc_roc", "frame_f1", "event_precision_avg", "event_recall_avg")):
    """results_by_model: {"ModelName": evaluate_scores(...) dict, ...}"""
    names = list(results_by_model.keys())
    # Fall back to alternative metric names if necessary
    actual_metrics = []
    for m in metrics:
        if any(m in results_by_model[n] for n in names):
            actual_metrics.append(m)
        elif m.startswith("frame_") and any(m.replace("frame_", "") in results_by_model[n] for n in names):
            actual_metrics.append(m.replace("frame_", ""))

    if not actual_metrics:
        actual_metrics = list(next(iter(results_by_model.values())).keys())[:4]

    x = np.arange(len(actual_metrics))
    width = 0.8 / len(names)

    fig = plt.figure(figsize=(9, 5))
    for i, name in enumerate(names):
        values = [results_by_model[name].get(m, 0.0) for m in actual_metrics]
        plt.bar(x + i * width, values, width, label=name)

    clean_labels = [m.replace("_", " ").title() for m in actual_metrics]
    plt.xticks(x + width * (len(names) - 1) / 2, clean_labels, rotation=15)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Model Performance Comparison (UCSD Pedestrian Dataset)")
    plt.legend()
    plt.tight_layout()
    return fig
