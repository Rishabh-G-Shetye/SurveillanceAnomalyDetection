"""
Inference Pipeline for Surveillance Video Anomaly Detection.

Runs a trained anomaly detection model on any surveillance video file (.mp4, .avi)
or directory of image frames. Produces:
1. An annotated video with real-time alert banner, score gauge, and spatial anomaly heatmap.
2. A temporal score timeline plot (.png).
3. A structured JSON report of detected anomaly events with timestamps and frame numbers.

Usage:
    python src/inference.py --input data/raw/UCSD_Anomaly_Dataset/UCSDped1/Test/Test003 --model ConvAE
    python src/inference.py --input path/to/video.mp4 --model ConvLSTM-AE --threshold 0.5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.ucsd_dataset import PreprocessConfig, build_transform
from models.conv_ae import ConvAE
from models.convlstm_ae import ConvLSTMAE
from models.transformer_ae import TransformerAE
from models.frame_prediction import FramePredictionNet
from models.memory_ae import MemAE
from utils.persistence import load_checkpoint

MODEL_REGISTRY = {
    "ConvAE": (lambda: ConvAE(use_skip=False, score_agg="mean"), "conv_ae_baseline"),
    "ConvLSTM-AE": (lambda: ConvLSTMAE(), "convlstm_ae_baseline"),
    "TransformerAE": (lambda: TransformerAE(), "transformer_ae_baseline"),
    "FramePrediction": (lambda: FramePredictionNet(), "frame_prediction_baseline"),
    "MemAE": (lambda: MemAE(), "memae_baseline"),
}


def load_frames_from_source(source_path: Path) -> list[np.ndarray]:
    """Load video frames from a video file (.mp4, .avi) or a folder of images."""
    frames = []
    source_path = Path(source_path)

    if source_path.is_dir():
        # Directory of image frames
        img_suffixes = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted(
            [p for p in source_path.iterdir() if p.is_file() and p.suffix.lower() in img_suffixes]
        )
        for p in files:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
    elif source_path.is_file():
        # Video file
        cap = cv2.VideoCapture(str(source_path))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    else:
        raise FileNotFoundError(f"Input source not found: {source_path}")

    if not frames:
        raise ValueError(f"No frames could be read from {source_path}")

    return frames


def run_inference(
    model: torch.nn.Module,
    raw_frames: list[np.ndarray],
    config: PreprocessConfig,
    device: str,
    stride: int = 1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Run model on sliding windows of frames and compute per-frame anomaly scores and spatial heatmaps.
    """
    model.eval()
    transform = build_transform(config, is_train=False)
    T = config.window_length
    num_frames = len(raw_frames)

    # Pre-transform all frames to grayscale tensors
    gray_tensors = []
    for f in raw_frames:
        pil_img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        gray_tensors.append(transform(pil_img))  # (1, H, W)

    frame_scores_map = [[] for _ in range(num_frames)]
    heatmaps_map = [[] for _ in range(num_frames)]
    is_prediction = (model.__class__.__name__ == "FramePredictionNet")

    with torch.no_grad():
        for start in range(0, num_frames - T + 1, stride):
            window_tensors = gray_tensors[start : start + T]
            clip = torch.stack(window_tensors).unsqueeze(0).to(device)  # (1, T, 1, H, W)

            if is_prediction:
                pred = model(clip)  # (1, 1, H, W)
                target = clip[:, -1]
                diff = (pred - target).abs().squeeze().cpu().numpy()  # (H, W)
                score = float(((pred - target) ** 2).mean().item())
                last_idx = start + T - 1
                frame_scores_map[last_idx].append(score)
                heatmaps_map[last_idx].append(diff)
            else:
                recon = model(clip)  # (1, T, 1, H, W)
                diff = (recon - clip).abs().squeeze(2).squeeze(0).cpu().numpy()  # (T, H, W)
                for t in range(T):
                    idx = start + t
                    err = float(((recon[:, t] - clip[:, t]) ** 2).mean().item())
                    frame_scores_map[idx].append(err)
                    heatmaps_map[idx].append(diff[t])

    # Aggregate overlapping window scores
    final_scores = np.zeros(num_frames, dtype=float)
    final_heatmaps = []

    for i in range(num_frames):
        if frame_scores_map[i]:
            final_scores[i] = np.mean(frame_scores_map[i])
            # Average spatial difference map
            avg_diff = np.mean(np.stack(heatmaps_map[i]), axis=0)
            final_heatmaps.append(avg_diff)
        else:
            final_scores[i] = final_scores[max(0, i - 1)]
            final_heatmaps.append(np.zeros(config.target_size, dtype=float))

    # Min-max normalization for the sequence
    s_min, s_max = final_scores.min(), final_scores.max()
    if s_max - s_min > 1e-8:
        norm_scores = (final_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = final_scores

    return norm_scores, final_heatmaps


def annotate_video(
    raw_frames: list[np.ndarray],
    scores: np.ndarray,
    heatmaps: list[np.ndarray],
    output_path: Path,
    threshold: float = 0.5,
    fps: int = 15,
) -> list[dict]:
    """
    Annotate frames with anomaly status, gauge, and spatial heatmap overlay, and write to video.
    Returns list of detected anomaly events.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    H, W, _ = raw_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    events = []
    current_event = None

    for i, frame in enumerate(raw_frames):
        score = scores[i]
        is_anom = score >= threshold
        annotated = frame.copy()

        # Generate spatial heatmap overlay
        hm = heatmaps[i]
        hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        hm_resized = cv2.resize((hm_norm * 255).astype(np.uint8), (W, H))
        color_hm = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)

        # Blend heatmap if anomaly score is significant
        if score > 0.3:
            alpha = min(0.6, float(score * 0.7))
            annotated = cv2.addWeighted(annotated, 1 - alpha, color_hm, alpha, 0)

        # Status Banner
        banner_color = (0, 0, 220) if is_anom else (0, 180, 0)  # Red vs Green (BGR)
        status_text = "ANOMALY ALERT!" if is_anom else "NORMAL"
        cv2.rectangle(annotated, (10, 10), (320, 65), (20, 20, 20), -1)
        cv2.rectangle(annotated, (10, 10), (320, 65), banner_color, 2)
        cv2.putText(
            annotated, status_text, (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner_color, 2, cv2.LINE_AA
        )

        # Score Gauge
        cv2.putText(
            annotated, f"Score: {score:.2f} (Thresh: {threshold:.2f})", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
        )

        # Progress bar under banner
        bar_x, bar_y, bar_w, bar_h = 10, 70, 310, 10
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * np.clip(score, 0.0, 1.0))
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), banner_color, -1)

        # Frame counter
        cv2.putText(
            annotated, f"Frame: {i+1}/{len(raw_frames)}", (W - 160, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )

        writer.write(annotated)

        # Track events
        if is_anom:
            if current_event is None:
                current_event = {"start_frame": i + 1, "max_score": float(score), "peak_frame": i + 1}
            else:
                if score > current_event["max_score"]:
                    current_event["max_score"] = float(score)
                    current_event["peak_frame"] = i + 1
        else:
            if current_event is not None:
                current_event["end_frame"] = i
                events.append(current_event)
                current_event = None

    if current_event is not None:
        current_event["end_frame"] = len(raw_frames)
        events.append(current_event)

    writer.release()
    print(f"Annotated video saved: {output_path}")
    return events


def save_timeline_plot(scores: np.ndarray, threshold: float, output_path: Path):
    """Save score timeline plot to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    frames = np.arange(1, len(scores) + 1)
    ax.plot(frames, scores, color="steelblue", linewidth=1.8, label="Anomaly score")
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.2, label=f"Alert threshold ({threshold})")
    ax.fill_between(frames, threshold, scores, where=(scores >= threshold), color="red", alpha=0.3, label="Alert active")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Anomaly Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Video Anomaly Detection Score Timeline")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Timeline plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Surveillance Video Anomaly Detection - Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file or image directory")
    parser.add_argument("--model", type=str, default="ConvAE", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video (.mp4)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly alert threshold [0.0 - 1.0]")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    factory, default_ckpt = MODEL_REGISTRY[args.model]
    model = factory()
    ckpt_name = args.checkpoint or str(PROJECT_ROOT / "models" / f"{default_ckpt}.pt")

    if Path(ckpt_name).exists():
        model.load_state_dict(torch.load(ckpt_name, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {ckpt_name}")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_name}. Running with initialized weights.")

    model.to(device)

    # Load Video Frames
    print(f"Loading input: {args.input}...")
    frames = load_frames_from_source(Path(args.input))
    print(f"Loaded {len(frames)} frames.")

    config = PreprocessConfig(target_size=(128, 128), window_length=8, stride=1)

    # Run Anomaly Scoring
    print(f"Running inference with {args.model}...")
    scores, heatmaps = run_inference(model, frames, config, device, stride=1)

    # Default Output paths
    in_name = Path(args.input).name
    out_video = Path(args.output) if args.output else (PROJECT_ROOT / "outputs" / "videos" / f"{in_name}_{args.model}_annotated.mp4")
    out_timeline = PROJECT_ROOT / "outputs" / "figures" / f"{in_name}_{args.model}_timeline.png"
    out_json = PROJECT_ROOT / "outputs" / "logs" / f"{in_name}_{args.model}_events.json"

    # Annotate Video
    events = annotate_video(frames, scores, heatmaps, out_video, threshold=args.threshold)
    save_timeline_plot(scores, args.threshold, out_timeline)

    # Save Event Report
    report = {
        "input_source": str(args.input),
        "model": args.model,
        "total_frames": len(frames),
        "threshold": args.threshold,
        "max_score": float(scores.max()),
        "mean_score": float(scores.mean()),
        "detected_events": events,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Event summary saved: {out_json}")

    print("\n--- Detection Summary ---")
    print(f"Total Detected Anomaly Events: {len(events)}")
    for idx, ev in enumerate(events, 1):
        print(f"  Event #{idx}: Frames {ev['start_frame']} - {ev['end_frame']} | Peak: Frame {ev['peak_frame']} (Score: {ev['max_score']:.2f})")


if __name__ == "__main__":
    main()
