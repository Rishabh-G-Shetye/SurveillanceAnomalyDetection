"""
Test Sequence Anomaly Detection Visualizer.

Demonstrates how the anomaly detection models detect anomalies by generating:
1. Multi-panel comparison grid figures showing:
   - Raw surveillance frame
   - Ground-truth pixel mask (from _gt bmp files)
   - Continuous reconstruction/prediction anomaly heatmap
   - Thresholded detection overlay
2. Side-by-side 4-panel annotated video / GIF illustrating frame-by-frame detection.

Usage:
    python src/evaluation/visualize_detections.py --sequence Test003 --model ConvAE
    python src/evaluation/visualize_detections.py --sequence Test004 --model FramePrediction
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.ucsd_dataset import PreprocessConfig, build_transform
from models.conv_ae import ConvAE
from models.convlstm_ae import ConvLSTMAE
from models.transformer_ae import TransformerAE
from models.frame_prediction import FramePredictionNet
from models.memory_ae import MemAE
from utils.persistence import load_checkpoint

MODEL_FACTORIES = {
    "ConvAE": (lambda: ConvAE(use_skip=False, score_agg="mean"), "conv_ae_baseline"),
    "ConvLSTM-AE": (lambda: ConvLSTMAE(), "convlstm_ae_baseline"),
    "TransformerAE": (lambda: TransformerAE(embed_dim=256, nhead=4), "transformer_ae_baseline"),
    "FramePrediction": (lambda: FramePredictionNet(), "frame_prediction_baseline"),
    "MemAE": (lambda: MemAE(), "memae_baseline"),
}


def load_sequence_data(sequence_name: str, dataset: str = "Ped1"):
    """Load frames and corresponding ground-truth masks for a given sequence."""
    ped_dir = f"UCSD{dataset.lower()}"
    seq_dir = PROJECT_ROOT / "data" / "raw" / "UCSD_Anomaly_Dataset" / ped_dir / "Test" / sequence_name
    gt_dir = PROJECT_ROOT / "data" / "raw" / "UCSD_Anomaly_Dataset" / ped_dir / "Test" / f"{sequence_name}_gt"

    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    frame_files = sorted([p for p in seq_dir.glob("*") if p.suffix.lower() in {".tif", ".png", ".jpg"}])
    gt_files = sorted([p for p in gt_dir.glob("*.bmp")]) if gt_dir.exists() else []

    frames = [cv2.imread(str(p)) for p in frame_files]
    masks = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in gt_files] if gt_files else None

    return frames, masks, frame_files


def run_spatial_inference(model, frames, config, device):
    """Run model on frames and compute spatial error maps for every frame."""
    model.eval()
    transform = build_transform(config, is_train=False)
    T = config.window_length
    num_frames = len(frames)
    H, W = frames[0].shape[:2]

    # Pre-transform frames to normalized tensors
    gray_tensors = []
    for f in frames:
        pil_img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        gray_tensors.append(transform(pil_img))

    heatmaps = [[] for _ in range(num_frames)]
    frame_scores = [[] for _ in range(num_frames)]
    is_prediction = (model.__class__.__name__ == "FramePredictionNet")

    with torch.no_grad():
        for start in range(0, num_frames - T + 1, 1):
            window = gray_tensors[start : start + T]
            clip = torch.stack(window).unsqueeze(0).to(device)

            if is_prediction:
                pred = model(clip)
                target = clip[:, -1]
                diff = (pred - target).abs().squeeze().cpu().numpy()  # (128, 128)
                score = float(((pred - target) ** 2).mean().item())
                last_idx = start + T - 1
                heatmaps[last_idx].append(diff)
                frame_scores[last_idx].append(score)
            else:
                recon = model(clip)
                diff = (recon - clip).abs().squeeze(2).squeeze(0).cpu().numpy()  # (T, 128, 128)
                for t in range(T):
                    idx = start + t
                    err = float(((recon[:, t] - clip[:, t]) ** 2).mean().item())
                    heatmaps[idx].append(diff[t])
                    frame_scores[idx].append(err)

    # Aggregate spatial heatmaps and scores
    final_heatmaps = []
    final_scores = np.zeros(num_frames)

    for i in range(num_frames):
        if heatmaps[i]:
            avg_hm = np.mean(np.stack(heatmaps[i]), axis=0)
            final_scores[i] = np.mean(frame_scores[i])
        else:
            avg_hm = final_heatmaps[-1] if final_heatmaps else np.zeros((128, 128))
            final_scores[i] = final_scores[max(0, i - 1)]

        # Resize heatmap back to original frame dimensions
        hm_full = cv2.resize(avg_hm, (W, H))
        final_heatmaps.append(hm_full)

    # Normalize scores
    s_min, s_max = final_scores.min(), final_scores.max()
    if s_max - s_min > 1e-8:
        norm_scores = (final_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = final_scores

    return norm_scores, final_heatmaps


def generate_comparison_grid(
    frames: list[np.ndarray],
    masks: list[np.ndarray] | None,
    heatmaps: list[np.ndarray],
    scores: np.ndarray,
    key_frame_indices: list[int],
    sequence_name: str,
    model_name: str,
    output_path: Path,
):
    """
    Generate high-resolution grid figure showing key frames with:
    Column 1: Raw Frame
    Column 2: Ground-Truth Pixel Mask
    Column 3: Model Anomaly Heatmap
    Column 4: Anomaly Detection Overlay
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_rows = len(key_frame_indices)
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 3.2 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, f_idx in enumerate(key_frame_indices):
        frame = frames[f_idx]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]

        # 1. Raw Frame
        axes[row_idx, 0].imshow(rgb_frame)
        axes[row_idx, 0].set_title(f"Frame #{f_idx + 1} (Score: {scores[f_idx]:.2f})", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 2. Ground-Truth Mask
        if masks and f_idx < len(masks):
            mask = masks[f_idx]
            gt_rgb = np.zeros_like(rgb_frame)
            gt_rgb[mask > 0] = [255, 50, 50]  # Red highlight for anomalous pixels
            blended_gt = cv2.addWeighted(rgb_frame, 0.7, gt_rgb, 0.3, 0)
            axes[row_idx, 1].imshow(blended_gt)
            axes[row_idx, 1].set_title("Ground-Truth Mask (Pixel GT)" if (mask > 0).any() else "Ground-Truth (Normal)", fontsize=11)
        else:
            axes[row_idx, 1].text(0.5, 0.5, "No Pixel Mask", ha="center", va="center")
        axes[row_idx, 1].axis("off")

        # 3. Model Anomaly Heatmap
        hm = heatmaps[f_idx]
        hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        axes[row_idx, 2].imshow(hm_norm, cmap="inferno")
        axes[row_idx, 2].set_title(f"Reconstruction Error Heatmap", fontsize=11)
        axes[row_idx, 2].axis("off")

        # 4. Detection Overlay
        hm_resized = cv2.resize((hm_norm * 255).astype(np.uint8), (W, H))
        color_hm = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(rgb_frame, 0.65, cv2.cvtColor(color_hm, cv2.COLOR_BGR2RGB), 0.35, 0)

        # Draw status badge
        status = "ALERT" if scores[f_idx] >= 0.5 else "NORMAL"
        badge_col = "red" if status == "ALERT" else "green"
        axes[row_idx, 3].imshow(overlay)
        axes[row_idx, 3].set_title(f"Detection Overlay: [{status}]", fontsize=11, color=badge_col, fontweight="bold")
        axes[row_idx, 3].axis("off")

    fig.suptitle(f"Anomaly Detection Visual Breakdown -- {sequence_name} ({model_name})", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison grid saved: {output_path}")


def generate_side_by_side_video(
    frames: list[np.ndarray],
    masks: list[np.ndarray] | None,
    heatmaps: list[np.ndarray],
    scores: np.ndarray,
    output_path: Path,
    sequence_name: str,
    model_name: str,
    threshold: float = 0.5,
    fps: int = 15,
):
    """
    Generate a 4-panel video showing:
    Top-Left: Original Footage
    Top-Right: Ground-Truth Pixel Mask Overlay
    Bottom-Left: Model Spatial Anomaly Heatmap
    Bottom-Right: Combined Detection Alert & Score Gauge
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    H, W, _ = frames[0].shape
    panel_w, panel_h = 320, int(320 * (H / W))
    canvas_w, canvas_h = panel_w * 2, panel_h * 2 + 50  # Extra space for bottom timeline bar

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))

    for i, frame in enumerate(frames):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # 1. Top-Left: Original Frame
        f_small = cv2.resize(frame, (panel_w, panel_h))
        cv2.putText(f_small, f"Raw Frame #{i+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        canvas[0:panel_h, 0:panel_w] = f_small

        # 2. Top-Right: Ground-Truth Mask Overlay
        if masks and i < len(masks):
            mask = masks[i]
            m_small = cv2.resize(mask, (panel_w, panel_h))
            gt_overlay = f_small.copy()
            gt_colored = np.zeros_like(gt_overlay)
            gt_colored[m_small > 0] = [0, 255, 255]  # Yellow/Cyan highlight
            gt_overlay = cv2.addWeighted(gt_overlay, 0.7, gt_colored, 0.3, 0)
            cv2.putText(gt_overlay, "Ground-Truth Anomaly Mask", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        else:
            gt_overlay = f_small.copy()
            cv2.putText(gt_overlay, "Ground-Truth (No mask)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        canvas[0:panel_h, panel_w:panel_w*2] = gt_overlay

        # 3. Bottom-Left: Spatial Heatmap
        hm = heatmaps[i]
        hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        hm_small = cv2.resize((hm_norm * 255).astype(np.uint8), (panel_w, panel_h))
        color_hm = cv2.applyColorMap(hm_small, cv2.COLORMAP_JET)
        cv2.putText(color_hm, f"Spatial Error Heatmap ({model_name})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        canvas[panel_h:panel_h*2, 0:panel_w] = color_hm

        # 4. Bottom-Right: Detection Alert Overlay
        is_anom = scores[i] >= threshold
        det_overlay = cv2.addWeighted(f_small, 0.65, color_hm, 0.35, 0)
        status_text = "ANOMALY DETECTED!" if is_anom else "NORMAL"
        badge_color = (0, 0, 255) if is_anom else (0, 200, 0)
        cv2.rectangle(det_overlay, (10, 10), (panel_w - 10, 45), (20, 20, 20), -1)
        cv2.putText(det_overlay, status_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_color, 2)
        cv2.putText(det_overlay, f"Score: {scores[i]:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        canvas[panel_h:panel_h*2, panel_w:panel_w*2] = det_overlay

        # 5. Bottom Timeline Bar
        bar_y = panel_h * 2 + 10
        cv2.putText(canvas, f"{sequence_name} | Frame {i+1}/{len(frames)}", (20, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        bar_start_x = 220
        bar_w = canvas_w - bar_start_x - 30
        cv2.rectangle(canvas, (bar_start_x, bar_y + 10), (bar_start_x + bar_w, bar_y + 25), (60, 60, 60), -1)
        progress_w = int(bar_w * ((i + 1) / len(frames)))
        cv2.rectangle(canvas, (bar_start_x, bar_y + 10), (bar_start_x + progress_w, bar_y + 25), (0, 180, 255), -1)

        writer.write(canvas)

    writer.release()
    print(f"Side-by-side video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Sequence Anomaly Detection Visualizer")
    parser.add_argument("--sequence", type=str, default="Test003", help="Sequence name (e.g. Test003, Test004)")
    parser.add_argument("--dataset", type=str, default="Ped1", choices=["Ped1", "Ped2"])
    parser.add_argument("--model", type=str, default="ConvAE", choices=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--key-frames", nargs="+", type=int, default=[30, 95, 140, 185], help="0-indexed key frames")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running visualizer on {device}...")

    # Load Model
    factory, default_ckpt = MODEL_FACTORIES[args.model]
    model = factory()
    ckpt_path = args.checkpoint or str(PROJECT_ROOT / "models" / f"{default_ckpt}.pt")

    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found. Running with untrained weights.")

    model.to(device)

    # Load Sequence Data
    print(f"Loading sequence {args.sequence} ({args.dataset})...")
    frames, masks, frame_files = load_sequence_data(args.sequence, args.dataset)
    print(f"Loaded {len(frames)} frames. Has pixel masks: {masks is not None}")

    config = PreprocessConfig(target_size=(128, 128), window_length=8, stride=1)
    scores, heatmaps = run_spatial_inference(model, frames, config, device)

    # Generate Outputs
    out_grid = PROJECT_ROOT / "outputs" / "figures" / f"anomaly_detection_grid_{args.sequence}_{args.model}.png"
    generate_comparison_grid(frames, masks, heatmaps, scores, args.key_frames, args.sequence, args.model, out_grid)

    out_video = PROJECT_ROOT / "outputs" / "videos" / f"detection_breakdown_{args.sequence}_{args.model}.mp4"
    generate_side_by_side_video(frames, masks, heatmaps, scores, out_video, args.sequence, args.model, threshold=args.threshold)


if __name__ == "__main__":
    main()
