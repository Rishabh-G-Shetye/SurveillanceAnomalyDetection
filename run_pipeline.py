"""
run_pipeline.py - Master Pipeline for Surveillance Video Anomaly Detection.

Trains all 5 models on normal surveillance footage, evaluates them using
the canonical UCSD benchmark protocol (frame-level AUC-ROC, EER, F1, event-level IoU,
and inference latency), and persists checkpoints, logs, and comparison visualizations.

Supports:
    --dataset Ped1  : Train & evaluate on UCSD Ped1 (36 test sequences)
    --dataset Ped2  : Train & evaluate on UCSD Ped2 (12 test sequences)
    --dataset both  : Train & evaluate on BOTH Ped1 and Ped2, outputting a comparative benchmark table

Usage:
    python run_pipeline.py --dataset both --epochs 10 --generate-video
    python run_pipeline.py --dataset Ped2 --epochs 10
    python run_pipeline.py --dataset both --skip-train
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.ucsd_dataset import PreprocessConfig, UCSDClipDataset
from data.generate_metadata import generate_metadata
from models.conv_ae import ConvAE
from models.convlstm_ae import ConvLSTMAE
from models.transformer_ae import TransformerAE
from models.frame_prediction import FramePredictionNet
from models.memory_ae import MemAE
from training.trainer import train_model
from evaluation.metrics import frame_and_event_level_eval, measure_inference_time
from evaluation.visualize import plot_roc_curves, plot_metric_comparison, plot_score_timeline
from utils.persistence import (
    save_checkpoint, load_checkpoint, save_results,
    save_figure, save_metrics_only,
)

MODEL_REGISTRY = {
    "ConvAE": {
        "factory": lambda: ConvAE(use_skip=False, score_agg="mean"),
        "base_name": "conv_ae",
    },
    "ConvLSTM-AE": {
        "factory": lambda: ConvLSTMAE(),
        "base_name": "convlstm_ae",
    },
    "TransformerAE": {
        "factory": lambda: TransformerAE(embed_dim=256, nhead=4),
        "base_name": "transformer_ae",
    },
    "FramePrediction": {
        "factory": lambda: FramePredictionNet(),
        "base_name": "frame_prediction",
    },
    "MemAE": {
        "factory": lambda: MemAE(num_slots=100, shrink_thresh=0.002, entropy_weight=0.0002),
        "base_name": "memae",
    },
}


def setup_data_for_dataset(dataset_name: str, args):
    """Generate metadata if needed and build datasets & loaders for a specific dataset."""
    metadata_path = PROJECT_ROOT / "data" / "metadata.csv"
    data_root = PROJECT_ROOT / "data" / "raw" / "UCSD_Anomaly_Dataset"

    if not data_root.exists():
        print(f"ERROR: Dataset not found at {data_root}")
        print("Please extract UCSD_Anomaly_Dataset.zip into data/raw/")
        sys.exit(1)

    if not metadata_path.exists():
        print("Generating metadata.csv with canonical ground truth...")
        generate_metadata(data_root, metadata_path)

    df = pd.read_csv(metadata_path)
    config = PreprocessConfig(
        target_size=(128, 128),
        window_length=8,
        stride=4,
        horizontal_flip_prob=0.5 if not args.skip_train else 0.0,
    )

    print(f"\nLoading datasets for {dataset_name}...")
    train_ds = UCSDClipDataset(df, data_root, dataset_name, "Train", config, is_train=True)
    test_ds = UCSDClipDataset(df, data_root, dataset_name, "Test", config, is_train=False)

    print(f"  [{dataset_name}] Train clips: {len(train_ds)} | Test clips: {len(test_ds)}")
    print(f"  [{dataset_name}] Skipped (corrupted) train: {train_ds.skipped_count}")
    print(f"  [{dataset_name}] Skipped (corrupted) test:  {test_ds.skipped_count}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    return train_ds, test_ds, train_loader, test_loader


def train_and_evaluate_single_model(
    model_name, model_info, dataset_name, train_loader, test_loader, test_ds, args, device
):
    """Train a single model, evaluate using canonical frame-level protocol, and persist."""
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name} | MODEL: {model_name}")
    print(f"{'='*60}")

    model = model_info["factory"]()
    ckpt_name = f"{model_info['base_name']}_{dataset_name.lower()}"
    ckpt_file = PROJECT_ROOT / "models" / f"{ckpt_name}.pt"

    if (args.skip_existing or args.skip_train) and ckpt_file.exists():
        print(f"\n  Found existing checkpoint: {ckpt_name}.pt (loading without training)")
        model = load_checkpoint(model, ckpt_name, device=device)
    elif not args.skip_train:
        print(f"\n  Training {model_name} on {dataset_name} for {args.epochs} epochs...")
        history = train_model(
            model, train_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device,
            patience=args.patience,
        )
        save_checkpoint(model, ckpt_name)
    else:
        print(f"  WARNING: Checkpoint {ckpt_name}.pt not found, skipping {model_name}")
        return None

    # Run batched frame-level and event-level evaluation
    print(f"\n  Running Frame & Event-Level Evaluation for {model_name} on {dataset_name}...")
    eval_results = frame_and_event_level_eval(
        model, test_ds, device=device, batch_size=args.batch_size, normalize_per_seq=True, smooth=True
    )
    timing = measure_inference_time(model, test_ds[0]["clip"], device=device)

    all_metrics = {k: v for k, v in eval_results.items() if not isinstance(v, np.ndarray)}
    all_metrics.update(timing)
    save_metrics_only(f"{ckpt_name}_metrics", all_metrics)

    print(f"\n  {model_name} ({dataset_name}) Results:")
    print(f"    Frame AUC-ROC:  {eval_results['frame_auc_roc']:.4f}")
    print(f"    Frame EER:      {eval_results['frame_eer']:.4f}")
    print(f"    Frame F1:       {eval_results['frame_f1']:.4f}")
    print(f"    Frame Precision:{eval_results['frame_precision']:.4f}")
    print(f"    Frame Recall:   {eval_results['frame_recall']:.4f}")
    print(f"    Event Precision:{eval_results['event_precision_avg']:.4f}")
    print(f"    Event Recall:   {eval_results['event_recall_avg']:.4f}")
    print(f"    Event F1:       {eval_results['event_f1_avg']:.4f}")
    print(f"    Inference Time: {timing['ms_per_frame']:.2f} ms/frame ({timing['fps']:.1f} FPS)")

    # Timeline visualization for sample sequence
    sample_seq = "Test003" if dataset_name == "Ped1" else "Test004"
    try:
        fig_timeline = plot_score_timeline(model, test_ds, sample_seq, device=device)
        save_figure(fig_timeline, f"score_timeline_{sample_seq}_{ckpt_name}")
    except Exception as e:
        print(f"    Timeline plot skipped: {e}")

    return {
        "model": model,
        "metrics": all_metrics,
        "scores": eval_results["pooled_scores"],
        "labels": eval_results["pooled_labels"],
        "ckpt_name": ckpt_name,
    }


def generate_dataset_comparison_outputs(all_results, dataset_name, test_ds, device):
    """Generate ROC curves, comparison bar charts, and summary JSON for one dataset."""
    print(f"\n{'='*60}")
    print(f"  Generating Visualizations for {dataset_name}")
    print(f"{'='*60}")

    # ROC curve comparison
    model_curves = {name: (r["labels"], r["scores"]) for name, r in all_results.items() if "scores" in r and "labels" in r}
    if model_curves:
        fig_roc = plot_roc_curves(model_curves)
        save_figure(fig_roc, f"roc_comparison_all_models_{dataset_name.lower()}")

    results_by_model = {name: r["metrics"] for name, r in all_results.items()}

    # Metric comparison bar chart
    fig_bar = plot_metric_comparison(
        results_by_model,
        metrics=("frame_auc_roc", "frame_f1", "event_precision_avg", "event_recall_avg"),
    )
    save_figure(fig_bar, f"metric_comparison_all_models_{dataset_name.lower()}")

    # Summary table
    print(f"\n  Summary Comparison Table ({dataset_name}):")
    summary_rows = {}
    for name, r in all_results.items():
        m = r["metrics"]
        summary_rows[name] = {
            "Frame AUC-ROC": f"{m['frame_auc_roc']:.4f}",
            "Frame EER": f"{m['frame_eer']:.4f}",
            "Frame F1": f"{m['frame_f1']:.4f}",
            "Frame Prec": f"{m['frame_precision']:.4f}",
            "Frame Recall": f"{m['frame_recall']:.4f}",
            "Event Prec": f"{m['event_precision_avg']:.4f}",
            "Event Recall": f"{m['event_recall_avg']:.4f}",
            "ms/frame": f"{m['ms_per_frame']:.2f}",
            "FPS": f"{m['fps']:.1f}",
        }

    summary_df = pd.DataFrame(summary_rows).T
    print(summary_df.to_string())

    summary_path = PROJECT_ROOT / "outputs" / "logs" / f"full_comparison_{dataset_name.lower()}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {k: {kk: float(vv) for kk, vv in r["metrics"].items()} for k, r in all_results.items()},
            f, indent=2,
        )
    print(f"\n  Full comparison JSON saved to: {summary_path}")


def run_pipeline_for_dataset(dataset_name: str, args, device):
    """Run full training & evaluation for a single dataset (Ped1 or Ped2)."""
    train_ds, test_ds, train_loader, test_loader = setup_data_for_dataset(dataset_name, args)

    models_to_run = args.models or list(MODEL_REGISTRY.keys())
    all_results = {}

    for model_name in models_to_run:
        model_info = MODEL_REGISTRY[model_name]
        result = train_and_evaluate_single_model(
            model_name, model_info, dataset_name, train_loader, test_loader, test_ds, args, device
        )
        if result is not None:
            all_results[model_name] = result

    if len(all_results) > 1:
        generate_dataset_comparison_outputs(all_results, dataset_name, test_ds, device)

    # Optional video generation
    if args.generate_video and all_results:
        best_name = list(all_results.keys())[0]
        sample_seq = "Test003" if dataset_name == "Ped1" else "Test004"
        sample_dir = PROJECT_ROOT / "data" / "raw" / "UCSD_Anomaly_Dataset" / f"UCSD{dataset_name.lower()}" / "Test" / sample_seq
        if sample_dir.exists():
            print(f"\nGenerating sample alert video using {best_name} on {sample_seq} ({dataset_name})...")
            from inference import load_frames_from_source, run_inference, annotate_video
            frames = load_frames_from_source(sample_dir)
            cfg = PreprocessConfig(target_size=(128, 128), window_length=8, stride=1)
            scores, heatmaps = run_inference(all_results[best_name]["model"], frames, cfg, device)
            out_vid = PROJECT_ROOT / "outputs" / "videos" / f"sample_alert_{sample_seq}_{best_name}_{dataset_name.lower()}.mp4"
            annotate_video(frames, scores, heatmaps, out_vid, threshold=0.5)

    return all_results


def print_dual_dataset_comparison(ped1_results, ped2_results):
    """Print clean side-by-side comparison table for Ped1 vs Ped2."""
    print(f"\n{'='*75}")
    print("  DUAL DATASET BENCHMARK SUMMARY: UCSD Ped1 vs UCSD Ped2")
    print(f"{'='*75}")

    rows = []
    for model_name in ped1_results.keys():
        if model_name in ped2_results:
            m1 = ped1_results[model_name]["metrics"]
            m2 = ped2_results[model_name]["metrics"]
            rows.append({
                "Model": model_name,
                "Ped1 AUC": f"{m1['frame_auc_roc']:.4f}",
                "Ped1 EER": f"{m1['frame_eer']:.4f}",
                "Ped1 F1":  f"{m1['frame_f1']:.4f}",
                "Ped2 AUC": f"{m2['frame_auc_roc']:.4f}",
                "Ped2 EER": f"{m2['frame_eer']:.4f}",
                "Ped2 F1":  f"{m2['frame_f1']:.4f}",
                "ms/frame": f"{m1['ms_per_frame']:.2f}",
                "FPS":      f"{m1['fps']:.1f}",
            })

    df = pd.DataFrame(rows).set_index("Model")
    print(df.to_string())

    # Save dual comparison json
    dual_json = PROJECT_ROOT / "outputs" / "logs" / "dual_dataset_comparison.json"
    dual_data = {
        "Ped1": {k: {kk: float(vv) for kk, vv in r["metrics"].items()} for k, r in ped1_results.items()},
        "Ped2": {k: {kk: float(vv) for kk, vv in r["metrics"].items()} for k, r in ped2_results.items()},
    }
    with open(dual_json, "w") as f:
        json.dump(dual_data, f, indent=2)
    print(f"\n  Dual dataset comparison JSON saved to: {dual_json}")


def main():
    parser = argparse.ArgumentParser(description="Surveillance Video Anomaly Detection Pipeline")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--dataset", type=str, default="both", choices=["Ped1", "Ped2", "both"])
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for Windows)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, evaluate saved checkpoints")
    parser.add_argument("--skip-existing", action="store_true", help="Skip training if checkpoint already exists on disk")
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Which models to train/evaluate (default: all)"
    )
    parser.add_argument("--generate-video", action="store_true", help="Generate sample alert videos")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if args.dataset in ("Ped1", "Ped2"):
        run_pipeline_for_dataset(args.dataset, args, device)
    elif args.dataset == "both":
        print("\n" + "#" * 60)
        print("  STEP 1: RUNNING UCSD PED1 BENCHMARK")
        print("#" * 60)
        ped1_results = run_pipeline_for_dataset("Ped1", args, device)

        print("\n" + "#" * 60)
        print("  STEP 2: RUNNING UCSD PED2 BENCHMARK")
        print("#" * 60)
        ped2_results = run_pipeline_for_dataset("Ped2", args, device)

        if ped1_results and ped2_results:
            print_dual_dataset_comparison(ped1_results, ped2_results)

    print(f"\n{'='*60}")
    print("  Pipeline Execution Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
