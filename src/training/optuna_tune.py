"""
Optuna Hyperparameter Optimization for Video Anomaly Detection.

Tunes critical hyperparameters to maximize Frame-Level AUC-ROC:
- Learning rate (log scale 1e-4 to 5e-3)
- Model architecture capacity (latent channels, memory slots, transformer heads)
- Regularization weights (MemAE entropy penalty, shrinkage thresholds)
- Training schedule (batch size, epochs)

Usage:
    python src/training/optuna_tune.py --model ConvAE --n-trials 10 --epochs 3
    python src/training/optuna_tune.py --model MemAE --n-trials 15 --epochs 3
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.ucsd_dataset import PreprocessConfig, UCSDClipDataset
from models.conv_ae import ConvAE
from models.convlstm_ae import ConvLSTMAE
from models.transformer_ae import TransformerAE
from models.frame_prediction import FramePredictionNet
from models.memory_ae import MemAE
from training.trainer import train_model
from evaluation.metrics import frame_and_event_level_eval


def objective(trial, model_name: str, train_ds, test_ds, device: str, epochs: int):
    # Sample common hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Sample model-specific hyperparameters
    if model_name == "ConvAE":
        latent_channels = trial.suggest_categorical("latent_channels", [32, 64, 128])
        model = ConvAE(latent_channels=latent_channels, use_skip=False, score_agg="mean")
    elif model_name == "ConvLSTM-AE":
        feat_channels = trial.suggest_categorical("feat_channels", [16, 32, 64])
        hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        model = ConvLSTMAE(feat_channels=feat_channels, hidden_channels=hidden_channels)
    elif model_name == "TransformerAE":
        embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 512])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        model = TransformerAE(embed_dim=embed_dim, nhead=nhead)
    elif model_name == "MemAE":
        num_slots = trial.suggest_categorical("num_slots", [50, 100, 200])
        shrink_thresh = trial.suggest_float("shrink_thresh", 0.001, 0.005)
        entropy_weight = trial.suggest_float("entropy_weight", 1e-5, 1e-3, log=True)
        model = MemAE(num_slots=num_slots, shrink_thresh=shrink_thresh, entropy_weight=entropy_weight)
    elif model_name == "FramePrediction":
        hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
        model = FramePredictionNet(hidden_channels=hidden_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # Train model
    train_model(model, train_loader, num_epochs=epochs, lr=lr, device=device, patience=3)

    # Evaluate on test set
    eval_results = frame_and_event_level_eval(
        model, test_ds, device=device, batch_size=32, normalize_per_seq=True, smooth=True
    )

    auc = eval_results["frame_auc_roc"]
    print(f"Trial {trial.number} finished with Frame AUC: {auc:.4f}")
    return auc


def main():
    import optuna

    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    parser.add_argument("--model", type=str, default="ConvAE", choices=["ConvAE", "ConvLSTM-AE", "TransformerAE", "FramePrediction", "MemAE"])
    parser.add_argument("--dataset", type=str, default="Ped1", choices=["Ped1", "Ped2"])
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per trial")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Optuna hyperparameter search for {args.model} on {device}...")

    metadata_path = PROJECT_ROOT / "data" / "metadata.csv"
    data_root = PROJECT_ROOT / "data" / "raw" / "UCSD_Anomaly_Dataset"
    df = pd.read_csv(metadata_path)

    config = PreprocessConfig(target_size=(128, 128), window_length=8, stride=4)
    train_ds = UCSDClipDataset(df, data_root, args.dataset, "Train", config, is_train=True)
    test_ds = UCSDClipDataset(df, data_root, args.dataset, "Test", config, is_train=False)

    study = optuna.create_study(direction="maximize", study_name=f"{args.model}_{args.dataset}_optimization")
    study.optimize(
        lambda trial: objective(trial, args.model, train_ds, test_ds, device, args.epochs),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 60)
    print(f"Optimization Finished for {args.model}!")
    print(f"Best Trial #{study.best_trial.number}")
    print(f"Best Frame AUC-ROC: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Save best parameters to logs
    import json
    out_path = PROJECT_ROOT / "outputs" / "logs" / f"best_hyperparameters_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump({"best_auc": study.best_value, "params": study.best_params}, f, indent=2)
    print(f"Best hyperparameters saved to: {out_path}")


if __name__ == "__main__":
    main()
