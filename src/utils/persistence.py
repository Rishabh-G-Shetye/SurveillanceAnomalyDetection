"""
Model/result persistence -- local version (replaces Colab Drive paths).

Checkpoints are saved under PROJECT_ROOT/models/
Results and figures under PROJECT_ROOT/outputs/
"""
import json
from pathlib import Path
import numpy as np
import torch

# Project root is two levels up from this file (src/utils/persistence.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CKPT_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "logs"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def save_checkpoint(model: torch.nn.Module, name: str) -> Path:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{name}.pt"
    torch.save(model.state_dict(), path)
    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(model: torch.nn.Module, name: str, device=None) -> torch.nn.Module:
    path = CKPT_DIR / f"{name}.pt"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device)


def save_results(name: str, results: dict, scores: np.ndarray, labels: np.ndarray) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    np.savez(RESULTS_DIR / f"{name}_scores.npz", scores=scores, labels=labels)
    print(f"  Results saved: {RESULTS_DIR / name}")


def load_results(name: str):
    with open(RESULTS_DIR / f"{name}_metrics.json") as f:
        results = json.load(f)
    npz = np.load(RESULTS_DIR / f"{name}_scores.npz")
    return results, npz["scores"], npz["labels"]


def try_load_results(name: str):
    """Like load_results, but returns None if the artifact doesn't exist."""
    try:
        return load_results(name)
    except FileNotFoundError:
        return None


def save_metrics_only(name: str, results: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_metrics.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return path


def save_figure(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {path}")
    return path
