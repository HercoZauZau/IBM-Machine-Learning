from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def save_plot(x, y, xlabel: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    args = parser.parse_args()
    df = pd.read_csv(args.history)
    out = args.history.parent / "plots"
    out.mkdir(exist_ok=True)

    if "mean_loss" in df:
        series = df["mean_loss"].rolling(100, min_periods=1).mean()
        save_plot(df["episode"], series, "Episódio", "Loss", "Loss média móvel", out / "loss.png")
    if "epsilon" in df:
        save_plot(df["episode"], df["epsilon"], "Episódio", "Epsilon", "Exploração epsilon-greedy", out / "epsilon.png")
    if "moves" in df:
        series = df["moves"].rolling(100, min_periods=1).mean()
        save_plot(df["episode"], series, "Episódio", "Jogadas", "Duração média móvel das partidas", out / "game_length.png")
    print(f"Gráficos em {out}")


if __name__ == "__main__":
    main()
