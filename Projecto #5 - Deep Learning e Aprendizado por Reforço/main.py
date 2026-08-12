from pathlib import Path

from ntxuva.gui.app import run_gui


if __name__ == "__main__":
    model = Path(__file__).resolve().parent / "models" / "ntxuva_dqn_best.pt"
    run_gui(model if model.exists() else None)
