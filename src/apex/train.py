"""Command-line entry point for the reproducible training workflow."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from apex.data import load_dataset, save_dataset
from apex.training import train_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate APEX churn models.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV using the APEX feature contract.")
    parser.add_argument("--rows", type=int, default=1800, help="Synthetic rows when --csv is not supplied.")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--tracking-uri", default=None, help="MLflow URI; defaults to sqlite:///mlflow.db.")
    parser.add_argument("--disable-mlflow", action="store_true")
    args = parser.parse_args()
    frame = load_dataset(args.csv, rows=args.rows)
    if args.csv is None:
        frame.attrs["synthetic_data"] = True
        save_dataset(frame, Path("data/synthetic_churn.csv"))
    result = train_models(frame, args.output, track_mlflow=not args.disable_mlflow,
                          tracking_uri=args.tracking_uri)
    print(json.dumps({
        "selected_model": result["selected_model"],
        "metrics": result["metrics"][result["selected_model"]],
        "model_path": result["model_path"],
        "tracking_uri": result["tracking_uri"],
        "tracking": "disabled" if args.disable_mlflow else "enabled",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
