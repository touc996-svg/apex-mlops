"""Controlled retraining from a newly collected, contract-valid CSV dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from apex.data import load_dataset
from apex.training import train_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain APEX and evaluate a registry candidate.")
    parser.add_argument("--csv", type=Path, required=True, help="New labeled CSV using the APEX feature contract.")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    frame = load_dataset(args.csv)
    result = train_models(frame, args.output, track_mlflow=True, tracking_uri=args.tracking_uri)
    print(json.dumps({"selected_model": result["selected_model"],
                      "selected_metrics": result["metrics"][result["selected_model"]],
                      "model_registry": result["model_registry"],
                      "model_path": result["model_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
