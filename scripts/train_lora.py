#!/usr/bin/env python
"""
Train a LoRA adapter on Stable Diffusion 1.5.

Usage:
  python scripts/train_lora.py --config configs/default.yaml
  python scripts/train_lora.py --config configs/default.yaml \
      dataset.image_dir=data/raw/my_concept dataset.name=null \
      training.max_train_steps=800
"""
import argparse
import sys
from pathlib import Path

# Make src importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf  # noqa: E402

from src.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "overrides", nargs="*",
        help="OmegaConf-style overrides, e.g. training.learning_rate=5e-5",
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
