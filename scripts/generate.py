#!/usr/bin/env python
"""
Generate images from a prompt list, optionally with a trained LoRA.

Usage:
  python scripts/generate.py --prompts "a green dragon pokemon" "a blue robot pokemon" \
      --lora outputs/checkpoints/lora_run/lora/lora_step_001000.safetensors \
      --out outputs/samples/gen
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import generate, load_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora", default=None, help="Path to .safetensors LoRA weights.")
    ap.add_argument("--out", default="outputs/samples/gen")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pipe = load_pipeline(model_id=args.model, lora_path=args.lora)
    generate(
        pipe, args.prompts,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        out_dir=args.out,
    )
    print(f"[generate] {len(args.prompts)} image(s) → {args.out}")


if __name__ == "__main__":
    main()
