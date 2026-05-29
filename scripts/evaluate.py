#!/usr/bin/env python
"""
Evaluate base vs LoRA-fine-tuned model with CLIP score.

Usage:
  python scripts/evaluate.py \
      --lora outputs/checkpoints/lora_run/lora/lora_step_001000.safetensors \
      --prompts-file configs/eval_prompts.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate  # noqa: E402


DEFAULT_PROMPTS = [
    "a drawing of a green pokemon with red eyes",
    "a cartoon character with a face on it",
    "a yellow and blue toy with a red nose",
    "a small pink pokemon with big ears",
    "a fire-type pokemon with orange flames",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora", default=None, help="Path to LoRA .safetensors file.")
    ap.add_argument("--prompts-file", default=None, help="One prompt per line.")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.prompts_file:
        prompts = [
            ln.strip()
            for ln in Path(args.prompts_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        prompts = DEFAULT_PROMPTS

    evaluate(
        prompts=prompts,
        model_id=args.model,
        lora_path=args.lora,
        out_dir=args.out,
        clip_model=args.clip_model,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
