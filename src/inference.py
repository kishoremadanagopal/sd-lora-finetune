"""
Inference utilities.

Loads a `StableDiffusionPipeline`, optionally attaches a trained LoRA adapter,
and generates images. Used both by the CLI script and the evaluation pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline


def load_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_path: Optional[str] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """Build the pipeline; attach LoRA weights if a path is given."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16 isn't supported on CPU.
    if device == "cpu":
        dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    if lora_path:
        # diffusers >=0.27 understands PEFT-format LoRA via load_lora_weights
        pipe.load_lora_weights(lora_path)
        print(f"[inference] loaded LoRA: {lora_path}")

    return pipe


def generate(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
    out_dir: Optional[str] = None,
) -> list:
    """Generate one image per prompt with a deterministic seed."""
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    images = []
    for i, p in enumerate(prompts):
        img = pipe(
            p,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]
        images.append(img)
        if out_dir:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            img.save(out / f"{i:03d}.png")
    return images
