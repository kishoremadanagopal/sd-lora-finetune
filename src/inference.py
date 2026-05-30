"""
Inference utilities.

Loads a `StableDiffusionPipeline`, optionally attaches a trained LoRA adapter
via PEFT (matching the training-time config), and generates images.

Uses PEFT injection directly rather than `pipe.load_lora_weights()` because
the .safetensors files saved during training use PEFT key format
(`base_model.model.<...>.lora_A.weight`), which the diffusers loader silently
ignores for some peft/diffusers version combos.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline


# Must match the LoRA config used in training (see configs/default.yaml).
_LORA_RANK = 4
_LORA_ALPHA = 4
_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
_LORA_DROPOUT = 0.0


def load_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_path: Optional[str] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """Build the pipeline; attach LoRA weights to the UNet via PEFT if a path is given."""
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
        from safetensors.torch import load_file
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

        state_dict = load_file(lora_path)
        print(f"[inference] LoRA file has {len(state_dict)} keys")

        # Re-wrap the UNet with the same LoRA config used at training time, then
        # load the trained adapter weights into it.
        lora_config = LoraConfig(
            r=_LORA_RANK,
            lora_alpha=_LORA_ALPHA,
            target_modules=_LORA_TARGETS,
            lora_dropout=_LORA_DROPOUT,
            bias="none",
        )
        pipe.unet = get_peft_model(pipe.unet, lora_config)
        set_peft_model_state_dict(pipe.unet, state_dict)
        pipe.unet = pipe.unet.to(device=device, dtype=dtype)
        print(f"[inference] LoRA loaded: {lora_path}")

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
