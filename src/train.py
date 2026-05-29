"""
LoRA fine-tuning loop for Stable Diffusion 1.5.

Strategy:
  - Freeze VAE, text encoder, and UNet base weights.
  - Inject LoRA adapters into UNet cross-attention projections via PEFT.
  - Train only LoRA params with standard noise-prediction (eps) loss.
  - Periodically sample validation images and save LoRA checkpoints.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .dataset import build_dataset, collate_fn


# --------------------------------------------------------------------------- #
# Model setup
# --------------------------------------------------------------------------- #
def load_models(model_id: str, revision: str | None = None):
    """Load the four SD components. UNet and text encoder are returned for training context."""
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", revision=revision)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", revision=revision)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def inject_lora(unet: UNet2DConditionModel, lora_cfg) -> UNet2DConditionModel:
    """Wrap UNet with PEFT LoRA adapters on cross-attention projections."""
    config = LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        target_modules=list(lora_cfg.target_modules),
        lora_dropout=lora_cfg.dropout,
        bias="none",
    )
    unet = get_peft_model(unet, config)
    unet.print_trainable_parameters()
    return unet


# --------------------------------------------------------------------------- #
# Validation sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_validation(
    cfg,
    accelerator: Accelerator,
    unet,
    text_encoder,
    vae,
    tokenizer,
    step: int,
    save_dir: Path,
) -> List[Path]:
    """Sample validation prompts with the current LoRA-adapted UNet."""
    unet_eval = accelerator.unwrap_model(unet)
    unet_eval.eval()

    pipeline = StableDiffusionPipeline.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        unet=unet_eval,
        text_encoder=accelerator.unwrap_model(text_encoder),
        vae=vae,
        tokenizer=tokenizer,
        safety_checker=None,
        torch_dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32,
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    sample_dir = save_dir / f"step_{step:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    gen = torch.Generator(device=accelerator.device).manual_seed(cfg.training.seed)

    for i, prompt in enumerate(cfg.training.validation_prompts):
        for k in range(cfg.training.num_validation_images):
            img = pipeline(
                prompt,
                num_inference_steps=cfg.eval.num_inference_steps,
                guidance_scale=cfg.eval.guidance_scale,
                generator=gen,
            ).images[0]
            out_path = sample_dir / f"p{i:02d}_s{k}.png"
            img.save(out_path)
            saved.append(out_path)

    del pipeline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    unet_eval.train()
    return saved


# --------------------------------------------------------------------------- #
# Save / load LoRA weights
# --------------------------------------------------------------------------- #
def save_lora(unet, save_dir: Path, step: int) -> Path:
    """Save just the LoRA adapter state — small, portable file."""
    save_dir.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(unet)
    path = save_dir / f"lora_step_{step:06d}.safetensors"
    from safetensors.torch import save_file
    save_file(state, str(path))
    return path


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
def train(cfg) -> None:
    out_dir = Path(cfg.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with="tensorboard",
        project_dir=str(out_dir / "logs"),
    )
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)

    # ----- models ------------------------------------------------------- #
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(
        cfg.model.pretrained_model_name_or_path, cfg.model.revision
    )

    # Freeze base weights — only LoRA params will train.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet = inject_lora(unet, cfg.lora)

    # Cast frozen models to a memory-efficient dtype for forward passes.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if cfg.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if cfg.training.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            accelerator.print("[train] xformers enabled.")
        except Exception as e:  # pragma: no cover
            accelerator.print(f"[train] xformers unavailable, continuing without ({e}).")

    # ----- data --------------------------------------------------------- #
    dataset = build_dataset(cfg, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ----- optimizer & schedule ---------------------------------------- #
    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.training.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    steps_per_epoch = math.ceil(len(loader) / cfg.training.gradient_accumulation_steps)
    max_steps = cfg.training.max_train_steps or (cfg.training.num_train_epochs * steps_per_epoch)

    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps * cfg.training.gradient_accumulation_steps,
        num_training_steps=max_steps * cfg.training.gradient_accumulation_steps,
    )

    unet, optimizer, loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, loader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("sd_lora_finetune")
        accelerator.print(
            f"[train] dataset={len(dataset)} | steps/epoch={steps_per_epoch} | max_steps={max_steps}"
        )

    # ----- training loop ---------------------------------------------- #
    progress = tqdm(
        range(max_steps),
        disable=not accelerator.is_local_main_process,
        desc="train",
    )
    global_step = 0
    samples_dir = out_dir / "samples"

    for epoch in range(cfg.training.num_train_epochs):
        unet.train()
        for batch in loader:
            with accelerator.accumulate(unet):
                # Encode images to latents (VAE is frozen, half precision).
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                # Sample noise + random timestep per latent.
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device,
                ).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text conditioning.
                enc_hidden = text_encoder(batch["input_ids"])[0]

                # Predict noise; target is the noise added (eps-prediction).
                pred = unet(noisy, timesteps, enc_hidden).sample

                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # Periodic checkpoint
                if global_step % cfg.training.checkpoint_every == 0 and accelerator.is_main_process:
                    p = save_lora(accelerator.unwrap_model(unet), out_dir / "lora", global_step)
                    accelerator.print(f"[train] saved LoRA → {p}")

                # Periodic validation
                if global_step % cfg.training.validation_every == 0 and accelerator.is_main_process:
                    run_validation(
                        cfg, accelerator, unet, text_encoder, vae, tokenizer, global_step, samples_dir,
                    )

                if global_step >= max_steps:
                    break
        if global_step >= max_steps:
            break

    # Final save + validation
    if accelerator.is_main_process:
        save_lora(accelerator.unwrap_model(unet), out_dir / "lora", global_step)
        run_validation(
            cfg, accelerator, unet, text_encoder, vae, tokenizer, global_step, samples_dir,
        )
        accelerator.print(f"[train] done. LoRA saved under {out_dir / 'lora'}")

    accelerator.end_training()
