"""
Dataset loading for LoRA fine-tuning.

Supports two modes:
  1. Public HuggingFace datasets (e.g. `lambdalabs/pokemon-blip-captions`)
  2. Local image folders, with captions read from sibling .txt files or
     generated automatically with BLIP.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# --------------------------------------------------------------------------- #
# Image transforms
# --------------------------------------------------------------------------- #
def build_image_transform(resolution: int, center_crop: bool, random_flip: bool):
    """Standard SD pre-processing: resize → crop → flip → normalize to [-1, 1]."""
    ops = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
    ]
    if random_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    return transforms.Compose(ops)


# --------------------------------------------------------------------------- #
# Local folder dataset
# --------------------------------------------------------------------------- #
class LocalImageCaptionDataset(Dataset):
    """
    Reads images from a folder. For each `<name>.<ext>`, looks for `<name>.txt`
    holding the caption. Falls back to `instance_prompt` if no caption file is
    present, or auto-generates one with BLIP when `auto_caption=True`.
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(
        self,
        image_dir: str,
        tokenizer,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        instance_prompt: Optional[str] = None,
        auto_caption: bool = False,
    ):
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

        self.paths = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in self.IMG_EXTS
        )
        if not self.paths:
            raise ValueError(f"No images found in {image_dir}")

        self.tokenizer = tokenizer
        self.transform = build_image_transform(resolution, center_crop, random_flip)
        self.instance_prompt = instance_prompt
        self.captions = self._resolve_captions(auto_caption)

    # ----- captioning ------------------------------------------------------ #
    def _resolve_captions(self, auto_caption: bool) -> list[str]:
        """Read sibling .txt files; fall back to BLIP or instance_prompt."""
        captions: list[str] = []
        missing: list[int] = []
        for i, p in enumerate(self.paths):
            txt = p.with_suffix(".txt")
            if txt.exists():
                captions.append(txt.read_text(encoding="utf-8").strip())
            else:
                captions.append("")
                missing.append(i)

        if not missing:
            return captions

        if auto_caption:
            print(f"[dataset] Auto-captioning {len(missing)} images with BLIP…")
            blip_caps = self._blip_caption([self.paths[i] for i in missing])
            for idx, cap in zip(missing, blip_caps):
                captions[idx] = cap
        elif self.instance_prompt:
            for idx in missing:
                captions[idx] = self.instance_prompt
        else:
            raise ValueError(
                f"{len(missing)} image(s) have no caption .txt file. "
                "Either provide captions, set `instance_prompt`, or enable `auto_caption`."
            )
        return captions

    @staticmethod
    def _blip_caption(paths) -> list[str]:
        """Lazy import — BLIP is only needed for the auto-caption path."""
        from transformers import BlipForConditionalGeneration, BlipProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

        caps = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            inputs = proc(img, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=40)
            caps.append(proc.decode(out[0], skip_special_tokens=True))
        return caps

    # ----- pytorch dataset interface -------------------------------------- #
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.paths[idx]).convert("RGB")
        caption = self.captions[idx]
        return {
            "pixel_values": self.transform(img),
            "input_ids": self._tokenize(caption),
        }

    def _tokenize(self, caption: str) -> torch.Tensor:
        return self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]


# --------------------------------------------------------------------------- #
# HuggingFace hub dataset wrapper
# --------------------------------------------------------------------------- #
class HFCaptionDataset(Dataset):
    """Wraps a `datasets.Dataset` that yields {image, caption} rows."""

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        image_column: str = "image",
        caption_column: str = "text",
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.image_column = image_column
        self.caption_column = caption_column
        self.transform = build_image_transform(resolution, center_crop, random_flip)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        img = row[self.image_column]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        caption = row[self.caption_column]
        if isinstance(caption, list):  # some datasets store multi-caption lists
            caption = caption[0]
        return {
            "pixel_values": self.transform(img),
            "input_ids": self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0],
        }


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_dataset(cfg, tokenizer):
    """Pick the right dataset class based on config."""
    ds_cfg = cfg.dataset

    if ds_cfg.name:  # HF hub mode
        from datasets import load_dataset

        ds = load_dataset(ds_cfg.name, split="train")
        if ds_cfg.max_train_samples:
            ds = ds.select(range(min(ds_cfg.max_train_samples, len(ds))))
        return HFCaptionDataset(
            ds,
            tokenizer=tokenizer,
            image_column=ds_cfg.image_column,
            caption_column=ds_cfg.caption_column,
            resolution=ds_cfg.resolution,
            center_crop=ds_cfg.center_crop,
            random_flip=ds_cfg.random_flip,
        )

    if not ds_cfg.image_dir:
        raise ValueError("Set either dataset.name (HF hub) or dataset.image_dir (local).")

    return LocalImageCaptionDataset(
        image_dir=ds_cfg.image_dir,
        tokenizer=tokenizer,
        resolution=ds_cfg.resolution,
        center_crop=ds_cfg.center_crop,
        random_flip=ds_cfg.random_flip,
        instance_prompt=ds_cfg.instance_prompt,
        auto_caption=ds_cfg.auto_caption,
    )


def collate_fn(batch):
    """Stack tokenized captions and pixel tensors into a single batch."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]).to(
            memory_format=torch.contiguous_format
        ).float(),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
    }
