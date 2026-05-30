"""
Evaluation pipeline.

Measures prompt–image alignment with CLIP score (cosine similarity between
text and image embeddings, scaled by 100). Compares the base SD model against
the LoRA-adapted model on the same prompts + same seeds for a fair before/after
readout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .inference import generate, load_pipeline


# --------------------------------------------------------------------------- #
# CLIP score
# --------------------------------------------------------------------------- #
class CLIPScorer:
    """Wraps a CLIP model to score image–text alignment.

    Uses the full forward pass (rather than `get_image_features` /
    `get_text_features` separately) for robustness across transformers
    versions — `outputs.image_embeds` and `outputs.text_embeds` are
    guaranteed plain tensors.
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def score(self, images: List[Image.Image], prompts: List[str]) -> List[float]:
        """Return per-pair CLIP score = 100 * cos(emb(img), emb(txt))."""
        assert len(images) == len(prompts), "images and prompts must be 1:1"

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        img_feats = outputs.image_embeds
        txt_feats = outputs.text_embeds

        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        cos = (img_feats * txt_feats).sum(dim=-1)
        return (cos * 100.0).cpu().tolist()


# --------------------------------------------------------------------------- #
# Before / after evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    prompts: List[str],
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_path: str | None = None,
    out_dir: str = "outputs/eval",
    clip_model: str = "openai/clip-vit-base-patch32",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> Dict:
    """
    Generate images with the base model AND the LoRA-adapted model on the same
    prompts + same seed, score both with CLIP, and write a JSON report plus
    side-by-side PNGs.
    """
    out = Path(out_dir)
    (out / "base").mkdir(parents=True, exist_ok=True)
    (out / "lora").mkdir(parents=True, exist_ok=True)
    (out / "comparison").mkdir(parents=True, exist_ok=True)

    scorer = CLIPScorer(clip_model)

    # ----- base ------------------------------------------------------- #
    print("[eval] generating with base model...")
    base_pipe = load_pipeline(model_id=model_id, lora_path=None)
    base_imgs = generate(
        base_pipe, prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        out_dir=str(out / "base"),
    )
    base_scores = scorer.score(base_imgs, prompts)
    del base_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----- lora ------------------------------------------------------- #
    if lora_path:
        print("[eval] generating with LoRA model...")
        lora_pipe = load_pipeline(model_id=model_id, lora_path=lora_path)
        lora_imgs = generate(
            lora_pipe, prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            out_dir=str(out / "lora"),
        )
        lora_scores = scorer.score(lora_imgs, prompts)
        del lora_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        lora_imgs, lora_scores = [], []

    # ----- side-by-side comparison ------------------------------------ #
    if lora_imgs:
        for i, (b, l, p) in enumerate(zip(base_imgs, lora_imgs, prompts)):
            combined = Image.new("RGB", (b.width + l.width, max(b.height, l.height)))
            combined.paste(b, (0, 0))
            combined.paste(l, (b.width, 0))
            combined.save(out / "comparison" / f"{i:03d}.png")

    # ----- report ----------------------------------------------------- #
    mean_base = sum(base_scores) / len(base_scores)
    mean_lora = (sum(lora_scores) / len(lora_scores)) if lora_scores else None

    report = {
        "model_id": model_id,
        "lora_path": lora_path,
        "clip_model": clip_model,
        "num_prompts": len(prompts),
        "per_prompt": [
            {
                "prompt": p,
                "base_clip_score": base_scores[i],
                "lora_clip_score": (lora_scores[i] if lora_scores else None),
                "delta": (lora_scores[i] - base_scores[i] if lora_scores else None),
            }
            for i, p in enumerate(prompts)
        ],
        "mean_base_clip_score": mean_base,
        "mean_lora_clip_score": mean_lora,
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    if mean_lora is not None:
        print("[eval] mean CLIP - base: {:.2f} | lora: {:.2f}".format(mean_base, mean_lora))
    else:
        print("[eval] mean CLIP - base: {:.2f}".format(mean_base))
    print(f"[eval] report -> {out / 'report.json'}")
    return report
