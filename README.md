# Stable Diffusion 1.5 — LoRA Fine-Tuning Pipeline

End-to-end pipeline for fine-tuning Stable Diffusion 1.5 with **LoRA** adapters on a small custom dataset, plus a **CLIP-score evaluation harness** that compares the base model against the fine-tuned model on a held-out prompt set.

Built on 🤗 **Diffusers**, **PEFT**, **Accelerate**, and **PyTorch**. Runs locally on an 8 GB+ NVIDIA GPU or on a free Colab T4.

---

## Architecture

![architecture](docs/images/architecture.svg)

**Why LoRA?** A LoRA adapter on the UNet's cross-attention projections trains only ~0.5–1 M parameters (vs. ~860 M for full fine-tuning), produces a 3–6 MB checkpoint, runs on 8 GB VRAM, and preserves the base model's generality so it can be swapped in or out without re-downloading anything.

---

## Repo layout

```
sd-lora-finetune/
├── src/
│   ├── dataset.py        # HF hub + local-folder loaders, BLIP auto-captioning
│   ├── train.py          # LoRA injection (PEFT), accelerate-based training loop
│   ├── inference.py      # Pipeline loader + generate() helper
│   └── evaluate.py       # CLIP scoring, base vs LoRA comparison
├── scripts/
│   ├── train_lora.py     # CLI: training entrypoint
│   ├── generate.py       # CLI: single-prompt inference
│   └── evaluate.py       # CLI: before/after CLIP eval
├── configs/
│   ├── default.yaml      # All training hyperparameters
│   └── eval_prompts.txt  # Held-out evaluation prompts
├── notebooks/
│   └── colab_quickstart.ipynb
├── tests/
│   └── test_smoke.py     # Offline sanity checks
└── docs/images/
    ├── architecture.svg
    ├── comparison_green_pokemon.png
    └── comparison_pink_pokemon.png
```

---

## Quickstart

### Option 1 — Colab (free, recommended for first run)

The `notebooks/colab_quickstart.ipynb` notebook runs the full pipeline end-to-end on a free Colab T4. Open it directly via **File → Open notebook → GitHub** in Colab, paste `kishoremadanagopal/sd-lora-finetune`, and run the cells top to bottom. Total time ≈ 45 minutes.

### Option 2 — Local (CUDA)

```bash
git clone https://github.com/kishoremadanagopal/sd-lora-finetune.git
cd sd-lora-finetune

# Install PyTorch matching your CUDA version (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install xformers  # optional, recommended on CUDA
```

#### Smoke test (no GPU, no downloads)

```bash
python tests/test_smoke.py
```

#### Train on the public demo dataset

```bash
python scripts/train_lora.py --config configs/default.yaml
```

This trains a Pokémon-style LoRA on the `reach-vb/pokemon-blip-captions` dataset (~833 images). On a single RTX 3060 (8 GB) at default settings, ~1000 steps take roughly 30 minutes.

#### Train on your own images

Put images (and optional `<name>.txt` caption files) in a folder:

```bash
python scripts/train_lora.py --config configs/default.yaml \
  dataset.name=null \
  dataset.image_dir=data/raw/my_concept \
  dataset.instance_prompt="a photo of sks dog" \
  dataset.auto_caption=false \
  training.num_train_epochs=100
```

Set `dataset.auto_caption=true` to caption images automatically with BLIP.

#### Generate images with the trained LoRA

```bash
python scripts/generate.py \
  --prompts "a green dragon pokemon with red eyes" "a small pink pokemon with big ears" \
  --lora outputs/checkpoints/lora_run/lora/lora_step_000250.safetensors \
  --out outputs/samples/gen
```

#### Evaluate base vs LoRA with CLIP score

```bash
python scripts/evaluate.py \
  --lora outputs/checkpoints/lora_run/lora/lora_step_000250.safetensors \
  --prompts-file configs/eval_prompts.txt
```

Output:

- `outputs/eval/base/` — base SD images
- `outputs/eval/lora/` — fine-tuned model images (same prompts, same seed)
- `outputs/eval/comparison/` — side-by-side PNGs
- `outputs/eval/report.json` — per-prompt + mean CLIP score, plus delta

---

## Training steps in detail

1. **Load components** — VAE, text encoder, UNet, scheduler from `runwayml/stable-diffusion-v1-5`.
2. **Freeze everything**, then inject PEFT LoRA adapters (`rank=4`, `α=4`) into the UNet's cross-attention `to_q`, `to_k`, `to_v`, `to_out.0` projections.
3. **Encode each batch**: images → VAE latents (×0.18215); captions → CLIP text embeddings.
4. **Diffusion objective**: sample noise ε and a random timestep t, form `x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε`, predict ε̂ with the UNet, optimize `MSE(ε̂, ε)`.
5. **Periodic checkpoint** — save the PEFT state-dict as a small `.safetensors` file (~3–6 MB).

Mixed-precision (fp16), gradient checkpointing, and xformers attention are on by default. AdamW with constant LR `1e-4` is a solid starting point; reduce to `5e-5` for larger datasets, raise the LoRA `rank` (8–16) for more capacity at the cost of file size.

---

## Evaluation methodology

**CLIP score** measures how well a generated image matches its prompt: the cosine similarity between CLIP image and text embeddings, multiplied by 100. Higher is better. Implementation: `openai/clip-vit-base-patch32`.

The evaluator generates images with the **base model** and the **LoRA-adapted model** on identical prompts and identical seeds, then computes CLIP score for each, plus the per-prompt and mean delta. Side-by-side PNGs make qualitative inspection trivial.

**Limitations.** CLIP score rewards semantic alignment, not fidelity to a specific style. After domain-specific fine-tuning, CLIP score on the **target domain's prompts** may rise (better stylistic match) or fall slightly (less generic compositions) — pair it with visual inspection of the comparison grids. For style fine-tuning, also consider FID against a held-out set of real images from the target domain.

---

## Results

Trained 250 steps on `reach-vb/pokemon-blip-captions` (400 images subset, resolution 512, fp16) on a Colab T4.

| Metric                | Base SD 1.5 | + LoRA (250 steps) | Δ      |
| --------------------- | ----------- | ------------------ | ------ |
| Mean CLIP score (n=8) | 33.98       | 33.41              | −0.57  |
| Checkpoint size       | ~3.4 GB     | ~3 MB              | −99.9% |
| Trainable params      | ~860 M      | ~797 K             | −99.9% |
| VRAM peak (training)  | —           | ~7.5 GB            | —      |

### Base (left) vs LoRA (right) on held-out prompts

Prompt: *"a drawing of a green pokemon with red eyes"*

![comparison green pokemon](docs/images/comparison_green_pokemon.png)

Prompt: *"a small pink pokemon with big ears"*

![comparison pink pokemon](docs/images/comparison_pink_pokemon.png)

### Discussion

The fine-tuned LoRA shows a measurable drop in CLIP score (−0.57), and inspection of the generated images explains why: the LoRA has clearly learned the target Pokémon aesthetic (rounded shapes, flat illustrative coloring, simpler compositions), but in doing so it loses some literal prompt precision — for example, attributes like "red eyes" may not always render, and color terms can be reinterpreted toward what the model saw most often in training.

This is the expected style-vs-prompt-following tradeoff for short-run domain fine-tuning, sometimes called *prompt drift*. The base model wins on literal CLIP alignment because it draws on a vastly larger and more diverse training distribution; the LoRA wins on stylistic conformity, which CLIP doesn't directly measure.

Practical mitigations:

- **Train longer** (≥1500 steps) to let the model better separate style from content
- **Lower the learning rate** to `5e-5` for less aggressive drift
- **Increase LoRA rank** (8–16) for more representational capacity
- **Add regularization images** of the base distribution alongside training data
- **Use a paired evaluation** like DreamBooth's prior-preservation metrics in addition to CLIP

---

## Troubleshooting

- **OOM on 8 GB GPU.** Lower `dataset.resolution` to 384, keep `train_batch_size=1`, raise `gradient_accumulation_steps`, and confirm `mixed_precision: fp16` + `gradient_checkpointing: true`.
- **xformers fails to import.** It's optional — leave `training.enable_xformers=true`, the code falls back silently.
- **`safety_checker` warnings.** Disabled deliberately for evaluation — re-enable in `src/inference.py` if you redistribute generations.
- **CLIP score doesn't improve.** Expected for short-run style fine-tuning (see Discussion above). Train longer, lower the LR, increase rank, or evaluate with style-aware metrics.
- **Mid-training validation sampling crashes with a dtype mismatch.** Non-essential; checkpoints save before validation runs. Reduce `training.validation_every` or set it very high to skip.

---

## Tech stack

`diffusers` · `transformers` · `peft` · `accelerate` · `torch` · `torchvision` · `datasets` · `safetensors` · `omegaconf`

---

## License

MIT — see `LICENSE`. Stable Diffusion 1.5 weights are governed by their own [CreativeML Open RAIL-M license](https://huggingface.co/runwayml/stable-diffusion-v1-5).
