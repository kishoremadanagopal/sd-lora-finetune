"""
Sanity checks that don't require downloading the SD weights or a GPU.

Run:  python tests/test_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_config_loads():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/default.yaml")
    assert cfg.model.pretrained_model_name_or_path
    assert cfg.lora.rank > 0
    assert cfg.training.train_batch_size > 0
    print("[ok] config loads")


def test_imports():
    from src import dataset, evaluate, inference, train  # noqa: F401
    print("[ok] all src modules import")


def test_image_transform_shape():
    import torch
    from PIL import Image
    from src.dataset import build_image_transform

    t = build_image_transform(resolution=64, center_crop=True, random_flip=False)
    img = Image.new("RGB", (128, 96), color=(120, 50, 200))
    out = t(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 64, 64), out.shape
    assert -1.01 <= out.min().item() and out.max().item() <= 1.01
    print("[ok] image transform produces (3,64,64) in [-1,1]")


def test_collate():
    import torch
    from src.dataset import collate_fn

    batch = [
        {"pixel_values": torch.randn(3, 32, 32), "input_ids": torch.zeros(77, dtype=torch.long)}
        for _ in range(3)
    ]
    out = collate_fn(batch)
    assert out["pixel_values"].shape == (3, 3, 32, 32)
    assert out["input_ids"].shape == (3, 77)
    print("[ok] collate_fn stacks batch correctly")


if __name__ == "__main__":
    test_config_loads()
    test_imports()
    test_image_transform_shape()
    test_collate()
    print("\nAll smoke tests passed.")
