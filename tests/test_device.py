import pytest
import torch

from scanlag.device import autocast_dtype, torch_device


def test_device_order_and_override(monkeypatch):
    monkeypatch.delenv("SCANLAG_DEVICE", raising=False)
    for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        assert torch_device().type == expected
    monkeypatch.setenv("SCANLAG_DEVICE", " CPU ")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch_device().type == "cpu"
    monkeypatch.setenv("SCANLAG_DEVICE", "gpu")
    with pytest.raises(ValueError, match="cuda, mps, cpu"):
        torch_device()


def test_autocast_dtype():
    assert autocast_dtype(torch.device("mps")) == torch.float16
    assert autocast_dtype(torch.device("cuda")) == autocast_dtype(torch.device("cpu")) == torch.bfloat16
