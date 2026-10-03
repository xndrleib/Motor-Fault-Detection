import subprocess
import sys

import pytest
import torch

from src.utils import configure_deterministic_operations


def test_determinism_keeps_seeded_randomness_and_dropout():
    # A separate process keeps backend settings isolated from other tests.
    source = """
import os, torch
from src.utils import configure_deterministic_operations
torch.manual_seed(42)
before = torch.get_rng_state().clone()
settings = configure_deterministic_operations()
assert torch.equal(before, torch.get_rng_state())
assert settings['deterministic_algorithms']
assert settings['cudnn_deterministic'] and not settings['cudnn_benchmark']
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
x = torch.nn.Dropout(.5)(torch.ones(100))
assert (x == 0).any() and (x != 0).any()
assert settings == configure_deterministic_operations()
"""
    subprocess.run([sys.executable, "-c", source], check=True)


def test_determinism_rejects_late_cuda_workspace_change(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        configure_deterministic_operations()
