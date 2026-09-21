import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from atlas.common import selected_entity_mean


def test_duplicate_selected_entities_preserve_per_entity_loss_scale():
    losses = torch.tensor([1.0, 3.0], requires_grad=True)
    duplicated = losses.repeat(5)
    torch.testing.assert_close(selected_entity_mean(losses), selected_entity_mean(duplicated))


def test_weights_are_detached_and_denominator_is_selected_count():
    losses = torch.tensor([1.0, 3.0], requires_grad=True)
    weights = torch.tensor([0.5, 1.0], requires_grad=True)
    value = selected_entity_mean(losses, weights)
    value.backward()
    torch.testing.assert_close(value.detach(), torch.tensor(1.75))
    assert weights.grad is None


def test_no_selected_entity_requires_caller_to_skip():
    try:
        selected_entity_mean(torch.empty(0))
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty selection must not silently change the normalizer")
