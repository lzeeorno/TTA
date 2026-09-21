import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from atlas.common import destroy_sensitivity, insufficient_destroy_penalty


def test_raw_confident_destroy_drops_is_positive_support():
    sensitivity = destroy_sensitivity(torch.tensor([0.9]), torch.tensor([0.2]))
    torch.testing.assert_close(sensitivity, torch.tensor([0.7]))
    torch.testing.assert_close(insufficient_destroy_penalty(sensitivity, 0.2), torch.zeros(1))


def test_raw_confident_destroy_unchanged_is_risky():
    sensitivity = destroy_sensitivity(torch.tensor([0.9]), torch.tensor([0.9]))
    torch.testing.assert_close(sensitivity, torch.zeros(1))
    torch.testing.assert_close(insufficient_destroy_penalty(sensitivity, 0.2), torch.tensor([0.2]))


def test_noisy_destroy_cannot_create_negative_sensitivity():
    sensitivity = destroy_sensitivity(torch.tensor([0.4]), torch.tensor([0.8]))
    torch.testing.assert_close(sensitivity, torch.zeros(1))


def test_misleading_view_has_more_risk_than_supported_view():
    raw = torch.tensor([0.8, 0.8])
    destroy = torch.tensor([0.75, 0.1])
    risk = insufficient_destroy_penalty(destroy_sensitivity(raw, destroy), 0.2)
    assert risk[0] > risk[1]
