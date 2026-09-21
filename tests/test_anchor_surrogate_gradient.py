import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from atlas.common import arc_surrogate, class_centered_anchor_loss


@pytest.mark.parametrize(
    "ratio,advantage,expected_sign",
    [(0.5, 1.0, 1), (1.0, 1.0, 1), (1.5, 1.0, 0),
     (0.5, -1.0, 0), (1.0, -1.0, -1), (1.5, -1.0, -1)],
)
def test_signed_two_sided_ppo_gradient(ratio, advantage, expected_sign):
    r = torch.tensor([ratio], requires_grad=True)
    value = arc_surrogate(r, torch.tensor([advantage]), eps_low=0.2, eps_high=0.2)
    value.sum().backward()
    sign = int(torch.sign(r.grad).item())
    assert sign == expected_sign


def test_class_centered_anchor_step_reduces_forward_kl():
    logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    center = torch.tensor([[0.6, 0.3, 0.1]])
    before = torch.nn.functional.kl_div(
        logits.log_softmax(1), center, reduction="batchmean"
    )
    loss = class_centered_anchor_loss(logits, center).mean()
    gradient = torch.autograd.grad(loss, logits)[0]
    after_logits = logits.detach() - 0.1 * gradient
    after = torch.nn.functional.kl_div(
        after_logits.log_softmax(1), center, reduction="batchmean"
    )
    assert after < before
