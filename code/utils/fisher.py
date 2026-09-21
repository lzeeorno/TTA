"""
Compute Fisher Information Matrix for EATA anti-forgetting regularization
Based on EATA ICML 2022, Section 4.2, Eqn. (9)
"""
import torch
import torch.nn as nn
from tqdm import tqdm


def compute_fishers(model, dataloader, num_samples=2000, device='cuda'):
    """
    Compute diagonal Fisher Information using unlabeled ID samples.

    Args:
        model: pre-trained model (original, before any adaptation)
        dataloader: ImageNet validation dataloader
        num_samples: number of samples for Fisher computation (paper uses 2000)
        device: cuda/cpu

    Returns:
        fishers: dict {param_name: (fisher_value, param_initial_value)}
    """
    model.eval()
    fishers = {}

    # Initialize
    for name, param in model.named_parameters():
        if param.requires_grad:
            fishers[name] = [torch.zeros_like(param), param.clone().detach()]

    # Accumulate gradients
    sample_count = 0
    for images, _ in tqdm(dataloader, desc='Computing Fisher', leave=False):
        if sample_count >= num_samples:
            break

        images = images.to(device)
        model.zero_grad()

        # Forward pass
        outputs = model(images)

        # Use pseudo-labels (hard prediction)
        pseudo_labels = outputs.argmax(dim=1)

        # Cross-entropy loss
        loss = nn.CrossEntropyLoss()(outputs, pseudo_labels)
        loss.backward()

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fishers[name][0] += param.grad.detach() ** 2

        sample_count += images.size(0)

    # Average
    for name in fishers:
        fishers[name][0] /= sample_count

    print(f"Fisher information computed from {sample_count} samples")
    return fishers
