"""The masked multi-label loss.

Everything else in train.py is a stochastic training loop, not something to
unit-test in the traditional sense - but the masking arithmetic is a pure
function and it is exactly the piece ADR-0005's whole design depends on:
get the mask backwards and "unknown, don't supervise" silently becomes
"known negative," corrupting the model's binary detection signal with the
same noise the mask exists to keep out.
"""

from __future__ import annotations

import torch

from portcullis.training.l2.train import masked_bce_loss


def test_fully_masked_row_gets_zero_gradient_contribution() -> None:
    """mask=0 must mean 'no opinion', not 'push toward this target'."""
    logits = torch.tensor([[5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0]], requires_grad=True)
    target = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)

    loss = masked_bce_loss(logits, target, mask)
    assert loss.item() == 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad == 0.0)


def test_fully_supervised_row_matches_plain_bce() -> None:
    logits = torch.tensor([[2.0, -1.0]])
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.ones_like(logits)

    got = masked_bce_loss(logits, target, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert torch.isclose(got, expected)


def test_masked_dimensions_do_not_affect_the_loss_value() -> None:
    """Two rows differing only in a masked-out dimension must produce the
    same loss - if they don't, the mask is leaking into the average."""
    logits = torch.tensor([[1.0, 99.0]])  # dim 1 will be masked out
    target_a = torch.tensor([[1.0, 0.0]])
    target_b = torch.tensor([[1.0, 1.0]])  # different target, but masked
    mask = torch.tensor([[1.0, 0.0]])

    loss_a = masked_bce_loss(logits, target_a, mask)
    loss_b = masked_bce_loss(logits, target_b, mask)
    assert torch.isclose(loss_a, loss_b)


def test_partially_masked_batch_averages_only_active_elements() -> None:
    """The ADR-0005 case directly: one benign row (fully supervised), one
    excluded-attack row (only the benign dim supervised). The loss must be
    the mean over exactly the 9 active elements (8 + 1), not 16."""
    logits = torch.zeros((2, 8))
    target = torch.zeros((2, 8))
    target[0, -1] = 1.0  # row 0: benign row, benign bit = 1
    mask = torch.zeros((2, 8))
    mask[0, :] = 1.0  # row 0: fully supervised
    mask[1, -1] = 1.0  # row 1: only the benign dim supervised

    got = masked_bce_loss(logits, target, mask)

    per_element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    active = per_element[mask.bool()]
    expected = active.mean()
    assert torch.isclose(got, expected)


def test_empty_batch_mask_does_not_divide_by_zero() -> None:
    logits = torch.zeros((1, 8))
    target = torch.zeros((1, 8))
    mask = torch.zeros((1, 8))
    loss = masked_bce_loss(logits, target, mask)
    assert torch.isfinite(loss)
