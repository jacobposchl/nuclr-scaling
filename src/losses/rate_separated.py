"""Contrastive content features with supervised rate and linear leakage penalties."""

import math

import torch

from .nuclr_loss import NuCLRLoss


def mean_squared_rate_correlation(content, rates, epsilon=1e-8):
    """Mean squared Pearson correlation across content coordinates in one micro-batch."""
    h = content - content.mean(dim=0)
    r = rates - rates.mean()
    covariance = (h * r[:, None]).mean(dim=0)
    variance_product = h.square().mean(dim=0) * r.square().mean()
    return (covariance.square() / variance_product.clamp_min(epsilon)).mean()


class RateSeparatedNuCLRLoss(NuCLRLoss):
    requires_firing_rate = True

    def __init__(self, dim_in, tau, dcl, projector=True, full_denom=False,
                 rate_supervision_weight=.1, rate_correlation_weight=.1, rate_scale_hz=10.):
        if dim_in < 3:
            raise ValueError("Need a rate coordinate and at least two content coordinates")
        for name, value in (("rate_supervision_weight", rate_supervision_weight),
                            ("rate_correlation_weight", rate_correlation_weight), ("rate_scale_hz", rate_scale_hz)):
            if not math.isfinite(value) or value < 0 or (name == "rate_scale_hz" and value == 0):
                raise ValueError(f"Invalid {name}: {value}")
        super().__init__(dim_in - 1, tau, dcl, projector, full_denom)
        self.rate_supervision_weight = rate_supervision_weight
        self.rate_correlation_weight = rate_correlation_weight
        self.rate_scale_hz = rate_scale_hz

    def forward(self, x, seqlen, unit_ids, probe_ids, prefix="", *, firing_rate_hz):
        if firing_rate_hz.shape != (len(x),):
            raise ValueError("Expected one firing-rate target per neuron/view")
        contrastive, logs = super().forward(x[:, 1:], seqlen, unit_ids, probe_ids, prefix)
        # Keep correlations and rate errors in float32 under bf16 training.
        with torch.autocast(device_type=x.device.type, enabled=False):
            rates = firing_rate_hz.detach().float()
            error = x[:, 0].float() - rates
            mse = error.square().mean()
            supervised = mse / self.rate_scale_hz ** 2
            correlation = mean_squared_rate_correlation(x[:, 1:].float(), rates)
            loss = (contrastive + self.rate_supervision_weight * supervised
                    + self.rate_correlation_weight * correlation)
        logs.update({f"{prefix}loss": loss.item(), f"{prefix}contrastive_loss": contrastive.item(),
                     f"{prefix}rate_mse_hz2": mse.item(), f"{prefix}rate_mae_hz": error.abs().mean().item(),
                     f"{prefix}rate_supervision_loss": supervised.item(),
                     f"{prefix}rate_correlation_penalty": correlation.item(),
                     f"{prefix}rate_target_mean_hz": rates.mean().item(),
                     f"{prefix}rate_prediction_mean_hz": x[:, 0].detach().float().mean().item()})
        return loss, logs
