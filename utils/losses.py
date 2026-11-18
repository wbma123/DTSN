from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_arctangent_weights(pred_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # [T, 1] weights broadcast across batches and channels.
    steps = torch.arange(1, pred_len + 1, device=device, dtype=dtype)
    return (-torch.atan(steps) + math.pi / 4 + 1.0).unsqueeze(-1)


def apply_time_weights(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
    return pred * weights, target * weights


class ArctanMAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss(reduction='mean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = build_arctangent_weights(
            pred_len=pred.size(1),
            device=pred.device,
            dtype=pred.dtype,
        )
        pred, target = apply_time_weights(pred, target, weights)
        return self.mae(pred, target)


class DLossEMA(nn.Module):

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"DLossEMA expects [B, T, C], got {tuple(x.shape)}")

        orig_dtype = x.dtype
        # Accumulate EMA in float64 to limit underflow over long horizons.
        work = x.to(torch.float64)

        _, seq_len, _ = work.shape
        if seq_len == 0:
            return x

        powers = torch.arange(seq_len - 1, -1, -1, device=work.device, dtype=torch.float64)
        weights = torch.pow(1.0 - self.alpha, powers)
        divisor = weights.clone()

        if seq_len > 1:
            weights = weights.clone()
            weights[1:] = weights[1:] * self.alpha

        weights = weights.view(1, seq_len, 1)
        divisor = divisor.view(1, seq_len, 1)

        weighted_cumsum = torch.cumsum(work * weights, dim=1)
        trend = weighted_cumsum / divisor

        return trend.to(orig_dtype)


class DLossHorizonDecomposition(nn.Module):

    def __init__(self, alpha: float):
        super().__init__()
        self.ema = DLossEMA(alpha)

    def forward(self, x: torch.Tensor):
        trend = self.ema(x)
        seasonal = x - trend
        return seasonal, trend


class DLoss(nn.Module):

    def __init__(self, alpha: float = 0.2, beta: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.decomp = DLossHorizonDecomposition(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.mse = nn.MSELoss(reduction='mean')
        self.mae = nn.L1Loss(reduction='mean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_season, pred_trend = self.decomp(pred)
        true_season, true_trend = self.decomp(target)

        season_loss = self.mse(pred_season, true_season)
        trend_loss = self.mae(pred_trend, true_trend)
        # Balance loss scales without differentiating through the scale factor.
        trend_loss = trend_loss * (season_loss / (trend_loss + self.eps)).detach()

        return self.beta * season_loss + (1.0 - self.beta) * trend_loss


ForecastHorizonDecomp = DLossHorizonDecomposition
