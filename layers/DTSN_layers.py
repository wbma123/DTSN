from __future__ import annotations

import torch
from torch import nn


class InstanceNormalizer(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        if mode == 'denorm':
            return self._denormalize(x)
        raise NotImplementedError

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class ExponentialMovingAverage(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"EMA expects [B, T, C], got shape {tuple(x.shape)}")

        _, seq_len, _ = x.shape
        if seq_len == 0:
            return x

        dtype = x.dtype
        device = x.device
        one_minus_alpha = torch.as_tensor(1.0 - self.alpha, dtype=dtype, device=device)
        alpha = torch.as_tensor(self.alpha, dtype=dtype, device=device)

        powers = torch.arange(seq_len - 1, -1, -1, dtype=dtype, device=device)
        weights = one_minus_alpha.pow(powers)
        divisor = weights.clone()
        if seq_len > 1:
            weights = weights.clone()
            weights[1:] = weights[1:] * alpha

        weights = weights.view(1, seq_len, 1)
        divisor = divisor.view(1, seq_len, 1)
        weighted_cumsum = torch.cumsum(x * weights, dim=1)
        return weighted_cumsum / divisor


class DoubleExponentialMovingAverage(nn.Module):
    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"DEMA expects [B, T, C], got shape {tuple(x.shape)}")
        if x.size(1) < 2:
            return x

        alpha = torch.as_tensor(self.alpha, dtype=x.dtype, device=x.device)
        beta = torch.as_tensor(self.beta, dtype=x.dtype, device=x.device)

        level_prev = x[:, 0, :]
        trend = x[:, 1, :] - level_prev
        outputs = [level_prev.unsqueeze(1)]

        for step in range(1, x.size(1)):
            current = x[:, step, :]
            level = alpha * current + (1.0 - alpha) * (level_prev + trend)
            trend = beta * (level - level_prev) + (1.0 - beta) * trend
            level_prev = level
            outputs.append(level.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class TimeSeriesDecomposition(nn.Module):
    def __init__(self, ma_type: str, alpha: float, beta: float):
        super().__init__()
        if ma_type == 'ema':
            self.moving_average = ExponentialMovingAverage(alpha)
        elif ma_type == 'dema':
            self.moving_average = DoubleExponentialMovingAverage(alpha, beta)
        else:
            raise ValueError(f"Unsupported decomposition type: {ma_type}")

    def forward(self, x: torch.Tensor):
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class VectorGate(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, init_bias: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)
        self._last_gate = None

    def gate_values(self):
        return self._last_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.net(x))
        self._last_gate = gate.detach().mean().cpu()
        return gate


class DTSNEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        padding_patch: str,
        use_gated_fusion: bool = True,
        fusion_hidden: int = 64,
        fusion_bias: float = 0.0,
    ):
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        if patch_len > seq_len:
            raise ValueError("patch_len cannot exceed seq_len")

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.use_gated_fusion = bool(use_gated_fusion)
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1

        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        elif padding_patch in (None, 'none', 'None'):
            self.padding_patch_layer = None
        else:
            raise ValueError(f"Unsupported padding_patch mode: {padding_patch}")

        self._build_patch_path(patch_len, pred_len)
        self._build_trend_path(seq_len, pred_len)
        self._build_fusion(pred_len, fusion_hidden, fusion_bias)

    def _build_patch_path(self, patch_len, pred_len):
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num, patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        self.fc2 = nn.Linear(self.dim, patch_len)

        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

    def _build_trend_path(self, seq_len, pred_len):
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

    def _build_fusion(self, pred_len, fusion_hidden, fusion_bias):
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

        self.fusion_gate = VectorGate(
            in_dim=pred_len * 4,
            hidden_dim=fusion_hidden,
            out_dim=pred_len,
            init_bias=fusion_bias,
        )

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_patch_layer is not None:
            x = self.padding_patch_layer(x)
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def encode_seasonal(self, seasonal: torch.Tensor) -> torch.Tensor:
        patches = self._extract_patches(seasonal)

        patches = self.fc1(patches)
        patches = self.gelu1(patches)
        patches = self.bn1(patches)

        residual = patches

        patches = self.conv1(patches)
        patches = self.gelu2(patches)
        patches = self.bn2(patches)

        residual = self.fc2(residual)
        patches = patches + residual

        patches = self.conv2(patches)
        patches = self.gelu3(patches)
        patches = self.bn3(patches)

        patches = self.flatten1(patches)
        patches = self.fc3(patches)
        patches = self.gelu4(patches)
        return self.fc4(patches)

    def encode_trend(self, trend: torch.Tensor) -> torch.Tensor:
        trend = self.fc5(trend)
        trend = self.avgpool1(trend)
        trend = self.ln1(trend)

        trend = self.fc6(trend)
        trend = self.avgpool2(trend)
        trend = self.ln2(trend)
        return self.fc7(trend)

    def fuse(self, seasonal_features, trend_features):
        if self.use_gated_fusion:
            fusion_input = torch.cat(
                (
                    seasonal_features,
                    trend_features,
                    seasonal_features - trend_features,
                    seasonal_features * trend_features,
                ),
                dim=1,
            )
            gate = self.fusion_gate(fusion_input)
            fused = trend_features + gate * (seasonal_features - trend_features)
        else:
            fused = torch.cat((seasonal_features, trend_features), dim=1)
            fused = self.fc8(fused)

        return fused

    def forward(self, seasonal, trend):
        return self.encode_seasonal(seasonal), self.encode_trend(trend)
