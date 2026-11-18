from __future__ import annotations

import torch
from torch import nn

from layers.DTSN_layers import InstanceNormalizer, TimeSeriesDecomposition, DTSNEncoder


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.revin = bool(configs.revin)

        self.normalizer = InstanceNormalizer(configs.enc_in, affine=True, subtract_last=False)
        self.decomposition = None
        if configs.ma_type != 'reg':
            self.decomposition = TimeSeriesDecomposition(configs.ma_type, configs.alpha, configs.beta)
        self.backbone = DTSNEncoder(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            patch_len=configs.patch_len,
            stride=configs.stride,
            padding_patch=configs.padding_patch,
            use_gated_fusion=bool(configs.use_gated_fusion),
            fusion_hidden=configs.fusion_hidden,
            fusion_bias=float(configs.fusion_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, channels = x.shape

        # Normalize and separate the input components: [B, T, C].
        if self.revin:
            x = self.normalizer(x, 'norm')
        if self.decomposition is None:
            seasonal, trend = x, x
        else:
            seasonal, trend = self.decomposition(x)

        # Apply the same temporal network independently to each channel.
        seasonal = seasonal.permute(0, 2, 1).reshape(batch_size * channels, length)
        trend = trend.permute(0, 2, 1).reshape(batch_size * channels, length)
        seasonal_features = self.backbone.encode_seasonal(seasonal)
        trend_features = self.backbone.encode_trend(trend)
        forecast = self.backbone.fuse(seasonal_features, trend_features)

        # Restore the forecast layout and original scale: [B, H, C].
        forecast = forecast.reshape(batch_size, channels, self.pred_len)
        forecast = forecast.permute(0, 2, 1)
        if self.revin:
            forecast = self.normalizer(forecast, 'denorm')
        return forecast
