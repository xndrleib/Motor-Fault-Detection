# src/models.py
"""Model definitions for 1D spectral classification with optional priors.

This module provides CNN/ResNet/MLP architectures for 1D signals with optional
spectral prior attention. Each model exposes a `forward_features` method to
retrieve penultimate embeddings and an `embedding_dim` property to query their
dimension.

Notes
-----
- All models return **logits** from `forward(x)` and **embeddings** from
  `forward_features(x)`.
- Embeddings are taken **after** dropout, matching the representation used by
  the classifier during training.
"""
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel-wise reweighting.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    reduction : int, default=16
        Reduction ratio for the bottleneck MLP.

    Notes
    -----
    Applies global average pooling over the temporal dimension and uses a
    two-layer MLP with ReLU/sigmoid to produce channel-wise gates.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, T)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, C, T)`` reweighted channel-wise.
        """
        s = x.mean(dim=-1)  # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # (B, C)
        s = s.unsqueeze(-1)  # (B, C, 1)
        return x * s


class BandWeightAttention(nn.Module):
    """Band-weight attention using inside/outside scalar gains.

    Implements ``x * w`` with
    ``w = w_out * (1 - mask) + w_in * mask``, where weights are constrained
    to be non-negative (and ``w_in >= 1``) via softplus parameterization.

    Parameters
    ----------
    mask : torch.Tensor
        Importance mask of shape ``(1, 1, L)`` with values in ``[0, 1]``
        (1 inside diagnostic bands, 0 elsewhere).
    w_in_init : float, default=1.0
        Initial inside-band weight (constrained to be ``>= 1``).
    w_out_init : float, default=0.2
        Initial outside-band weight (constrained to be ``>= 0``).
    learnable_in : bool, default=True
        Whether the inside-band weight is trainable.
    learnable_out : bool, default=True
        Whether the outside-band weight is trainable.

    Notes
    -----
    The parameters are stored in log-space and passed through softplus
    to enforce constraints.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        w_in_init: float = 1.0,
        w_out_init: float = 0.2,
        learnable_in: bool = True,
        learnable_out: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

        if learnable_out:
            init = torch.log(torch.tensor(w_out_init + 1e-6))
            self.log_out = nn.Parameter(init)
        else:
            self.register_buffer(
                "log_out", torch.log(torch.tensor(w_out_init + 1e-6)), persistent=False
            )

        if learnable_in:
            init = torch.log(torch.tensor(w_in_init - 1.0 + 1e-6))
            self.log_in = nn.Parameter(init)
        else:
            self.register_buffer(
                "log_in",
                torch.log(torch.tensor(w_in_init - 1.0 + 1e-6)),
                persistent=False,
            )

    def _weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        w_out = F.softplus(self.log_out)  # ≥0
        w_in = 1.0 + F.softplus(self.log_in)  # ≥1
        return w_in, w_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply band-weight attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, L)``.

        Returns
        -------
        torch.Tensor
            Reweighted tensor of shape ``(B, C, L)``.
        """
        w_in, w_out = self._weights()
        weight = w_out + (w_in - w_out) * self.mask
        return x * weight


class SpectralPriorAttention(nn.Module):
    """Fixed prior mask with learnable global gain.

    Parameters
    ----------
    mask : torch.Tensor
        Prior mask of shape ``(1, 1, L)`` with values in ``[0, 1]``.
    init_gain : float, default=2.0
        Initial gain for the prior.

    Notes
    -----
    Multiplies input by ``(1 + gain * mask)``.
    """

    def __init__(self, mask: torch.Tensor, init_gain: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.gain = nn.Parameter(torch.full((1,), init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C, L)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, C, L)``.
        """
        return x * (1.0 + self.gain * self.mask)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with optional SE.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, default=1
        Convolution stride for the first conv.
    downsample : nn.Module or None, optional
        Optional downsampling layer applied to the residual path.
    use_se : bool, default=False
        Whether to apply an SEBlock after the second convolution.

    Notes
    -----
    Follows the pre-activation order BN → ReLU → Conv for each conv layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = downsample
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, C_in, L)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, C_out, L_out)``.
        """
        residual = x
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        if self.use_se:
            out = self.se(out)

        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """A lightweight 1D ResNet for spectral inputs.

    Parameters
    ----------
    block : type[ResidualBlock]
        Residual block class to instantiate.
    layers : list[int]
        Number of blocks per stage (length must be 4).
    num_classes : int, default=2
        Number of output classes.
    dropout_rate : float, default=0.5
        Dropout probability applied to embeddings.
    block_kwargs : dict, optional
        Extra keyword arguments forwarded to each `ResidualBlock`.
    prior_kwargs : dict or None, optional
        If given, enables :class:`BandWeightAttention` with these kwargs.

    Attributes
    ----------
    num_classes : int
        Number of output classes.
    in_channels : int
        Current channel width for layer construction.
    fc : nn.Linear
        Final classification layer.

    Notes
    -----
    - `forward(x)` returns logits of shape ``(B, num_classes)``.
    - `forward_features(x)` returns embeddings of shape ``(B, D)``.
    """

    def __init__(
        self,
        block: type[ResidualBlock],
        layers: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        block_kwargs: Optional[Dict] = None,
        prior_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        if block_kwargs is None:
            block_kwargs = {}

        self.num_classes = num_classes
        self.in_channels = 64
        self.dropout_rate = dropout_rate

        self.conv = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm1d(64)
        self.prior_attn = (
            BandWeightAttention(**prior_kwargs)
            if prior_kwargs is not None
            else nn.Identity()
        )

        self.layer1 = self.make_layer(block, 64, layers[0], stride=1, **block_kwargs)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, **block_kwargs)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, **block_kwargs)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, **block_kwargs)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    @property
    def embedding_dim(self) -> int:
        """Dimension of the penultimate embedding."""
        return self.fc.in_features

    def make_layer(
        self,
        block: type[ResidualBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1,
        **block_kwargs,
    ) -> nn.Sequential:
        """Construct a ResNet stage.

        Parameters
        ----------
        block : type[ResidualBlock]
            Residual block class to instantiate.
        out_channels : int
            Output channels for this stage.
        blocks : int
            Number of residual blocks in this stage.
        stride : int, default=1
            Stride for the first block.

        Returns
        -------
        nn.Sequential
            A sequential container of residual blocks.
        """
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

        layers = [
            block(self.in_channels, out_channels, stride, downsample, **block_kwargs)
        ]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, **block_kwargs))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Kaiming/normal initialization for conv/linear; BN gamma=1, beta=0."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute penultimate embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 1, L)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(B, 512)``.
        """
        out = self.prior_attn(x)
        out = F.relu(self.bn(self.conv(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out).squeeze(-1)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 1, L)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        feats = self.forward_features(x)
        return self.fc(feats)


class MLP(nn.Module):
    """Simple MLP for flattened inputs.

    Parameters
    ----------
    input_dim : int
        Input feature dimensionality.
    hidden_dims : list[int]
        Hidden layer sizes.
    num_classes : int, default=2
        Number of output classes.
    dropout_rate : float, default=0.5
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ]
            in_dim = h
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)
        self._initialize_weights()

    @property
    def embedding_dim(self) -> int:
        """Dimension of the penultimate embedding."""
        return self.classifier.in_features

    def _initialize_weights(self) -> None:
        """Normal initialization for linear layers; BN gamma=1, beta=0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute penultimate embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, 1, D)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(B, H)`` where ``H`` is last hidden size.
        """
        x = x.squeeze(1)
        x = self.feature_extractor(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, 1, D)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        feats = self.forward_features(x)
        return self.classifier(feats)


class CNN(nn.Module):
    """Pyramidal 1D CNN with optional band prior.

    Parameters
    ----------
    num_classes : int, default=2
        Number of output classes.
    dropout_rate : float, default=0.5
        Dropout probability applied to embeddings.
    prior_kwargs : dict or None, optional
        If given, enables :class:`BandWeightAttention` with these kwargs.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        prior_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.prior_attn = (
            BandWeightAttention(**prior_kwargs)
            if prior_kwargs is not None
            else nn.Identity()
        )
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.block2 = self._make_block(64, 128)
        self.block3 = self._make_block(128, 256)
        self.block4 = self._make_block(256, 512)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout_rate = dropout_rate
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    @property
    def embedding_dim(self) -> int:
        """Dimension of the penultimate embedding."""
        return self.fc.in_features

    def _make_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Conv → BN → ReLU block with stride 2.

        Parameters
        ----------
        in_ch : int
            Input channels.
        out_ch : int
            Output channels.

        Returns
        -------
        nn.Sequential
            A sequential block.
        """
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _initialize_weights(self) -> None:
        """Kaiming/normal initialization for conv/linear; BN gamma=1, beta=0."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute penultimate embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 1, L)``.

        Returns
        -------
        torch.Tensor
            Embeddings of shape ``(B, 512)``.
        """
        x = self.prior_attn(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_pool(x).squeeze(-1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 1, L)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        feats = self.forward_features(x)
        return self.fc(feats)
