import torch
import torch.nn as nn
import torch.nn.functional as F

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        # Global average pooling over the temporal dimension: [B, C, T] -> [B, C]
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        s = s.unsqueeze(-1)  # reshape to [B, C, 1] for channel-wise scaling
        return x * s
    
class BandWeightAttention(nn.Module):
    """
    x · w,  with  w = w_out·(1-mask) + w_in·mask

    Parameters
    ----------
    mask : (1,1,L) float tensor with 1 inside diagnostic bands, 0 elsewhere.
    w_in_init  : start value (≥1)  for inside‑band weight
    w_out_init : start value (≤1)  for outside‑band weight (can be 0)
    learnable_in / learnable_out : whether each scalar is trainable.
    """
    def __init__(self,
                 mask: torch.Tensor,
                 w_in_init: float  = 1.0,
                 w_out_init: float = 0.2,
                 learnable_in: bool  = True,
                 learnable_out: bool = True):
        super().__init__()
        self.register_buffer("mask", mask)

        # ----- OUTSIDE weight ---------------------------------------------------
        if learnable_out:
            # softplus‑param ⇒ always ≥0
            init = torch.log(torch.tensor(w_out_init + 1e-6))
            self.log_out = nn.Parameter(init)
        else:
            self.register_buffer("log_out",
                                 torch.log(torch.tensor(w_out_init + 1e-6)),
                                 persistent=False)        # fixed scalar

        # ----- INSIDE weight ----------------------------------------------------
        if learnable_in:
            # inside weight ≥1 : parameterise w_in-1 with softplus
            init = torch.log(torch.tensor(w_in_init - 1.0 + 1e-6))
            self.log_in  = nn.Parameter(init)
        else:
            self.register_buffer("log_in",
                                 torch.log(torch.tensor(w_in_init - 1.0 + 1e-6)),
                                 persistent=False)

    # -----------------------------------------------------------------------
    def _weights(self):
        w_out = F.softplus(self.log_out)          # ≥0
        w_in  = 1.0 + F.softplus(self.log_in)     # ≥1
        return w_in, w_out
    
    def forward(self, x):                                    # x: (B,C,L)
        w_in, w_out = self._weights()
        weight = w_out + (w_in - w_out) * self.mask          # broadcast
        return x * weight
    
class SpectralPriorAttention(nn.Module):
    """
    Multiplies feature maps by (1 + gain * mask) where `mask` is a fixed
    (1,1,L) tensor with 0/1 or smoothed values. `gain` is learnable so the
    network can down-weight or up-weight the prior during training.
    """
    def __init__(self, mask: torch.Tensor, init_gain: float = 2.0):
        super().__init__()
        self.register_buffer("mask", mask)            # shape (1,1,L)
        self.gain = nn.Parameter(torch.full((1,), init_gain))

    def forward(self, x):                              # x: (B,C,L)
        return x * (1.0 + self.gain * self.mask)

class ResidualBlock(nn.Module):
    """
    Definition of the Residual Block used in the ResNet model.
    - use_se: If True, integrates a Squeeze-and-Excitation (SE) block.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_se=False):
        super(ResidualBlock, self).__init__()
        # Pre-activation ordering: BN -> ReLU -> Conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = downsample
        self.use_se = use_se

        if use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        
        # Pre-activation: first BN and ReLU on input
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
    def __init__(self, block, layers, num_classes=2,
                 dropout_rate=0.5, block_kwargs=None, 
                 prior_kwargs: dict | None = None,
                 init_gain: float = 2.0):
        super().__init__()
        if block_kwargs is None:
            block_kwargs = {}

        self.in_channels = 64
        self.dropout_rate = dropout_rate
        self.conv = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn   = nn.BatchNorm1d(64)

        if prior_kwargs is not None:
            self.prior_attn = BandWeightAttention(**prior_kwargs)
        else:
            self.prior_attn = nn.Identity()

        self.layer1 = self.make_layer(block, 64,  layers[0], stride=1, **block_kwargs)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, **block_kwargs)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, **block_kwargs)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, **block_kwargs)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def make_layer(self, block, out_channels, blocks, stride=1, **block_kwargs):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
                )
            
        layers = [block(self.in_channels, out_channels, stride, downsample, **block_kwargs)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, **block_kwargs))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.prior_attn(x)
        out = F.relu(self.bn(self.conv(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out).squeeze(-1)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        return self.fc(out)

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model 
    """
    def __init__(self, input_dim, latent_dim, dropout_p=0.2):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),

            nn.Flatten(),
            nn.Linear(128 * input_dim, 128), 
            nn.LeakyReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),

            nn.Linear(128, 128 * input_dim),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, input_dim)),
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            
            nn.ConvTranspose1d(64, 1, kernel_size=3, padding=1),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def generate(self, z):
        x_decoded_input = self.decoder_input(z)
        x_decoded = self.decoder(x_decoded_input)
        return x_decoded

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        z = self.reparameterize(mu, logvar)
        x_decoded_input = self.decoder_input(z)
        x_decoded = self.decoder(x_decoded_input)
        return x_decoded, mu, logvar
