import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Definition of the Residual Block used in the ResNet model.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """
    Definition of the ResNet model for time-series classification.
    """
    def __init__(self, block, layers, num_classes=2, dropout_rate=0.5):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.dropout_rate = dropout_rate
        self.conv = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm1d(64)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
                )
            
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.squeeze(-1)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.fc(out)
        return out

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

def total_variation_loss(signal, weight=1e-3):
    """
    Encourages smoothness in the 1D output signal.
    signal shape: [B, 1, L]
    """
    diff = signal[:, :, 1:] - signal[:, :, :-1]
    tv = torch.mean(torch.abs(diff))
    return weight * tv


def vae_loss(x, x_decoded, mu, logvar, beta=1.0, smoothness_weight=0.0, delta=1.0):
    """
    VAE loss using Huber (Smooth L1) for reconstruction + KL divergence + optional TV smoothing.
    
    Args:
      x (Tensor): Original input of shape [B, 1, L].
      x_decoded (Tensor): Model's reconstruction of shape [B, 1, L].
      mu (Tensor): Mean vector from the encoder.
      logvar (Tensor): Log variance from the encoder.
      beta (float): Weight for the KL term (for beta-VAE).
      smoothness_weight (float): If > 0, apply total variation penalty at that weight.
      delta (float): Huber threshold (nn.SmoothL1Loss).
    """
    # 1) Huber (Smooth L1) reconstruction
    huber_fn = nn.SmoothL1Loss(reduction='sum', beta=delta)
    recon_loss = huber_fn(x_decoded, x) / x.size(0)

    # 2) KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # 3) Optional total variation for smoothing
    tv_loss = 0.0
    if smoothness_weight > 0:
        tv_loss = total_variation_loss(x_decoded, weight=smoothness_weight)

    return recon_loss + beta * kl_loss + tv_loss
