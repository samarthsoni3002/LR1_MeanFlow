import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_coord_channels(z):
    B, C, H, W = z.shape
    device = z.device
    dtype = z.dtype

    y_coords = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x_coords = torch.linspace(-1, 1, W, device=device, dtype=dtype)

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    xx = xx[None, None, :, :].expand(B, 1, H, W)
    yy = yy[None, None, :, :].expand(B, 1, H, W)

    return torch.cat([z, xx, yy], dim=1)

class SinusoidalEmbedding(nn.Module):

  def __init__(self, dim: int):
    super().__init__()
    self.dim = dim

  def forward(self, x):

    half_dim = self.dim // 2 
    device = x.device

    freqs = torch.exp(
        -math.log(1000) * torch.arange(half_dim, device=device) / (half_dim - 1)
    ) 

    args = x[:, None] * freqs[None, :] 

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim = -1)

    if self.dim % 2 == 1:
      emb = F.pad(emb, (0,1)) 

    return emb


class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, temb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)


        time_bias = self.time_proj(temb)[:, :, None, None]
        h = h + time_bias

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)
    

class MeanFlowUNet(nn.Module):

    def __init__(
        self,
        img_channels=3,
        base_channels=64,
        time_emb_dim=128,
        num_classes=3,
    ):
        super().__init__()

        self.time_embed = SinusoidalEmbedding(time_emb_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim*2, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.label_embed = nn.Embedding(num_classes, time_emb_dim)

        self.in_conv = nn.Conv2d(img_channels + 2, base_channels, kernel_size=3, padding=1)

        self.res1 = ResBlock(base_channels, base_channels, time_emb_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.res2 = ResBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.mid = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.res_up1 = ResBlock(base_channels * 2, base_channels * 2, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.res_up2 = ResBlock(base_channels, base_channels, time_emb_dim)

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)

    def forward(self, z, r, t, y=None):

        dt = t - r

        t_emb = self.time_embed(t)
        dt_emb = self.time_embed(dt)

        temb = torch.cat([t_emb, dt_emb], dim=-1)
        temb = self.time_mlp(temb)

        if y is not None:
            temb = temb + self.label_embed(y)
            
        z = add_coord_channels(z)

        h1 = self.in_conv(z)
        h1 = self.res1(h1, temb)

        h2 = self.down1(h1)
        h2 = self.res2(h2, temb)

        h3 = self.down2(h2)
        h3 = self.mid(h3, temb)

        h = self.up1(h3)
        h = h + h2
        h = self.res_up1(h, temb)

        h = self.up2(h)
        h = h + h1
        h = self.res_up2(h, temb)

        h = self.out_norm(h)
        h = F.silu(h)
        out = self.out_conv(h)

        return out