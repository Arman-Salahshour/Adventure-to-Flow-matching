"""Conditional U-Net predicting the flow matching velocity on 4x12x16 latents."""
import math

from constants import *


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d, max_period=10000, scale=1000):
        super().__init__()
        assert d % 2 == 0
        self.d = d
        # t lives in [0, 1], scale spreads it out so nearby times stay distinguishable
        self.scale = scale
        # h as half of dimension
        self.h = d // 2

        k = torch.arange(self.h, dtype=torch.float32)
        freq = torch.exp(-math.log(max_period) * k / self.h)
        self.register_buffer('freq', freq, persistent=False)

    def forward(self, t):
        args = self.scale * t[:, None] * self.freq[None, :]
        return torch.concat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.sin = SinusoidalEmbedding(d)
        self.fc = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.SiLU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, t):
        return self.fc(self.sin(t))


class ConEmbedding(nn.Module):
    """Embeds the 6 scores; rows where drop is True get the learned null vector (for CFG)."""
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 4 * d),
            nn.SiLU(),
            nn.Linear(4 * d, d),
        )

        self.null_embd = nn.Parameter(torch.zeros(d))

    def forward(self, x, drop):
        x = self.fc(x)
        x = torch.where(drop[:, None], self.null_embd, x)

        return x


class ResBlock(nn.Module):
    def __init__(self, C_in, C_out, d):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=C_in)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=C_out)

        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)

        self.embd_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, C_out * 2),
        )
        # FiLM starts as identity
        nn.init.zeros_(self.embd_proj[1].weight)
        nn.init.zeros_(self.embd_proj[1].bias)

        if C_in == C_out:
            self.skip_path = nn.Identity()
        else:
            self.skip_path = nn.Conv2d(C_in, C_out, kernel_size=1)

        self.act = nn.SiLU()

    def forward(self, x, embd):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        embd = self.embd_proj(embd)

        scale = embd[..., :embd.shape[-1] // 2]
        shift = embd[..., embd.shape[-1] // 2:]

        scale = scale[..., None, None]
        shift = shift[..., None, None]

        # Modulate after the norm, otherwise the norm would erase the shift
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip_path(x)


class AttenBlock(nn.Module):
    def __init__(self, heads, channels):
        super().__init__()

        self.norm = nn.GroupNorm(32, channels)
        self.multi_atten = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)

    def forward(self, x):
        batch, channel, height, width = x.shape
        h = self.norm(x)
        # Each spatial position becomes a token
        h = h.reshape(batch, channel, -1)
        h = torch.permute(h, (0, 2, 1))
        h = self.multi_atten(h, h, h)[0]
        h = torch.permute(h, (0, 2, 1))
        h = h.reshape(batch, channel, height, width)

        return x + h


class UNet(nn.Module):
    def __init__(self, d=256, blocks_per_level=2, width_levels=(128, 256, 384)):
        super().__init__()
        self.blocks_per_level = blocks_per_level
        self.con_embd = ConEmbedding(d)
        self.time_mlp = TimeMLP(d)

        current = width_levels[0]
        self.conv1 = nn.Conv2d(4, current, kernel_size=3, padding=1)

        # skip_ch mirrors the skips list in forward, so up-block widths come out right
        skip_ch = list()
        res_blocks_downward = list()
        downsamplers = list()
        for i, width in enumerate(width_levels):
            for _ in range(blocks_per_level):
                res_blocks_downward.append(ResBlock(current, width, d))
                current = width
                skip_ch.append(width)

            if i != len(width_levels) - 1:
                downsamplers.append(nn.Conv2d(current, current, kernel_size=3, stride=2, padding=1))

        self.res_blocks_downward = nn.ModuleList(res_blocks_downward)
        self.downsamplers = nn.ModuleList(downsamplers)

        self.res_block1 = ResBlock(current, current, d)
        self.att_block = AttenBlock(8, current)
        self.res_block2 = ResBlock(current, current, d)

        res_blocks_upward = list()
        upsamplers = list()
        for i, width in enumerate(width_levels[::-1]):
            for _ in range(blocks_per_level):
                popped = skip_ch.pop(-1)
                res_blocks_upward.append(ResBlock(current + popped, width, d))
                current = width

            if i != len(width_levels) - 1:
                upsamplers.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(current, current, kernel_size=3, padding=1),
                ))

        self.res_blocks_upward = nn.ModuleList(res_blocks_upward)
        self.upsamplers = nn.ModuleList(upsamplers)

        assert len(skip_ch) == 0

        self.head = nn.Sequential(
            nn.GroupNorm(32, current),
            nn.SiLU(),
            nn.Conv2d(current, 4, kernel_size=3, padding=1),
        )

        # Starts by predicting zero velocity
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, x, t, c, drop):
        t = self.time_mlp(t)
        c = self.con_embd(c, drop)
        emb = t + c
        h = self.conv1(x)

        skips = list()
        n_levels = len(self.downsamplers) + 1

        k = 0
        for i in range(n_levels):
            for _ in range(self.blocks_per_level):
                h = self.res_blocks_downward[k](h, emb)
                skips.append(h)
                k += 1

            if i < n_levels - 1:
                h = self.downsamplers[i](h)

        h = self.res_block1(h, emb)
        h = self.att_block(h)
        h = self.res_block2(h, emb)

        k = 0
        for i in range(n_levels):
            for _ in range(self.blocks_per_level):
                popped = skips.pop(-1)
                h = torch.concat([h, popped], dim=1)
                h = self.res_blocks_upward[k](h, emb)
                k += 1

            if i < n_levels - 1:
                h = self.upsamplers[i](h)

        h = self.head(h)

        return h
