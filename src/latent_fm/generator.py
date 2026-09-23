"""Sampler and the full generator: embedding -> image.

An embedding is [z, c]: z is the flattened starting noise (768 values),
c the six scores in 0-9 units. With a fixed number of Euler steps and a
fixed guidance scale the map is deterministic, which is what the graders
need when they re-decode embeddings.pth.
"""
from constants import *
from model_fm import UNet
from diffusers import AutoencoderKL

STEPS = 50
GUIDANCE = 2.0


@torch.no_grad()
def euler_sample(unet, x0, c, steps=STEPS, guidance=GUIDANCE):
    """Integrates the velocity from noise (t=0) to data (t=1). c is already divided by 9."""
    B = x0.shape[0]
    dt = 1.0 / steps
    keep = torch.zeros(B, dtype=torch.bool, device=x0.device)
    drop = torch.ones(B, dtype=torch.bool, device=x0.device)

    x = x0
    for i in range(steps):
        t = torch.full((B,), i * dt, device=x0.device)
        # Conditioned and unconditioned in one batch; GroupNorm is per sample, so they don't mix
        v = unet(torch.concat([x, x]), torch.concat([t, t]), torch.concat([c, c]), torch.concat([keep, drop]))
        v_cond, v_null = v[:B], v[B:]
        v = v_null + guidance * (v_cond - v_null)
        x = x + v * dt

    return x


class CarGenerator(nn.Module):
    def __init__(self, unet, vae, scaling_factor, steps=STEPS, guidance=GUIDANCE):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.steps = steps
        self.guidance = guidance
        self.register_buffer('mean', torch.tensor(IN_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(IN_STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def latent(self, z, c):
        x0 = z.reshape(-1, *LATENT_SHAPE)
        return euler_sample(self.unet, x0, c / MAX_SCORE, self.steps, self.guidance)

    @torch.no_grad()
    def decode_latent(self, x1):
        img = self.vae.decode(x1 / self.scaling_factor).sample
        img = ((img + 1) / 2).clamp(0, 1)
        # Same space as the dataset loader gives
        return (img - self.mean) / self.std

    @torch.no_grad()
    def decode(self, z, c):
        return self.decode_latent(self.latent(z, c))

    def forward(self, z, c=None):
        # Accepts either (z, c) or one full embedding [N, 774]
        if c is None:
            z, c = z[:, :LATENT_DIM], z[:, LATENT_DIM:]
        return self.decode(z, c)

    def checkpoint(self):
        # Only the U-Net is trained; the VAE is the public pretrained one
        return {
            'unet': self.unet.state_dict(),
            'vae_name': VAE_NAME,
            'scaling_factor': self.scaling_factor,
            'steps': self.steps,
            'guidance': self.guidance,
        }

    @classmethod
    def from_checkpoint(cls, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        unet = UNet()
        unet.load_state_dict(ck['unet'])
        vae = AutoencoderKL.from_pretrained(ck['vae_name'])
        gen = cls(unet, vae, ck['scaling_factor'], ck['steps'], ck['guidance'])
        return gen.to(device).eval()


def build_generator(ema_path, latents_path):
    unet = UNet()
    unet.load_state_dict(torch.load(ema_path, map_location='cpu'))
    vae = AutoencoderKL.from_pretrained(VAE_NAME)
    scaling_factor = torch.load(latents_path, map_location='cpu', weights_only=False)['scaling_factor']
    return CarGenerator(unet, vae, scaling_factor).to(device).eval()
