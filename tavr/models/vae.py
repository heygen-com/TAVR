import re
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan, WanRMS_norm

from tavr.contract import LATENT_CHANNELS

__all__ = ["WanVAE"]

LATENT_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
LATENT_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)

_RENAMES = (
    (r"^(encoder|decoder)\.conv1\.", r"\1.conv_in."),
    (r"^encoder\.downsamples\.", "encoder.down_blocks."),
    (r"^conv1\.", "quant_conv."),
    (r"^conv2\.", "post_quant_conv."),
    (r"\.middle\.0\.", ".mid_block.resnets.0."),
    (r"\.middle\.1\.", ".mid_block.attentions.0."),
    (r"\.middle\.2\.", ".mid_block.resnets.1."),
    (r"\.head\.0\.", ".norm_out."),
    (r"\.head\.2\.", ".conv_out."),
    (r"\.residual\.0\.", ".norm1."),
    (r"\.residual\.2\.", ".conv1."),
    (r"\.residual\.3\.", ".norm2."),
    (r"\.residual\.6\.", ".conv2."),
    (r"\.shortcut\.", ".conv_shortcut."),
)


def _regroup_upsample(match: re.Match) -> str:
    slot = int(match.group(1))
    tail = "upsamplers.0" if slot % 4 == 3 else f"resnets.{slot % 4}"
    return f"decoder.up_blocks.{slot // 4}.{tail}."


def to_diffusers_key(key: str) -> str:
    key = re.sub(r"^decoder\.upsamples\.(\d+)\.", _regroup_upsample, key)
    for pattern, replacement in _RENAMES:
        key = re.sub(pattern, replacement, key)
    return key


class Fp32RMSNorm(WanRMS_norm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(x.float(), dim=1 if self.channel_first else -1)
        return (normed * self.scale * self.gamma + self.bias).to(x.dtype)


def _upcast_rms_norms(root: torch.nn.Module) -> None:
    for module in root.modules():
        if type(module) is WanRMS_norm:
            module.__class__ = Fp32RMSNorm


class WanVAE:
    def __init__(self, checkpoint_path: str, dtype: torch.dtype = torch.float32, device: str = "cpu"):
        with torch.device("meta"):
            core = AutoencoderKLWan()
        assert tuple(core.config.latents_mean) == LATENT_MEAN, "diffusers moved the latent statistics"
        assert tuple(core.config.latents_std) == LATENT_STD, "diffusers moved the latent statistics"

        original = torch.load(checkpoint_path, map_location="cpu", mmap=True)
        renamed = {to_diffusers_key(name): tensor for name, tensor in original.items()}
        assert len(renamed) == len(original), "the rename table collided two checkpoint entries"
        core.load_state_dict(renamed, assign=True, strict=True)

        self.dtype = dtype
        self.device = device
        self.core = core.eval().requires_grad_(False).to(dtype).to(device)
        _upcast_rms_norms(self.core)

        self.mean = torch.tensor(LATENT_MEAN, dtype=dtype, device=device)
        self.inv_std = 1.0 / torch.tensor(LATENT_STD, dtype=dtype, device=device)

    def _channels(self, stats: torch.Tensor) -> torch.Tensor:
        return stats.view(1, LATENT_CHANNELS, 1, 1, 1)

    @torch.no_grad()
    def encode(self, videos: list[torch.Tensor]) -> list[torch.Tensor]:
        latents = []
        for video in videos:
            mu = self.core.encode(video.unsqueeze(0).to(self.dtype).to(self.device)).latent_dist.mode()
            latents.append(((mu - self._channels(self.mean)) * self._channels(self.inv_std)).squeeze(0))
        return latents

    @torch.no_grad()
    def decode_stream(
        self,
        latents: torch.Tensor,
        max_frames: int = 250,
    ) -> Iterator[torch.Tensor]:
        core = self.core
        core.clear_cache()

        z = latents.unsqueeze(0).to(self.dtype).to(self.device)
        z = z / self._channels(self.inv_std) + self._channels(self.mean)
        z = core.post_quant_conv(z)

        held: list[torch.Tensor] = []
        held_frames = 0
        for step in range(z.shape[2]):
            core._conv_idx = [0]
            pixels = core.decoder(z[:, :, step : step + 1], feat_cache=core._feat_map, feat_idx=core._conv_idx)
            held.append(pixels.clamp(-1, 1).squeeze(0))
            held_frames += held[-1].shape[1]
            if held_frames >= max_frames:
                yield torch.cat(held, dim=1)
                held, held_frames = [], 0
        if held:
            yield torch.cat(held, dim=1)
        core.clear_cache()
