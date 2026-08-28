import math

import torch
import torch.nn as nn
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.normalization import RMSNorm
from safetensors.torch import load_file

from tavr.contract import (
    DIT_IN_CHANNELS,
    FRAME_BUCKET,
    LATENT_CHANNELS,
    NUM_REF,
    NUM_VIDEOREF,
    PATCH_SIZE,
    TEXT_DIM,
    TEXT_LEN,
)
from tavr.models.attention import flash_attention, flash_attention_videoref

__all__ = ["WanModel", "load_dit"]

DIM = 5120
FFN_DIM = 13824
NUM_HEADS = 40
NUM_LAYERS = 40
FREQ_DIM = 256
EPS = 1e-6
AUDIO_DIM = 8192
VIDEOREF_AUDIO_DIM = 2048
ADAPTER_DIM = 256
ROPE_SPATIAL_STRIDE = 720 / 480
VIDEOREF_ROPE_OFFSET = 256
AUDIO_PER_LATENT = 4
AUDIO_WINDOW = 3 * AUDIO_PER_LATENT

NUM_VIDEO_LATENT = (FRAME_BUCKET - 1) // 4 + 1
NUM_VIDEOREF_LATENT = NUM_VIDEOREF // 4


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    frequencies = torch.pow(10000, -torch.arange(half).to(position).div(half))
    angles = torch.outer(position, frequencies)
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)


@torch.autocast("cuda", enabled=False)
def rope_table(max_seq_len: int, dim: int, theta: float = 10000, stride: float = 1) -> torch.Tensor:
    positions = torch.arange(max_seq_len, device="cpu") * stride
    return get_1d_rotary_pos_embed(dim, positions, theta, repeat_interleave_real=False, freqs_dtype=torch.float64)


@torch.autocast("cuda", enabled=False)
def apply_rope(x: torch.Tensor, grid: tuple[int, int, int], freqs: torch.Tensor) -> torch.Tensor:
    frames, height, width = grid
    heads, half = x.size(2), x.size(3) // 2
    freqs_t, freqs_h, freqs_w = freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)

    per_token = torch.cat(
        [
            freqs_t[:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            freqs_h[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            freqs_w[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(frames * height * width, 1, -1)

    seq_len = frames * height * width
    rotated = []
    for sample in x:
        pairs = torch.view_as_complex(sample[:seq_len].to(torch.float64).reshape(seq_len, heads, -1, 2))
        turned = torch.view_as_real(pairs * per_token.to(pairs.device)).flatten(2)
        rotated.append(torch.cat([turned, sample[seq_len:]]))
    return torch.stack(rotated).float()


class WanLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dtype=torch.bfloat16)


class WanRMSNorm(RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__(dim, eps)
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


def audio_adapter(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(dim), WanLinear(dim, ADAPTER_DIM), nn.GELU(), WanLinear(ADAPTER_DIM, dim))


class WanSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = WanLinear(dim, dim)
        self.k = WanLinear(dim, dim)
        self.v = WanLinear(dim, dim)
        self.o = WanLinear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def out_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.o(x.flatten(2))


class WanVideoReferenceSelfAttention(WanSelfAttention):
    def forward(self, x, grid, freqs, valid_tokens):
        batch, total, heads, head_dim = *x.shape[:2], self.num_heads, self.head_dim
        assert batch == 1, "valid-token self-attention is written for a single sequence"
        frames, height, width = grid
        len_videoref = NUM_VIDEOREF_LATENT * height * width
        len_sample = (frames - NUM_VIDEOREF_LATENT) * height * width

        kept = valid_tokens["all"]
        narrow = x[:, kept]
        q_kept = self.norm_q(self.q(narrow)).view(batch, -1, heads, head_dim)
        k_kept = self.norm_k(self.k(narrow)).view(batch, -1, heads, head_dim)
        v_kept = self.v(narrow).view(batch, -1, heads, head_dim)
        q = torch.zeros((batch, total, heads, head_dim), dtype=q_kept.dtype, device=q_kept.device)
        k, v = q.clone(), q.clone()
        q[:, kept], k[:, kept], v[:, kept] = q_kept, k_kept, v_kept

        q = apply_rope(q, grid, freqs).to(x.dtype)
        k = apply_rope(k, grid, freqs).to(x.dtype)
        q, q_videoref = torch.split(q, [len_sample, len_videoref], dim=1)
        k, k_videoref = torch.split(k, [len_sample, len_videoref], dim=1)
        v, v_videoref = torch.split(v, [len_sample, len_videoref], dim=1)

        valid_videoref = valid_tokens["videoref"]
        valid_sample = valid_tokens["sample_ref"]
        out_sample = torch.zeros_like(q)
        attended, attended_videoref = flash_attention_videoref(
            q=q[:, valid_sample].contiguous(),
            k=k[:, valid_sample].contiguous(),
            v=v[:, valid_sample].contiguous(),
            q_videoref=q_videoref[:, valid_videoref].contiguous(),
            k_videoref=k_videoref[:, valid_videoref].contiguous(),
            v_videoref=v_videoref[:, valid_videoref].contiguous(),
        )
        out_videoref = torch.zeros(
            (batch, len_videoref, heads, head_dim), dtype=attended_videoref.dtype, device=attended_videoref.device
        )
        out_sample[:, valid_sample] = attended
        out_videoref[:, valid_videoref] = attended_videoref
        return self.out_fn(torch.cat([out_sample, out_videoref], dim=1))


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, grid, valid_tokens):
        batch, total, heads, head_dim = x.size(0), x.size(1), self.num_heads, self.head_dim
        assert batch == 1
        frames, height, width = grid

        kept = valid_tokens["all"]
        q_kept = self.norm_q(self.q(x[:, kept])).view(batch, -1, heads, head_dim)
        q = torch.zeros((batch, total, heads, head_dim), dtype=q_kept.dtype, device=q_kept.device)
        q[:, kept] = q_kept
        k = self.norm_k(self.k(context)).view(batch, -1, heads, head_dim)
        v = self.v(context).view(batch, -1, heads, head_dim)

        attended = flash_attention(
            q.view(frames, height * width, heads, head_dim),
            k.view(frames, -1, heads, head_dim),
            v.view(frames, -1, heads, head_dim),
        ).view(1, -1, heads, head_dim)

        projected = self.out_fn(attended[:, kept])
        out = torch.zeros((batch, total, heads * head_dim), dtype=projected.dtype, device=projected.device)
        out[:, kept] = projected
        return out


class WanA2VCrossAttention(WanSelfAttention):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__(dim, num_heads, eps)
        self.videoref_aud_q_adapter = audio_adapter(dim)
        self.videoref_aud_k_adapter = audio_adapter(dim)
        self.videoref_aud_v_adapter = audio_adapter(dim)

    def forward(self, x, context, pos_emb, is_videoref_aud_mode, valid_tokens=None):
        batch, total, heads, head_dim = x.size(0), x.size(1), self.num_heads, self.head_dim
        latents = context.shape[1] // AUDIO_PER_LATENT

        if is_videoref_aud_mode:
            kept = valid_tokens["videoref"]
            narrow = x[:, kept]
            q_kept = self.norm_q(self.q(narrow) + self.videoref_aud_q_adapter(narrow)).view(batch, -1, heads, head_dim)
            q = torch.zeros((batch, total, heads, head_dim), dtype=q_kept.dtype, device=q_kept.device)
            q[:, kept] = q_kept
            q = q.view(batch * latents, -1, heads, head_dim)
            k = self.norm_k(self.k(context) + self.videoref_aud_k_adapter(context)).view(batch, -1, heads, head_dim)
            v = (self.v(context) + self.videoref_aud_v_adapter(context)).view(batch, -1, heads, head_dim)
            width = AUDIO_PER_LATENT
        else:
            q = self.norm_q(self.q(x)).view(batch * latents, -1, heads, head_dim)
            k = self.norm_k(self.k(context)).view(batch, -1, heads, head_dim)
            v = self.v(context).view(batch, -1, heads, head_dim)
            pad = k.new_zeros(batch, AUDIO_PER_LATENT, heads, head_dim)
            k = torch.cat([pad, k, pad], dim=1)
            v = torch.cat([pad, v, pad], dim=1)
            width = AUDIO_WINDOW

        k = self._windows(k, latents, width) + pos_emb.view(width, heads, head_dim)
        v = self._windows(v, latents, width)
        return self.out_fn(flash_attention(q, k, v).view(batch, -1, heads, head_dim))

    @staticmethod
    def _windows(tokens, latents, width):
        stacked = torch.stack(
            [tokens[:, AUDIO_PER_LATENT * i : AUDIO_PER_LATENT * i + width] for i in range(latents)], dim=1
        )
        return stacked.flatten(0, 1)


class WanAttentionBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanVideoReferenceSelfAttention(dim, num_heads, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(WanLinear(dim, ffn_dim), nn.GELU(approximate="tanh"), WanLinear(ffn_dim, dim))
        self.norm4 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.audio_attn = WanA2VCrossAttention(dim, num_heads, eps)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        e_videoref,
        grid,
        freqs,
        txt,
        null_txt,
        aud,
        aud_pos_emb,
        videoref_aud,
        videoref_aud_pos_emb,
        valid_tokens,
    ):
        assert e.dtype == torch.float32 and e_videoref.dtype == torch.float32
        with torch.autocast("cuda", dtype=torch.float32):
            shifts = (self.modulation + e).chunk(6, dim=1)
            shifts_videoref = (self.modulation + e_videoref).chunk(6, dim=1)

        frames, height, width = grid
        len_sample = NUM_VIDEO_LATENT * height * width
        len_ref = NUM_REF * height * width
        len_videoref = NUM_VIDEOREF_LATENT * height * width
        assert frames == NUM_VIDEO_LATENT + NUM_REF + NUM_VIDEOREF_LATENT
        assert len_sample + len_ref + len_videoref == x.shape[1]

        def modulation(i):
            return torch.cat(
                [shifts[i].repeat(1, len_sample + len_ref, 1), shifts_videoref[i].repeat(1, len_videoref, 1)], dim=1
            )

        attended = self.self_attn(self.norm1(x).float() * (1 + modulation(1)) + modulation(0), grid, freqs, valid_tokens)
        with torch.autocast("cuda", dtype=torch.float32):
            x = x + attended * modulation(2)

        captions = torch.cat(
            [
                txt.unsqueeze(1).repeat(1, NUM_VIDEO_LATENT + NUM_REF, 1, 1),
                null_txt.unsqueeze(1).repeat(1, NUM_VIDEOREF_LATENT, 1, 1),
            ],
            dim=1,
        )
        x = x + self.cross_attn(self.norm3(x), captions, grid, valid_tokens)

        x[:, :len_sample] = x[:, :len_sample] + self.audio_attn(
            self.norm4(x[:, :len_sample]), aud, aud_pos_emb, is_videoref_aud_mode=False
        )
        start, end = len_sample + len_ref, len_sample + len_ref + len_videoref
        x[:, start:end] = x[:, start:end] + self.audio_attn(
            self.norm4(x[:, start:end]),
            videoref_aud,
            videoref_aud_pos_emb,
            is_videoref_aud_mode=True,
            valid_tokens=valid_tokens,
        )

        kept = valid_tokens["all"]
        forwarded = self.ffn((self.norm2(x).float() * (1 + modulation(4)) + modulation(3))[:, kept])
        scattered = torch.zeros(x.shape, dtype=forwarded.dtype, device=forwarded.device)
        scattered[:, kept] = forwarded
        with torch.autocast("cuda", dtype=torch.float32):
            x = x + scattered * modulation(5)
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = WanLinear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        assert e.dtype == torch.float32
        with torch.autocast("cuda", dtype=torch.float32):
            shift, scale = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanModel(nn.Module):
    def __init__(
        self,
        dim: int = DIM,
        ffn_dim: int = FFN_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        freq_dim: int = FREQ_DIM,
        text_dim: int = TEXT_DIM,
        audio_dim: int = AUDIO_DIM,
        out_dim: int = LATENT_CHANNELS,
        eps: float = EPS,
        rope_spatial_stride: float = ROPE_SPATIAL_STRIDE,
    ):
        super().__init__()
        assert dim % num_heads == 0 and (dim // num_heads) % 2 == 0
        self.dim = dim
        self.num_heads = num_heads
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.patch_embedding = nn.Conv3d(DIT_IN_CHANNELS, dim, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.text_embedding = nn.Sequential(WanLinear(text_dim, dim), nn.GELU(approximate="tanh"), WanLinear(dim, dim))
        self.time_embedding = nn.Sequential(WanLinear(freq_dim, dim), nn.SiLU(), WanLinear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), WanLinear(dim, dim * 6))

        self.blocks = nn.ModuleList([WanAttentionBlock(dim, ffn_dim, num_heads, eps) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, PATCH_SIZE, eps)

        head_dim = dim // num_heads
        sixth = head_dim // 6
        self.freqs = torch.cat(
            [
                rope_table(1024, head_dim - 4 * sixth),
                rope_table(1024, 2 * sixth, stride=rope_spatial_stride),
                rope_table(1024, 2 * sixth, stride=rope_spatial_stride),
            ],
            dim=1,
        )

        self.aud_embedding = nn.Sequential(WanLinear(audio_dim, dim), nn.GELU(approximate="tanh"), WanLinear(dim, dim))
        self.audio_pos_embedding = nn.Embedding(AUDIO_WINDOW, dim)
        self.null_audio_feature = nn.Embedding(1, audio_dim)

        self.ref_pos_embedding = nn.Embedding(1, dim)

        self.cross_videoref_pos_embedding = nn.Embedding(1, dim)
        self.videoref_aud_embedding = nn.Sequential(
            WanLinear(VIDEOREF_AUDIO_DIM, dim), nn.GELU(approximate="tanh"), WanLinear(dim, dim)
        )
        self.null_videoref_audio_feature = nn.Embedding(1, VIDEOREF_AUDIO_DIM)
        self.videoref_audio_pos_embedding = nn.Embedding(AUDIO_PER_LATENT, dim)
        self.time_embedding_videoref = nn.Parameter(torch.randn(1, dim) * 0.02)

    def _time_condition(
        self,
        timestep: torch.Tensor,
        offset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.float32):
            embedding = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep).float()
            )
            if offset is not None:
                embedding = embedding + offset
            projection = self.time_projection(embedding).unflatten(
                1,
                (6, self.dim),
            )
        return embedding, projection

    def forward(self, x, t, txt, aud=None, null_txt=None, **kwargs):
        x = torch.stack(x)
        device = x.device
        batch = len(x)
        valid_tokens = self._valid_token_indices(kwargs["valid_masks"])
        assert x.shape[2] == NUM_VIDEO_LATENT + NUM_REF

        aud = self.aud_embedding(self._audio_tokens(aud, self.null_audio_feature, NUM_VIDEO_LATENT, batch, device))
        videoref_aud = self.videoref_aud_embedding(
            self._audio_tokens(
                kwargs.get("context_videoref_audio"),
                self.null_videoref_audio_feature,
                kwargs["context_videoref"][0].shape[1],
                batch,
                device,
            ).contiguous()
        )
        aud_pos_emb = self.audio_pos_embedding(torch.arange(AUDIO_WINDOW, device=device))
        videoref_aud_pos_emb = self.videoref_audio_pos_embedding(torch.arange(AUDIO_PER_LATENT, device=device))

        patched = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        _, height, width = patched[0].shape[2:]
        tokens = torch.cat([u.flatten(2).transpose(1, 2) for u in patched])
        videoref_patched = [self.patch_embedding(u.unsqueeze(0)) for u in kwargs["context_videoref"]]
        tokens = torch.cat([tokens, torch.cat([u.flatten(2).transpose(1, 2) for u in videoref_patched])], dim=1)
        grid = (NUM_VIDEO_LATENT + NUM_REF + NUM_VIDEOREF_LATENT, height, width)

        e, e0 = self._time_condition(t)
        _, e0_videoref = self._time_condition(
            torch.zeros_like(t),
            self.time_embedding_videoref,
        )
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        ref_pos_emb = self.ref_pos_embedding(torch.tensor(0, device=device))
        videoref_pos_emb = self.cross_videoref_pos_embedding(torch.tensor(0, device=device)).view(1, 1, -1)
        videoref_pos_emb = videoref_pos_emb.repeat(1, NUM_VIDEOREF_LATENT, 1).repeat_interleave(height * width, dim=1)

        ref_start = NUM_VIDEO_LATENT * height * width
        ref_end = ref_start + NUM_REF * height * width
        videoref_end = ref_end + NUM_VIDEOREF_LATENT * height * width
        assert tokens.shape[1] == videoref_end

        block_kwargs = {
            "e": e0,
            "e_videoref": e0_videoref,
            "grid": grid,
            "freqs": self._time_shifted_freqs(device),
            "txt": self._embed_text(txt),
            "null_txt": self._embed_text(null_txt),
            "aud": aud,
            "aud_pos_emb": aud_pos_emb,
            "videoref_aud": videoref_aud,
            "videoref_aud_pos_emb": videoref_aud_pos_emb,
            "valid_tokens": valid_tokens,
        }
        for block in self.blocks:
            tokens[:, ref_start:ref_end] = tokens[:, ref_start:ref_end] + ref_pos_emb
            tokens[:, ref_end:videoref_end] = tokens[:, ref_end:videoref_end] + videoref_pos_emb
            tokens = block(tokens, **block_kwargs)

        tokens = self.head(tokens, e)
        return [u[:, :NUM_VIDEO_LATENT].contiguous().float() for u in self.unpatchify(tokens, grid)]

    @staticmethod
    def _audio_tokens(features, null_embedding, latents, batch, device):
        null = null_embedding(torch.tensor(0, device=device))
        if features is None:
            features = null.view(1, 1, -1).expand(batch, latents * AUDIO_PER_LATENT, -1)
        elif features.shape[1] % AUDIO_PER_LATENT == 0:
            features = features + 0 * null
        else:
            head = null.view(1, 1, -1).expand(batch, AUDIO_PER_LATENT - 1, -1)
            features = torch.cat([head, features], dim=1)
        assert features.shape[1] == latents * AUDIO_PER_LATENT
        return features

    def _time_shifted_freqs(self, device):
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        half = self.dim // self.num_heads // 2
        freqs_t, freqs_h, freqs_w = self.freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)
        shifted = freqs_t.clone()
        shifted[NUM_VIDEO_LATENT : NUM_VIDEO_LATENT + NUM_REF] = freqs_t[:NUM_REF]
        start = NUM_VIDEO_LATENT + NUM_REF
        shifted[start : start + NUM_VIDEOREF_LATENT] = freqs_t[
            VIDEOREF_ROPE_OFFSET : VIDEOREF_ROPE_OFFSET + NUM_VIDEOREF_LATENT
        ]
        return torch.cat([shifted, freqs_h, freqs_w], dim=1)

    def _embed_text(self, texts):
        padded = torch.stack([torch.cat([u, u.new_zeros(TEXT_LEN - u.size(0), u.size(1))]) for u in texts])
        return self.text_embedding(padded)

    @staticmethod
    def _valid_token_indices(valid_masks):
        return {
            "all": torch.where(valid_masks.flatten())[0],
            "videoref": torch.where(valid_masks[-NUM_VIDEOREF_LATENT:].flatten())[0],
            "sample_ref": torch.where(valid_masks[:-NUM_VIDEOREF_LATENT].flatten())[0],
        }

    def unpatchify(self, x, grid):
        out = []
        for sample in x:
            u = sample[: math.prod(grid)].view(*grid, *PATCH_SIZE, self.out_dim)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            out.append(u.reshape(self.out_dim, *[a * b for a, b in zip(grid, PATCH_SIZE, strict=True)]))
        return out


def load_dit(checkpoint_path: str, device: str = "cpu", **overrides) -> WanModel:
    with torch.device("meta"):
        model = WanModel(**overrides)
    model.load_state_dict(load_file(checkpoint_path, device="cpu"), strict=True, assign=True)
    return model.eval().requires_grad_(False).to(device)
