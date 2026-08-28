import math
from dataclasses import dataclass

import torch

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)
LATENT_CHANNELS = 16
DIT_IN_CHANNELS = 18
TEXT_LEN = 512
TEXT_DIM = 4096

FRAME_BUCKET = 81
NUM_REF = 1
NUM_VIDEOREF = 20
VIDEOREF_FPS = 25
OUTPUT_FPS = 25

AUDIO_SR = 16000
AUDIO_DIM_DRIVING = 8192

ROPE_SPATIAL_STRIDE = 720 / 480

BUCKETS = {"landscape": (480, 896), "portrait": (896, 480)}

MOTIONFRAME_LATENTS = 2
MOTIONFRAME_CHUNK_LEN = (FRAME_BUCKET - 1) // VAE_STRIDE[0] + 1
MOTIONFRAME_CHUNK_STRIDE = MOTIONFRAME_CHUNK_LEN - MOTIONFRAME_LATENTS


def latent_len(num_frames: int) -> int:
    return (num_frames - 1) // VAE_STRIDE[0] + 1


def num_motionframe_chunks(latents: int) -> int:
    return math.ceil((latents - MOTIONFRAME_CHUNK_LEN) / MOTIONFRAME_CHUNK_STRIDE) + 1


SAMPLE_STEPS = 24
FLOW_SHIFT = 5.0
TXT_GUIDE_SCALE = 5.0
AUD_GUIDE_SCALE = 1.8
BASE_SEED = 42
MOTIONFRAME_TAIL_EXTENSION = 4

NEGATIVE_PROMPT = (
    "Worst quality, low quality, blurry details, unclear details, "
    "poorly drawn face, "
    "blurry eyes, unclear eyes, "
    "poorly drawn hands, blurry hands, fat hands, swollen hands, "
    "extra fingers, fused fingers, deformed fingers, fat fingers, swollen fingers, "
    "missing limbs, malformed limbs, three legs, "
    "Moire pattern, stitching artifacts, stitching beard. "
)


@dataclass
class Reference:
    latents: torch.Tensor
    audio: torch.Tensor | None
    face_masks: torch.Tensor


@dataclass
class Target:
    latents: torch.Tensor
    foreground: torch.Tensor


@dataclass
class Driving:
    features: torch.Tensor
    num_frames: int
