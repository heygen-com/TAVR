from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import UniPCMultistepScheduler

from tavr import contract

logger = logging.getLogger(__name__)

NUM_VIDEO_LATENT = (contract.FRAME_BUCKET - 1) // contract.VAE_STRIDE[0] + 1
NUM_VIDEOREF_LATENT = contract.NUM_VIDEOREF // contract.VAE_STRIDE[0]
MOTIONFRAME_LATENTS = contract.MOTIONFRAME_LATENTS
MOTIONFRAME_CHUNK_LEN = NUM_VIDEO_LATENT
MOTIONFRAME_CHUNK_STRIDE = contract.MOTIONFRAME_CHUNK_STRIDE
AUDIO_PER_LATENT = contract.VAE_STRIDE[0]
MOTIONFRAME_AUDIO_LEN = (MOTIONFRAME_CHUNK_LEN - 1) * AUDIO_PER_LATENT + 1
NUM_SLOTS = NUM_VIDEO_LATENT + contract.NUM_REF + NUM_VIDEOREF_LATENT


@dataclass(frozen=True)
class SamplingConfig:
    steps: int = contract.SAMPLE_STEPS
    shift: float = contract.FLOW_SHIFT
    txt_scale: float = contract.TXT_GUIDE_SCALE
    aud_scale: float = contract.AUD_GUIDE_SCALE
    seed: int = contract.BASE_SEED
    start_use_videoref_step: int = 0


@dataclass(frozen=True)
class TextConditioning:
    prompt: list[torch.Tensor]
    negative: list[torch.Tensor]
    null: list[torch.Tensor]


@dataclass(frozen=True)
class MotionframePlan:
    count: int
    stride: int
    length: int
    canvas_len: int

    def read_at(self, index: int) -> int:
        return index * self.stride

    def write_span(self, index: int) -> tuple[int, int, int]:
        offset = MOTIONFRAME_LATENTS if index else 0
        start = index * self.stride + offset
        stop = min(index * self.stride + self.length, self.canvas_len)
        return offset, start, stop


def plan_motionframe_chunks(latent_len: int) -> MotionframePlan:
    count = contract.num_motionframe_chunks(latent_len)
    if count < 1:
        raise ValueError("driving audio is too short for motionframe generation")
    return MotionframePlan(
        count=count,
        stride=MOTIONFRAME_CHUNK_STRIDE,
        length=MOTIONFRAME_CHUNK_LEN,
        canvas_len=latent_len + contract.MOTIONFRAME_TAIL_EXTENSION,
    )


def draw_noise(
    plan: MotionframePlan,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    canvas = torch.randn(
        contract.LATENT_CHANNELS,
        plan.canvas_len,
        height,
        width,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    chunks = [canvas[:, plan.read_at(i) : plan.read_at(i) + plan.length].clone() for i in range(plan.count)]

    tail = chunks[-1]
    assert tail.shape[1] > 0, f"chunk {plan.count - 1} starts past the noise canvas"
    if tail.shape[1] < plan.length:
        missing = torch.randn(
            (contract.LATENT_CHANNELS, plan.length - tail.shape[1], height, width),
            device=device,
            generator=generator,
        )
        chunks[-1] = torch.cat([tail, missing], dim=1)
    return canvas, chunks


def build_scheduler(steps: int, shift: float, device: torch.device) -> UniPCMultistepScheduler:
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=shift,
    )
    scheduler.set_timesteps(steps + 1, device=device)
    scheduler.timesteps = scheduler.timesteps[:-1]
    scheduler.sigmas = torch.cat([scheduler.sigmas[:-2], scheduler.sigmas[-1:]])
    return scheduler


def build_valid_masks(face_masks: torch.Tensor, foreground: torch.Tensor, has_motionframes: bool) -> torch.Tensor:
    background = 1.0 - foreground.reshape(1, 1, *foreground.shape[-2:])
    ref_row = F.max_pool2d(background, kernel_size=2, stride=2)[0].bool()
    if has_motionframes:
        ref_row = torch.ones_like(ref_row)
    sample_rows = torch.ones(
        (NUM_VIDEO_LATENT, *ref_row.shape[-2:]),
        dtype=torch.bool,
        device=ref_row.device,
    )
    masks = torch.cat([sample_rows, ref_row, face_masks.to(ref_row.device).bool()], dim=0)
    assert masks.shape[0] == NUM_SLOTS, f"expected {NUM_SLOTS} mask slots, got {masks.shape[0]}"
    return masks


def _videoref_at_step(
    videoref: list[torch.Tensor],
    videoref_audio: torch.Tensor | None,
    valid_masks: torch.Tensor,
    step_index: int,
    num_steps: int,
    start_step: int,
) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor]:
    boundary = num_steps if start_step < 0 else start_step
    if step_index >= boundary:
        return videoref, videoref_audio, valid_masks

    hidden = [item.clone() for item in videoref]
    for item in hidden:
        item[: contract.LATENT_CHANNELS] = 0
    hidden_masks = valid_masks.clone()
    hidden_masks[-NUM_VIDEOREF_LATENT:] = False
    return hidden, None, hidden_masks


def slice_audio(features: torch.Tensor, chunk_index: int, plan: MotionframePlan) -> torch.Tensor:
    start = chunk_index * plan.stride * AUDIO_PER_LATENT
    window = features[:, start : start + MOTIONFRAME_AUDIO_LEN]
    assert window.shape[1] > 0, f"chunk {chunk_index} starts past the end of the driving audio"
    if window.shape[1] < MOTIONFRAME_AUDIO_LEN:
        held = window[:, -1:].expand(-1, MOTIONFRAME_AUDIO_LEN - window.shape[1], -1)
        window = torch.cat([window, held], dim=1)
    return window


def _dit_input(
    chunk: torch.Tensor,
    motionframes: torch.Tensor | None,
    ref_row: torch.Tensor,
) -> list[torch.Tensor]:
    body = (
        chunk
        if motionframes is None
        else torch.cat([motionframes, chunk[:, MOTIONFRAME_LATENTS:]], dim=1)
    )
    latents = torch.cat([body, ref_row], dim=1)

    ref_mask = torch.ones_like(latents[:1])
    ref_mask[:, : -contract.NUM_REF] = 0.0
    motionframe_mask = torch.zeros_like(latents[:1])
    if motionframes is not None:
        motionframe_mask[:, :MOTIONFRAME_LATENTS] = 1.0
    return [torch.cat([latents, ref_mask, motionframe_mask], dim=0)]


def _guided_prediction(
    dit,
    model_input: list[torch.Tensor],
    timestep: torch.Tensor,
    text: TextConditioning,
    audio: torch.Tensor,
    videoref: list[torch.Tensor],
    videoref_audio: torch.Tensor | None,
    valid_masks: torch.Tensor,
    config: SamplingConfig,
) -> torch.Tensor:
    def forward(txt: list[torch.Tensor], aud: torch.Tensor | None, ref_audio: torch.Tensor | None) -> torch.Tensor:
        return dit(
            model_input,
            t=timestep,
            txt=txt,
            null_txt=text.null,
            aud=aud,
            context_videoref=videoref,
            context_videoref_audio=ref_audio,
            valid_masks=valid_masks,
        )[0]

    cond = forward(text.prompt, audio, videoref_audio)
    prediction = cond.clone()
    if config.aud_scale != 1:
        aud_uncond = forward(text.prompt, None, None)
        prediction = prediction + (config.aud_scale - 1) * (cond - aud_uncond)
    if config.txt_scale != 1:
        txt_uncond = forward(text.negative, audio, videoref_audio)
        prediction = prediction + (config.txt_scale - 1) * (cond - txt_uncond)
    return prediction


@torch.no_grad()
def sample(
    dit,
    reference: contract.Reference,
    target: contract.Target,
    driving: contract.Driving,
    text: TextConditioning,
    config: SamplingConfig,
    device: torch.device,
) -> torch.Tensor:
    height, width = target.latents.shape[-2:]
    latent_len = (driving.num_frames - 1) // contract.VAE_STRIDE[0] + 1
    plan = plan_motionframe_chunks(latent_len)

    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    canvas, chunks = draw_noise(plan, height, width, device, generator)

    logger.info("%d motionframe chunks of %d latents, stride %d", plan.count, plan.length, plan.stride)

    assert reference.latents.shape[1] == NUM_VIDEOREF_LATENT, f"expected one window, got {reference.latents.shape}"
    videoref = [
        torch.cat(
            [reference.latents, torch.ones_like(reference.latents[:1]), torch.zeros_like(reference.latents[:1])],
            dim=0,
        )
    ]
    audio_len = NUM_VIDEOREF_LATENT * AUDIO_PER_LATENT
    videoref_audio = reference.audio if reference.audio is not None and reference.audio.shape[1] == audio_len else None

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(plan.count):
            has_motionframes = index > 0
            masks = build_valid_masks(reference.face_masks, target.foreground, has_motionframes)
            audio = slice_audio(driving.features, index, plan)
            ref_row = chunks[0][:, :1] if has_motionframes else target.latents
            motionframes = chunks[index - 1][:, -MOTIONFRAME_LATENTS:] if has_motionframes else None

            scheduler = build_scheduler(config.steps, config.shift, device)
            chunk = chunks[index]
            for step_index, timestep in enumerate(scheduler.timesteps):
                step_videoref, step_videoref_audio, step_masks = _videoref_at_step(
                    videoref,
                    videoref_audio,
                    masks,
                    step_index,
                    len(scheduler.timesteps),
                    config.start_use_videoref_step,
                )
                prediction = _guided_prediction(
                    dit,
                    _dit_input(chunk, motionframes, ref_row),
                    torch.stack([timestep]),
                    text,
                    audio,
                    step_videoref,
                    step_videoref_audio,
                    step_masks,
                    config,
                )
                if has_motionframes:
                    prediction = prediction[:, MOTIONFRAME_LATENTS:]
                noisy = chunk[:, MOTIONFRAME_LATENTS:] if has_motionframes else chunk
                stepped = scheduler.step(prediction.unsqueeze(0), timestep, noisy.unsqueeze(0), return_dict=False)[0]
                scheduler._step_index = None
                chunk = (
                    torch.cat([chunk[:, :MOTIONFRAME_LATENTS], stepped[0]], dim=1)
                    if has_motionframes
                    else stepped[0]
                )
                chunk = chunk.bfloat16()
            chunks[index] = chunk

    for index in range(plan.count):
        offset, start, stop = plan.write_span(index)
        if stop > start:
            canvas[:, start:stop] = chunks[index][:, offset : offset + stop - start].to(canvas.device)
    return canvas
