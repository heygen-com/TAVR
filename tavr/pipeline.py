from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from tavr import contract, sampling
from tavr.models.dit import load_dit
from tavr.models.text import TextEncoder
from tavr.models.vae import WanVAE
from tavr.preprocess import audio, masks, media, reference

logger = logging.getLogger(__name__)

VAE_WEIGHTS = "Wan2.1_VAE.pth"
T5_WEIGHTS = "models_t5_umt5-xxl-enc-bf16.pth"
T5_TOKENIZER = "google/umt5-xxl"
WAV2VEC_DIR = "wav2vec2-xlsr-53-espeak-cv-ft"
YOLO_WEIGHTS = "yolo11x.pt"
DWPOSE_WEIGHTS = "dw-ll_ucoco_384.onnx"
PRETRAINED_SUBDIR = "pretrained/Wan2.1-T2V-14B"

GESTURE_SENTENCE = "The person is talking with gestures."


@dataclass(frozen=True)
class Sample:
    name: str
    reference_video: Path
    target_image: Path
    audio: Path
    prompt: str

    @classmethod
    def from_dir(cls, sample_dir: Path) -> Sample:
        audio_path = sample_dir / "target.mp3"
        if not audio_path.exists():
            audio_path = sample_dir / "target.wav"
        caption = sample_dir / "target_caption.json"
        for label, path in (
            ("reference video", sample_dir / "ref.mp4"),
            ("target image", sample_dir / "target.png"),
            ("caption", caption),
            ("driving audio", audio_path),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        return cls(
            name=sample_dir.name,
            reference_video=sample_dir / "ref.mp4",
            target_image=sample_dir / "target.png",
            audio=audio_path,
            prompt=json.loads(caption.read_text(encoding="utf-8"))["caption"],
        )


class Tavr:
    def __init__(self, dit_ckpt: Path, ckpt_dir: Path, device: str = "cuda:0") -> None:
        pretrained = (ckpt_dir / PRETRAINED_SUBDIR).resolve()
        self.device = torch.device(device)

        self.vae_encoder = WanVAE(str(pretrained / VAE_WEIGHTS), torch.float32, device)
        self.vae_decoder = WanVAE(str(pretrained / VAE_WEIGHTS), torch.float16, device)
        self.text_encoder = TextEncoder(str(pretrained / T5_WEIGHTS), str(pretrained / T5_TOKENIZER), device)
        self.wav2vec = audio.Wav2Vec(str(pretrained / WAV2VEC_DIR), device)
        self.boxes = masks.PersonBoxes(str(pretrained / YOLO_WEIGHTS), device)
        self.pose = masks.DWPose(str(pretrained / DWPOSE_WEIGHTS), self.boxes, device)
        self.dit = load_dit(
            str(dit_ckpt),
            device,
            audio_dim=contract.AUDIO_DIM_DRIVING,
            rope_spatial_stride=contract.ROPE_SPATIAL_STRIDE,
        )

    def run(self, sample: Sample, output_path: Path, config: sampling.SamplingConfig) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        still = media.read_still(str(sample.target_image))
        driving = audio.build_driving(str(sample.audio), self.wav2vec)
        target = reference.build_target(still, self.vae_encoder, self.boxes)
        videoref = reference.build_reference(
            str(sample.reference_video),
            still,
            driving.num_frames,
            self.vae_encoder,
            self.pose,
            self.wav2vec,
            str(output_path.parent / "work"),
            null_audio=self.dit.null_videoref_audio_feature.weight.detach().float().cpu(),
        )
        logger.info("%s: %d frames, %d reference latents", sample.name, driving.num_frames, videoref.latents.shape[1])

        latents = sampling.sample(
            self.dit,
            _on_device(videoref, self.device),
            _on_device(target, self.device),
            _on_device(driving, self.device),
            self.encode_text(sample.prompt),
            config,
            self.device,
        )
        media.write_video(self.vae_decoder.decode_stream(latents), str(output_path), str(sample.audio))
        logger.info("%s: wrote %s", sample.name, output_path)

    @torch.no_grad()
    def encode_text(self, prompt: str) -> sampling.TextConditioning:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return sampling.TextConditioning(
                prompt=self.text_encoder([prompt.replace(GESTURE_SENTENCE, "")], self.device),
                negative=self.text_encoder([contract.NEGATIVE_PROMPT], self.device),
                null=self.text_encoder([""], self.device),
            )


def _on_device(bundle, device: torch.device):
    moved = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in vars(bundle).items()
    }
    return type(bundle)(**moved)


