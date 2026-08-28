import librosa
import numpy as np
import pyloudnorm
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from tavr.contract import AUDIO_SR, Driving

DRIVING_LAYERS = (0, 8, 16, 24)
TARGET_LUFS = -16.0
TAIL_SILENCE_S = 0.2
VIDEOREF_HEAD_PAD = 3
MAX_DRIVING_S = 30
LOUDNESS_CHUNK_S = 1.0
LOUDNESS_OVERLAP = 0.1
MIN_LOUDNESS_CHUNK_S = 0.1


def _pair_to_25hz(feature: torch.Tensor) -> torch.Tensor:
    even = feature[: len(feature) // 2 * 2]
    return even.reshape(len(even) // 2, -1)


def _normalise_loudness(wave: np.ndarray) -> np.ndarray:
    step = int(LOUDNESS_CHUNK_S * AUDIO_SR)
    overlap = int(step * LOUDNESS_OVERLAP)
    meter = pyloudnorm.Meter(AUDIO_SR)

    pieces: list[np.ndarray] = []
    start = 0
    while start < len(wave):
        chunk = wave[start : start + step]
        if len(chunk) < AUDIO_SR * MIN_LOUDNESS_CHUNK_S:
            if pieces:
                pieces[-1] = np.concatenate([pieces[-1], chunk])
            break
        loudness = meter.integrated_loudness(chunk)
        quiet = not np.isfinite(loudness) or loudness <= -70
        pieces.append(chunk if quiet else pyloudnorm.normalize.loudness(chunk, loudness, TARGET_LUFS))
        start += step - overlap

    if not pieces:
        return wave
    if len(pieces) == 1:
        return pieces[0]

    output = np.zeros(sum(len(piece) for piece in pieces) - overlap * (len(pieces) - 1))
    output[: len(pieces[0])] = pieces[0]
    cursor = len(pieces[0]) - overlap
    for piece in pieces[1:]:
        if overlap > 0 and cursor > 0 and cursor + overlap <= len(output) and overlap <= len(piece):
            output[cursor : cursor + overlap] *= np.linspace(1, 0, overlap)
            output[cursor : cursor + overlap] += piece[:overlap] * np.linspace(0, 1, overlap)
            tail = min(len(piece) - overlap, len(output) - cursor - overlap)
            if tail > 0:
                output[cursor + overlap : cursor + overlap + tail] = piece[overlap : overlap + tail]
        else:
            end = min(cursor + len(piece), len(output))
            output[cursor:end] = piece[: end - cursor]
        cursor += len(piece) - overlap
    return output[: len(wave)] if len(output) > len(wave) else np.pad(output, (0, len(wave) - len(output)))


def _load(path: str) -> np.ndarray:
    wave, _ = librosa.load(path, sr=AUDIO_SR)
    assert len(wave) <= MAX_DRIVING_S * AUDIO_SR, f"{path} is longer than the {MAX_DRIVING_S}s this release supports"
    return wave


class Wav2Vec:
    def __init__(self, path: str, device: str = "cpu"):
        self.processor = Wav2Vec2Processor.from_pretrained(path, local_files_only=True, do_phonemize=False)
        self.model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=True).to(device).eval()
        self.device = device

    def encode(self, wave: np.ndarray, layers: tuple[int, ...] | None = None) -> torch.Tensor:
        values = self.processor(wave, sampling_rate=AUDIO_SR, return_tensors="pt").input_values
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.wav2vec2(values.to(self.device), output_hidden_states=layers is not None)
            if layers:
                stacked = torch.cat([output.hidden_states[i].to(torch.bfloat16) for i in layers], -1)
            else:
                stacked = output.last_hidden_state
        return _pair_to_25hz(stacked[0].cpu())


def build_driving(path: str, wav2vec: Wav2Vec) -> Driving:
    wave = _load(path)
    speech = _normalise_loudness(np.concatenate([wave, np.zeros(int(TAIL_SILENCE_S * AUDIO_SR))]))
    features = torch.cat(
        [wav2vec.encode(speech, DRIVING_LAYERS), wav2vec.encode(np.zeros_like(wave), DRIVING_LAYERS)], dim=0
    )
    num_frames = (len(features) // 2 + 3) // 4 * 4 + 1
    return Driving(features=features[None].to(torch.bfloat16), num_frames=num_frames)


def videoref_features(
    audio_path: str, num_frames: int, wav2vec: Wav2Vec, null_feature: torch.Tensor | None = None
) -> torch.Tensor:
    features = wav2vec.encode(_load(audio_path))[:num_frames]
    if len(features) < num_frames:
        assert null_feature is not None, "reference audio is short and no null embedding was supplied"
        pad = null_feature.reshape(1, -1).to(features.dtype).repeat(num_frames - len(features), 1)
        features = torch.cat([features, pad], dim=0)
    return torch.cat([features[:1].repeat(VIDEOREF_HEAD_PAD, 1), features], dim=0)[None]
