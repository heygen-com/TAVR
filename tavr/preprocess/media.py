import json
import math
import os
import subprocess
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal

import cv2
import imageio.v2 as iio
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tavr.contract import BUCKETS, OUTPUT_FPS

FFMPEG = os.environ.get("TAVR_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("TAVR_FFPROBE", "ffprobe")
_DECODE_BATCH = 64


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _frame_rate(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = (stream.get(key) or "0/0").partition("/")
        if num != "0" and den != "0":
            return float(Decimal(num) / Decimal(den))
    raise ValueError(f"no usable frame rate on the video stream of {stream.get('index')}")


def probe(path: str) -> dict:
    report = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    streams = json.loads(report)["streams"]
    video = max(
        (s for s in streams if s["codec_type"] == "video"),
        key=lambda s: s.get("disposition", {}).get("default", 0),
    )
    return {
        "fps": _frame_rate(video),
        "duration": float(video["duration"]),
        "height": int(video["height"]),
        "width": int(video["width"]),
        "has_audio": sum(s["codec_type"] == "audio" for s in streams) == 1,
    }


def read_still(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def bucket_for(still: np.ndarray) -> tuple[int, int]:
    return BUCKETS["landscape"] if still.shape[0] <= still.shape[1] else BUCKETS["portrait"]


def frames_rgb(path: str, count: int | None = None):
    meta = probe(path)
    fps, height, width = meta["fps"], meta["height"], meta["width"]
    end = meta["duration"] if count is None else min(meta["duration"], count / fps)
    span = (Decimal(str(end)) - Decimal("0.0")) * Decimal(str(fps))
    total = int(span.quantize(Decimal("0"), rounding=ROUND_HALF_UP))
    stride = height * width * 2
    command = [
        FFMPEG,
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        "0.0",
        "-to",
        str(end),
        "-i",
        path,
        "-filter_complex",
        f"[0]fps=fps={fps}:round=near[s0]",
        "-map",
        "[s0]",
        "-pix_fmt",
        "uyvy422",
        "-f",
        "rawvideo",
        "pipe:",
    ]
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        for _ in range(total):
            raw = pipe.stdout.read(stride)
            if len(raw) < stride:
                return
            yield cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(height, width, 2), cv2.COLOR_YUV2RGB_UYVY)
    finally:
        pipe.kill()
        pipe.stdout.close()
        pipe.wait()


def force_25fps(source: str, target: str) -> str:
    if probe(source)["fps"] == float(OUTPUT_FPS):
        return source
    os.makedirs(os.path.dirname(target), exist_ok=True)
    command = [
        FFMPEG,
        "-i",
        source,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-vf",
        f"fps={OUTPUT_FPS}",
        "-c:v",
        "libx264",
        "-crf",
        "17",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        target,
        "-y",
    ]
    _run(command)
    return target


def crop_video(source: str, target: str, box: tuple[int, int, int, int], max_frames: int) -> str:
    left, top, width, height = box
    command = [
        FFMPEG,
        "-y",
        "-t",
        str(max_frames / OUTPUT_FPS),
        "-i",
        source,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        f"crop={width}:{height}:{left}:{top}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "17",
        "-c:a",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        target,
    ]
    _run(command)
    return target


def extract_audio(source: str, target: str, seconds: float) -> str:
    _run([FFMPEG, "-i", source, "-ss", "0", "-t", str(seconds), "-vn", "-acodec", "mp3", "-y", target])
    return target


def _cover_size(size: tuple[int, int], bucket: tuple[int, int]) -> tuple[int, int]:
    height, width = size
    target_h, target_w = bucket
    if height / width < target_h / target_w:
        return target_h, round(width * target_h / height)
    return round(height * target_w / width), target_w


def fit_still(still: np.ndarray, bucket: tuple[int, int]) -> np.ndarray:
    for factor in (2, 1):
        height, width = bucket[0] * factor, bucket[1] * factor
        resized_h, resized_w = _cover_size(still.shape[:2], (height, width))
        scaled = Image.fromarray(still).resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        top, left = (resized_h - height) // 2, (resized_w - width) // 2
        still = np.asarray(scaled)[top : top + height, left : left + width]
    return still


def fit_frames(frames, bucket: tuple[int, int], dtype=torch.bfloat16) -> torch.Tensor:
    batch = torch.from_numpy(np.stack(frames).copy()).permute(0, 3, 1, 2).float().div_(255.0)
    height, width = bucket
    resized_h, resized_w = _cover_size(batch.shape[-2:], bucket)
    batch = F.interpolate(batch, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    top, left = (resized_h - height) // 2, (resized_w - width) // 2
    batch = batch[:, :, top : top + height, left : left + width]
    return batch.to(dtype).sub_(0.5).div_(0.5)


def load_frames(path: str, count: int, bucket: tuple[int, int], dtype=torch.bfloat16) -> torch.Tensor:
    pending: list[np.ndarray] = []
    fitted: list[torch.Tensor] = []
    for frame in frames_rgb(path, count):
        pending.append(frame)
        if len(pending) == _DECODE_BATCH:
            fitted.append(fit_frames(pending, bucket, dtype))
            pending = []
    if pending:
        fitted.append(fit_frames(pending, bucket, dtype))
    return torch.cat(fitted)[:count]


def _to_uint8(chunk: torch.Tensor) -> np.ndarray:
    levels = chunk.clamp(-1, 1).add(1).mul(0.5).mul(255).clamp(0, 255).to(torch.uint8)
    return levels.permute(1, 2, 3, 0).cpu().numpy()


def _frame_ceiling(audio_path: str) -> int:
    frame = 1.0 / OUTPUT_FPS
    padded = math.ceil(librosa.get_duration(path=audio_path) / frame) * frame
    return int(padded * OUTPUT_FPS)


def write_video(chunks: Iterable[torch.Tensor], path: str, audio_path: str | None = None) -> str:
    silent = path if audio_path is None else f"{os.path.splitext(path)[0]}_silent.mp4"
    ceiling = None if audio_path is None else _frame_ceiling(audio_path)
    writer = iio.get_writer(
        silent,
        fps=OUTPUT_FPS,
        codec="h264",
        output_params=["-crf", "18"],
        macro_block_size=2,
    )
    written = 0
    try:
        for chunk in chunks:
            frames = _to_uint8(chunk)
            if ceiling is not None:
                frames = frames[: max(ceiling - written, 0)]
            for frame in frames:
                writer.append_data(frame)
            written += len(frames)
    finally:
        writer.close()
    if audio_path is None:
        return path
    _run(
        [
            FFMPEG,
            "-i",
            silent,
            "-i",
            audio_path,
            "-map",
            "0",
            "-map",
            "1",
            "-acodec",
            "aac",
            "-strict",
            "experimental",
            "-vcodec",
            "copy",
            path,
            "-y",
        ]
    )
    os.remove(silent)
    return path
