import os

import numpy as np
import torch
import torch.nn.functional as F

from tavr import contract
from tavr.contract import (
    NUM_VIDEOREF,
    OUTPUT_FPS,
    VAE_STRIDE,
    VIDEOREF_FPS,
    Reference,
    Target,
)
from tavr.preprocess import audio, masks, media

VAE_PREFIX_FRAMES = 25
SMART_CROP_INTERVAL = 5
SMART_CROP_MAX_SAMPLES = 100
SMART_CROP_MAX_FRAMES = 500
SMART_CROP_BATCH = 16
VIDEOREF_POSE_BATCH = 32
SMART_CROP_MARGIN = 1.0
SMART_CROP_MIN_RATIO = 0.3
SMART_CROP_MAX_AREA = 0.95
FOREHEAD_FACTOR = 1.2
MIN_CROP_FACE_POINTS = 10
MIN_LOAD_FRAMES = 100
LOOKAHEAD_FRAMES = 200

assert VIDEOREF_FPS == OUTPUT_FPS
NUM_VIDEOREF_LATENT = NUM_VIDEOREF // VAE_STRIDE[0]


def videoref_window(total_latents: int, num_chunks: int) -> tuple[int, int]:
    if num_chunks <= 1 or total_latents <= NUM_VIDEOREF_LATENT:
        return 0, min(total_latents, NUM_VIDEOREF_LATENT)
    stride = (total_latents - NUM_VIDEOREF_LATENT) // num_chunks
    if stride > NUM_VIDEOREF_LATENT:
        start = stride // 2 - NUM_VIDEOREF_LATENT // 2
        return start, start + NUM_VIDEOREF_LATENT
    return 0, NUM_VIDEOREF_LATENT


def _forehead(points: torch.Tensor) -> torch.Tensor:
    probe = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 33]
    if (points[probe, 2] < masks.FACE_CONF).any():
        return points[:, :2]
    lifted = points[:, :2].clone()
    lifted[17:27] += (lifted[27] - lifted[33]) * FOREHEAD_FACTOR
    return lifted


def _face_union(keypoints: torch.Tensor, size: tuple[int, int]) -> tuple[float, float, float, float] | None:
    height, width = size
    corners = []
    for frame_points in keypoints:
        face = frame_points[masks.FACE68]
        visible = face[:, 2] > masks.FACE_CONF
        points = _forehead(face)[visible]
        if len(points) < MIN_CROP_FACE_POINTS:
            continue
        low_x, low_y = points[:, 0].min().item(), points[:, 1].min().item()
        high_x, high_y = points[:, 0].max().item(), points[:, 1].max().item()
        margin_x, margin_y = (high_x - low_x) * SMART_CROP_MARGIN, (high_y - low_y) * SMART_CROP_MARGIN
        corners.append(
            (
                max(0.0, low_x - margin_x),
                max(0.0, low_y - margin_y),
                min(width, high_x + margin_x),
                min(height, high_y + margin_y),
            )
        )
    if len(corners) < len(keypoints) * 0.5:
        return None
    box = np.asarray(corners)
    return box[:, 0].min(), box[:, 1].min(), box[:, 2].max(), box[:, 3].max()


def _crop_box(face, video_size: tuple[int, int], bucket: tuple[int, int]) -> tuple[int, int, int, int] | None:
    height, width = video_size
    ratio = bucket[1] / bucket[0]
    crop_w = max(face[2] - face[0], (face[3] - face[1]) * ratio)
    crop_h = crop_w / ratio
    shortfall = SMART_CROP_MIN_RATIO / min(crop_w / width, crop_h / height)
    if shortfall > 1:
        crop_w, crop_h = crop_w * shortfall, crop_h * shortfall
    if crop_w > width:
        crop_w, crop_h = width, width / ratio
    if crop_h > height:
        crop_w, crop_h = height * ratio, height
    if ratio > 1:
        crop_h = int(crop_h) // 2 * 2
        crop_w = int(crop_h * ratio) // 2 * 2
    else:
        crop_w = int(crop_w) // 2 * 2
        crop_h = int(crop_w / ratio) // 2 * 2
    if crop_w * crop_h > SMART_CROP_MAX_AREA * width * height:
        return None
    left = int(max(0, min((face[0] + face[2]) / 2 - crop_w / 2, width - crop_w))) // 2 * 2
    top = int(max(0, min((face[1] + face[3]) / 2 - crop_h / 2, height - crop_h))) // 2 * 2
    return left, top, crop_w, crop_h


def smart_crop(video: str, bucket: tuple[int, int], pose: masks.DWPose, target: str) -> str:
    meta = media.probe(video)
    sampled = [
        frame
        for index, frame in enumerate(media.frames_rgb(video, SMART_CROP_MAX_FRAMES))
        if index % SMART_CROP_INTERVAL == 0
    ][:SMART_CROP_MAX_SAMPLES]
    frames = torch.from_numpy(np.stack(sampled)).permute(0, 3, 1, 2).float().div_(255.0)
    size = (meta["height"], meta["width"])
    face = _face_union(pose.keypoints(frames, batch_size=SMART_CROP_BATCH), size)
    assert face is not None, "fewer than half the sampled frames show a usable face"
    box = _crop_box(face, size, bucket)
    if box is None:
        return video
    return media.crop_video(video, target, box, SMART_CROP_MAX_FRAMES)


def encode_videoref(frames: torch.Tensor, vae) -> torch.Tensor:
    keep = frames.shape[0] // VAE_STRIDE[0]
    assert frames.shape[-2] % 16 == 0 and frames.shape[-1] % 16 == 0, frames.shape
    prefixed = torch.cat([frames[:1].expand(VAE_PREFIX_FRAMES, -1, -1, -1), frames], dim=0)
    latents = vae.encode([prefixed.permute(1, 0, 2, 3)])[0]
    assert latents.shape[1] == (VAE_PREFIX_FRAMES + frames.shape[0] - 1) // VAE_STRIDE[0] + 1, latents.shape
    return latents[:, -keep:]


def build_reference(
    ref_video: str,
    still: np.ndarray,
    num_frames: int,
    vae,
    pose: masks.DWPose,
    wav2vec: audio.Wav2Vec,
    work_dir: str,
    null_audio: torch.Tensor | None = None,
) -> Reference:
    os.makedirs(work_dir, exist_ok=True)
    bucket = media.bucket_for(still)
    clip = media.force_25fps(ref_video, os.path.join(work_dir, "ref_25fps.mp4"))
    clip = smart_crop(clip, bucket, pose, os.path.join(work_dir, "ref_cropped.mp4"))

    meta = media.probe(clip)
    count = min(max(num_frames + LOOKAHEAD_FRAMES, MIN_LOAD_FRAMES), int(meta["duration"] * OUTPUT_FPS))
    frames = media.load_frames(clip, count, bucket)

    track = None
    if meta["has_audio"]:
        mp3 = media.extract_audio(clip, os.path.join(work_dir, "ref_audio.mp3"), count / OUTPUT_FPS + 3)
        track = audio.videoref_features(mp3, count, wav2vec, null_audio)

    unsigned = frames.add(1).div(2)
    keypoints = pose.keypoints(unsigned, batch_size=VIDEOREF_POSE_BATCH)
    frames = unsigned.mul(2).sub(1).clamp(-1, 1)

    face_pixels = masks.face_masks(keypoints, *bucket)
    latents = encode_videoref(frames, vae)
    start, end = videoref_window(latents.shape[1], contract.num_motionframe_chunks(contract.latent_len(num_frames)))
    first_frame = start * VAE_STRIDE[0]
    window = face_pixels[first_frame : first_frame + NUM_VIDEOREF]
    assert len(window) == NUM_VIDEOREF and end - start == NUM_VIDEOREF_LATENT, (len(window), start, end)
    return Reference(
        latents=latents[:, start:end],
        audio=None if track is None else track[:, first_frame : first_frame + NUM_VIDEOREF],
        face_masks=masks.to_patch_grid(window),
    )


def build_target(still: np.ndarray, vae, boxes: masks.PersonBoxes) -> Target:
    bucket = media.bucket_for(still)
    pixels = torch.from_numpy(media.fit_still(still, bucket).copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    foreground = boxes.foreground(pixels)
    background = 1.0 - F.interpolate(foreground[None], scale_factor=VAE_STRIDE[1], mode="nearest").bool().float()[0]
    latents = vae.encode([(pixels * background).unsqueeze(1)])[0]
    return Target(latents=latents, foreground=foreground)
