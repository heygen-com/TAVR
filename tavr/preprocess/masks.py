from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from tavr.contract import PATCH_SIZE, VAE_STRIDE

FACE68 = slice(23, 91)
FACE_CONF = 0.3
FACE_EXPAND_LATENT_PX = 3
MIN_FACE_POINTS = 2
PERSON_EXPAND = (0.1, 0.2)
EDGE_SNAP = 0.05
DWPOSE_SHAPE = (384, 288)
DWPOSE_MEAN = (123.675, 116.28, 103.53)
DWPOSE_STD = (58.395, 57.12, 57.375)
DWPOSE_BOX_PAD = 1.25
SIMCC_SPLIT = 2.0


class PersonBoxes:
    def __init__(self, weights: str, device: str = "cpu", expand: tuple[float, float] = PERSON_EXPAND):
        if not Path(weights).is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        self.model = YOLO(weights, verbose=False).to(device)
        self.device = device
        self.expand_h, self.expand_w = expand

    def _letterbox(self, frames: torch.Tensor):
        height, width = frames.shape[-2:]
        target_h, target_w = (640, 480) if height > width else (480, 640)
        scale = min(target_h / height, target_w / width)
        inner_h, inner_w = int(height * scale), int(width * scale)
        pad_top, pad_left = (target_h - inner_h) // 2, (target_w - inner_w) // 2
        resized = F.interpolate(frames.add(1).div(2), size=(inner_h, inner_w), mode="bilinear", align_corners=False)
        padded = F.pad(resized, (pad_left, target_w - inner_w - pad_left, pad_top, target_h - inner_h - pad_top))
        return padded, (height / inner_h, width / inner_w, pad_top, pad_left)

    def _pick(self, result, centre):
        if result.boxes is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        if centre is None:
            scores = areas
        else:
            middle = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
            scores = areas / (1 + 0.01 * np.linalg.norm(middle - np.asarray(centre), axis=1))
        return boxes[int(scores.argmax())]

    def _anchor(self, letterboxed: torch.Tensor, pad_top: int, pad_left: int):
        raw = self._pick(self.model(letterboxed[None], classes=[0], verbose=False, device=self.device)[0], None)
        if raw is None or (raw[2] - raw[0]) * (raw[3] - raw[1]) <= 0:
            return None
        return ((raw[0] + raw[2]) / 2 - pad_left, (raw[1] + raw[3]) / 2 - pad_top)

    def detect(self, frames: torch.Tensor, batch_size: int = 16) -> list[tuple[int, int, int, int] | None]:
        padded, (scale_y, scale_x, pad_top, pad_left) = self._letterbox(frames)
        height, width = frames.shape[-2:]
        centre = self._anchor(padded[0], pad_top, pad_left)
        if centre is None:
            return [(0, 0, width, height)] * len(frames)

        boxes: list[tuple[int, int, int, int] | None] = []
        for start in range(0, len(padded), batch_size):
            for result in self.model(
                padded[start : start + batch_size], classes=[0], verbose=False, device=self.device
            ):
                raw = self._pick(result, centre if not boxes else None)
                if raw is None:
                    boxes.append(None)
                    continue
                left, top, right, bottom = (int(edge) for edge in raw)
                grow_w, grow_h = (right - left) * self.expand_w, (bottom - top) * self.expand_h
                left, right = int(left - grow_w) - pad_left, int(right + grow_w) - pad_left
                top, bottom = int(top - grow_h) - pad_top, int(bottom + grow_h) - pad_top
                boxes.append(
                    (
                        max(0, int(left * scale_x)),
                        max(0, int(top * scale_y)),
                        min(width, int(right * scale_x)),
                        min(height, int(bottom * scale_y)),
                    )
                )
        return boxes

    def foreground(self, still: torch.Tensor) -> torch.Tensor:
        height, width = still.shape[-2:]
        box = self.detect(still[None])[0]
        assert box is not None, "no person detected on the target still"
        left, top, right, bottom = box
        left, top = max(0, left), max(0, top)
        right, bottom = min(width - 1, right), min(height - 1, bottom)
        if left < int(width * EDGE_SNAP):
            left = 0
        if top < int(height * EDGE_SNAP):
            top = 0
        if right > width - int(width * EDGE_SNAP):
            right = width
        if bottom > height - int(height * EDGE_SNAP):
            bottom = height
        pixels = torch.zeros((1, height, width), dtype=torch.float32)
        pixels[0, top:bottom, left:right] = 1.0
        return F.max_pool2d(pixels, VAE_STRIDE[1], VAE_STRIDE[1])


class DWPose:
    """COCO-WholeBody 133-point pose, RTMPose-l with SimCC heads (DWPose release)."""

    def __init__(self, weights: str, boxes: PersonBoxes, device: str = "cpu"):
        if not Path(weights).is_file():
            raise FileNotFoundError(f"DWPose weights not found: {weights}")
        if device.startswith("cuda"):
            ordinal = int(device.split(":")[1]) if ":" in device else 0
            providers = [("CUDAExecutionProvider", {"device_id": ordinal}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(weights, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.boxes = boxes
        self.device = device

    def _crop(self, frame: torch.Tensor, box):
        left, top, right, bottom = box
        centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
        span_w, span_h = (right - left) * DWPOSE_BOX_PAD, (bottom - top) * DWPOSE_BOX_PAD
        target_ratio = DWPOSE_SHAPE[1] / DWPOSE_SHAPE[0]
        if span_w > span_h * target_ratio:
            span_h = span_w / target_ratio
        else:
            span_w = span_h * target_ratio
        left, top = int(round(centre_x - span_w / 2)), int(round(centre_y - span_h / 2))
        right, bottom = left + int(round(span_w)), top + int(round(span_h))
        height, width = frame.shape[-2:]
        window = frame[:, max(0, top) : min(height, bottom), max(0, left) : min(width, right)]
        window = F.pad(window, (max(0, -left), max(0, right - width), max(0, -top), max(0, bottom - height)))
        return window, (left, top, right - left, bottom - top)

    @staticmethod
    def _decode(simcc_x: np.ndarray, simcc_y: np.ndarray, window) -> torch.Tensor:
        left, top, span_w, span_h = window
        scores = np.minimum(simcc_x.max(axis=1), simcc_y.max(axis=1))
        points = np.stack([simcc_x.argmax(axis=1), simcc_y.argmax(axis=1)], axis=1) / SIMCC_SPLIT
        points[scores <= 0] = -1
        points[:, 0] = points[:, 0] / DWPOSE_SHAPE[1] * span_w + left
        points[:, 1] = points[:, 1] / DWPOSE_SHAPE[0] * span_h + top
        return torch.from_numpy(np.concatenate([points, scores[:, None]], axis=1).astype(np.float32))

    def keypoints(self, frames: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        mean = torch.tensor(DWPOSE_MEAN).view(1, -1, 1, 1)
        std = torch.tensor(DWPOSE_STD).view(1, -1, 1, 1)

        found = {}
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            boxes = self.boxes.detect(chunk.mul(2).sub(1), batch_size=len(chunk))
            scaled = chunk.float().mul(255)

            crops, windows, seen = [], [], []
            for index, box in enumerate(boxes):
                if box is None:
                    continue
                window, geometry = self._crop(scaled[index], box)
                crops.append(F.interpolate(window[None], size=DWPOSE_SHAPE, mode="bilinear", align_corners=False)[0])
                windows.append(geometry)
                seen.append(start + index)
            if not crops:
                continue

            batch = ((torch.stack(crops) - mean) / std).numpy()
            simcc_x, simcc_y = self.session.run(None, {self.input_name: batch})
            for offset in range(len(crops)):
                found[seen[offset]] = self._decode(simcc_x[offset], simcc_y[offset], windows[offset])

        blank = torch.zeros((133, 3))
        return torch.stack([found.get(i, blank) for i in range(len(frames))])


def face_masks(keypoints: torch.Tensor, height: int, width: int) -> torch.Tensor:
    pad = FACE_EXPAND_LATENT_PX * VAE_STRIDE[1]
    masks = torch.zeros((len(keypoints), height, width), dtype=torch.bool)
    for index, frame_points in enumerate(keypoints):
        face = frame_points[FACE68]
        visible = face[face[:, 2] > FACE_CONF, :2]
        if len(visible) < MIN_FACE_POINTS:
            continue
        low_x, low_y = visible[:, 0].min().item(), visible[:, 1].min().item()
        high_x, high_y = visible[:, 0].max().item(), visible[:, 1].max().item()
        top = round(low_y - 0.5 * (high_y - low_y)) - pad
        masks[
            index,
            int(np.clip(top, 0, height)) : int(np.clip(round(high_y) + pad, 0, height)),
            int(np.clip(round(low_x) - pad, 0, width)) : int(np.clip(round(high_x) + pad, 0, width)),
        ] = True
    return masks


def to_patch_grid(pixel_masks: torch.Tensor) -> torch.Tensor:
    pooled = F.max_pool3d(pixel_masks[None, None].float(), VAE_STRIDE, VAE_STRIDE)
    pooled = F.max_pool3d(pooled, PATCH_SIZE, PATCH_SIZE)
    return pooled[0, 0].bool()
