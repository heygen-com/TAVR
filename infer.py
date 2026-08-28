#!/usr/bin/env python3


from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tavr import contract
from tavr.pipeline import Sample, Tavr
from tavr.sampling import SamplingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-dir", required=True, type=Path, help="Directory containing one inference sample")
    parser.add_argument("--dit-ckpt", required=True, type=Path, help="TAVR transformer .safetensors")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Results land in <output-dir>/<sample name>/generated_target.mp4"
    )
    parser.add_argument(
        "--ckpt-dir", default=Path("."), type=Path, help="Root holding pretrained/Wan2.1-T2V-14B and the detectors"
    )
    parser.add_argument("--sample-steps", type=int, default=contract.SAMPLE_STEPS)
    parser.add_argument("--sample-shift", type=float, default=contract.FLOW_SHIFT)
    parser.add_argument("--txt-guide-scale", type=float, default=contract.TXT_GUIDE_SCALE)
    parser.add_argument("--aud-guide-scale", type=float, default=contract.AUD_GUIDE_SCALE)
    parser.add_argument(
        "--start-use-videoref-step",
        type=int,
        default=0,
        help="Denoising step that enables video-reference conditioning; negative disables it",
    )
    parser.add_argument("--seed", type=int, default=contract.BASE_SEED)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    sample = Sample.from_dir(args.sample_dir.resolve())
    config = SamplingConfig(
        steps=args.sample_steps,
        shift=args.sample_shift,
        txt_scale=args.txt_guide_scale,
        aud_scale=args.aud_guide_scale,
        seed=args.seed,
        start_use_videoref_step=args.start_use_videoref_step,
    )

    tavr = Tavr(args.dit_ckpt.resolve(), args.ckpt_dir.resolve())
    tavr.run(sample, args.output_dir.resolve() / sample.name / "generated_target.mp4", config)


if __name__ == "__main__":
    main()
