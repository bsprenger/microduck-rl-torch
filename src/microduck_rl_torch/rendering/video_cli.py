"""Command-line entry point for MP4-to-GIF conversion."""

from __future__ import annotations

import argparse

from .video import convert_video_to_gif


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--colors", type=int, default=48)
    parser.add_argument("--dither", default="bayer")
    args = parser.parse_args(argv)
    convert_video_to_gif(
        args.input,
        args.output,
        fps=args.fps,
        width=args.width,
        colors=args.colors,
        dither=args.dither,
    )
    print(f"GIF: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
