"""Small, streaming ffmpeg helpers for rollout videos and GIFs."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


def _require_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError(
            "ffmpeg is required for rollout rendering. Install it and make sure it is on PATH."
        )
    return executable


class VideoWriter:
    """Stream RGB frames into an H.264 MP4 without retaining the rollout."""

    def __init__(
        self,
        output: str | Path,
        *,
        width: int,
        height: int,
        fps: int = 25,
        crf: int = 18,
        preset: str = "medium",
        timeout: int = 120,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("Video dimensions must be positive")
        if fps < 1:
            raise ValueError("Video fps must be positive")
        self.output = Path(output)
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.timeout = timeout
        self.frame_count = 0
        self._process: subprocess.Popen[bytes] | None = None
        self._started_at = 0.0

    def _command(self, executable: str) -> list[str]:
        return [
            executable,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            str(self.output),
        ]

    def start(self) -> None:
        if self._process is not None:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(  # noqa: S603
            self._command(_require_ffmpeg()),
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._started_at = time.monotonic()

    def write(self, frame: np.ndarray) -> None:
        """Write one contiguous uint8 ``(height, width, 3)`` frame."""

        array = np.asarray(frame)
        expected_shape = (self.height, self.width, 3)
        if array.shape != expected_shape:
            raise ValueError(f"Expected frame shape {expected_shape}, got {array.shape}")
        if array.dtype != np.uint8:
            array = array.clip(0, 255).astype(np.uint8)
        self.start()
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(np.ascontiguousarray(array).tobytes())
        except BrokenPipeError as exc:
            stderr_pipe = self._process.stderr
            stderr = stderr_pipe.read().decode(errors="replace") if stderr_pipe else ""
            raise RuntimeError(f"ffmpeg stopped while encoding video:\n{stderr}") from exc
        self.frame_count += 1

    def close(self) -> float:
        """Finish encoding and return the wall-clock encoding time."""

        if self._process is None:
            return 0.0
        process = self._process
        assert process.stdin is not None
        if self.frame_count == 0:
            process.kill()
            process.wait()
            return time.monotonic() - self._started_at
        process.stdin.close()
        try:
            return_code = process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise TimeoutError(f"ffmpeg did not finish within {self.timeout}s") from exc
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while writing {self.output}:\n{stderr}")
        return time.monotonic() - self._started_at

    def abort(self) -> None:
        """Terminate a failed encoder without leaving a live child process."""

        if self._process is None or self._process.poll() is not None:
            return
        self._process.kill()
        self._process.wait()


def convert_video_to_gif(
    video: str | Path,
    gif: str | Path,
    *,
    fps: int = 12,
    width: int = 720,
    colors: int = 48,
    dither: str = "bayer",
    timeout: int = 120,
) -> Path:
    """Convert an MP4 to a looping, palette-optimized GIF."""

    video_path = Path(video)
    gif_path = Path(gif)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if fps < 1 or width < 1 or colors < 2:
        raise ValueError("GIF fps, width, and colors must be positive")
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    # A one-frame smoke-test MP4 can be shorter than one GIF frame period;
    # clone the final frame for one period so ffmpeg does not drop the stream.
    pad_duration = 1.0 / fps
    command = [
        _require_ffmpeg(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-filter_complex",
        (
            f"tpad=stop_mode=clone:stop_duration={pad_duration:.9g},"
            f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
            f"[s0]palettegen=max_colors={colors}:stats_mode=diff[p];"
            f"[s1][p]paletteuse=dither={dither}"
        ),
        "-loop",
        "0",
        str(gif_path),
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while writing {gif_path}:\n{completed.stderr}")
    return gif_path
