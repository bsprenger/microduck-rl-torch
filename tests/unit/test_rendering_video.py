import json
import shutil
import subprocess

import numpy as np
import pytest

from microduck_rl_torch.rendering.video import VideoWriter, convert_video_to_gif


def _probe(path):
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_name,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


@pytest.mark.integration
def test_video_writer_and_gif_conversion(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")

    video_path = tmp_path / "rollout.mp4"
    gif_path = tmp_path / "rollout.gif"
    writer = VideoWriter(video_path, width=8, height=6, fps=5)
    writer.write(np.zeros((6, 8, 3), dtype=np.uint8))
    writer.write(np.full((6, 8, 3), 255, dtype=np.uint8))
    writer.close()

    assert video_path.is_file()
    assert video_path.stat().st_size > 0
    convert_video_to_gif(video_path, gif_path, fps=5, width=8, colors=8)
    assert gif_path.is_file()
    assert gif_path.stat().st_size > 0

    video_probe = _probe(video_path)
    video_stream = video_probe["streams"][0]
    assert video_stream["codec_name"] == "h264"
    assert video_stream["nb_frames"] == "2"
    assert float(video_probe["format"]["duration"]) == pytest.approx(0.4, abs=0.05)

    gif_probe = _probe(gif_path)
    gif_stream = gif_probe["streams"][0]
    assert gif_stream["codec_name"] == "gif"
    assert gif_stream["nb_frames"] == "2"
    assert float(gif_probe["format"]["duration"]) == pytest.approx(0.4, abs=0.05)
