import shutil

import numpy as np
import pytest

from microduck_rl_torch.rendering.video import VideoWriter, convert_video_to_gif


@pytest.mark.integration
def test_video_writer_and_gif_conversion(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")

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
