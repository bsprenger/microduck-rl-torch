import torch

from microduck_rl_torch.envs.config import CommandConfig, sample_twist


def test_turn_in_place_sampling_matches_upstream_bucket():
    config = CommandConfig(turn_in_place_fraction=1.0, standing_fraction=0.0)
    generator = torch.Generator().manual_seed(7)
    for _ in range(32):
        twist = sample_twist(
            config,
            generator=generator,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert torch.equal(twist[:2], torch.zeros(2))
        assert 0.4 <= float(torch.abs(twist[2])) <= 1.0
