"""Backend-independent derived kinematics used by sensors and task terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def site_linear_velocity(data: Any, bundle: Any, site_ids: Sequence[int]) -> torch.Tensor:
    """Return world-frame linear velocity at the selected sites.

    ``mujoco-torch`` exposes body spatial velocity (``cvel``), but not the
    optional ``site_xvelp`` convenience array exposed by MuJoCo's Python
    binding.  Site velocity is an exact kinematic derivation from the body
    spatial velocity and the current site offset; keeping it here gives both
    the generic sensor API and velocity rewards the same backend-neutral
    semantics.
    """

    if not site_ids:
        return torch.zeros(
            (*data.site_xpos.shape[:-2], 0, 3),
            dtype=data.site_xpos.dtype,
            device=data.site_xpos.device,
        )
    site_index = tuple(int(index) for index in site_ids)
    native_body_ids = tuple(int(bundle.native_model.site_bodyid[index]) for index in site_index)
    body_ids = torch.as_tensor(native_body_ids, dtype=torch.long, device=data.site_xpos.device)
    site_position = data.site_xpos[..., site_index, :]
    body_spatial = data.cvel[..., body_ids, :]
    torch_model = getattr(bundle, "torch_model", None)
    if torch_model is not None and hasattr(torch_model, "body_rootid_t"):
        body_rootid = torch_model.body_rootid_t
    else:
        body_rootid = torch.as_tensor(
            bundle.native_model.body_rootid, dtype=torch.long, device=data.site_xpos.device
        )
    if hasattr(data, "subtree_com"):
        root_ids = torch.as_tensor(
            body_rootid, dtype=torch.long, device=data.site_xpos.device
        ).index_select(0, body_ids)
        reference_position = data.subtree_com[..., root_ids, :]
    else:
        # Keep the helper usable with small synthetic data objects that only
        # provide body COMs; real mujoco-torch data follows the native
        # subtree-COM path above.
        reference_position = data.xipos[..., body_ids, :]
    return body_spatial[..., 3:] - torch.cross(
        body_spatial[..., :3], reference_position - site_position, dim=-1
    )
