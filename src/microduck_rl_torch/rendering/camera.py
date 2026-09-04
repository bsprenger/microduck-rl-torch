"""Camera-name resolution for composed MuJoCo scenes."""

from __future__ import annotations

import mujoco


def resolve_named_camera(
    model: mujoco.MjModel,
    name: str,
    *,
    entity_name: str | None = None,
) -> tuple[int, str]:
    """Resolve an authored or entity-qualified camera name.

    Scene composition scopes entity-local names as ``<entity>/<name>``. A
    unique scoped match is a valid fallback when the unqualified name is absent.
    """

    candidates = [name]
    if entity_name:
        candidates.append(f"{entity_name}/{name}")
    for candidate in candidates:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, candidate)
        if camera_id >= 0:
            return int(camera_id), candidate

    if "/" not in name:
        matches = []
        for camera_id in range(int(model.ncam)):
            camera_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
            if camera_name and camera_name.endswith(f"/{name}"):
                matches.append((camera_id, camera_name))
        if len(matches) == 1:
            return matches[0]
        if matches:
            names = ", ".join(camera_name for _camera_id, camera_name in matches)
            raise ValueError(f"Camera {name!r} is ambiguous; matches: {names}")

    raise ValueError(f"Camera {name!r} was not found in the model")
