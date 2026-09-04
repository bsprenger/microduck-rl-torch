# Policy parity notes

The initial golden artifact is `alpha_walking.onnx` from the official public model repository:

- repository: `pollen-robotics/microduck-policies`
- policy API: 61 observations to 14 actions
- deployment frequency: 50 Hz
- model format: ONNX

The repository manifest is fetched at the same resolved revision as the policy. The fetcher checks
the model API dimensions, robot identifier, control frequency, ONNX graph dimensions, and SHA-256
digest. It then writes a small local artifact metadata file alongside the policy.

This is intentionally a deployment-level golden policy rather than a raw trainer checkpoint. The
upstream Hugging Face Jobs flow uploads `logs/rsl_rl/**/model_*.pt` snapshots to private model
repositories. A future checkpoint adapter should be added only after the exact run configuration,
network architecture, observation normalization, and action scaling are identified; loading an
arbitrary `.pt` file would not establish parity.

The current environment uses the upstream walk XML's ordinary position actuators. The upstream
inference path changes the compiled timestep to 0.005 s and applies each action for four physics
steps, so those settings are explicit in the validation bundle. Actuator backlash, response delay,
and reward/training semantics are not silently approximated by this first validator. The local
`mujoco-torch` dependency is pinned to the sibling checkout so its collision and renderer fixes
are immediately used after `uv sync`.

The target keeps the full CAD mesh arrays for rendering but only derives convex collision data for
geoms enabled by MuJoCo contact masks or explicit pairs. This matches the upstream separation of
visual and curated collision geometry. The default CPU smoke command still disables contact
processing for a bounded quick check; `CONTACTS=enabled` runs the curated contact path.
