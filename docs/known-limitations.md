# Known limitations

The current implementation has these documented boundaries:

- Terrain contact is not numerically equivalent to MuJoCo-Warp. The CPU Torch path supports the
  rough-task heightfield, box, and mesh pairs, while its heightfield manifold is an explicit prism
  approximation. Mesh-versus-terrain contact can be expensive; disabling mesh-to-mesh contacts
  does not remove that cost.
- Terrain physics heights are preserved, but heightfield texture and material buffers are not
  transferred to the current backend path. Terrain visualization uses a fallback color where
  necessary.
- The contact sensor exposes the backend contact count and native contact fields for force, torque,
  position, and normal. Full MuJoCo `CONTACT` dataspec support is not implemented in the Torch path.
- Actuator delay state uses one shared buffer for the primary actuated entity. Independent delay
  groups for multiple actuated entities are not supported.
- The rough-task foot-clearance reward is defined but contributes zero when peak-height state is
  not configured. Force normalization and sensor cadence are not covered by a reference
  trajectory check for critic-level parity.
- The public task catalog has one constructible velocity task. Other task identifiers are explicit
  names for lookup and reporting, but do not imply a complete executable task configuration.
- Direct MuJoCo-Warp comparison is optional and requires a separately provisioned `microduck_rl`
  environment. MuJoCo-Warp does not provide an MPS backend.
