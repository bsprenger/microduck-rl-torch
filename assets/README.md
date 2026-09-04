# MicroDuck model assets

The XML files and mesh assets in this directory are copied from the MicroDuck robot model used by
the upstream simulation. The directory structure is preserved because MuJoCo resolves `<include>`
and mesh paths relative to the XML files.

The initial validation uses `scene_walk.xml` and `robot_walk.xml`. Other scenes are retained so
future parity work can add the backlash, roller, and apartment configurations without changing the
asset provenance.

Code and XML provenance: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl).
Check the upstream repository and asset-specific notices before redistributing mesh files.

