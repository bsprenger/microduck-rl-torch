"""MicroDuck robot model specifications.

The public constants are loaded lazily so the backend can import shared joint
names without recursively importing the entire environment package.
"""

__all__ = [
    "MICRODUCK_BACKLASH_ROBOT_CFG",
    "MICRODUCK_ALLCOLLISIONS_ROBOT_CFG",
    "MICRODUCK_BALL_CFG",
    "MICRODUCK_GROUND_PICK_ROBOT_CFG",
    "MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG",
    "MICRODUCK_STANDUP_ROBOT_CFG",
    "MICRODUCK_WALK_BACKLASH_ROBOT_CFG",
    "MICRODUCK_WALK_ROBOT_CFG",
    "MICRODUCK_WALK_ROLLERS_ROBOT_CFG",
    "SERVO_JOINT_NAMES",
]


def __getattr__(name: str):
    if name in __all__:
        from . import model_variants

        return getattr(model_variants, name)
    raise AttributeError(name)
