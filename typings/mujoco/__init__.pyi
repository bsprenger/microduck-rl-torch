"""Minimal typing surface for MuJoCo's dynamic Python bindings.

The published MuJoCo package does not expose complete annotations for the
``MjSpec`` builder and enum-style constants used by this project.  Runtime
behavior still comes from the installed package; this stub only supplies the
names and dynamic fields needed by the static checker.
"""

from typing import Any


class MjModel:
    @classmethod
    def from_xml_path(cls, path: str) -> MjModel: ...
    @classmethod
    def from_xml_string(cls, xml: str) -> MjModel: ...

    def __getattr__(self, name: str) -> Any: ...


class MjData:
    time: Any

    def __init__(self, model: MjModel) -> None: ...

    def __getattr__(self, name: str) -> Any: ...


class MjSpec:
    @classmethod
    def from_file(cls, path: str) -> MjSpec: ...
    @classmethod
    def from_string(cls, xml: str) -> MjSpec: ...

    def compile(self) -> MjModel: ...
    def to_xml(self) -> str: ...
    def __getattr__(self, name: str) -> Any: ...


class MjvCamera:
    azimuth: float
    distance: float
    elevation: float
    fixedcamid: int
    lookat: Any
    type: int
    trackbodyid: int

    def __getattr__(self, name: str) -> Any: ...


class Renderer:
    def __init__(self, model: MjModel, *, height: int, width: int) -> None: ...

    def __getattr__(self, name: str) -> Any: ...


class mjtGeom:
    mjGEOM_BOX: Any
    mjGEOM_CAPSULE: Any
    mjGEOM_CYLINDER: Any
    mjGEOM_ELLIPSOID: Any
    mjGEOM_HFIELD: Any
    mjGEOM_MESH: Any
    mjGEOM_PLANE: Any
    mjGEOM_SPHERE: Any


class mjtJoint:
    mjJNT_HINGE: Any


class mjtObj:
    mjOBJ_ACTUATOR: Any
    mjOBJ_BODY: Any
    mjOBJ_CAMERA: Any
    mjOBJ_GEOM: Any
    mjOBJ_JOINT: Any
    mjOBJ_KEY: Any
    mjOBJ_MATERIAL: Any
    mjOBJ_SENSOR: Any
    mjOBJ_SITE: Any
    mjOBJ_TENDON: Any
    mjOBJ_XBODY: Any


class mjtSensor:
    mjSENS_CONTACT: Any


class mjtCamera:
    mjCAMERA_FREE: Any
    mjCAMERA_TRACKING: Any


class mjtTrn:
    mjTRN_JOINT: Any
    mjTRN_TENDON: Any


class mjtDisableBit:
    mjDSBL_CONTACT: Any


def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_id2name(model: MjModel, object_type: Any, object_id: int) -> str | None: ...
def mj_name2id(model: MjModel, object_type: Any, name: str) -> int: ...
def mj_jacSite(*args: Any, **kwargs: Any) -> Any: ...
def mj_setConst(*args: Any, **kwargs: Any) -> Any: ...
def mjv_defaultFreeCamera(*args: Any, **kwargs: Any) -> Any: ...
