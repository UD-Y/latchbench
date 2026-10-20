"""Arena assets for the RJ45 task: the 3-port panel (one per clearance tier), its stand, and the plug.

Pattern: ``isaaclab_arena_environments/isaac_cap/usbc_insertion/assets.py`` (``LibraryObject`` subclasses registered
with ``@register_asset``), with a custom spawner instead of ``spawn_cfg_addon`` so the PhysX settings chosen by contact
tuning (``patchbench/physics.py``) are authored exactly as in ``scripts/check_contact.py``: contact/rest offset on every
collider, one plastic physics material, contact reports and a depenetration limit on the plug links.

The plug is not a free scene object. Under the v1 grasp assumption (CLAUDE.md section 5) it is part of the robot: the
SO-101 spawner below references ``plug.usd`` under the robot prim and welds ``Plug/body`` to the ``gripper`` link with a
fixed joint, so PhysX makes the plug body and latch extra links of the SO-101 articulation (joint ``latch_joint``).
``RJ45Plug`` is registered so the asset can be found by name; ``RJ45Plug.attach_to(robot_cfg)`` does the welding.

Import after the simulation app has started (Isaac Lab / pxr / Arena imports).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from patchbench import physics
from patchbench.assets.build_assets import TIERS, USD_DIR, RJ45Dims, jack_geometry

DIMS = RJ45Dims()

LOOK = True
"""Use the render-only look layers (usd/look/*.usd: realistic plug, cable, jacks; build_visuals.py). They reference
the physics USDs and only add visual meshes and materials, so physics is identical either way. Set by the env
(``--look``) before any asset is constructed."""

CABLE_SEGMENTS = 24
"""Render-only trailing cable: number of segments re-posed each control step (cable_visual.py) when cameras are on."""


def plug_usd_path() -> str:
    return str(USD_DIR / "look" / "plug_look.usd") if LOOK else str(USD_DIR / "medium" / "plug.usd")


def panel_usd_path(tier: str) -> str:
    return str(USD_DIR / "look" / f"panel_3port_{tier}_look.usd") if LOOK else str(USD_DIR / tier / f"panel_3port_{tier}.usd")

PLUG_GRASP_BEHIND = 0.020
"""m. The jaws hold the cable 20 mm behind the plug's rear face (a boot/cable stub, visual only), so the gripper stays
>= 26 mm in front of the panel face at full insertion (seat depth 16 mm). Grasp assumption, CLAUDE.md section 5."""

PLUG_IN_HAND_ROT_XYZW = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
"""Plug frame in the ``gripper`` link frame: 180 deg about (1, 1, 0)/sqrt(2), i.e. plug +X (insertion) = gripper +Y,
plug +Y = gripper +X (the jaw closing direction clamps the plug's sides), plug +Z (latch side) = gripper -Z (along the
jaws). With the jaws pointing straight down the plug axis is horizontal and the latch faces down, as the panel's ports
expect; wrist roll then sets the plug's yaw freely (the 5-DOF arm's gimbal case, :func:`wrist_roll_for`), which is why
the insertion axis can be horizontal for every port. The wrist camera (isaaclab_so101) then looks along the plug at
the panel (checked in results/arena_env_v0/medium_wrist.png; the opposite mounting shows only the cable)."""

TCP_IN_GRIPPER = (-0.007, 0.0, -0.102)
"""isaaclab_so101.constants.TCP_OFFSET: between the jaw tips, ``gripper`` link frame."""

PANEL_POS = (0.22, 0.0, 0.05)
"""m, nominal panel frame (port_1 mouth) in the env frame; insertion along env +X. Chosen with the closed-form 5-DOF IK
(tests/test_env_layout.py): every joint stays >= 0.35 rad inside its limits for the plug tip from 15 mm in front of
any mouth to 3 mm past the seat, under the full panel randomization below, jaws pointing straight down."""

PANEL_POS_RANGE = 0.010
"""m, +- per axis (CLAUDE.md 6.2)."""

PANEL_ROT_RANGE = math.radians(3.0)
"""rad, +- yaw and pitch (CLAUDE.md 6.2)."""


def plug_in_hand_pos() -> tuple[float, float, float]:
    """Plug frame origin (tip centre) in the ``gripper`` frame: TCP + (plug length + grasp offset) along plug +X."""
    reach = DIMS.plug_length + PLUG_GRASP_BEHIND
    return (TCP_IN_GRIPPER[0], TCP_IN_GRIPPER[1] + reach, TCP_IN_GRIPPER[2])  # plug +X = gripper +Y


def wrist_roll_for(psi: float, yaw_b: float = -math.pi / 2) -> float:
    """Natural-pose roll (isaaclab_so101.ee_pose, tilt 0 = jaws down) that points the plug axis at base yaw ``yaw_b``.

    ``psi`` is the TCP azimuth about the pan axis. With tilt 0 the gripper's +Y axis has base yaw psi + roll + pi/2,
    and the plug axis is gripper +Y. Default ``yaw_b`` = -pi/2: base -Y = env +X, the nominal insertion direction.
    """
    return math.remainder(yaw_b - psi - math.pi / 2, math.tau)


# --------------------------------------------------------------- Isaac Lab / Arena part (needs the simulation app)

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.sim.utils.prims import clone
    from isaaclab.utils.configclass import configclass
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object_library import LibraryObject
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.register import register_asset
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    HAVE_ISAAC = True
except ImportError:  # CPU-only tests: the pure geometry above stays importable
    HAVE_ISAAC = False

if HAVE_ISAAC:
    # ------------------------------------------------------------------------------------------- USD physics authoring


    def author_contact_physics(stage: Usd.Stage, root: str, *, plug_links: bool) -> None:
        """physics.py contact settings on every collider under ``root`` (check_contact.py's ``build_scene``).

        Binds one plastic material (static = dynamic friction ``CONTACT_FRICTION``, restitution 0). With ``plug_links``,
        rigid bodies also get a contact report API (threshold 0) and the depenetration velocity limit.
        """
        mat = UsdShade.Material.Define(stage, f"{root}/rj45_plastic")
        m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        m.CreateStaticFrictionAttr(physics.CONTACT_FRICTION)
        m.CreateDynamicFrictionAttr(physics.CONTACT_FRICTION)
        m.CreateRestitutionAttr(0.0)
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                c = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                c.CreateContactOffsetAttr(physics.CONTACT_OFFSET)
                c.CreateRestOffsetAttr(physics.REST_OFFSET)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            if plug_links and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(physics.MAX_DEPENETRATION_VELOCITY)


    def attach_plug(stage: Usd.Stage, robot_path: str, plug_usd: str, pos, rot_xyzw, parent_link: str = "gripper",
                    cable_segments: int = 0) -> None:
        """Reference ``plug_usd`` at ``<robot>/Plug`` and weld ``Plug/body`` to ``<robot>/<parent_link>`` at (pos, rot).

        The plug prim is posed consistently with the link's current USD pose so the articulation starts closed.
        """
        plug = UsdGeom.Xform.Define(stage, f"{robot_path}/Plug")
        plug.GetPrim().GetReferences().AddReference(plug_usd)
        rot = Gf.Quatd(rot_xyzw[3], Gf.Vec3d(*rot_xyzw[:3]))
        in_hand = Gf.Matrix4d().SetTransform(Gf.Rotation(rot), Gf.Vec3d(*pos))
        cache = UsdGeom.XformCache()
        link_w = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{robot_path}/{parent_link}"))
        robot_w = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(robot_path))
        UsdGeom.Xformable(plug).MakeMatrixXform().Set(in_hand * link_w * robot_w.GetInverse())  # row-vector convention

        joint = UsdPhysics.FixedJoint.Define(stage, f"{robot_path}/plug_grasp")
        joint.CreateBody0Rel().SetTargets([f"{robot_path}/{parent_link}"])
        joint.CreateBody1Rel().SetTargets([f"{robot_path}/Plug/body"])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pos))
        joint.CreateLocalRot0Attr(Gf.Quatf(rot))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))

        look = "/look/" in plug_usd
        if not look:
            # Visual cable stub from the plug's rear face back through the jaws (no collider, no mass).
            length = PLUG_GRASP_BEHIND + 0.015
            cable = UsdGeom.Cylinder.Define(stage, f"{robot_path}/Plug/body/cable_stub")
            cable.CreateAxisAttr(UsdGeom.Tokens.x)
            cable.CreateRadiusAttr(0.003)
            cable.CreateHeightAttr(length)
            cable.AddTranslateOp().Set(Gf.Vec3d(-DIMS.plug_length - length / 2, 0.0, 0.0))
            cable.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.35, 0.8)])
        if cable_segments:
            # Trailing cable (render only): segments posed every control step by cable_visual.cable_follow; the
            # look layer's static cable is hidden. Each segment: unit-length cylinder along +X, scaled in X.
            if look:
                UsdGeom.Imageable(stage.OverridePrim(f"{robot_path}/Plug/body/look/cable")).CreateVisibilityAttr(
                    UsdGeom.Tokens.invisible
                )
            jacket = UsdShade.Material(stage.GetPrimAtPath(f"{robot_path}/Plug/Looks/jacket_pvc"))
            for k in range(cable_segments):
                seg = UsdGeom.Xform.Define(stage, f"{robot_path}/cable_vis/seg_{k:02d}")
                seg.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, -1.0))
                seg.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0))
                seg.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.01, 1.0, 1.0))
                cyl = UsdGeom.Cylinder.Define(stage, f"{robot_path}/cable_vis/seg_{k:02d}/tube")
                cyl.CreateAxisAttr(UsdGeom.Tokens.x)
                cyl.CreateRadiusAttr(0.003)
                cyl.CreateHeightAttr(1.0)
                cyl.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
                if jacket:
                    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(jacket)
                else:
                    cyl.CreateDisplayColorAttr([Gf.Vec3f(0.09, 0.30, 0.74)])
        author_contact_physics(stage, f"{robot_path}/Plug", plug_links=True)


    @clone
    def spawn_robot_with_plug(prim_path: str, cfg: RobotWithPlugUsdFileCfg, translation=None, orientation=None, **kwargs):
        """``spawn_from_usd`` for the robot, then :func:`attach_plug` (cloned to every env together with the robot)."""
        prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
        attach_plug(prim.GetStage(), prim_path, cfg.plug_usd_path, cfg.plug_pos, cfg.plug_rot_xyzw, cfg.plug_parent_link,
                    cfg.cable_segments)
        return prim


    @configclass
    class RobotWithPlugUsdFileCfg(UsdFileCfg):
        """A robot USD with the RJ45 plug welded to one of its links."""

        func: Callable = spawn_robot_with_plug
        plug_usd_path: str = str(USD_DIR / "medium" / "plug.usd")
        plug_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        plug_rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        plug_parent_link: str = "gripper"
        cable_segments: int = 0


    @clone
    def spawn_rj45_usd(prim_path: str, cfg: RJ45UsdFileCfg, translation=None, orientation=None, **kwargs):
        """``spawn_from_usd`` plus :func:`author_contact_physics` (panel: no contact reports)."""
        prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
        author_contact_physics(prim.GetStage(), prim_path, plug_links=False)
        return prim


    @configclass
    class RJ45UsdFileCfg(UsdFileCfg):
        func: Callable = spawn_rj45_usd


    # ----------------------------------------------------------------------------------------------- registered assets


    class _RJ45Panel(LibraryObject):
        """3-port panel, one kinematic rigid body; frame at the plate front face, centre of port_1, +X = insertion."""

        tags = ["object", "rj45"]
        object_type = ObjectType.RIGID
        tier: str = "medium"

        def __init__(self, instance_name: str | None = "rj45_panel", **kwargs):
            spawner = RJ45UsdFileCfg(usd_path=panel_usd_path(self.tier))
            super().__init__(instance_name=instance_name, spawner_cfg=spawner, **kwargs)
            self.geometry = jack_geometry(DIMS, self.tier)

        @staticmethod
        def port_positions() -> list[tuple[float, float, float]]:
            """Port mouth origins in the panel frame (port_0 = left = +Y)."""
            return [(0.0, (1 - i) * DIMS.pitch, 0.0) for i in range(DIMS.n_ports)]

        @staticmethod
        def extent_z() -> tuple[float, float]:
            """Panel z range in its own frame [m] (ports flipped latch side down; build_assets.build_panel)."""
            g = jack_geometry(DIMS, "loose")
            return -g.z_top_outer - DIMS.plate_margin, -g.z_bottom_outer + DIMS.plate_margin

        @staticmethod
        def extent_y() -> tuple[float, float]:
            g = jack_geometry(DIMS, "loose")
            ys = [p[1] for p in _RJ45Panel.port_positions()]
            return min(ys) - g.outer_width / 2 - DIMS.plate_margin, max(ys) + g.outer_width / 2 + DIMS.plate_margin


    @register_asset
    class RJ45PanelLoose(_RJ45Panel):
        name = "rj45_panel_loose"
        tier = "loose"


    @register_asset
    class RJ45PanelMedium(_RJ45Panel):
        name = "rj45_panel_medium"
        tier = "medium"


    @register_asset
    class RJ45PanelSpec(_RJ45Panel):
        name = "rj45_panel_spec"
        tier = "spec"


    PANEL_CLASSES = {"loose": RJ45PanelLoose, "medium": RJ45PanelMedium, "spec": RJ45PanelSpec}
    assert set(PANEL_CLASSES) == set(TIERS)

    STAND_DEPTH = DIMS.seat_depth + DIMS.back_wall
    """m, panel x extent (front face to back stop): the stand is as deep as the jacks."""


    def stand_size(panel_bottom_z_w: float) -> tuple[float, float, float]:
        """Stand box (x, y, z) from the ground (z = 0) up to the panel's lowest point."""
        y0, y1 = _RJ45Panel.extent_y()
        return (STAND_DEPTH, y1 - y0, panel_bottom_z_w)


    @register_asset
    class RJ45PanelStand(LibraryObject):
        """Kinematic box under the panel; the env's reset event moves it with the panel."""

        name = "rj45_panel_stand"
        tags = ["object", "rj45"]
        object_type = ObjectType.RIGID

        def __init__(self, instance_name: str | None = None, size=(0.02, 0.06, 0.03), **kwargs):
            spawner = sim_utils.CuboidCfg(
                size=tuple(size),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            )
            super().__init__(instance_name=instance_name, spawner_cfg=spawner, **kwargs)


    @register_asset
    class RJ45Plug(Asset):
        """The RJ45 plug (``/Plug``: tip-centre origin, +X insertion, +Z latch side; links ``body`` + ``latch``).

        Not spawned on its own: :meth:`attach_to` welds it into a robot articulation (v1 grasp assumption).
        """

        name = "rj45_plug"
        tags = ["object", "rj45"]
        def __init__(self, instance_name: str | None = None):
            super().__init__(name=instance_name or self.name, tags=list(self.tags))
            self.usd_path = plug_usd_path()  # identical for every tier

        def attach_to(
            self,
            robot_cfg: ArticulationCfg,
            pos: tuple[float, float, float] | None = None,
            rot_xyzw: tuple[float, float, float, float] = PLUG_IN_HAND_ROT_XYZW,
            parent_link: str = "gripper",
            cable_segments: int = 0,
        ) -> ArticulationCfg:
            """Copy of ``robot_cfg`` whose USD spawner also adds this plug, welded to ``parent_link``."""
            spawn = robot_cfg.spawn
            assert isinstance(spawn, UsdFileCfg), "attach_to needs a USD-spawned robot"
            fields = {k: getattr(spawn, k) for k in spawn.__dataclass_fields__ if k != "func"}
            new_spawn = RobotWithPlugUsdFileCfg(
                **fields,
                plug_usd_path=self.usd_path,
                plug_pos=pos if pos is not None else plug_in_hand_pos(),
                plug_rot_xyzw=rot_xyzw,
                plug_parent_link=parent_link,
                cable_segments=cable_segments,
            )
            return robot_cfg.replace(spawn=new_spawn)
