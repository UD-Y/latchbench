"""Parametric pxr builder: writes usd/<tier>/plug.usd and panel_3port_<tier>.usd (CLAUDE.md section 5).

Only ``pxr`` (UsdGeom, UsdPhysics) is used, so this runs in any env with usd-core, without Isaac Sim.
PhysX-specific tuning (contact/rest offsets, solver iterations, CCD) is not baked in here; it is applied by
``patchbench/physics.py`` after contact tuning (Block 2).

Conventions (all SI, metres / kg / s; USD joint angles and angular-drive gains are per *degree*):

* **Plug frame** ``/Plug``: origin at the centre of the plug's front face (the tip), +X = insertion direction,
  +Z = latch side. The body spans x in [-plug_length, 0].
* **Jack frame** (``port_i``): origin at the centre of the body opening on the panel front face (the mouth),
  +X = insertion direction, +Z = latch side. "top/bottom/left(+Y)/right(-Y)" prim names are in this frame.
  A plug is inserted correctly when its frame coincides with the port frame (plug tip at the mouth), then
  moves along +X; it is seated when the tip reaches ``seat`` (x = seat_depth).
* **Panel frame** ``/Panel``: origin at the plate front face, centre of port_1. Ports sit at y = (1 - i) * pitch,
  so port_0 is on the left (+Y) when looking along +X with +Z up. Each port is rotated 180 deg about X, so the
  latch side faces **down** in the panel frame (standard orientation, section 5).

Latch mechanism (primitives only): the latch is a thin box on a revolute joint at its root, near the plug tip,
rising backwards at ``latch_rest_deg``. Its root sits in a recess (pivot ``latch_thickness`` below the body top),
so when it is pressed flat it is flush with the body top. The jack ceiling has a slot over the latch; the slot is
closed at the mouth by the lead-in wedge, the ``top_lip`` and the ``catch_ledge``, which press the latch flat on
the way in. Behind the ledge is a pocket. The latch tip clears the ledge's back face (x = catch_depth) when the
tip is ``snap_margin`` short of ``seat_depth``, so it springs up into the pocket just before the plug hits the
back stop; pulling back, the tip catches on the ledge (latched axial play = snap_margin <= 1.14 mm).

Run: ``python -m patchbench.assets.build_assets [--out DIR]``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

# Lateral clearance per side added to plug width and body height (CLAUDE.md section 5).
TIERS: dict[str, float] = {"loose": 0.50e-3, "medium": 0.25e-3, "spec": 0.115e-3}

USD_DIR = Path(__file__).resolve().parent / "usd"


@dataclass(frozen=True)
class RJ45Dims:
    """All dimensions in metres, masses in kg. Sources: IEC 60603-7 / FCC 68 notes as summarised in CLAUDE.md 5."""

    # --- plug (IEC 60603-7 reference values) ---
    plug_width: float = 11.68e-3
    plug_body_height: float = 6.60e-3  # insertion-area height; body + latch at rest is ~8.0 mm (spec 8.00)
    plug_length: float = 22.48e-3
    plug_mass: float = 5.0e-3  # section 5: plug ~5 g
    # --- latch (cantilever beam estimate, section 5) ---
    latch_length: float = 12.0e-3  # L
    latch_width: float = 3.5e-3  # b
    latch_thickness: float = 1.0e-3  # t
    latch_youngs_modulus: float = 2.3e9  # polycarbonate E
    latch_rest_deg: float = 9.0  # section 5: angled up ~8-10 deg
    latch_root_offset: float = 1.0e-3  # pivot distance behind the plug tip ("rooted near the plug front")
    latch_mass: float = 0.3e-3  # section 5: latch ~0.3 g
    latch_lower_deg: float = -12.0  # joint limits relative to rest (P2 spec)
    latch_upper_deg: float = 2.0
    latch_damping_ratio: float = 0.3  # "small damping"; not tuned (Block 2)
    # --- jack ---
    seat_depth: float = 16.0e-3  # mouth -> back stop; plug tip position when fully seated
    snap_margin: float = 0.5e-3  # latch clears the ledge this far before the back stop; = latched axial play (<= 1.14)
    catch_length: float = 1.40e-3  # axial length of the catch ledge (spec: latch catch length >= 1.40 mm)
    lead_in_width: float = 0.5e-3  # chamfer widens the opening by this much per side at the mouth
    lead_in_angle_deg: float = 30.0  # chamfer face angle to the insertion axis
    latch_slot_clearance: float = 0.5e-3  # per side, latch slot in the jack ceiling
    pocket_clearance: float = 0.5e-3  # pocket height above the latch's rest height
    side_wall: float = 1.0e-3  # thin so a jack fits the 15 mm pitch
    top_wall: float = 2.0e-3
    bottom_wall: float = 2.0e-3
    back_wall: float = 2.0e-3
    # --- panel ---
    n_ports: int = 3
    pitch: float = 15.0e-3
    plate_thickness: float = 2.0e-3
    plate_margin: float = 6.0e-3

    @property
    def latch_k_theta(self) -> float:
        """Torsional stiffness at the latch root, N*m/rad: k_tip = 3EI/L^3, k_theta = k_tip*L^2 = 3EI/L (section 5)."""
        inertia = self.latch_width * self.latch_thickness**3 / 12.0
        return 3.0 * self.latch_youngs_modulus * inertia / self.latch_length

    @property
    def latch_inertia_root(self) -> float:
        """Latch moment of inertia about its root (thin rod, m L^2 / 3), kg*m^2."""
        return self.latch_mass * self.latch_length**2 / 3.0

    @property
    def latch_damping(self) -> float:
        """Drive damping in N*m*s/rad from the damping ratio: c = 2 zeta sqrt(k I)."""
        return 2.0 * self.latch_damping_ratio * math.sqrt(self.latch_k_theta * self.latch_inertia_root)

    @property
    def lead_in_length(self) -> float:
        """Axial length of the lead-in chamfer."""
        return self.lead_in_width / math.tan(math.radians(self.lead_in_angle_deg))

    @property
    def latch_rise(self) -> float:
        """Height of the latch's highest point above the body top at rest (root recessed by its thickness)."""
        a = math.radians(self.latch_rest_deg)
        return self.latch_length * math.sin(a) + self.latch_thickness * math.cos(a) - self.latch_thickness


@dataclass(frozen=True)
class JackGeometry:
    """Derived jack dimensions for one clearance tier (jack frame, metres)."""

    tier: str
    clearance: float  # per side
    width: float  # opening width = plug_width + 2c
    height: float  # body opening height = plug_body_height + 2c
    slot_width: float
    pocket_height: float  # pocket above the cavity ceiling, behind the ledge
    lip_start: float  # = lead-in length
    catch_start: float
    catch_depth: float  # x of the ledge's back face (the catch)
    seat_depth: float
    snap_depth: float  # plug-tip depth at which the (flat) latch tip clears the ledge
    latched_play: float
    outer_width: float
    z_top_outer: float
    z_bottom_outer: float


def jack_geometry(dims: RJ45Dims, tier: str, clearance: float | None = None) -> JackGeometry:
    c = TIERS[tier] if clearance is None else clearance
    w = dims.plug_width + 2 * c
    h = dims.plug_body_height + 2 * c
    pocket = dims.latch_rise + dims.pocket_clearance
    snap = dims.seat_depth - dims.snap_margin
    # Flat latch tip sits latch_root_offset + latch_length behind the plug tip.
    catch_depth = snap - dims.latch_root_offset - dims.latch_length
    catch_start = catch_depth - dims.catch_length
    g = JackGeometry(
        tier=tier,
        clearance=c,
        width=w,
        height=h,
        slot_width=dims.latch_width + 2 * dims.latch_slot_clearance,
        pocket_height=pocket,
        lip_start=dims.lead_in_length,
        catch_start=catch_start,
        catch_depth=catch_depth,
        seat_depth=dims.seat_depth,
        snap_depth=snap,
        latched_play=dims.snap_margin,
        outer_width=w + 2 * dims.side_wall,
        z_top_outer=h / 2 + pocket + dims.top_wall,
        z_bottom_outer=-h / 2 - dims.bottom_wall,
    )
    # Geometry sanity: fail loudly instead of writing an impossible jack.
    assert g.catch_start >= g.lip_start, f"{tier}: lip/ledge does not fit between lead-in and catch depth"
    assert dims.lead_in_width < dims.side_wall, "lead-in chamfer deeper than the side wall"
    assert g.slot_width < w, "latch slot wider than the opening"
    assert g.outer_width <= dims.pitch, f"{tier}: jack outer width {g.outer_width * 1e3:.2f} mm exceeds pitch"
    assert g.latched_play <= 1.14e-3, "latched axial play exceeds the 1.14 mm spec"
    return g


# ----------------------------------------------------------------------------------------------- pxr helpers


def _new_stage(path: Path, default_prim: str) -> tuple[Usd.Stage, Usd.Prim]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    # ASCII inside a .usd file: diffable in git and independent of the crate version of the reader.
    layer = Sdf.Layer.CreateNew(str(path), args={"format": "usda"})
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{default_prim}").GetPrim()
    stage.SetDefaultPrim(root)
    return stage, root


def _xform(stage, path, pos=(0.0, 0.0, 0.0), quat: Gf.Quatd | None = None) -> Usd.Prim:
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat or Gf.Quatd(1.0))
    return x.GetPrim()


def _box(stage, path, lo, hi, color, collide=True) -> Usd.Prim:
    """Axis-aligned box collider from corner ``lo`` to ``hi`` in the parent frame (unit cube + scale)."""
    lo, hi = Gf.Vec3d(*lo), Gf.Vec3d(*hi)
    size = hi - lo
    assert min(size) > 0, f"degenerate box {path}: {size}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set((lo + hi) / 2)
    cube.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(size)
    cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def _prism(stage, path, quad, axis: int, extent, color) -> Usd.Prim:
    """Convex quadrilateral prism: ``quad`` = 4 (a, b) points in the plane normal to ``axis``, extruded over
    ``extent`` = (lo, hi) along ``axis``. Collides as a convex hull."""
    others = [i for i in range(3) if i != axis]
    pts = []
    for s in extent:
        for a, b in quad:
            p = [0.0, 0.0, 0.0]
            p[axis], p[others[0]], p[others[1]] = s, a, b
            pts.append(Gf.Vec3f(*p))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * 6))
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    )
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    return mesh.GetPrim()


def _quat_y(deg: float) -> Gf.Quatd:
    h = math.radians(deg) / 2
    return Gf.Quatd(math.cos(h), Gf.Vec3d(0.0, math.sin(h), 0.0))


def _set_meta(prim: Usd.Prim, data: dict) -> None:
    for k, v in data.items():
        prim.SetCustomDataByKey(f"patchbench:{k}", v)


PLUG_COLOR = (0.75, 0.82, 0.90)
LATCH_COLOR = (0.95, 0.55, 0.10)
JACK_COLOR = (0.30, 0.30, 0.33)
LEDGE_COLOR = (0.80, 0.15, 0.15)
PLATE_COLOR = (0.55, 0.57, 0.60)

# ----------------------------------------------------------------------------------------------- plug


def build_plug(path: Path, dims: RJ45Dims) -> Path:
    stage, root = _new_stage(path, "Plug")
    d = dims
    _set_meta(
        root,
        {
            "frame": "origin at plug tip centre, +X insertion, +Z latch side",
            "latch_k_theta_Nm_per_rad": d.latch_k_theta,
            "latch_rest_deg": d.latch_rest_deg,
        },
    )

    body = _xform(stage, "/Plug/body")
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(d.plug_mass)
    hw, hh = d.plug_width / 2, d.plug_body_height / 2
    _box(stage, "/Plug/body/geom", (-d.plug_length, -hw, -hh), (0.0, hw, hh), PLUG_COLOR)
    _xform(stage, "/Plug/body/tip_frame")
    _xform(stage, "/Plug/body/attach_frame", (-d.plug_length, 0.0, 0.0))  # cable end; fixed joint to the gripper

    # Latch link frame: origin at the pivot, latch along -X, bottom face on the pivot line; rest = +rest_deg about Y.
    pivot = Gf.Vec3d(-d.latch_root_offset, 0.0, hh - d.latch_thickness)
    rest = _quat_y(d.latch_rest_deg)
    latch = _xform(stage, "/Plug/latch", pivot, rest)
    UsdPhysics.RigidBodyAPI.Apply(latch)
    UsdPhysics.MassAPI.Apply(latch).CreateMassAttr(d.latch_mass)
    lw = d.latch_width / 2
    _box(stage, "/Plug/latch/geom", (-d.latch_length, -lw, 0.0), (0.0, lw, d.latch_thickness), LATCH_COLOR)

    # Joint angle 0 = rest (joint frame on the body is pre-rotated by the rest angle); + = latch tip up.
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Plug/latch_joint")
    joint.CreateBody0Rel().SetTargets([body.GetPath()])
    joint.CreateBody1Rel().SetTargets([latch.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(pivot))
    joint.CreateLocalRot0Attr(Gf.Quatf(rest))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    joint.CreateAxisAttr(UsdGeom.Tokens.y)
    joint.CreateLowerLimitAttr(d.latch_lower_deg)
    joint.CreateUpperLimitAttr(d.latch_upper_deg)
    # USD angular drive gains are per degree: k[N*m/deg] = k[N*m/rad] * pi/180.
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateStiffnessAttr(d.latch_k_theta * math.pi / 180.0)
    drive.CreateDampingAttr(d.latch_damping * math.pi / 180.0)
    _set_meta(joint.GetPrim(), {"rest_deg_from_body_axis": d.latch_rest_deg, "zero": "rest"})

    stage.GetRootLayer().Save()
    return path


# ----------------------------------------------------------------------------------------------- panel


def _build_jack(stage, port: str, d: RJ45Dims, g: JackGeometry) -> None:
    """Jack colliders under ``port`` in the jack frame (latch side +Z)."""
    hw, hh = g.width / 2, g.height / 2
    ow = g.outer_width / 2
    ll, lw = g.lip_start, d.lead_in_width
    zt, zb = g.z_top_outer, g.z_bottom_outer
    sd, sw = g.seat_depth, g.slot_width / 2

    _box(stage, f"{port}/wall_left", (ll, hw, zb), (sd, ow, zt), JACK_COLOR)
    _box(stage, f"{port}/wall_right", (ll, -ow, zb), (sd, -hw, zt), JACK_COLOR)
    _box(stage, f"{port}/wall_bottom", (ll, -hw, zb), (sd, hw, -hh), JACK_COLOR)
    # Ceiling: shoulders either side of the latch slot hold the body; the slot is closed by lip + ledge at the mouth.
    _box(stage, f"{port}/shoulder_left", (ll, sw, hh), (sd, hw, zt), JACK_COLOR)
    _box(stage, f"{port}/shoulder_right", (ll, -hw, hh), (sd, -sw, zt), JACK_COLOR)
    _box(stage, f"{port}/top_lip", (ll, -sw, hh), (g.catch_start, sw, zt), JACK_COLOR)
    _box(stage, f"{port}/catch_ledge", (g.catch_start, -sw, hh), (g.catch_depth, sw, zt), LEDGE_COLOR)
    _box(stage, f"{port}/pocket_roof", (g.catch_depth, -sw, hh + g.pocket_height), (sd, sw, zt), JACK_COLOR)
    _box(stage, f"{port}/back_stop", (sd, -ow, zb), (sd + d.back_wall, ow, zt), JACK_COLOR)

    # Lead-in wedges (x in [0, lead-in length]): chamfer from (opening + lead_in_width) at the mouth to the opening.
    _prism(stage, f"{port}/lead_in_top", [(0, hh + lw), (ll, hh), (ll, zt), (0, zt)], 1, (-ow, ow), JACK_COLOR)
    _prism(stage, f"{port}/lead_in_bottom", [(0, -hh - lw), (0, zb), (ll, zb), (ll, -hh)], 1, (-ow, ow), JACK_COLOR)
    # axis z: quad points are (x, y)
    _prism(stage, f"{port}/lead_in_left", [(0, hw + lw), (0, ow), (ll, ow), (ll, hw)], 2, (-hh, hh), JACK_COLOR)
    _prism(stage, f"{port}/lead_in_right", [(0, -hw - lw), (ll, -hw), (ll, -ow), (0, -ow)], 2, (-hh, hh), JACK_COLOR)

    _xform(stage, f"{port}/mouth")
    _xform(stage, f"{port}/seat", (g.seat_depth, 0.0, 0.0))
    _xform(stage, f"{port}/catch", (g.catch_depth, 0.0, hh))


def build_panel(path: Path, dims: RJ45Dims, tier: str, kinematic: bool = True) -> Path:
    g = jack_geometry(dims, tier)
    stage, root = _new_stage(path, "Panel")
    meta = {k: v for k, v in asdict(g).items()}
    meta["frame"] = "origin at plate front face, centre of port_1; ports rotated 180 deg about X (latch side down)"
    _set_meta(root, meta)
    # One rigid body for the whole panel; kinematic so the env can teleport it (pose randomization).
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(kinematic)

    flip = Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))  # 180 deg about X: jack +Z (latch side) -> panel -Z
    port_y = [(1 - i) * dims.pitch for i in range(dims.n_ports)]
    for i, y in enumerate(port_y):
        port = f"/Panel/port_{i}"
        _xform(stage, port, (0.0, y, 0.0), flip)
        _build_jack(stage, port, dims, g)

    # Plate (panel frame): frame around the jack housings. The flip maps jack z in [zb, zt] to [-zt, -zb].
    ow = g.outer_width / 2
    z_lo, z_hi = -g.z_top_outer, -g.z_bottom_outer
    m, t = dims.plate_margin, dims.plate_thickness
    y_min, y_max = port_y[-1] - ow - m, port_y[0] + ow + m
    _box(stage, "/Panel/plate/top", (0, y_min, z_hi), (t, y_max, z_hi + m), PLATE_COLOR)
    _box(stage, "/Panel/plate/bottom", (0, y_min, z_lo - m), (t, y_max, z_lo), PLATE_COLOR)
    edges = [y_max] + [e for y in port_y for e in (y + ow, y - ow)] + [y_min]
    for k in range(0, len(edges), 2):
        hi_, lo_ = edges[k], edges[k + 1]
        if hi_ - lo_ > 1e-5:
            _box(stage, f"/Panel/plate/strip_{k // 2}", (0, lo_, z_lo), (t, hi_, z_hi), PLATE_COLOR)

    stage.GetRootLayer().Save()
    return path


# ----------------------------------------------------------------------------------------------- CLI


def build_all(out_dir: Path = USD_DIR, dims: RJ45Dims = RJ45Dims()) -> dict[str, dict[str, Path]]:
    out = {}
    for tier in TIERS:
        tier_dir = Path(out_dir) / tier
        out[tier] = {
            "plug": build_plug(tier_dir / "plug.usd", dims),
            "panel": build_panel(tier_dir / f"panel_3port_{tier}.usd", dims, tier),
        }
    return out


def summary_table(dims: RJ45Dims = RJ45Dims()) -> str:
    mm = 1e3
    rows = [
        "| tier | clearance/side | opening W x H | total lateral | catch depth | snap depth | seat depth | latched play |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tier in TIERS:
        g = jack_geometry(dims, tier)
        rows.append(
            f"| {tier} | {g.clearance * mm:.3f} | {g.width * mm:.3f} x {g.height * mm:.3f} | "
            f"{2 * g.clearance * mm:.3f} | {g.catch_depth * mm:.2f} | {g.snap_depth * mm:.2f} | "
            f"{g.seat_depth * mm:.2f} | {g.latched_play * mm:.2f} |"
        )
    k = dims.latch_k_theta
    rows += [
        "",
        f"plug {dims.plug_width * mm:.2f} x {dims.plug_body_height * mm:.2f} x {dims.plug_length * mm:.2f} mm, "
        f"{dims.plug_mass * 1e3:.1f} g; body + latch at rest {(dims.plug_body_height + dims.latch_rise) * mm:.2f} mm tall",
        f"latch {dims.latch_length * mm:.1f} x {dims.latch_width * mm:.1f} x {dims.latch_thickness * mm:.1f} mm, "
        f"{dims.latch_mass * 1e3:.1f} g, rest {dims.latch_rest_deg:.1f} deg, limits "
        f"[{dims.latch_lower_deg:+.0f}, {dims.latch_upper_deg:+.0f}] deg from rest",
        f"latch k_theta {k:.4f} N*m/rad = {k * math.pi / 180:.3e} N*m/deg (USD drive), "
        f"damping {dims.latch_damping:.2e} N*m*s/rad (zeta {dims.latch_damping_ratio})",
        f"lead-in {dims.lead_in_width * mm:.2f} mm x {dims.lead_in_angle_deg:.0f} deg "
        f"(axial {dims.lead_in_length * mm:.3f} mm), pitch {dims.pitch * mm:.1f} mm",
    ]
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=USD_DIR)
    args = ap.parse_args()
    for tier, files in build_all(args.out).items():
        for p in files.values():
            print(f"wrote {p}")
    print(summary_table())


if __name__ == "__main__":
    main()
