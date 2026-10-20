"""Newton builder helpers for the RJ45 panel and plug (no Isaac Sim): same geometry, masses, latch spring and contact
settings as ``scripts/check_contact_newton.py``'s validated configuration (Newton collision pipeline + MuJoCo-Warp,
``impratio`` 1000; results/newton_vs_physx/REPORT.md), placed at arbitrary world poses.

Frames as in build_assets: plug frame at the tip centre (+X insertion, +Z latch side); panel frame at the plate front
face, centre of port_1; each port is its jack frame rotated 180 deg about X (latch side down).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
from patchbench import physics
from patchbench.assets.build_assets import RJ45Dims, jack_geometry, jack_parts, plate_parts, prism_points

DIMS = RJ45Dims()
SOLREF_MODE_RAW = 1  # newton/_src/solvers/mujoco/constants.py
FLIP = wp.quat(1.0, 0.0, 0.0, 0.0)  # 180 deg about X (xyzw)
_PRISM_FACES = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]  # build_assets._prism


@dataclass
class ContactCfg:
    """Validated Newton contact settings (results/newton_vs_physx: nwc_s8_imp1000)."""

    substep_dt: float = physics.PHYSICS_DT / 8
    solimp: tuple = (0.99, 0.999, 0.0002, 0.5, 2.0)
    friction: float = physics.CONTACT_FRICTION
    gap: float = 0.5e-3
    condim: int = 3

    def shape(self, density: float) -> newton.ModelBuilder.ShapeConfig:
        return newton.ModelBuilder.ShapeConfig(density=density, mu=self.friction, restitution=0.0, mu_torsional=0.0,
                                               mu_rolling=0.0, margin=0.0, gap=self.gap)  # fmt: skip

    def attrs(self) -> dict:
        return {"mujoco:solref": wp.vec2(2 * self.substep_dt, 1.0), "mujoco:solref_mode": SOLREF_MODE_RAW,
                "mujoco:geom_solimp": self.solimp, "mujoco:condim": self.condim}  # fmt: skip


def _tf(p=(0.0, 0.0, 0.0), q=None) -> wp.transform:
    return wp.transform(wp.vec3(*map(float, p)), q if q is not None else wp.quat_identity())


def _box(b, body, parent: wp.transform, lo, hi, cfg, attrs, label):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    xf = wp.transform_multiply(parent, _tf(c))
    return b.add_shape_box(body, xform=xf, hx=h[0], hy=h[1], hz=h[2], cfg=cfg, label=label, custom_attributes=attrs)


def _prism(b, parent: wp.transform, pts, cfg, attrs, label):
    tris = []
    for f in range(6):
        a, bb, c, d = _PRISM_FACES[4 * f : 4 * f + 4]
        tris += [a, bb, c, a, c, d]
    mesh = newton.Mesh(np.asarray(pts, np.float32), np.asarray(tris, np.int32))
    return b.add_shape_convex_hull(-1, xform=parent, mesh=mesh, cfg=cfg, label=label, custom_attributes=attrs)


def add_panel(b: newton.ModelBuilder, tier: str, panel_w: wp.transform, cc: ContactCfg) -> None:
    """Static 3-port panel (all ports + plate) at world pose ``panel_w``."""
    g = jack_geometry(DIMS, tier)
    cfg, attrs = cc.shape(0.0), cc.attrs()
    for i in range(DIMS.n_ports):
        port_w = wp.transform_multiply(panel_w, _tf((0.0, (1 - i) * DIMS.pitch, 0.0), FLIP))
        for kind, name, *spec in jack_parts(DIMS, g):
            if kind == "box":
                _box(b, -1, port_w, spec[0], spec[1], cfg, attrs, f"panel/port_{i}/{name}")
            else:
                _prism(b, port_w, prism_points(*spec[:3]), cfg, attrs, f"panel/port_{i}/{name}")
    for _, name, lo, hi, _c in plate_parts(DIMS, g):
        _box(b, -1, panel_w, lo, hi, cfg, attrs, f"panel/plate/{name}")


def add_plug(b: newton.ModelBuilder, parent: int, parent_to_plug: wp.transform, cc: ContactCfg) -> dict:
    """Plug body welded to ``parent`` at ``parent_to_plug`` (fixed joint) + latch on its spring hinge.
    Returns body and joint ids; the caller adds the joints to an articulation."""
    d = DIMS
    hw, hh = d.plug_width / 2, d.plug_body_height / 2
    attrs = cc.attrs()
    plug = b.add_link(label="plug_body")
    vol = d.plug_length * d.plug_width * d.plug_body_height
    _box(b, plug, _tf(), (-d.plug_length, -hw, -hh), (0.0, hw, hh), cc.shape(d.plug_mass / vol), attrs, "plug_geom")
    j_weld = b.add_joint_fixed(parent, plug, parent_xform=parent_to_plug, child_xform=_tf(), label="plug_weld")
    rest = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(d.latch_rest_deg))
    pivot = (-d.latch_root_offset, 0.0, hh - d.latch_thickness)
    latch = b.add_link(label="latch")
    lw = d.latch_width / 2
    vol = d.latch_length * d.latch_width * d.latch_thickness
    _box(b, latch, _tf(), (-d.latch_length, -lw, 0.0), (0.0, lw, d.latch_thickness),
         cc.shape(d.latch_mass / vol), attrs, "latch_geom")  # fmt: skip
    j_latch = b.add_joint_revolute(plug, latch, parent_xform=_tf(pivot, rest), child_xform=_tf(), axis="Y",
                                   target_pos=0.0, target_ke=d.latch_k_theta, target_kd=d.latch_damping,
                                   limit_lower=math.radians(d.latch_lower_deg),
                                   limit_upper=math.radians(d.latch_upper_deg), label="latch_joint")  # fmt: skip
    return {"plug": plug, "latch": latch, "joints": [j_weld, j_latch]}


def extend_last_articulation(b: newton.ModelBuilder, joints: list[int]) -> None:
    """Append ``joints`` (created right after the last articulation's joints) to that articulation, so a part
    welded to an imported robot (MJCF) is simulated as part of it. ``add_articulation`` cannot extend one."""
    art = b.articulation_count - 1
    assert joints == list(range(b.articulation_end[art], b.articulation_end[art] + len(joints))), "not contiguous"
    for j in joints:
        b.joint_articulation[j] = art
    b.articulation_end[art] += len(joints)
