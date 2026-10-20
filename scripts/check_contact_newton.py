"""Virtual-wrist contact test on Newton (MuJoCo-Warp solver): the PhysX probe of check_contact.py on a second engine.

Standalone Newton, no Isaac Sim / Kit. Same geometry (``build_assets.jack_parts`` / ``plate_parts``, the full 3-port
panel), same plug and latch (masses, beam spring and damping, limits), same compliant 6-DOF wrist (2000 N/m, 5 N*m/rad
drives about the plug tip), same insertion schedule and the same metrics (``patchbench.contact_probe``). Everything
is built in port_1's jack frame (+X insertion, +Z latch side); gravity is +Z there because the panel mounts the
latch side down (check_contact.py flips the port by 180 deg about X).

MuJoCo-Warp needs one model per tier (worlds must share their geometry), so each tier is its own model with
offsets x tilts worlds. Contacts are MuJoCo's own (``use_mujoco_contacts``): soft constraints set per geom by
``solref`` (time constant, damping ratio) and ``solimp``; box margins are forced to 0 by Newton for native CCD,
so there is no equivalent of PhysX's contact offset. One probe step (``--dt``, 1/480 as PhysX) is ``--substeps``
solver steps.

    python scripts/check_contact_newton.py --substeps 4 --solref 0.004,1 --run_id nw_s4_tc4
Writes results/newton_vs_physx/<run_id>/{summary.json, traces.csv.gz, force_vs_depth.png, jam_map.png}.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

import newton
from newton.sensors import SensorContact
from patchbench import physics
from patchbench.assets.build_assets import RJ45Dims, jack_geometry, jack_parts, plate_parts, prism_points
from patchbench.contact_probe import CHAIN, D_LIN, D_ROT, DIMS, K_LIN, K_ROT, WRIST_LIMITS, Probe

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dt", type=float, default=physics.PHYSICS_DT, help="probe step [s] (PhysX physics dt)")
parser.add_argument("--substeps", type=int, default=4, help="MuJoCo steps per probe step")
parser.add_argument("--solref", default="auto", help="geom solref 'timeconst,dampratio'; auto = 2 x substep dt, 1.0 "
                    "(MuJoCo's stiffest stable setting)")
parser.add_argument("--solimp", default="0.99,0.999,0.0002,0.5,2.0",
                    help="geom solimp; Arena usbc_insertion uses 0.95,0.999,0.0005,0.5,2 (too soft for the 0.3 g latch)")
parser.add_argument("--iterations", type=int, default=100, help="MuJoCo solver iterations (MuJoCo default 100)")
parser.add_argument("--ls_iterations", type=int, default=50)
parser.add_argument("--njmax", type=int, default=400, help="constraint rows per world (overflow drops contacts)")
parser.add_argument("--nconmax", type=int, default=128, help="contacts per world")
parser.add_argument("--cone", default="elliptic", choices=["pyramidal", "elliptic"])
parser.add_argument("--impratio", type=float, default=1.0)
parser.add_argument("--condim", type=int, default=3)
parser.add_argument("--multiccd", action="store_true", help="up to 4 contacts per geom pair (box-box)")
parser.add_argument("--contacts", default="mujoco", choices=["mujoco", "newton"],
                    help="mujoco: MuJoCo-Warp collision (one point per box-mesh pair); newton: Newton's collision "
                    "pipeline (contact manifolds) fed to the MuJoCo-Warp solver")
parser.add_argument("--gap", type=float, default=0.5e-3, help="m, contact detection gap (newton contacts only; "
                    "PhysX contact offset analogue)")
parser.add_argument("--latch_armature", type=float, default=0.0, help="kg*m^2 added to the latch joint")
parser.add_argument("--tiers", default="loose,medium")
parser.add_argument("--offsets_mm", default="0,0.2,0.5,1.0")
parser.add_argument("--tilts_deg", default="0,1,2")
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--friction", type=float, default=physics.CONTACT_FRICTION)
parser.add_argument("--no_graph", action="store_true", help="do not capture the step in a CUDA graph")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--run_id", default=None)
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "newton_vs_physx")
args = parser.parse_args()

SOLREF = (2 * args.dt / args.substeps, 1.0) if args.solref == "auto" else tuple(float(v) for v in args.solref.split(","))
SOLIMP = tuple(float(v) for v in args.solimp.split(","))
RUN_ID = args.run_id or (("nwc_" if args.contacts == "newton" else "") + f"nw_s{args.substeps}_tc{SOLREF[0] * 1e3:.2g}_d{SOLIMP[0]:g}_{args.cone[:3]}"
                         + ("_mccd" if args.multiccd else ""))
GRAVITY_JACK = (0.0, 0.0, 9.81)  # world -Z seen from the flipped port_1 frame
SOLREF_MODE_RAW = 1  # newton/_src/solvers/mujoco/constants.py


def _shape_cfg(density: float) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=density, mu=args.friction, restitution=0.0, mu_torsional=0.0, mu_rolling=0.0, margin=0.0,
        gap=args.gap if args.contacts == "newton" else 0.0,
    )


def _mj_attrs() -> dict:
    return {
        "mujoco:solref": wp.vec2(*SOLREF),
        "mujoco:solref_mode": SOLREF_MODE_RAW,
        "mujoco:geom_solimp": SOLIMP,
        "mujoco:condim": args.condim,
    }


def _add_box(b: newton.ModelBuilder, body: int, lo, hi, cfg, label: str) -> int:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    return b.add_shape_box(body, xform=wp.transform(wp.vec3(*c), wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2],
                           cfg=cfg, label=label, custom_attributes=_mj_attrs())  # fmt: skip


_PRISM_FACES = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]  # build_assets._prism


def _add_prism(b: newton.ModelBuilder, pts, cfg, label: str) -> int:
    tris = []
    for f in range(6):
        a, bb, c, d = _PRISM_FACES[4 * f : 4 * f + 4]
        tris += [a, bb, c, a, c, d]
    mesh = newton.Mesh(np.asarray(pts, np.float32), np.asarray(tris, np.int32))
    return b.add_shape_convex_hull(-1, mesh=mesh, cfg=cfg, label=label, custom_attributes=_mj_attrs())


def build_world(tier: str) -> tuple[newton.ModelBuilder, dict]:
    """One world: static 3-port panel + wrist chain + plug + latch, in port_1's jack frame."""
    d: RJ45Dims = DIMS
    g = jack_geometry(d, tier)
    b = newton.ModelBuilder(gravity=GRAVITY_JACK)
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    static = _shape_cfg(0.0)
    # Panel. Port i sits at panel y_i with the same flip as port_1, so in port_1's jack frame it is a pure shift
    # by -y_i; plate parts are in the panel frame: (x, y, z) -> (x, -y, -z).
    for i in range(d.n_ports):
        dy = -(1 - i) * d.pitch
        for kind, name, *spec in jack_parts(d, g):
            if kind == "box":
                lo, hi = np.array(spec[0], float), np.array(spec[1], float)
                lo[1] += dy
                hi[1] += dy
                _add_box(b, -1, lo, hi, static, f"port_{i}/{name}")
            else:
                pts = [(x, y + dy, z) for x, y, z in prism_points(*spec[:3])]
                _add_prism(b, pts, static, f"port_{i}/{name}")
    for _, name, lo, hi, _c in plate_parts(d, g):
        _add_box(b, -1, (lo[0], -hi[1], -hi[2]), (hi[0], -lo[1], -lo[2]), static, f"plate/{name}")

    # Wrist chain: x, y, z prismatic then rx, ry, rz revolute, all about the plug tip (check_contact.py rig).
    small = wp.mat33(1e-7, 0.0, 0.0, 0.0, 1e-7, 0.0, 0.0, 0.0, 1e-7)
    joints, parent = [], -1
    hw, hh = d.plug_width / 2, d.plug_body_height / 2
    for k, n in enumerate(CHAIN):
        if k < 5:
            child = b.add_link(mass=1e-3, inertia=small, label=f"l_{n}", lock_inertia=True)
        else:
            child = b.add_link(label="plug_body")
            vol = d.plug_length * d.plug_width * d.plug_body_height
            _add_box(b, child, (-d.plug_length, -hw, -hh), (0.0, hw, hh), _shape_cfg(d.plug_mass / vol), "plug_geom")
        axis = "XYZ"[k % 3]
        if k < 3:
            lim = WRIST_LIMITS["x"] if k == 0 else WRIST_LIMITS[n]
            j = b.add_joint_prismatic(parent, child, axis=axis, target_ke=K_LIN, target_kd=D_LIN,
                                      limit_lower=lim[0], limit_upper=lim[1], label=n)  # fmt: skip
        else:
            lim = [math.radians(v) for v in WRIST_LIMITS["rot_deg"]]
            j = b.add_joint_revolute(parent, child, axis=axis, target_ke=K_ROT, target_kd=D_ROT,
                                     limit_lower=lim[0], limit_upper=lim[1], label=n)  # fmt: skip
        joints.append(j)
        parent = child
    plug = parent

    # Latch: pivot latch_root_offset behind the tip, latch_thickness below the body top, rest angle about +Y.
    a = math.radians(d.latch_rest_deg)
    rest = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), a)
    pivot = wp.vec3(-d.latch_root_offset, 0.0, hh - d.latch_thickness)
    latch = b.add_link(label="latch")
    vol = d.latch_length * d.latch_width * d.latch_thickness
    lw = d.latch_width / 2
    _add_box(b, latch, (-d.latch_length, -lw, 0.0), (0.0, lw, d.latch_thickness), _shape_cfg(d.latch_mass / vol),
             "latch_geom")  # fmt: skip
    j = b.add_joint_revolute(plug, latch, parent_xform=wp.transform(pivot, rest), child_xform=wp.transform_identity(),
                             axis="Y", target_pos=0.0, target_ke=d.latch_k_theta, target_kd=d.latch_damping,
                             limit_lower=math.radians(d.latch_lower_deg), limit_upper=math.radians(d.latch_upper_deg),
                             armature=args.latch_armature, label="latch_joint")  # fmt: skip
    joints.append(j)
    b.add_articulation(joints, label="rig")
    return b, {"plug": plug, "latch": latch}


def run_tier(tier: str, full: Probe) -> tuple[Probe, float, int, dict]:
    """Run all seeds of one tier. Initial offsets/tilts come from ``full`` (all tiers, like the PhysX run) so each
    variant gets the same random direction per seed on both engines."""
    world, ids = build_world(tier)
    probe = Probe([tier], full.offsets, full.tilts, args.dt, full.dev)
    rows = [i for i, (t, _, _) in enumerate(full.variants) if t == tier]
    n = probe.n
    main = newton.ModelBuilder(gravity=GRAVITY_JACK)
    newton.solvers.SolverMuJoCo.register_custom_attributes(main)
    main.replicate(world, n)
    model = main.finalize(device=args.device)
    ndof = model.joint_dof_count // n
    assert ndof == 7, ndof  # x y z rx ry rz latch
    bodies_per_world = model.body_count // n
    sense = [w * bodies_per_world + ids[k] for w in range(n) for k in ("plug", "latch")]
    sensor = SensorContact(model, sensing_bodies=sense, measure_total=True)
    solver = newton.solvers.SolverMuJoCo(
        model, iterations=args.iterations, ls_iterations=args.ls_iterations, cone=args.cone, impratio=args.impratio,
        enable_multiccd=args.multiccd, use_mujoco_contacts=args.contacts == "mujoco", njmax=args.njmax,
        nconmax=args.nconmax,
    )  # fmt: skip
    # Sensor buffer: MuJoCo's active contacts with their solved forces (update_contacts), for either collision path.
    contacts = newton.Contacts(solver.get_max_contact_count(), 0,
                               requested_attributes=model.get_requested_contact_attributes())  # fmt: skip
    pipeline = newton.CollisionPipeline(model) if args.contacts == "newton" else None
    nw_contacts = pipeline.contacts() if pipeline else None
    state, control = model.state(), model.control()
    q_t = wp.to_torch(state.joint_q).view(n, ndof)
    qd_t = wp.to_torch(state.joint_qd).view(n, ndof)
    tgt_t = wp.to_torch(control.joint_target_q).view(n, ndof)
    sub_dt = args.dt / args.substeps

    def simulate():
        for _ in range(args.substeps):
            if pipeline:
                pipeline.collide(state, nw_contacts)
            solver.step(state, state, control, nw_contacts, sub_dt)
        solver.update_contacts(contacts, state)
        sensor.update(state, contacts)

    def reset(q0):
        q_t.copy_(q0)
        qd_t.zero_()
        tgt_t.copy_(q0)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    t_wall, steps, graph = 0.0, 0, None
    for seed in range(args.seeds):
        init = full.initial_targets(seed)
        q0 = torch.stack([init[c][rows] for c in CHAIN] + [torch.zeros(n, device=full.dev)], 1).to(q_t.device)
        if graph is None and not args.no_graph:
            reset(q0)
            simulate()  # compile kernels outside the capture
            with wp.ScopedCapture(device=args.device) as cap:
                simulate()
            graph = cap.graph
        reset(q0)
        solver.reset(state)  # clears MuJoCo warm-start buffers; keeps state.joint_q
        reset(q0)
        probe.begin(seed)
        wp.synchronize()
        t0 = time.perf_counter()
        for k in range(probe.n_steps):
            tgt_t[:, 0] = probe.x_target(k * args.dt)
            if graph is not None:
                wp.capture_launch(graph)
            else:
                simulate()
            f = wp.to_torch(sensor.total_force).view(n, 2, 3).sum(1)  # on plug body + latch, jack frame
            q = q_t.clone()
            probe.step(k, {c: q[:, i] for i, c in enumerate(CHAIN)}, q[:, 6], qd_t[:, 0].clone(), f)
        wp.synchronize()
        t_wall += time.perf_counter() - t0
        steps += probe.n_steps
        nl = probe.end(q_t[:, 0].clone())
        print(f"[newton] {tier} seed {seed}: latched {nl}/{n}, exploded {int(probe.exploded.sum())}", flush=True)
    info = {"njmax": int(solver.mjw_data.njmax), "naconmax": int(solver.mjw_data.naconmax),
            "shapes_per_world": int(model.shape_count) // n}  # fmt: skip
    return probe, t_wall, steps, info


def main():
    tiers = args.tiers.split(",")
    offsets = [float(x) * 1e-3 for x in args.offsets_mm.split(",")]
    tilts = [math.radians(float(x)) for x in args.tilts_deg.split(",")]
    full = Probe(tiers, offsets, tilts, args.dt, args.device)
    t_wall, steps, infos = 0.0, 0, {}
    for tier in tiers:
        tp, tw, st, info = run_tier(tier, full)
        full.rows += tp.rows
        full.results += tp.results
        t_wall += tw
        steps += st
        infos[tier] = info | {"wall_s_per_sim_s": tw / (st * args.dt), "num_worlds": tp.n}
    config = {"engine": "newton", "newton": newton.__version__, "warp": wp.__version__, "solver": "SolverMuJoCo",
              "dt": args.dt, "substeps": args.substeps, "solref": SOLREF, "solimp": SOLIMP,
              "iterations": args.iterations, "ls_iterations": args.ls_iterations, "cone": args.cone,
              "impratio": args.impratio, "condim": args.condim, "multiccd": args.multiccd,
              "latch_armature": args.latch_armature, "contacts": args.contacts,
              "gap_m": args.gap if args.contacts == "newton" else 0.0, "friction": args.friction, "cuda_graph": not args.no_graph}  # fmt: skip
    # wall time per simulated second for one model holding every tier's worlds is not measurable on MuJoCo-Warp;
    # report the sum over tiers (sequential models), i.e. comparable to PhysX's single all-tier scene.
    full.save(args.out / RUN_ID, RUN_ID, config, steps / max(t_wall, 1e-9), args.seeds,
              {"wall_s_per_sim_s": t_wall / (steps / len(tiers) * args.dt), "per_tier": infos})  # fmt: skip


if __name__ == "__main__":
    main()
