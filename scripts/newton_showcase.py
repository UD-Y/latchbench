"""Newton showcase: the SO-101 inserts the RJ45 plug while a physical Cat6 cable (VBD rod) hangs from it.

Newton only (no Isaac Sim). Two Newton models stepped together:
  * rigid: MuJoCo Menagerie ``robotstudio_so101`` (same kinematics as isaaclab-so101; checked by FK on random poses),
    our plug + spring latch welded to the gripper and our 3-port panel, solved by MuJoCo-Warp with Newton's collision
    pipeline and the contact settings validated in results/newton_vs_physx (``impratio`` 1000, 8 substeps per 1/480 s).
  * cable: a 0.6 m Cat6 rod (6 mm, 40 g/m, bending stiffness EI = 5 mN*m^2) of capsule bodies and cable joints, solved
    by VBD, lying on the table. Its first segments are kinematic, carried by the plug through the jaws, so the cable
    follows the plug (one-way coupling: the cable does not pull on the arm).

The arm replays the joint positions of a PhysX expert episode recorded by gen_data.py (30 Hz, interpolated); the panel
is placed where that trajectory seats the plug (FK of the final frame), so the insertion is the expert's own.

    python scripts/newton_showcase.py --out results/newton_showcase/sim_loose_port1.npz
Writes the per-frame poses (30 Hz) consumed by scripts/showcase_usd.py.
"""

from __future__ import annotations

import argparse
import collections
import math
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation as Rot

import newton
from newton._src.utils.download_assets import MENAGERIE_REF, MENAGERIE_URL, download_git_folder
from latchbench import physics
from latchbench.contact_probe import DIMS
from latchbench.controllers.ik import SO101Kinematics
from latchbench.newton_rj45 import ContactCfg, add_panel, add_plug, extend_last_articulation
from latchbench.objects import PLUG_IN_HAND_ROT_XYZW, plug_in_hand_pos

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--traj", type=Path, default=ROOT / "results/newton_showcase/expert_traj_loose_port1.npz")
ap.add_argument("--tier", default="loose")
ap.add_argument("--settle_s", type=float, default=1.0, help="hold the first pose while the cable settles")
ap.add_argument("--hold_s", type=float, default=1.5, help="hold the last pose")
ap.add_argument("--final_depth_mm", type=float, default=17.5,
                help="target plug tip depth at the last frame: 1.5 mm past the seat (16.0), as the PhysX expert pushes")
ap.add_argument("--arm_kp", type=float, default=30.0, help="N*m/rad, compliant position drives (~300 N/m at the plug)")
ap.add_argument("--arm_kd", type=float, default=1.0, help="N*m*s/rad")
ap.add_argument("--cable_len", type=float, default=0.6)
ap.add_argument("--seg_len", type=float, default=0.01)
ap.add_argument("--kinematic_len", type=float, default=0.035, help="cable carried by the plug (boot end -> jaws)")
ap.add_argument("--bend_EI", type=float, default=5e-3, help="N*m^2, Cat6 bending stiffness")
ap.add_argument("--vbd_steps", type=int, default=2, help="VBD steps per 1/480 s physics step")
ap.add_argument("--max_frames", type=int, default=None)
ap.add_argument("--debug_contacts_at", type=int, default=None, help="print rigid contact pairs at this frame")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", type=Path, default=ROOT / "results/newton_showcase/sim_loose_port1.npz")
args = ap.parse_args()

FPS, PHYS_DT, SUBSTEPS = 30, physics.PHYSICS_DT, 8
STEPS_PER_FRAME = round(1 / (FPS * PHYS_DT))  # 16
BOOT_END = (-DIMS.plug_length - 0.014, 0.0, 0.0)  # plug frame: end of the look layer's boot


def so101_xml() -> Path:
    return download_git_folder(MENAGERIE_URL, "robotstudio_so101", ref=MENAGERIE_REF) / "so101.xml"


def T(R=None, p=None) -> np.ndarray:
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    if p is not None:
        M[:3, 3] = p
    return M


def to_wp(M: np.ndarray) -> wp.transform:
    q = Rot.from_matrix(M[:3, :3]).as_quat()  # xyzw
    return wp.transform(wp.vec3(*M[:3, 3]), wp.quat(*q))


def from_xform(x) -> np.ndarray:
    x = np.asarray(x, float)
    return T(Rot.from_quat(x[3:7]).as_matrix(), x[:3])


def gripper_tool_offset(xml: Path, grip_label: str) -> np.ndarray:
    """Constant transform G with T_ours(q) = B @ T_newton(q) @ G (gripper link frames of isaaclab-so101's URDF and
    Menagerie's MJCF). Rotation from the axes of relative motions (Kabsch), translation checked ~0."""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(str(xml), floating=False)
    m = b.finalize(device="cpu")
    gi = [i for i, lab in enumerate(m.body_label) if lab.endswith(grip_label)][0]
    kin = SO101Kinematics(dtype=torch.float64)

    def t_nw(q):
        st = m.state()
        jq = m.joint_q.numpy().copy()
        jq[:5] = q
        st.joint_q.assign(jq)
        newton.eval_fk(m, st.joint_q, st.joint_qd, st)
        return from_xform(st.body_q.numpy()[gi])

    def t_us(q):
        p, R, _, _ = kin.forward(torch.tensor(q)[None])
        return T(R[0].numpy(), p[0].numpy())

    rng = np.random.default_rng(1)
    Q = rng.uniform(-1.0, 1.0, (12, 5))
    a_us, a_nw = [], []
    for i in range(len(Q) - 1):
        ru = np.linalg.inv(t_us(Q[i])) @ t_us(Q[i + 1])
        rn = np.linalg.inv(t_nw(Q[i])) @ t_nw(Q[i + 1])
        a_us.append(Rot.from_matrix(ru[:3, :3]).as_rotvec())
        a_nw.append(Rot.from_matrix(rn[:3, :3]).as_rotvec())
    # rotvec_nw = R_G @ rotvec_us  (relative motions conjugated by G)
    R_G, rmsd = Rot.align_vectors(np.array(a_nw), np.array(a_us))
    G = T(R_G.as_matrix())
    # verify: with G, relative motions agree including translation
    err = 0.0
    for i in range(len(Q) - 1):
        ru = np.linalg.inv(t_us(Q[i])) @ t_us(Q[i + 1])
        rn = np.linalg.inv(G) @ np.linalg.inv(t_nw(Q[i])) @ t_nw(Q[i + 1]) @ G
        err = max(err, np.abs(ru - rn).max())
    print(f"[showcase] gripper frame offset: rot {np.degrees(np.linalg.norm(R_G.as_rotvec())):.2f} deg, "
          f"max relative-motion mismatch {err:.2e}")  # fmt: skip
    assert err < 1e-4, "isaaclab-so101 and Menagerie SO-101 kinematics differ"
    B = t_us(Q[0]) @ np.linalg.inv(G) @ np.linalg.inv(t_nw(Q[0]))
    return G, B


def plan_targets(q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """30 Hz joint targets [rad] (F, 6) and the jack_1 pose (our kinematic frame).

    The jack is where the recorded PhysX episode ended (plug tip at 15.9 mm, i.e. seated), oriented like the plug
    there. Targets: settle on the first pose; the recorded expert frames until the plug tip is 15 mm in front of the
    mouth; 0.5 s to line up on the jack axis; a straight insertion at 10 mm/s (our open-loop controller's motion) to
    ``--final_depth_mm``; hold. The straight segment is IK on the shared SO-101 kinematics: replaying the recorded
    joints instead reproduces the PhysX path's servo-sag tilt, which a position-controlled arm cannot correct."""
    from latchbench.controllers.ik import IKCfg, solve_ik

    kin = SO101Kinematics()  # float32, as the IK
    F32 = torch.float32
    tool_p = torch.tensor(plug_in_hand_pos(), dtype=F32)
    tool_R = torch.tensor(Rot.from_quat(PLUG_IN_HAND_ROT_XYZW).as_matrix(), dtype=F32)
    q = torch.tensor(np.radians(q_deg[:, :5]), dtype=F32)
    p, R, _, _ = kin.forward(q, tool_p, tool_R)
    p_f, R_f = p[-1].double().numpy(), R[-1].double().numpy()
    x = R_f[:, 0]
    mouth = p_f - 15.9e-3 * x
    depth = (p.double().numpy() - mouth) @ x
    k_pre = int(np.nonzero(depth <= -15e-3)[0][-1]) if (depth <= -15e-3).any() else 0
    jaw = np.radians(q_deg[-1, 5])
    out = [np.radians(q_deg[0])] * round(args.settle_s * FPS) + [np.radians(r) for r in q_deg[: k_pre + 1]]
    cfg = IKCfg(iterations=30)
    qk = q[k_pre : k_pre + 1].clone()
    p0, R0 = p[k_pre].double().numpy(), R[k_pre].double().numpy()
    pre = mouth - 15e-3 * x
    n_align = round(0.5 * FPS)
    for i in range(1, n_align + 1):  # line up: position and orientation slerp to the axis
        a = i / n_align
        Rt = (Rot.from_matrix(R0) * Rot.from_rotvec(a * (Rot.from_matrix(R0).inv() * Rot.from_matrix(R_f)).as_rotvec()))
        tgt_p = torch.tensor((1 - a) * p0 + a * pre, dtype=F32)[None]
        qk = solve_ik(kin, qk, tgt_p, torch.tensor(Rt.as_matrix(), dtype=F32)[None], tool_p, tool_R, cfg)
        out.append(np.r_[qk[0].double().numpy(), jaw])
    n_ins = round((15e-3 + args.final_depth_mm * 1e-3) / 0.010 * FPS)
    for i in range(1, n_ins + 1):  # straight insertion at 10 mm/s
        tgt_p = torch.tensor(pre + x * (i / n_ins) * (15e-3 + args.final_depth_mm * 1e-3), dtype=F32)[None]
        qk = solve_ik(kin, qk, tgt_p, torch.tensor(R_f, dtype=F32)[None], tool_p, tool_R, cfg)
        out.append(np.r_[qk[0].double().numpy(), jaw])
    out += [out[-1]] * round(args.hold_s * FPS)
    pe, Re, _, _ = kin.forward(qk, tool_p, tool_R)
    print(f"[showcase] plan: {len(out)} frames, recorded up to frame {k_pre} (tip {depth[k_pre] * 1e3:.1f} mm), "
          f"final IK error {np.linalg.norm(pe[0].double().numpy() - tgt_p[0].double().numpy()) * 1e3:.2f} mm "
          f"{np.degrees(Rot.from_matrix(Re[0].double().numpy().T @ R_f).magnitude()):.2f} deg")  # fmt: skip
    return np.array(out), T(R_f, mouth), k_pre


@wp.kernel
def sync_anchors(robot_q: wp.array[wp.transform], plug: int, ids: wp.array[int], rel: wp.array[wp.transform],
                 cable_q: wp.array[wp.transform], cable_qd: wp.array[wp.spatial_vector]):  # fmt: skip
    i = wp.tid()
    cable_q[ids[i]] = wp.transform_multiply(robot_q[plug], rel[i])
    cable_qd[ids[i]] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def main():
    traj = np.load(args.traj)
    q_deg = traj["state_deg"]  # (F, 6) shoulder_pan..wrist_roll, gripper; isaaclab joint conventions = Menagerie's
    xml = so101_xml()
    G, B = gripper_tool_offset(xml, "/gripper")
    targets, jack_us, n_rec = plan_targets(q_deg)

    # ---------------------------------------------------------------- rigid model (MuJoCo-Warp)
    cc = ContactCfg(substep_dt=PHYS_DT / SUBSTEPS)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.rigid_gap = 1e-3  # contact detection gap for the arm's MJCF shapes (default would add far-field contacts)
    # Mount Menagerie's base at B: Newton's world is then the PhysX env frame (isaaclab-so101 base on the table, z = 0)
    b.add_mjcf(str(xml), xform=to_wp(B), floating=False, enable_self_collisions=False)
    grip = [i for i, lab in enumerate(b.body_label) if lab.endswith("/gripper")][0]
    in_hand = T(Rot.from_quat(PLUG_IN_HAND_ROT_XYZW).as_matrix(), plug_in_hand_pos())  # isaaclab gripper frame
    in_hand_nw = G @ in_hand
    ids = add_plug(b, grip, to_wp(in_hand_nw), cc)
    extend_last_articulation(b, ids["joints"])

    # panel: jack_1 frame from plan_targets; our kinematic frame is Newton's world (robot mounted at B)
    jack1 = jack_us
    panel_w = jack1 @ T(Rot.from_quat([1.0, 0.0, 0.0, 0.0]).as_matrix())  # jack1 = panel @ FLIP, FLIP self-inverse
    add_panel(b, args.tier, to_wp(panel_w), cc)  # no ground plane here: the arm never touches the table
    model = b.finalize(device=args.device)
    ndof = model.joint_dof_count
    assert ndof == 7, ndof  # 6 arm/jaw + latch
    # Compliant arm: the PhysX servos (LeRobot P = 16) let the chamfer steer a slightly offset plug; Menagerie's stiff
    # drives (kp 998) cannot, so the replayed path would jam at the lip. Gravity compensation keeps soft drives on path.
    ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
    ke[:5], kd[:5] = args.arm_kp, args.arm_kd
    model.joint_target_ke.assign(ke)
    model.joint_target_kd.assign(kd)
    model.mujoco.gravcomp.assign(np.ones(model.body_count, np.float32))
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, impratio=1000.0, cone="elliptic",
                                         iterations=100, ls_iterations=50, njmax=2000, nconmax=1000)  # fmt: skip
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state, control = model.state(), model.control()
    q0 = np.concatenate([targets[0], [0.0]])
    state.joint_q.assign(q0.astype(np.float32))
    control.joint_target_q.assign(q0.astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    plug_w0 = from_xform(state.body_q.numpy()[ids["plug"]])

    # ---------------------------------------------------------------- cable model (VBD)
    c = newton.ModelBuilder()
    newton.solvers.SolverVBD.register_custom_attributes(c)
    c.add_ground_plane()
    x_plug = plug_w0[:3, 0]
    start = (plug_w0 @ np.r_[BOOT_END, 1.0])[:3]
    side = np.cross([0.0, 0.0, 1.0], x_plug)
    side /= np.linalg.norm(side)
    n_kin = round(args.kinematic_len / args.seg_len)
    n_seg = round(args.cable_len / args.seg_len)
    pts = [start - x_plug * args.seg_len * k for k in range(n_kin + 1)]
    for k in range(1, n_seg - n_kin + 1):  # then sideways, horizontal; gravity drapes it during --settle_s
        pts.append(pts[n_kin] + side * args.seg_len * k)
    pts = [wp.vec3(*map(float, p)) for p in pts]
    radius, line_density = 0.003, 0.040  # m, kg/m (Cat6)
    density = line_density / (math.pi * radius**2)
    cable_cfg = newton.ModelBuilder.ShapeConfig(density=density, mu=0.8, ke=1.0e4, kd=0.0, mu_torsional=0.0,
                                                mu_rolling=0.0)  # fmt: skip
    rod, _ = c.add_rod(pts, quaternions=newton.utils.create_parallel_transport_cable_quaternions(pts), radius=radius,
                       cfg=cable_cfg, stretch_stiffness=1.0e5, bend_stiffness=args.bend_EI / args.seg_len,
                       bend_damping=2.0e-4, label="cable", body_frame_origin="com")  # fmt: skip
    kin_ids = rod[:n_kin]
    for i in kin_ids:
        c.body_mass[i] = 0.0
        c.body_inv_mass[i] = 0.0
        c.body_inertia[i] = wp.mat33(0.0)
        c.body_inv_inertia[i] = wp.mat33(0.0)
    c.color()
    cmodel = c.finalize(device=args.device)
    cq0 = cmodel.body_q.numpy()
    rel = [to_wp(np.linalg.inv(plug_w0) @ from_xform(cq0[i])) for i in kin_ids]
    rel_wp = wp.array(rel, dtype=wp.transform, device=args.device)
    kin_wp = wp.array(kin_ids, dtype=int, device=args.device)
    vbd = newton.solvers.SolverVBD(cmodel, iterations=12, rigid_contact_hard=False, rigid_body_contact_buffer_size=256)
    cpipe = newton.CollisionPipeline(cmodel)
    ccontacts = cpipe.contacts()
    cs0, cs1, cctrl = cmodel.state(), cmodel.state(), cmodel.control()

    # ---------------------------------------------------------------- run
    frames = list(range(len(targets)))
    if args.max_frames:
        frames = frames[: args.max_frames]
    tgt = targets
    rec_robot, rec_cable, rec_latch = [], [], []
    t0 = time.perf_counter()
    for f, k in enumerate(frames):
        k1 = frames[min(f + 1, len(frames) - 1)]
        for s in range(STEPS_PER_FRAME):
            a = s / STEPS_PER_FRAME
            q_t = np.concatenate([(1 - a) * tgt[k] + a * tgt[k1], [0.0]]).astype(np.float32)
            control.joint_target_q.assign(q_t)
            for _ in range(SUBSTEPS):
                pipeline.collide(state, contacts)
                solver.step(state, state, control, contacts, PHYS_DT / SUBSTEPS)
            for _ in range(args.vbd_steps):
                wp.launch(sync_anchors, dim=len(kin_ids), inputs=[state.body_q, ids["plug"], kin_wp, rel_wp],
                          outputs=[cs0.body_q, cs0.body_qd], device=args.device)  # fmt: skip
                cpipe.collide(cs0, ccontacts)
                vbd.step(cs0, cs1, cctrl, ccontacts, PHYS_DT / args.vbd_steps)
                cs0, cs1 = cs1, cs0
        if f == args.debug_contacts_at:
            n = int(contacts.rigid_contact_count.numpy()[0])
            s0, s1 = contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n]
            pairs = collections.Counter((model.shape_label[a].split("/")[-1], model.shape_label[b2].split("/")[-1])
                                        for a, b2 in zip(s0, s1, strict=True))  # fmt: skip
            print(f"[showcase] frame {f}: {n} contacts", pairs.most_common(15))
        rec_robot.append(state.body_q.numpy().copy())
        rec_cable.append(cs0.body_q.numpy().copy())
        rec_latch.append(float(state.joint_q.numpy()[6]))
        if f % 30 == 0:
            plug_tip = from_xform(rec_robot[-1][ids["plug"]])
            depth = float((np.linalg.inv(jack1) @ np.r_[plug_tip[:3, 3], 1.0])[0]) * 1e3
            zmin = rec_cable[-1][:, 2].min()
            print(f"[showcase] frame {f}/{len(frames)} depth {depth:6.2f} mm latch "
                  f"{math.degrees(rec_latch[-1]):6.2f} deg cable z_min {zmin * 1e3:6.1f} mm "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)  # fmt: skip
    rb = np.array(rec_robot)
    if not np.isfinite(rb).all() or not np.isfinite(np.array(rec_cable)).all():
        raise RuntimeError("non-finite state")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, fps=FPS, robot_body_q=rb, cable_body_q=np.array(rec_cable), latch=np.array(rec_latch),
             robot_body_label=np.array(model.body_label), plug_body=ids["plug"], latch_body=ids["latch"],
             panel_w=panel_w, tier=args.tier, cable_radius=radius, seg_len=args.seg_len, n_kinematic=n_kin,
             targets=targets, n_recorded=n_rec, so101_xml=str(xml))  # fmt: skip
    print(f"[showcase] wrote {args.out}: {len(frames)} frames in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
