"""SO-101 FK model and DLS IK (CPU only, no Isaac)."""

import math

import torch

from latchbench.controllers.ik import (
    TCP_OFFSET,
    CartesianServo,
    IKCfg,
    SO101Kinematics,
    dls_step,
    pose_error,
    rot_exp,
    rot_log,
    solve_ik,
)

HOME = torch.tensor([[-0.2736, -0.6109, -0.0745, 1.5148, -1.6034]], dtype=torch.float64)


def test_home_tcp_matches_isaaclab_so101():
    # isaaclab_so101.ee_pose.HOME_NATURAL_POSE (URDF FK at HOME_JOINT_POS): (0.0815, -0.2140, 0.2266)
    kin = SO101Kinematics(dtype=torch.float64)
    p, _, _, _ = kin.forward(HOME, torch.tensor(TCP_OFFSET, dtype=torch.float64))
    assert torch.allclose(p[0], torch.tensor([0.0815, -0.2140, 0.2266], dtype=torch.float64), atol=1e-4)


def test_jacobian_matches_finite_differences():
    kin = SO101Kinematics(dtype=torch.float64)
    q = HOME + 0.1
    tool = torch.tensor(TCP_OFFSET, dtype=torch.float64)
    p, R, a, o = kin.forward(q, tool)
    J = kin.jacobian(p, a, o)
    eps = 1e-6
    for i in range(5):
        dq = torch.zeros_like(q)
        dq[0, i] = eps
        p2, R2, _, _ = kin.forward(q + dq, tool)
        num = torch.cat([(p2 - p) / eps, rot_log(R2 @ R.transpose(-1, -2)) / eps], -1)
        assert torch.allclose(J[0, :, i], num[0], atol=1e-5)


def test_rot_log_exp_roundtrip():
    w = torch.tensor([[0.3, -1.2, 0.5], [1e-9, 0.0, 0.0], [0.0, 2.5, 0.0]], dtype=torch.float64)
    assert torch.allclose(rot_log(rot_exp(w)), w, atol=1e-6)


def test_ik_converges_to_reachable_poses():
    """Targets = FK of random joint vectors near the working pose; solve from home (batched)."""
    torch.manual_seed(0)
    kin = SO101Kinematics()
    E = 64
    q_true = kin.clamp(HOME.float() + 0.4 * (torch.rand(E, 5) - 0.5) - torch.tensor([0, 0, 0, 0.15, 0]))
    tool_p = torch.tensor([0.0, 0.0, -0.13])  # a plug tip ~3 cm past the TCP
    p_t, R_t, _, _ = kin.forward(q_true, tool_p)
    q = solve_ik(kin, HOME.float().expand(E, 5), p_t, R_t, tool_p, None, IKCfg(), iterations=60)
    p, R, _, _ = kin.forward(q, tool_p)
    err = pose_error(p, R, p_t, R_t)
    assert err[:, :3].norm(dim=-1).max() < 1e-4  # < 0.1 mm
    assert err[:, 3:].norm(dim=-1).max() < math.radians(0.5)


def test_position_wins_when_orientation_is_unreachable():
    """5-DOF: a yaw that does not face along the arm is unreachable; position must still converge."""
    kin = SO101Kinematics()
    p_t, R_t, _, _ = kin.forward(HOME.float())
    R_bad = rot_exp(torch.tensor([[0.0, 0.0, 0.6]])) @ R_t
    q = solve_ik(kin, HOME.float(), p_t, R_bad, iterations=100)
    p, _, _, _ = kin.forward(q)
    assert (p - p_t).norm() < 1e-4
    q_w = solve_ik(kin, HOME.float(), p_t, R_bad, cfg=IKCfg(mode="weighted"), iterations=100)
    assert (kin.forward(q_w)[0] - p_t).norm() > 5e-3  # why priority is the default


def test_dls_step_reduces_error_with_supplied_jacobian():
    kin = SO101Kinematics()
    q = HOME.float()
    p, R, a, o = kin.forward(q)
    err = torch.tensor([[0.005, -0.003, 0.002, 0.0, 0.0, 0.0]])
    dq = dls_step(kin.jacobian(p, a, o), err, 0.1, 0.005)
    p2, _, _, _ = kin.forward(q + dq)
    assert ((p + err[:, :3]) - p2).norm() < 0.2 * err[:, :3].norm()


def test_servo_sag_integral_removes_constant_droop():
    """Plant: the arm sags a constant 1 deg on Pitch and Elbow below its goal; the integral cancels it."""
    kin = SO101Kinematics()
    servo = CartesianServo(1, cfg=None)
    p_t, R_t, _, _ = kin.forward(HOME.float())
    sag = torch.tensor([[0.0, math.radians(1.0), math.radians(1.0), 0.0, 0.0]])
    q_meas = torch.cat([HOME.float(), torch.zeros(1, 1)], -1)
    servo.reset(torch.tensor([0]), q_meas)
    for _ in range(150):
        goal = servo.step(p_t, R_t, q_meas, torch.zeros(3), torch.eye(3), torch.tensor([True]))
        q_meas = torch.cat([goal + sag, torch.zeros(1, 1)], -1)
    p, _, _, _ = kin.forward(q_meas[:, :5])
    assert (p - p_t).norm() < 2e-4


def test_servo_tilt_integral_removes_wrist_droop():
    """Plant: Wrist_Pitch sags 2 deg below its goal; the orientation integral cancels the tilt."""
    kin = SO101Kinematics()
    servo = CartesianServo(1)
    p_t, R_t, _, _ = kin.forward(HOME.float())
    sag = torch.tensor([[0.0, 0.0, 0.0, math.radians(2.0), 0.0]])
    q_meas = torch.cat([HOME.float(), torch.zeros(1, 1)], -1)
    servo.reset(torch.tensor([0]), q_meas)
    for _ in range(200):
        goal = servo.step(p_t, R_t, q_meas, torch.zeros(3), torch.eye(3), torch.tensor([True]))
        q_meas = torch.cat([goal + sag, torch.zeros(1, 1)], -1)
    p, R, _, _ = kin.forward(q_meas[:, :5])
    assert (p - p_t).norm() < 2e-4
    assert rot_log(R_t @ R.transpose(-1, -2)).norm() < math.radians(0.2)
