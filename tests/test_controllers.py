"""Scripted controller state machines against a toy plant (CPU only, no Isaac).

The plant: the plug tip follows the commanded tool pose (perfect servo) except along the insertion axis, where it is
stopped by the panel face at depth 0 unless its lateral offset is within the clearance of the true hole, and by the
back stop at the seat depth. Axial force = ``k * blocked penetration`` plus a latch bump (rise over the cam, drop at
the snap). Telemetry = gravity-hold loads + ~90 counts/N on Pitch (hold test) + bus noise.
"""

import math

import pytest
import torch

from patchbench.controllers import (
    ControllerObs,
    OpenLoopCfg,
    OpenLoopController,
    SpiralCfg,
    SpiralController,
)
from patchbench.controllers.ik import SO101Kinematics, solve_ik
from patchbench.controllers.open_loop import DONE, HOLD, INIT, INSERT, SEARCH

HOME = torch.tensor([-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465])
SEAT = 0.016
# Plug frame in the gripper frame: insertion along the jaws (-Z), latch side +X.
TOOL_P = torch.tensor([-0.007, 0.0, -0.125])
TOOL_R = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


class Plant:
    def __init__(self, E, hole=(0.0, 0.0), wall=True, k=800.0, latch_peak=1.0, clearance=0.0005, noise=1, seed=0):
        self.kin = SO101Kinematics()
        self.E, self.wall, self.k, self.latch_peak, self.cl, self.noise = E, wall, k, latch_peak, clearance, noise
        q_port = (HOME[:5] + torch.tensor([0.1, 0.3, -0.2, -0.3, 0.0])).expand(E, 5)
        self.port_p, self.port_R, _, _ = self.kin.forward(q_port, TOOL_P, TOOL_R)
        self.hole = torch.tensor(hole).expand(E, 2)
        self.q = HOME.expand(E, 6).clone()
        self.depth = torch.full((E,), -0.1)
        self.F = torch.zeros(E)
        self.gen = torch.Generator().manual_seed(seed)

    def obs(self) -> ControllerObs:
        base = torch.tensor([0.0, -34.0, -60.0, -9.0, 0.0, 0.0])  # hold-test gravity loads
        load = base + 90.0 * self.F.unsqueeze(-1) * torch.tensor([0.0, 1.0, 0.3, 0.0, 0.0, 0.0])
        load = load + torch.randint(-self.noise, self.noise + 1, (self.E, 6), generator=self.gen)
        return ControllerObs(
            joint_pos=self.q.clone(), port_pos=self.port_p, port_rot=self.port_R,
            tool_p=TOOL_P.expand(self.E, 3), tool_R=TOOL_R.expand(self.E, 3, 3), seat_depth=SEAT,
            load=load.round(), force=-self.F.unsqueeze(-1) * self.port_R[..., 0],
        )

    def latch_force(self, d):
        rise = ((d - 0.0135) / (0.0153 - 0.0135)).clamp(0, 1)
        return torch.where((d > 0.0135) & (d < 0.0153), self.latch_peak * rise, torch.zeros_like(d))

    def step(self, goal):
        p_c, R_c, _, _ = self.kin.forward(goal[:, :5], TOOL_P, TOOL_R)
        rel = torch.einsum("eij,ei->ej", self.port_R, p_c - self.port_p)
        d_c, lat_c = rel[:, 0], rel[:, 1:]
        inside = (self.depth > 0.0005) & self.wall
        lat = torch.where(inside.unsqueeze(-1), torch.max(torch.min(lat_c, self.hole + self.cl), self.hole - self.cl), lat_c)
        in_hole = ((lat - self.hole).abs() < self.cl).all(-1)
        if self.wall:
            limit = torch.where(in_hole | inside, torch.full_like(d_c, SEAT), torch.zeros_like(d_c))
        else:
            limit = torch.full_like(d_c, 1.0)
        d = torch.minimum(d_c, limit)
        self.F = self.k * (d_c - d) + self.latch_force(d) * ((in_hole | inside) & self.wall)
        R = self.port_R
        p_act = self.port_p + d.unsqueeze(-1) * R[..., 0] + lat[:, :1] * R[..., 1] + lat[:, 1:] * R[..., 2]
        q_arm = solve_ik(self.kin, goal[:, :5], p_act, R_c, TOOL_P, TOOL_R, iterations=4)
        self.q = torch.cat([q_arm, goal[:, 5:]], -1)
        self.depth = d


def run(ctrl, plant, seconds=20.0, log=None):
    for _ in range(int(seconds * 30)):
        goal = ctrl.compute(plant.obs())
        plant.step(goal)
        if log is not None:
            log.append((ctrl.phase.clone(), plant.depth.clone(), plant.F.clone()))
        if bool((ctrl.phase == DONE).all()):
            break
    return ctrl


def test_open_loop_aligned_seats_and_finishes():
    plant = Plant(2)
    ctrl = run(OpenLoopController(2, cfg=OpenLoopCfg(pose_noise_mm=0.0)), plant)
    assert (ctrl.phase == DONE).all()
    assert torch.allclose(plant.depth, torch.full((2,), SEAT), atol=2e-4)  # stopped by the back stop


def test_open_loop_misaligned_stays_on_the_face():
    plant = Plant(2, hole=(0.0012, -0.0008))
    ctrl = run(OpenLoopController(2, cfg=OpenLoopCfg(pose_noise_mm=0.0)), plant)
    assert (ctrl.phase == DONE).all()
    assert (plant.depth.abs() < 2e-4).all()


@pytest.mark.parametrize("sensor", ["telemetry", "oracle"])
def test_spiral_contact_search_dropin_latch(sensor):
    E = 3
    plant = Plant(E, hole=(0.0012, -0.0008))
    ctrl = SpiralController(E, cfg=SpiralCfg(sensor=sensor, pose_noise_mm=0.0))
    log = []
    run(ctrl, plant, log=log)
    phases = torch.stack([l[0] for l in log])
    assert ctrl.contact_detected.all() and ctrl.dropped_in.all()
    assert (phases == SEARCH).any(0).all() and (phases == INSERT).any(0).all()
    # contact detected on the face (depth ~0), not deep
    assert (ctrl.contact_depth.abs() < 1.5e-3).all()
    assert (ctrl.latch_detected | ctrl.backstop_hit).all()
    assert (plant.depth >= SEAT - 0.00114).all()  # seated within the latched play
    assert (ctrl.phase == DONE).all()


def test_spiral_latch_snap_detected_before_backstop():
    """Face contact first (misaligned), so the latch cam happens inside INSERT; medium-tier peak 1.4 N."""
    plant = Plant(2, hole=(-0.001, 0.0006), latch_peak=1.4)
    ctrl = SpiralController(2, cfg=SpiralCfg(sensor="oracle", pose_noise_mm=0.0))
    run(ctrl, plant)
    assert ctrl.latch_detected.all() and not ctrl.backstop_hit.any()
    assert (plant.depth >= SEAT - 0.00114).all()


def test_no_contact_timeout_holds():
    plant = Plant(2, wall=False)
    ctrl = SpiralController(2, cfg=SpiralCfg(sensor="telemetry", pose_noise_mm=0.0))
    run(ctrl, plant)
    assert ctrl.no_contact.all() and not ctrl.contact_detected.any()
    assert (ctrl.phase == DONE).all()


def test_telemetry_ignores_gravity_offset_and_bus_noise():
    """Free-space approach with noisy, offset loads (noise +-2 counts) must not trigger contact."""
    plant = Plant(4, wall=False, noise=2)
    ctrl = SpiralController(4, cfg=SpiralCfg(sensor="telemetry", pose_noise_mm=0.0))
    run(ctrl, plant)
    assert not ctrl.contact_detected.any()


def test_reset_subset_and_noise_reproducible():
    a = OpenLoopController(3, cfg=OpenLoopCfg(pose_noise_mm=1.0, seed=7))
    b = OpenLoopController(3, cfg=OpenLoopCfg(pose_noise_mm=1.0, seed=7))
    pa, pb = Plant(3), Plant(3)
    for _ in range(5):
        pa.step(a.compute(pa.obs()))
        pb.step(b.compute(pb.obs()))
    assert torch.equal(a.noise, b.noise) and a.noise.abs().max() > 0
    assert a.noise.std() == pytest.approx(1e-3, rel=0.9)
    a.reset(torch.tensor([1]))
    assert a.phase[1] == INIT and a.phase[0] != INIT and a.phase[2] != INIT
    old = a.noise.clone()
    a.compute(pa.obs())
    assert not torch.equal(a.noise[1], old[1]) and torch.equal(a.noise[0], old[0])


def test_open_loop_with_noise_targets_estimate():
    """With pose noise the controller aims at the noisy estimate: the commanded lateral equals the noise."""
    plant = Plant(4, wall=False)
    ctrl = OpenLoopController(4, cfg=OpenLoopCfg(pose_noise_mm=2.0, seed=1))
    run(ctrl, plant, seconds=4.0)
    rel = torch.einsum("eij,ei->ej", plant.port_R, ctrl.est_p - plant.port_p)
    meas = torch.einsum("eij,ei->ej", plant.port_R, ctrl.servo.measured_pose(plant.q, TOOL_P, TOOL_R)[0] - plant.port_p)
    assert torch.allclose(meas[:, 1:], rel[:, 1:], atol=2e-4)


def test_telemetry_detects_slow_force_ramp():
    """Soft servo (100 N/m at the tip): the push builds ~1 N/s; a baseline still adapting at contact would hide it."""
    plant = Plant(2, hole=(0.0012, -0.0008), k=100.0)
    ctrl = SpiralController(2, cfg=SpiralCfg(sensor="telemetry", pose_noise_mm=0.0))
    run(ctrl, plant)
    assert ctrl.contact_detected.all()
    assert (ctrl.contact_depth.abs() < 1.5e-3).all()
