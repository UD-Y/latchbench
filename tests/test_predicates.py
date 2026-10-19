"""Stage predicates on synthetic insertions (no Isaac Sim)."""

import math

import torch

from patchbench.predicates import (
    ALIGNED,
    CONTACTED,
    LATCHED,
    REACHED,
    SEATED,
    PortGeometry,
    StageTracker,
    funnel,
)

PITCH, SEAT, CLEAR = 0.015, 0.020, 0.00025


def ports(n):
    pos = torch.zeros(n, 3, 3)
    pos[:, :, 1] = torch.tensor([-PITCH, 0.0, PITCH])
    rot = torch.eye(3).expand(n, 3, 3, 3).clone()  # insertion along world +X
    return PortGeometry(pos, rot, CLEAR, SEAT)


def insert(lateral=0.0, deflect_deg=6.0, final_latch_deg=0.5, port=1, aim_port=1, axis=(1.0, 0.0, 0.0), steps=200,
           depth_end=SEAT, contact=True):  # fmt: skip
    """Drive one env along +X through the mouth of ``aim_port``; return the tracker and per-step furthest stage."""
    tr = StageTracker(1)
    g = ports(1)
    hist = []
    for i in range(steps):
        x = -0.02 + (depth_end + 0.02) * i / (steps - 1)
        tip = torch.tensor([[x, (aim_port - 1) * PITCH + lateral, 0.0]])
        if x < 0.008:
            ang = 0.0
        elif x < SEAT - 0.0005:  # latch snaps just after the plug seats
            ang = math.radians(deflect_deg)
        else:
            ang = math.radians(final_latch_deg)
        f = torch.zeros(1, 3, 3)
        if contact and x > -0.001:
            f[0, aim_port, 0] = -2.0 - 10 * max(0.0, x)  # jack pushes back along -X
        tr.update(tip_pos=tip, plug_axis=torch.tensor([axis]), ports=g, target_port=torch.tensor([port]),
                  contact_force=f, latch_angle=torch.tensor([ang]), episode_time=i / 30)  # fmt: skip
        hist.append(int(tr.furthest[0]))
    return tr, hist


def test_full_insertion_latches_in_order():
    tr, hist = insert()
    assert hist == sorted(hist)  # monotone
    assert set(hist) == {0, REACHED, CONTACTED, ALIGNED, SEATED, LATCHED}
    assert bool(tr.success[0]) and not bool(tr.wrong_port[0])
    assert 0 < tr.time_to_latch[0] < 200 / 30
    assert abs(tr.peak_axial_force[0] - (2.0 + 10 * SEAT)) < 1e-4


def test_no_latch_without_deflection():
    tr, hist = insert(deflect_deg=0.0)
    assert max(hist) == SEATED


def test_latch_still_deflected_is_not_latched():
    tr, hist = insert(final_latch_deg=4.0)
    assert max(hist) == SEATED


def test_lateral_offset_blocks_alignment():
    tr, hist = insert(lateral=CLEAR + 0.0006)
    assert max(hist) == CONTACTED


def test_wrong_axis_never_reaches():
    tilt = math.radians(15)
    tr, hist = insert(axis=(math.cos(tilt), math.sin(tilt), 0.0))
    assert max(hist) == 0


def test_no_contact_stops_at_reached():
    tr, hist = insert(contact=False)
    assert max(hist) == REACHED


def test_shallow_push_stops_before_seat():
    tr, hist = insert(depth_end=SEAT - 0.003)
    assert max(hist) == ALIGNED


def test_wrong_port_detected():
    tr, hist = insert(port=1, aim_port=2)
    assert bool(tr.wrong_port[0]) and max(hist) == 0


def test_batch_reset_and_funnel():
    n = 4
    tr = StageTracker(n)
    g = ports(n)
    tip = torch.tensor([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    f = torch.zeros(n, 3, 3)
    f[0, 1, 0] = -1.0
    tr.update(tip_pos=tip, plug_axis=torch.tensor([[1.0, 0, 0]] * n), ports=g, target_port=torch.tensor([1, 1, 1, 0]),
              contact_force=f, latch_angle=torch.zeros(n), episode_time=0.0)  # fmt: skip
    assert tr.furthest.tolist() == [CONTACTED, REACHED, 0, 0]
    assert funnel(tr.furthest)["reached"] == 0.5 and funnel(tr.furthest)["contacted"] == 0.25
    tr.reset(torch.tensor([0]))
    assert tr.furthest.tolist() == [0, REACHED, 0, 0]


def test_overload_rising_edges_counted():
    tr = StageTracker(1)
    g = ports(1)
    for ov in [0, 1, 1, 0, 1, 0]:
        tr.update(tip_pos=torch.tensor([[-0.1, 0, 0]]), plug_axis=torch.tensor([[1.0, 0, 0]]), ports=g,
                  target_port=torch.tensor([1]), contact_force=torch.zeros(1, 3, 3), latch_angle=torch.zeros(1),
                  episode_time=0.0, overload_active=torch.tensor([[bool(ov), False]]))  # fmt: skip
    assert tr.overload_events.item() == 2
