"""Pure-torch stage logic (reach -> contact -> align -> seat -> latch), CLAUDE.md section 6.1. No Isaac imports.

All inputs are world-frame tensors batched over environments; ports are batched as ``(num_envs, num_ports, ...)``.
A port's frame is given as a rotation matrix whose columns are the port's ``mouth`` axes: column 0 (+X) is the
insertion direction, columns 1 and 2 span the opening (width, height). Quaternion conventions stay in the caller.

Stages are sequential and monotone: stage k counts only once stage k-1 has been reached in this episode (earlier in
the same step included), and an episode reports its furthest stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

STAGES = ("none", "reached", "contacted", "aligned", "seated", "latched")
REACHED, CONTACTED, ALIGNED, SEATED, LATCHED = 1, 2, 3, 4, 5


@dataclass
class StageThresholds:
    """CLAUDE.md section 6.1 (SI units)."""

    reach_dist: float = 0.010  # plug tip within 10 mm of the target mouth
    reach_axis_deg: float = 10.0  # plug axis within 10 deg of the port axis
    contact_force: float = 0.2  # N, plug (incl. latch) vs target jack
    align_margin: float = 0.0005  # lateral |e| < clearance + 0.5 mm
    align_depth: float = 0.002  # depth > 2 mm
    seat_margin: float = 0.001  # depth >= seat_depth - 1.0 mm
    latch_deflect_deg: float = 3.0  # latch must first deflect more than 3 deg
    latch_return_deg: float = 2.0  # ... then return within 2 deg of rest
    latch_depth_margin: float = 0.00114  # depth >= seat_depth - 1.14 mm (IEC latched axial play)


@dataclass
class PortGeometry:
    """Per-env geometry of the jacks.

    Attributes:
        mouth_pos: ``(E, P, 3)`` opening centre of each port [m].
        mouth_rot: ``(E, P, 3, 3)`` rotation matrices, columns = (insertion, width, height) axes.
        clearance: ``(E,)`` or scalar lateral clearance per side of the tier [m].
        seat_depth: ``(E,)`` or scalar insertion depth at which the plug is fully seated [m].
    """

    mouth_pos: torch.Tensor
    mouth_rot: torch.Tensor
    clearance: torch.Tensor | float
    seat_depth: torch.Tensor | float


def port_coordinates(tip_pos: torch.Tensor, ports: PortGeometry) -> torch.Tensor:
    """Plug tip in each port's mouth frame: ``(E, P, 3)`` = (depth, lateral width, lateral height) [m]."""
    d = tip_pos.unsqueeze(1) - ports.mouth_pos
    return torch.einsum("epij,epi->epj", ports.mouth_rot, d)


def _inside(coords: torch.Tensor, clearance: torch.Tensor, margin: float, min_depth: float) -> torch.Tensor:
    lim = clearance + margin  # broadcastable to coords[..., 0]
    return (coords[..., 1].abs() < lim) & (coords[..., 2].abs() < lim) & (coords[..., 0] > min_depth)


class StageTracker:
    """Per-episode furthest stage, latch history and episode metrics for ``num_envs`` environments."""

    def __init__(self, num_envs: int, device: str | torch.device = "cpu", thresholds: StageThresholds | None = None):
        self.th = thresholds or StageThresholds()
        self.num_envs = num_envs
        self.device = torch.device(device)
        z = torch.zeros(num_envs, device=self.device)
        self.furthest = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.latch_deflected = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.wrong_port = torch.zeros_like(self.latch_deflected)
        self.peak_axial_force = z.clone()
        self.time_to_latch = torch.full_like(z, float("nan"))
        self.overload_events = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._overload_prev = torch.zeros_like(self.latch_deflected)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.furthest[ids] = 0
        self.latch_deflected[ids] = False
        self.wrong_port[ids] = False
        self.peak_axial_force[ids] = 0.0
        self.time_to_latch[ids] = float("nan")
        self.overload_events[ids] = 0
        self._overload_prev[ids] = False

    @property
    def success(self) -> torch.Tensor:
        return self.furthest >= LATCHED

    def update(
        self,
        *,
        tip_pos: torch.Tensor,
        plug_axis: torch.Tensor,
        ports: PortGeometry,
        target_port: torch.Tensor,
        contact_force: torch.Tensor,
        latch_angle: torch.Tensor,
        episode_time: torch.Tensor | float,
        overload_active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance one control step; returns the stage predicates satisfied *now* as ``(E, 5)`` bools.

        Args:
            tip_pos: ``(E, 3)`` plug tip [m].
            plug_axis: ``(E, 3)`` unit vector from cable end to tip.
            ports: jack geometry.
            target_port: ``(E,)`` target port index.
            contact_force: ``(E, P, 3)`` contact force of plug + latch on each jack [N].
            latch_angle: ``(E,)`` latch deflection from its rest angle [rad].
            episode_time: ``(E,)`` or scalar time since reset [s].
            overload_active: ``(E, J)`` servo overload flags; rising edges are counted.
        """
        th, e = self.th, torch.arange(self.num_envs, device=self.device)
        clearance = torch.as_tensor(ports.clearance, device=self.device, dtype=tip_pos.dtype).expand(self.num_envs)
        seat = torch.as_tensor(ports.seat_depth, device=self.device, dtype=tip_pos.dtype).expand(self.num_envs)
        coords = port_coordinates(tip_pos, ports)  # (E, P, 3)
        c_t = coords[e, target_port]
        axis_t = ports.mouth_rot[e, target_port, :, 0]
        depth = c_t[:, 0]

        cos_axis = (plug_axis * axis_t).sum(-1) / plug_axis.norm(dim=-1).clamp(min=1e-9)
        reached = (c_t.norm(dim=-1) < th.reach_dist) & (cos_axis > math.cos(math.radians(th.reach_axis_deg)))
        f_t = contact_force[e, target_port]
        contacted = f_t.norm(dim=-1) > th.contact_force
        aligned = _inside(c_t, clearance, th.align_margin, th.align_depth)
        seated = depth >= seat - th.seat_margin
        deflection = latch_angle.abs()
        self.latch_deflected |= deflection > math.radians(th.latch_deflect_deg)
        latched = self.latch_deflected & (deflection < math.radians(th.latch_return_deg))
        latched &= depth >= seat - th.latch_depth_margin

        now = torch.stack([reached, contacted, aligned, seated, latched], dim=-1)
        before = self.furthest.clone()
        for k in range(5):
            ok = now[:, k] & (self.furthest >= k)
            self.furthest = torch.where(ok, torch.clamp(self.furthest, min=k + 1), self.furthest)

        newly = (self.furthest >= LATCHED) & (before < LATCHED)
        t = torch.as_tensor(episode_time, device=self.device, dtype=tip_pos.dtype).expand(self.num_envs)
        self.time_to_latch = torch.where(newly, t, self.time_to_latch)

        others = torch.ones_like(coords[..., 0], dtype=torch.bool)
        others[e, target_port] = False
        in_other = _inside(coords, clearance.unsqueeze(-1), th.align_margin, th.align_depth) & others
        self.wrong_port |= in_other.any(dim=-1)

        axial = -(f_t * axis_t).sum(-1)  # jack pushes back against the insertion direction
        self.peak_axial_force = torch.maximum(self.peak_axial_force, axial)
        if overload_active is not None:
            ov = overload_active.any(dim=-1)
            self.overload_events += (ov & ~self._overload_prev).long()
            self._overload_prev = ov
        return now

    def summary(self) -> dict[str, torch.Tensor]:
        """Per-env episode results (call before :meth:`reset`)."""
        return {
            "furthest_stage": self.furthest.clone(),
            "success": self.success.clone(),
            "wrong_port": self.wrong_port.clone(),
            "peak_axial_force": self.peak_axial_force.clone(),
            "time_to_latch": self.time_to_latch.clone(),
            "overload_events": self.overload_events.clone(),
        }


def funnel(furthest: torch.Tensor) -> dict[str, float]:
    """Fraction of episodes that reached at least each stage."""
    n = max(1, furthest.numel())
    return {name: float((furthest >= k).sum()) / n for k, name in enumerate(STAGES) if k > 0}
