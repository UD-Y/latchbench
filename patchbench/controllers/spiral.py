"""Spiral search with contact/seat detection from telemetry (Present_Load) or oracle wrist force (CLAUDE.md 7).

Same motion as :mod:`open_loop` up to the approach; then:

``APPROACH`` until contact is detected -> ``SEARCH``: hold a light push (commanded depth = contact depth +
``push_depth``) and trace an Archimedean spiral in the plane normal to the insertion axis (pitch 0.3 mm, max radius
3 mm; restarts from the centre when it runs out) -> drop-in detected (measured depth > contact depth + 1.5 mm, or
the signal falls below ``drop_frac`` of its search level) -> ``INSERT``: push along the axis to seat + 3 mm,
watching for the latch snap (signal rise then drop) or the back stop (hard rise) -> ``HOLD`` -> ``DONE``. If the
insert jams near the contact depth (front face, not the jack) it returns to ``SEARCH``.

The only difference between the two variants is the scalar contact signal:

- ``telemetry``: ``sum_j w_j |Present_Load_j - baseline_j|`` [register counts], baseline = EMA of the free-space
  load during the approach, frozen at contact. Default weights favour ``Pitch`` (shoulder_lift), the joint that
  drives the push: in the hold test (results/hold_test_physx_joint) its load rose ~90 counts/N at the gripper while
  the elbow stayed flat until ~2 N in gearbox stiction.
- ``oracle``: ``|F|`` of the plug + latch contact force from the panel [N] (contact sensor, normal + friction).

Thresholds live in :class:`DetectCfg`, in the signal's own units. Virtual-wrist latch peaks (G1): loose ~0.5 N,
medium ~1.4 N, spec ~2.5 N, then a drop at the snap; the back stop gives a hard rise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .open_loop import APPROACH, DONE, HOLD, INSERT, SEARCH, ControllerObs, OpenLoopCfg, OpenLoopController


@dataclass
class DetectCfg:
    """Event thresholds, in the signal's units (telemetry: Present_Load counts; oracle: N)."""

    contact: float
    """Signal above this for ``confirm_steps`` consecutive steps = contact."""
    latch_min_rise: float
    """Latch snap: the signal first rises this much above its level at the start of INSERT ..."""
    latch_drop: float
    """... then falls this much below its peak."""
    backstop: float
    """Signal above this during INSERT = hard stop (seated against the back stop, or jammed)."""
    confirm_steps: int = 2
    drop_frac: float = 0.0
    """Drop-in by signal: below this fraction of the search level (0 disables; depth detection always on)."""


def telemetry_detect_cfg() -> DetectCfg:
    # ~90 counts/N on Pitch (hold test press); bus noise +-1 count; free-space start-of-motion transients up to
    # ~35 counts (env run 19 Oct, sw3_loose) -> 50 counts over 4 steps. Not yet calibrated against logged forces.
    return DetectCfg(contact=50.0, latch_min_rise=20.0, latch_drop=15.0, backstop=350.0, confirm_steps=4, drop_frac=0.0)


def oracle_detect_cfg() -> DetectCfg:
    return DetectCfg(contact=0.3, latch_min_rise=0.25, latch_drop=0.15, backstop=4.0, confirm_steps=2, drop_frac=0.0)


@dataclass
class SpiralCfg(OpenLoopCfg):
    sensor: str = "telemetry"
    """"telemetry" | "oracle"."""
    detect: DetectCfg | None = None
    """None -> the sensor's defaults."""
    load_weights: tuple[float, ...] = (0.0, 1.0, 0.5, 0.0, 0.0, 0.0)
    """Per-joint weights of |Present_Load - baseline| (sim order: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw)."""
    baseline_alpha: float = 0.1
    """EMA factor of the free-space load baseline per control step."""
    baseline_depth: float = -0.004
    """m: the baseline only updates while the measured tip is further than this in front of the estimated mouth. The
    push builds slowly through the soft servo (~1 N/s), so an EMA still running at contact tracks the rise and never
    crosses the threshold (env run 19 Oct: 7/20 telemetry episodes slid along the face undetected)."""
    baseline_settle_steps: int = 15
    """Approach steps before the baseline is trusted (start-of-motion friction transient)."""
    push_depth: float = 0.006
    """m commanded past the contact depth during the search (light push via the servo's position stiffness)."""
    spiral_pitch: float = 0.0003
    """m per turn (CLAUDE.md section 7)."""
    spiral_max_radius: float = 0.003
    """m (CLAUDE.md section 7); the spiral restarts from the centre beyond it."""
    spiral_speed: float = 0.004
    """m/s along the spiral."""
    drop_depth: float = 0.0015
    """m of measured depth gain over the contact depth = dropped in (CLAUDE.md section 7)."""
    jam_depth: float = 0.004
    """INSERT hitting the back-stop threshold less than this past the contact depth = jammed on the face -> SEARCH."""
    max_restarts: int = 3


class SpiralController(OpenLoopController):
    def __init__(self, num_envs: int, device: str | torch.device = "cpu", cfg: SpiralCfg | None = None):
        cfg = cfg or SpiralCfg()
        if cfg.detect is None:
            cfg.detect = telemetry_detect_cfg() if cfg.sensor == "telemetry" else oracle_detect_cfg()
        assert cfg.sensor in ("telemetry", "oracle"), cfg.sensor
        super().__init__(num_envs, device, cfg)
        E, dev = num_envs, self.device
        self.w = torch.tensor(cfg.load_weights, device=dev)
        self.baseline = torch.zeros(E, 6, device=dev)
        self.approach_steps = torch.zeros(E, dtype=torch.long, device=dev)
        self.above = torch.zeros(E, dtype=torch.long, device=dev)
        self.signal = torch.zeros(E, device=dev)
        self.contact_depth = torch.zeros(E, device=dev)
        self.contact_lat = torch.zeros(E, 2, device=dev)
        self.theta = torch.zeros(E, device=dev)
        self.search_level = torch.zeros(E, device=dev)
        self.insert_level = torch.zeros(E, device=dev)
        self.peak = torch.zeros(E, device=dev)
        self.restarts = torch.zeros(E, dtype=torch.long, device=dev)
        # episode flags (for logging / tests)
        self.contact_detected = torch.zeros(E, dtype=torch.bool, device=dev)
        self.dropped_in = torch.zeros_like(self.contact_detected)
        self.latch_detected = torch.zeros_like(self.contact_detected)
        self.backstop_hit = torch.zeros_like(self.contact_detected)
        self.no_contact = torch.zeros_like(self.contact_detected)
        self.gave_up = torch.zeros_like(self.contact_detected)

    def _init_extra(self, ids: torch.Tensor) -> None:
        for t in (self.baseline, self.approach_steps, self.above, self.signal, self.theta, self.peak, self.restarts):
            t[ids] = 0
        for t in (self.contact_detected, self.dropped_in, self.latch_detected, self.backstop_hit, self.no_contact,
                  self.gave_up):
            t[ids] = False

    def _integrate_mask(self) -> torch.Tensor:
        """Open-loop mask, plus the lateral axes during the search (the spiral has to be tracked to ~0.1 mm)."""
        m = super()._integrate_mask()
        s = (self.phase == SEARCH).float()
        return torch.maximum(m, torch.stack([torch.zeros_like(s), s, s], -1))

    # ------------------------------------------------------------------------------------------------ sensing
    def _update_signal(self, obs: ControllerObs) -> None:
        cfg: SpiralCfg = self.cfg  # type: ignore[assignment]
        if cfg.sensor == "oracle":
            assert obs.force is not None, "oracle spiral needs obs.force"
            self.signal = obs.force.norm(dim=-1).to(self.signal.dtype)
            return
        assert obs.load is not None, "telemetry spiral needs obs.load"
        load = obs.load.to(self.baseline.dtype)
        appr = self.phase == APPROACH
        first = appr & (self.approach_steps == 0)
        self.baseline = torch.where(first.unsqueeze(-1), load, self.baseline)
        ema = self.baseline + cfg.baseline_alpha * (load - self.baseline)
        upd = appr & ~first & (self.meas_depth < cfg.baseline_depth)  # frozen before any contact is possible
        self.baseline = torch.where(upd.unsqueeze(-1), ema, self.baseline)
        sig = ((load - self.baseline).abs() * self.w).sum(-1)
        trusted = (self.approach_steps >= cfg.baseline_settle_steps) | (self.phase > APPROACH)
        self.signal = torch.where(trusted, sig, torch.zeros_like(sig))
        self.approach_steps = self.approach_steps + appr.long()

    # ------------------------------------------------------------------------------------------- phase logic
    def _step_contact_phases(self, obs: ControllerObs) -> None:
        cfg: SpiralCfg = self.cfg  # type: ignore[assignment]
        d: DetectCfg = cfg.detect  # type: ignore[assignment]
        seat = torch.as_tensor(obs.seat_depth, device=self.device, dtype=self.depth.dtype).expand(self.num_envs)
        final = seat + cfg.push_past_seat
        ph = self.phase.clone()

        # The telemetry baseline must see the approach before its contact test (it freezes at contact).
        self._update_signal(obs)
        self.above = torch.where(self.signal > d.contact, self.above + 1, torch.zeros_like(self.above))
        contact = self.above >= d.confirm_steps

        # APPROACH: advance along the axis until contact; no contact by seat + 3 mm -> hold (no_contact).
        m = ph == APPROACH
        self.depth = torch.where(m, torch.minimum(self.depth + cfg.approach_speed * cfg.dt, final), self.depth)
        hit = m & contact
        self.contact_detected |= hit
        self.contact_depth = torch.where(hit, self.meas_depth, self.contact_depth)
        self.contact_lat = torch.where(hit.unsqueeze(-1), self.lat, self.contact_lat)
        self.theta = torch.where(hit, torch.zeros_like(self.theta), self.theta)
        self.search_level = torch.where(hit, self.signal, self.search_level)
        self._goto(hit, SEARCH)
        miss = m & ~hit & self._ramp_done(final)
        self.no_contact |= miss
        self._goto(miss, HOLD)

        # SEARCH: light push + Archimedean spiral r = b * theta around the contact point.
        m = ph == SEARCH
        b = cfg.spiral_pitch / (2 * math.pi)
        r = b * self.theta
        dtheta = cfg.spiral_speed * cfg.dt / torch.sqrt(r * r + b * b).clamp(min=1e-6)
        theta = torch.where(m, self.theta + dtheta, self.theta)
        over = m & (b * theta > cfg.spiral_max_radius)
        self.restarts = self.restarts + over.long()
        theta = torch.where(over, torch.zeros_like(theta), theta)
        self.theta = theta
        r = b * theta
        spiral = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], -1)
        self.lat = torch.where(m.unsqueeze(-1), self.contact_lat + spiral, self.lat)
        self.depth = torch.where(m, self.contact_depth + cfg.push_depth, self.depth)
        self.search_level = torch.where(m, torch.maximum(self.search_level, self.signal), self.search_level)
        drop = self.meas_depth > self.contact_depth + cfg.drop_depth
        if d.drop_frac > 0:
            drop |= self.signal < d.drop_frac * self.search_level
        drop = m & drop & (self.t_phase > 2 * cfg.dt)
        self.dropped_in |= drop
        self.insert_level = torch.where(drop, self.signal, self.insert_level)
        self.peak = torch.where(drop, self.signal, self.peak)
        self.depth = torch.where(drop, torch.maximum(self.depth, self.meas_depth), self.depth)
        self._goto(drop, INSERT)
        give_up = m & ~drop & (self.restarts > cfg.max_restarts)
        self.gave_up |= give_up
        self._goto(give_up, HOLD)

        # INSERT: push to seat + 3 mm; latch snap (rise then drop) or back stop -> hold; jammed on the face -> search.
        m = ph == INSERT
        self.depth = torch.where(m, torch.minimum(self.depth + cfg.approach_speed * cfg.dt, final), self.depth)
        self.peak = torch.where(m, torch.maximum(self.peak, self.signal), self.peak)
        snap = m & (self.peak - self.insert_level >= d.latch_min_rise) & (self.signal <= self.peak - d.latch_drop)
        hard = m & (self.signal >= d.backstop)
        jam = hard & (self.meas_depth < self.contact_depth + cfg.jam_depth)
        self.latch_detected |= snap & ~jam
        self.backstop_hit |= hard & ~jam
        self.depth = torch.where(jam, self.meas_depth, self.depth)
        self.contact_depth = torch.where(jam, self.meas_depth, self.contact_depth)
        self.contact_lat = torch.where(jam.unsqueeze(-1), self.lat, self.contact_lat)
        self.theta = torch.where(jam, torch.zeros_like(self.theta), self.theta)
        self._goto(jam, SEARCH)
        self._goto(m & ~jam & (snap | hard | (self.depth >= final)), HOLD)

        # HOLD keeps ramping to the final depth (bounded by the push lead), so an early or false snap cannot stall.
        m = self.phase >= HOLD
        self.depth = torch.where(m, torch.minimum(self.depth + cfg.approach_speed * cfg.dt, final), self.depth)
        self._goto((self.phase == HOLD) & (self.t_phase >= cfg.hold_s) & (ph == HOLD), DONE)
