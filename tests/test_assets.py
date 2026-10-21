"""pxr-only checks of the generated RJ45 assets (no Isaac Sim). Run: pytest -q tests/test_assets.py"""

import math
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from latchbench.assets.build_assets import TIERS, USD_DIR, RJ45Dims, build_all, jack_geometry  # noqa: E402

DIMS = RJ45Dims()
TOL = 1e-6  # m


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("usd"), DIMS)


def _bound(stage, path, rel_to):
    """Axis-aligned box of ``path`` expressed in the frame of ``rel_to``."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeRelativeBound(stage.GetPrimAtPath(path), stage.GetPrimAtPath(rel_to)).ComputeAlignedRange()
    return rng.GetMin(), rng.GetMax()


def _local(stage, path, rel_to):
    a = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    b = UsdGeom.Xformable(stage.GetPrimAtPath(rel_to)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return a * b.GetInverse()


@pytest.mark.parametrize("tier", list(TIERS))
def test_committed_assets_match_builder(built, tier):
    for key, fresh in built[tier].items():
        committed = USD_DIR / tier / Path(fresh).name
        assert committed.exists(), f"missing {committed}; run python -m latchbench.assets.build_assets"
        assert committed.read_text() == Path(fresh).read_text(), f"{committed} is stale; rebuild the assets"


@pytest.mark.parametrize("tier", list(TIERS))
def test_jack_clearances(built, tier):
    stage = Usd.Stage.Open(str(built[tier]["panel"]))
    c = TIERS[tier]
    g = jack_geometry(DIMS, tier)
    for i in range(3):
        port = f"/Panel/port_{i}"
        left_min, _ = _bound(stage, f"{port}/wall_left", port)
        _, right_max = _bound(stage, f"{port}/wall_right", port)
        _, bottom_max = _bound(stage, f"{port}/wall_bottom", port)
        shoulder_min, _ = _bound(stage, f"{port}/shoulder_left", port)
        lip_min, _ = _bound(stage, f"{port}/top_lip", port)
        width = left_min[1] - right_max[1]
        height = shoulder_min[2] - bottom_max[2]
        assert width == pytest.approx(DIMS.plug_width + 2 * c, abs=TOL)
        assert height == pytest.approx(DIMS.plug_body_height + 2 * c, abs=TOL)
        assert lip_min[2] == pytest.approx(shoulder_min[2], abs=TOL)  # lip flush with the ceiling
        # opening is centred on the mouth frame
        assert left_min[1] + right_max[1] == pytest.approx(0.0, abs=TOL)
        assert shoulder_min[2] + bottom_max[2] == pytest.approx(0.0, abs=TOL)
        # back stop at seat depth; catch ledge back face at catch depth
        stop_min, _ = _bound(stage, f"{port}/back_stop", port)
        _, ledge_max = _bound(stage, f"{port}/catch_ledge", port)
        ledge_min, _ = _bound(stage, f"{port}/catch_ledge", port)
        assert stop_min[0] == pytest.approx(DIMS.seat_depth, abs=TOL)
        assert ledge_max[0] == pytest.approx(g.catch_depth, abs=TOL)
        assert ledge_max[0] - ledge_min[0] >= 1.40e-3 - TOL
        # lead-in: 0.5 mm wider per side at the mouth plane
        lead_min, lead_max = _bound(stage, f"{port}/lead_in_left", port)
        assert lead_min[0] == pytest.approx(0.0, abs=TOL)
        assert lead_max[0] == pytest.approx(DIMS.lead_in_width / math.tan(math.radians(30)), abs=TOL)


def test_tier_ordering():
    widths = [jack_geometry(DIMS, t).width for t in ("loose", "medium", "spec")]
    assert widths == sorted(widths, reverse=True)
    assert jack_geometry(DIMS, "spec").width == pytest.approx(11.91e-3, abs=1e-6)  # IEC jack opening


def test_latch_snaps_at_seat_depth():
    """Flat latch tip clears the catch ledge snap_margin before the back stop; play within spec."""
    for tier in TIERS:
        g = jack_geometry(DIMS, tier)
        assert g.catch_depth + DIMS.latch_root_offset + DIMS.latch_length == pytest.approx(g.snap_depth)
        assert g.seat_depth - g.snap_depth <= 1.14e-3
        assert g.seat_depth - g.snap_depth < 1.0e-3  # snap happens inside the "seated" band (6.1)


@pytest.mark.parametrize("tier", list(TIERS))
def test_ports_and_frames(built, tier):
    stage = Usd.Stage.Open(str(built[tier]["panel"]))
    root = stage.GetDefaultPrim()
    assert root.GetPath() == "/Panel"
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get() is True
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    for i in range(3):
        port = f"/Panel/port_{i}"
        for frame in ("mouth", "seat", "catch"):
            assert stage.GetPrimAtPath(f"{port}/{frame}").IsA(UsdGeom.Xform)
        m = _local(stage, f"{port}/mouth", "/Panel")
        # port_0 is left (+Y), 15 mm pitch, mouths on the plate front face
        assert m.ExtractTranslation() == pytest.approx(Gf.Vec3d(0, (1 - i) * DIMS.pitch, 0), abs=TOL)
        assert m.TransformDir(Gf.Vec3d(1, 0, 0)) == pytest.approx(Gf.Vec3d(1, 0, 0), abs=1e-9)  # +X insertion
        assert m.TransformDir(Gf.Vec3d(0, 0, 1)) == pytest.approx(Gf.Vec3d(0, 0, -1), abs=1e-9)  # latch side down
        s = _local(stage, f"{port}/seat", f"{port}/mouth")
        assert s.ExtractTranslation() == pytest.approx(Gf.Vec3d(DIMS.seat_depth, 0, 0), abs=TOL)
    # every collider is a box or a convex hull
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if prim.IsA(UsdGeom.Mesh):
                assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull"
            else:
                assert prim.IsA(UsdGeom.Cube)


def test_plug(built):
    stage = Usd.Stage.Open(str(built["medium"]["plug"]))
    lo, hi = _bound(stage, "/Plug/body/geom", "/Plug")
    size = hi - lo
    assert size == pytest.approx(Gf.Vec3d(DIMS.plug_length, DIMS.plug_width, DIMS.plug_body_height), abs=TOL)
    assert hi[0] == pytest.approx(0.0, abs=TOL)  # tip at the origin
    body, latch = stage.GetPrimAtPath("/Plug/body"), stage.GetPrimAtPath("/Plug/latch")
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(5e-3)
    assert UsdPhysics.MassAPI(latch).GetMassAttr().Get() == pytest.approx(0.3e-3)
    assert body.HasAPI(UsdPhysics.RigidBodyAPI) and latch.HasAPI(UsdPhysics.RigidBodyAPI)
    attach = _local(stage, "/Plug/body/attach_frame", "/Plug").ExtractTranslation()
    assert attach == pytest.approx(Gf.Vec3d(-DIMS.plug_length, 0, 0), abs=TOL)

    # latch at rest: angled up ~9 deg, rises above the body, and fits the jack pocket
    llo, lhi = _bound(stage, "/Plug/latch/geom", "/Plug")
    rise = lhi[2] - DIMS.plug_body_height / 2
    assert rise == pytest.approx(DIMS.latch_rise, abs=TOL)
    assert 1.4e-3 <= rise <= 2.0e-3
    g = jack_geometry(DIMS, "spec")
    assert lhi[2] < g.height / 2 + g.pocket_height
    assert lhi[1] - llo[1] < g.slot_width

    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/Plug/latch_joint"))
    assert joint
    assert joint.GetBody0Rel().GetTargets() == [body.GetPath()]
    assert joint.GetBody1Rel().GetTargets() == [latch.GetPath()]
    assert joint.GetAxisAttr().Get() == "Y"
    assert joint.GetLowerLimitAttr().Get() == pytest.approx(-12.0)
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(2.0)
    # joint frame on the body is pre-rotated by the rest angle (angle 0 = rest)
    rot = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get()))
    assert rot.GetAngle() == pytest.approx(DIMS.latch_rest_deg, abs=1e-4)
    drive = UsdPhysics.DriveAPI(joint.GetPrim(), "angular")
    assert drive
    assert drive.GetTargetPositionAttr().Get() == 0.0
    k_deg = drive.GetStiffnessAttr().Get()
    assert k_deg * 180 / math.pi == pytest.approx(0.17, rel=0.05)  # section 5: k_theta ~0.17 N*m/rad
    assert 0 < drive.GetDampingAttr().Get() < k_deg
