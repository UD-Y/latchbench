"""Render-only "look" layers for the RJ45 plug and panels: realistic geometry and materials for RTX stills/video.

Physics is untouched. Each look USD *references* the physics USD (``usd/<tier>/plug.usd``,
``panel_3port_<tier>.usd``) and only adds visual meshes (no CollisionAPI, no mass) and materials, and hides the
plug's collision boxes. Benchmark runs keep using the physics USDs; renders use ``usd/look/*.usd``.

All visual dimensions derive from ``RJ45Dims`` (IEC 60603-7 reference values, CLAUDE.md section 5), so the visuals
match the colliders. Additions are cosmetic and documented here:

* Plug: clear polycarbonate shell (0.4 mm edge radius), 8 gold contact blades at 1.02 mm pitch on the face
  opposite the latch, the 8 insulated conductors visible through the shell (T568B colours), a strain-relief boot and
  a 6.0 mm Cat6 jacket (blue PVC), latch with a ribbed thumb tab.
* Jack: black housing (the collider boxes themselves, re-materialled), nickel shield rim around the mouth, 8 gold
  spring contacts on the wall opposite the latch slot, green/amber link LEDs above each port, dark anodised faceplate.

Run: ``python -m patchbench.assets.build_visuals`` (pxr + numpy + scipy; no Isaac Sim).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from scipy.spatial import ConvexHull

from patchbench.assets.build_assets import TIERS, USD_DIR, RJ45Dims, jack_geometry, jack_parts

LOOK_DIR = USD_DIR / "look"
MM = 1e-3

# T568B pin order 1..8: (base colour, striped with white)
T568B = [("orange", True), ("orange", False), ("green", True), ("blue", False),
         ("blue", True), ("green", False), ("brown", True), ("brown", False)]  # fmt: skip
WIRE_RGB = {"orange": (0.95, 0.42, 0.05), "green": (0.10, 0.55, 0.20), "blue": (0.08, 0.25, 0.80),
            "brown": (0.36, 0.20, 0.10)}  # fmt: skip

# name: (diffuse rgb, roughness, metallic, opacity, ior, emissive rgb)
MATERIALS = {
    "clear_pc": ((0.92, 0.94, 0.96), 0.04, 0.0, 0.18, 1.58, None),  # polycarbonate plug shell / latch
    "gold": ((1.00, 0.77, 0.34), 0.18, 1.0, 1.0, 1.5, None),
    "boot_tpu": ((0.10, 0.33, 0.78), 0.55, 0.0, 1.0, 1.5, None),
    "jacket_pvc": ((0.09, 0.30, 0.74), 0.42, 0.0, 1.0, 1.5, None),
    "black_pbt": ((0.025, 0.025, 0.028), 0.55, 0.0, 1.0, 1.5, None),  # jack housing
    "nickel": ((0.78, 0.77, 0.74), 0.28, 1.0, 1.0, 1.5, None),
    "anodised": ((0.06, 0.065, 0.075), 0.45, 0.3, 1.0, 1.5, None),  # faceplate
    "led_green": ((0.05, 0.9, 0.2), 0.2, 0.0, 1.0, 1.5, (0.1, 4.0, 0.6)),
    "led_amber": ((1.0, 0.55, 0.05), 0.2, 0.0, 1.0, 1.5, (4.0, 1.8, 0.1)),
    **{f"wire_{c}": (rgb, 0.45, 0.0, 1.0, 1.5, None) for c, rgb in WIRE_RGB.items()},
    **{f"wire_{c}_white": (tuple(0.7 + 0.3 * v for v in rgb), 0.45, 0.0, 1.0, 1.5, None) for c, rgb in WIRE_RGB.items()},
}


# ----------------------------------------------------------------------------------------------- mesh helpers


def rounded_box(lo, hi, r: float, seg: int = 4):
    """Box from ``lo`` to ``hi`` with edges and corners rounded by ``r``: convex hull of 8 corner spheres.
    Returns (points, face counts, face indices, normals) with per-vertex normals."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    r = min(r, 0.49 * float(np.min(hi - lo)))
    a, b = lo + r, hi - r
    th = np.linspace(0, math.pi / 2, seg + 1)
    oct_ = np.array([(math.cos(p) * math.cos(t), math.cos(p) * math.sin(t), math.sin(p)) for p in th for t in th])
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                c = np.array([(a if s < 0 else b)[i] for i, s in enumerate((sx, sy, sz))])
                pts.append(c + r * oct_ * np.array([sx, sy, sz]))
    pts = np.unique(np.round(np.concatenate(pts), 9), axis=0)
    hull = ConvexHull(pts)
    tris = _outward(hull, pts)
    used = np.unique(tris)
    remap = -np.ones(len(pts), int)
    remap[used] = np.arange(len(used))
    pts = pts[used]
    n = pts - np.clip(pts, a, b)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return pts, [3] * len(tris), remap[tris].ravel().tolist(), n


def _outward(hull, pts):
    tris = hull.simplices.copy()
    c = pts.mean(0)
    for i, t in enumerate(tris):
        p = pts[t]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p.mean(0) - c) < 0:
            tris[i] = t[[0, 2, 1]]
    return tris


def sweep(path, half_w, half_h=None, exponent=2.0, k: int = 24, caps: bool = True):
    """Superellipse tube along ``path`` (N x 3). ``half_w``/``half_h``/``exponent``: scalars or per-point arrays.
    Frames by parallel transport; returns (points, counts, indices, normals)."""
    path = np.asarray(path, float)
    n = len(path)
    hw = np.broadcast_to(np.asarray(half_w, float), (n,))
    hh = np.broadcast_to(np.asarray(half_w if half_h is None else half_h, float), (n,))
    ex = np.broadcast_to(np.asarray(exponent, float), (n,))
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[0][2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ref, t[0])
    u /= np.linalg.norm(u)
    us, vs = [], []
    for i in range(n):
        if i:
            u = u - np.dot(u, t[i]) * t[i]
            u /= np.linalg.norm(u)
        v = np.cross(t[i], u)
        us.append(u)
        vs.append(v)
    ang = np.linspace(0, 2 * math.pi, k, endpoint=False)
    pts, nrm = [], []
    for i in range(n):
        c, s = np.cos(ang), np.sin(ang)
        e = 2.0 / ex[i]
        x = hw[i] * np.sign(c) * np.abs(c) ** e
        y = hh[i] * np.sign(s) * np.abs(s) ** e
        nx = np.sign(c) * np.abs(c) ** (2 - e) / max(hw[i], 1e-9)
        ny = np.sign(s) * np.abs(s) ** (2 - e) / max(hh[i], 1e-9)
        pts.append(path[i] + np.outer(x, us[i]) + np.outer(y, vs[i]))
        nn = np.outer(nx, us[i]) + np.outer(ny, vs[i])
        nrm.append(nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12))
    pts, nrm = np.concatenate(pts), np.concatenate(nrm)
    counts, idx = [], []
    for i in range(n - 1):
        for j in range(k):
            a, b = i * k + j, i * k + (j + 1) % k
            counts.append(4)
            idx += [a, b, b + k, a + k]
    if caps:
        for i, sgn in ((0, -1), (n - 1, 1)):
            ci = len(pts)
            pts = np.vstack([pts, path[i]])
            nrm = np.vstack([nrm, sgn * t[i]])
            for j in range(k):
                a, b = i * k + j, i * k + (j + 1) % k
                counts.append(3)
                idx += [ci, b, a] if sgn < 0 else [ci, a, b]
    return pts, counts, idx, nrm


def _mesh(stage, path: str, geo, material: UsdShade.Material, xform=None) -> UsdGeom.Mesh:
    pts, counts, idx, nrm = geo
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in nrm]))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))]))
    if xform is not None:
        UsdGeom.Xformable(m).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*xform))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def _box_geo(lo, hi):
    """Sharp box: 6 quads with flat normals (24 vertices)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    pts, nrm, counts, idx = [], [], [], []
    for ax in range(3):
        for side in (0, 1):
            nvec = np.zeros(3)
            nvec[ax] = 1.0 if side else -1.0
            u, w = [i for i in range(3) if i != ax]
            quad = []
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = np.empty(3)
                p[ax] = hi[ax] if side else lo[ax]
                p[u] = hi[u] if a else lo[u]
                p[w] = hi[w] if b else lo[w]
                quad.append(p)
            if np.dot(np.cross(quad[1] - quad[0], quad[2] - quad[0]), nvec) < 0:
                quad = quad[::-1]
            idx += list(range(len(pts), len(pts) + 4))
            pts += quad
            nrm += [nvec] * 4
            counts.append(4)
    return np.array(pts), counts, idx, np.array(nrm)


def _materials(stage, root: str) -> dict[str, UsdShade.Material]:
    UsdGeom.Scope.Define(stage, f"{root}/Looks")
    out = {}
    for name, (rgb, rough, metal, opacity, ior, emit) in MATERIALS.items():
        mat = UsdShade.Material.Define(stage, f"{root}/Looks/{name}")
        sh = UsdShade.Shader.Define(stage, f"{root}/Looks/{name}/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
        if emit:
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emit))
        if opacity < 1.0:
            sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(1.0)
            sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.02)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        out[name] = mat
    return out


def _stage(path: Path, default_prim: str, ref: Path) -> Usd.Stage:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    layer = Sdf.Layer.CreateNew(str(path), args={"format": "usda"})
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = stage.DefinePrim(f"/{default_prim}")
    root.GetReferences().AddReference(Path(__import__("os").path.relpath(ref, path.parent)).as_posix())
    stage.SetDefaultPrim(root)
    return stage


# ----------------------------------------------------------------------------------------------- plug


def cable_path(d: RJ45Dims, length: float = 0.12, droop: float = 0.025, n: int = 40) -> np.ndarray:
    """Default (static) cable centreline in the plug frame: straight out of the boot, then sagging toward +Z
    (world down in the env, where the latch faces down). Replaced per frame by video scripts."""
    x0 = -d.plug_length - 12 * MM
    s = np.linspace(0, 1, n)
    return np.stack([x0 - length * s, np.zeros(n), droop * s**2], 1)


def build_plug_look(d: RJ45Dims = RJ45Dims()) -> Path:
    out = LOOK_DIR / "plug_look.usd"
    stage = _stage(out, "Plug", USD_DIR / "loose" / "plug.usd")
    mats = _materials(stage, "/Plug")
    L, hw, hh = d.plug_length, d.plug_width / 2, d.plug_body_height / 2
    # hide the collision boxes (colliders stay active)
    for p in ("/Plug/body/geom", "/Plug/latch/geom"):
        UsdGeom.Imageable(stage.OverridePrim(p)).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    v = "/Plug/body/look"
    UsdGeom.Xform.Define(stage, v)
    _mesh(stage, f"{v}/shell", rounded_box((-L, -hw, -hh), (0.0, hw, hh), 0.4 * MM, seg=4), mats["clear_pc"])
    # contact blades: on the -Z face (opposite the latch), front end, 1.02 mm pitch, 0.3 mm thick, recessed 0.1 mm
    pitch = 1.02 * MM
    for i in range(8):
        y = (i - 3.5) * pitch
        _mesh(stage, f"{v}/blade_{i + 1}", _box_geo((-3.8 * MM, y - 0.15 * MM, -hh + 0.1 * MM),
                                                     (-0.35 * MM, y + 0.15 * MM, -hh + 1.6 * MM)), mats["gold"])  # fmt: skip
    # conductors: from behind each blade, straight back, then converging into the jacket (2 x 4 bundle)
    for i, (col, striped) in enumerate(T568B):
        y0 = (i - 3.5) * pitch
        z0 = -hh + 1.1 * MM
        ang = math.pi * (i + 0.5) / 4
        y1, z1 = 1.6 * MM * math.cos(ang), 1.6 * MM * math.sin(ang) * 0.5
        xs = np.linspace(-3.6 * MM, -L - 14 * MM, 30)
        f = np.clip((-xs - 9 * MM) / ((L + 2) * MM - 9 * MM), 0, 1)
        f = f * f * (3 - 2 * f)
        path = np.stack([xs, y0 + (y1 - y0) * f, z0 + (z1 - z0) * f], 1)
        _mesh(stage, f"{v}/wire_{i + 1}", sweep(path, 0.45 * MM, k=12),
              mats[f"wire_{col}_white" if striped else f"wire_{col}"])  # fmt: skip
    # boot: rounded-rectangle sleeve over the plug's rear 8 mm, tapering to the jacket
    xs = np.linspace(-L + 8 * MM, -L - 14 * MM, 40)
    f = np.clip((-L + 8 * MM - xs) / (22 * MM), 0, 1)
    g = np.clip((f - 0.25) / 0.75, 0, 1) ** 0.8
    bw = (hw + 0.7 * MM) * (1 - g) + 3.45 * MM * g
    bh = (hh + 0.7 * MM) * (1 - g) + 3.45 * MM * g
    _mesh(stage, f"{v}/boot", sweep(np.stack([xs, 0 * xs, 0 * xs], 1), bw, bh, 4.0 * (1 - g) + 2.0 * g, k=48),
          mats["boot_tpu"])  # fmt: skip
    # jacket (static default pose; video scripts rewrite these points per frame)
    _mesh(stage, "/Plug/body/look/cable", sweep(cable_path(d), 3.0 * MM, k=32), mats["jacket_pvc"])

    # latch: clear, with three ribs on the thumb tab at the free end (latch frame: along -X, bottom face z = 0)
    lv = "/Plug/latch/look"
    UsdGeom.Xform.Define(stage, lv)
    lw, t, ll = d.latch_width / 2, d.latch_thickness, d.latch_length
    _mesh(stage, f"{lv}/arm", rounded_box((-ll, -lw, 0.0), (0.0, lw, t), 0.25 * MM), mats["clear_pc"])
    for k in range(3):
        x = -ll + (0.6 + 0.9 * k) * MM
        _mesh(stage, f"{lv}/rib_{k}", rounded_box((x - 0.25 * MM, -lw, t), (x + 0.25 * MM, lw, t + 0.4 * MM),
                                                  0.12 * MM, seg=2), mats["clear_pc"])  # fmt: skip
    stage.GetRootLayer().Save()
    return out


# ----------------------------------------------------------------------------------------------- panel


def build_panel_look(tier: str, d: RJ45Dims = RJ45Dims()) -> Path:
    out = LOOK_DIR / f"panel_3port_{tier}_look.usd"
    stage = _stage(out, "Panel", USD_DIR / tier / f"panel_3port_{tier}.usd")
    mats = _materials(stage, "/Panel")
    g = jack_geometry(d, tier)
    hw, hh, ow = g.width / 2, g.height / 2, g.outer_width / 2
    lw = d.lead_in_width
    # re-material the colliders: jack bodies black, faceplate anodised
    for i in range(d.n_ports):
        for _, name, *_ in jack_parts(d, g):
            UsdShade.MaterialBindingAPI.Apply(stage.OverridePrim(f"/Panel/port_{i}/{name}")).Bind(mats["black_pbt"])
    plate = stage.GetPrimAtPath("/Panel/plate")
    for child in plate.GetChildren():
        UsdShade.MaterialBindingAPI.Apply(stage.OverridePrim(child.GetPath())).Bind(mats["anodised"])

    for i in range(d.n_ports):
        v = f"/Panel/port_{i}/look"  # jack frame (latch side +Z = panel down)
        UsdGeom.Xform.Define(stage, v)
        # nickel shield rim around the chamfered mouth, 0.15 mm proud of the face
        x0, x1 = -0.15 * MM, 0.0
        zt, zb = g.z_top_outer, g.z_bottom_outer
        rim = [((x0, hw + lw, zb), (x1, ow, zt)), ((x0, -ow, zb), (x1, -hw - lw, zt)),
               ((x0, -hw - lw, hh + lw), (x1, hw + lw, zt)), ((x0, -hw - lw, zb), (x1, hw + lw, -hh - lw))]  # fmt: skip
        for k, (lo, hi) in enumerate(rim):
            _mesh(stage, f"{v}/shield_{k}", _box_geo(lo, hi), mats["nickel"])
        # 8 gold spring contacts on the wall opposite the latch slot, rising from the back toward the mouth
        for j in range(8):
            y = (j - 3.5) * 1.02 * MM
            path = np.array([(g.seat_depth - 1.5 * MM, y, -hh - 0.05 * MM), (6.0 * MM, y, -hh + 0.12 * MM),
                             (4.0 * MM, y, -hh - 0.2 * MM)])  # fmt: skip
            _mesh(stage, f"{v}/spring_{j + 1}", sweep(path, 0.17 * MM, k=10), mats["gold"])
        # link LEDs on the faceplate above the port (jack -Z side = panel up), 0.2 mm proud
        for k, (dy, mat) in enumerate(((-4.0 * MM, "led_green"), (4.0 * MM, "led_amber"))):
            z = g.z_bottom_outer - 2.2 * MM
            _mesh(stage, f"{v}/led_{k}", rounded_box((-0.2 * MM, dy - 1.0 * MM, z - 0.6 * MM),
                                                     (0.0, dy + 1.0 * MM, z + 0.6 * MM), 0.15 * MM, seg=2),
                  mats[mat])  # fmt: skip
    stage.GetRootLayer().Save()
    return out


def main() -> None:
    print(f"wrote {build_plug_look()}")
    for tier in TIERS:
        print(f"wrote {build_panel_look(tier)}")


if __name__ == "__main__":
    main()
