"""Newton showcase recording (newton_showcase.py npz) -> time-sampled USD scene for RTX rendering (render_showcase.py).

Robot: Menagerie SO-101 visual meshes and MJCF colours, one Xform per body with per-frame poses. Plug and panel: the
look layers (usd/look), plug pose and latch angle per frame, the look's static cable hidden. Cable: a 6 mm tube swept
along the simulated VBD rod every frame. Table: a matte top at z = 0 (the cable's ground plane). 30 time codes / s.

    python scripts/showcase_usd.py --sim results/newton_showcase/sim_loose_port1.npz
Writes results/newton_showcase/scene/showcase.usdc (regenerable, not committed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from scipy.spatial.transform import Rotation as Rot

import newton
from patchbench.assets.build_assets import USD_DIR, RJ45Dims
from patchbench.assets.build_visuals import sweep

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--sim", type=Path, default=ROOT / "results/newton_showcase/sim_loose_port1.npz")
ap.add_argument("--out", type=Path, default=ROOT / "results/newton_showcase/scene/showcase.usdc")
args = ap.parse_args()

DIMS = RJ45Dims()
BOOT_END = np.array([-DIMS.plug_length - 0.014, 0.0, 0.0])


def quatd(xyzw) -> Gf.Quatd:
    x, y, z, w = map(float, xyzw)
    return Gf.Quatd(w, Gf.Vec3d(x, y, z))


def preview(stage, path, rgb, rough=0.45, metal=0.0) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def mesh_prim(stage, path, verts, faces_flat, counts, material, xf=None, normals=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(verts, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(counts, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces_flat, np.int32)))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if normals is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals, np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if xf is not None:
        x = UsdGeom.Xformable(m)
        x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, xf[:3])))
        x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quatd(xf[3:7]))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def vertex_normals(v: np.ndarray, tris: np.ndarray) -> np.ndarray:
    fn = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def main():
    d = np.load(args.sim, allow_pickle=True)
    rb, cab, latch = d["robot_body_q"], d["cable_body_q"], d["latch"]
    F = len(rb)
    plug_b = int(d["plug_body"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    stage = Usd.Stage.CreateNew(str(args.out))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(F - 1)
    stage.SetTimeCodesPerSecond(float(d["fps"]))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")

    # robot: Menagerie visual meshes (same MJCF and body order as the sim model)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(str(d["so101_xml"]), floating=False)
    m = b.finalize(device="cpu")
    colors, mats = m.shape_color.numpy(), {}
    for i, lab in enumerate(m.shape_label):
        if not lab.endswith("_visual"):
            continue
        body = int(m.shape_body.numpy()[i])
        bp = f"/World/Robot/body_{body}"
        if not stage.GetPrimAtPath(bp):
            x = UsdGeom.Xform.Define(stage, bp)
            t_op = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            o_op = x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            for f in range(F):
                t_op.Set(Gf.Vec3d(*map(float, rb[f, body, :3])), f)
                o_op.Set(quatd(rb[f, body, 3:7]), f)
        rgb = tuple(np.round(colors[i], 3))
        if rgb not in mats:
            dark = max(rgb) < 0.3
            mats[rgb] = preview(stage, f"/World/Looks/robot_{len(mats)}", rgb, rough=0.35 if dark else 0.5)
        src = m.shape_source[i]
        v = np.asarray(src.vertices, float) * m.shape_scale.numpy()[i]
        tris = np.asarray(src.indices, int).reshape(-1, 3)
        mesh_prim(stage, f"{bp}/{lab.split('/')[-1]}", v, tris.ravel(), [3] * len(tris), mats[rgb],
                  m.shape_transform.numpy()[i], vertex_normals(v, tris))  # fmt: skip

    # plug (look layer) + latch angle; panel (look layer)
    plug = UsdGeom.Xform.Define(stage, "/World/Plug")
    plug.GetPrim().GetReferences().AddReference(str(USD_DIR / "look" / "plug_look.usd"))
    t_op = plug.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    o_op = plug.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    rest = Gf.Quatd(np.cos(np.radians(DIMS.latch_rest_deg) / 2), Gf.Vec3d(0, np.sin(np.radians(DIMS.latch_rest_deg) / 2), 0))
    lat_orient = stage.GetPrimAtPath("/World/Plug/latch").GetAttribute("xformOp:orient")
    for f in range(F):
        t_op.Set(Gf.Vec3d(*map(float, rb[f, plug_b, :3])), f)
        o_op.Set(quatd(rb[f, plug_b, 3:7]), f)
        a = float(latch[f]) / 2
        lat_orient.Set(rest * Gf.Quatd(np.cos(a), Gf.Vec3d(0, np.sin(a), 0)), f)
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Plug/body/look/cable")).MakeInvisible()
    panel = UsdGeom.Xform.Define(stage, "/World/Panel")
    panel.GetPrim().GetReferences().AddReference(str(USD_DIR / "look" / f"panel_3port_{d['tier']}_look.usd"))
    panel.AddTransformOp().Set(Gf.Matrix4d(np.asarray(d["panel_w"], float).T.tolist()))
    # stand under the panel, down to the table
    pz = float(d["panel_w"][2, 3])
    stand = UsdGeom.Cube.Define(stage, "/World/PanelStand")
    stand.CreateSizeAttr(1.0)
    sx = UsdGeom.Xformable(stand)
    centre = np.asarray(d["panel_w"])[:3, :3] @ [DIMS.seat_depth / 2, 0.0, 0.0] + np.asarray(d["panel_w"])[:3, 3]
    sx.AddTranslateOp().Set(Gf.Vec3d(float(centre[0]), float(centre[1]), (pz - 0.012) / 2))
    sx.AddScaleOp().Set(Gf.Vec3d(0.02, 0.07, max(pz - 0.012, 0.005)))
    UsdShade.MaterialBindingAPI.Apply(stand.GetPrim()).Bind(preview(stage, "/World/Looks/stand", (0.12, 0.12, 0.13), 0.5, 0.6))

    # cable: tube along the rod (boot end + body centres), constant topology, points per frame
    pts0, counts, idx, _ = sweep(np.r_[[BOOT_END], cab[0][:, :3]], 3.0e-3, k=24)
    cable = mesh_prim(stage, "/World/Cable", pts0, idx, counts, preview(stage, "/World/Looks/jacket", (0.09, 0.30, 0.74), 0.42))
    pa, na = cable.GetPointsAttr(), cable.GetNormalsAttr()
    for f in range(F):
        P = rb[f, plug_b]
        boot = Rot.from_quat(P[3:7]).apply(BOOT_END) + P[:3]
        pts, _, _, nrm = sweep(np.r_[[boot], cab[f][:, :3]], 3.0e-3, k=24)
        pa.Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)), f)
        na.Set(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)), f)

    # table top (z = 0 is the cable's ground plane)
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1.0)
    tx = UsdGeom.Xformable(table)
    tx.AddTranslateOp().Set(Gf.Vec3d(0.15, 0.0, -0.01))
    tx.AddScaleOp().Set(Gf.Vec3d(1.2, 0.9, 0.02))
    UsdShade.MaterialBindingAPI.Apply(table.GetPrim()).Bind(preview(stage, "/World/Looks/table", (0.55, 0.52, 0.48), 0.65))
    stage.GetRootLayer().Save()
    print(f"[showcase_usd] wrote {args.out} ({F} frames, {args.out.stat().st_size / 1e6:.1f} MB); "
          f"plug end {rb[-1, plug_b, :3]}, panel {np.asarray(d['panel_w'])[:3, 3]}")


if __name__ == "__main__":
    main()
