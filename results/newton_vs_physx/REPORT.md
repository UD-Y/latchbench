# RJ45 contact probe: PhysX vs Newton (MuJoCo-Warp)

20 Oct 2026, laptop (RTX 4060 8 GB), headless. Question: do the things PatchBench depends on (the latch force
signature, and jamming that grows as clearance shrinks) survive a change of physics engine, or are they PhysX artefacts?

## Setup

Both engines run the same probe (`patchbench/contact_probe.py`): the plug hangs on a compliant 6-DOF virtual wrist
(2000 N/m, 5 N·m/rad, pivot at the plug tip) and is pushed into port_1 at 10 mm/s to seat + 3 mm, held, then
pulled back 3 mm. The grid is 3 tiers × lateral offset {0, 0.2, 0.5, 1.0} mm × tilt {0, 1, 2}° × 5 seeds
(180 insertions per engine). Seeds draw the offset direction and tilt axis, and both engines get the same draws
for each variant. Both are scored by the same code. The geometry comes from one engine-neutral parts list
(`build_assets.jack_parts`/`plate_parts`; the USDs regenerate byte-identical), and the plug and latch masses, beam
spring and damping, and joint limits are identical.

| | PhysX | Newton |
|---|---|---|
| Script | `scripts/check_contact.py` (Isaac Lab, pinned `bb0c8e1`) | `scripts/check_contact_newton.py` (standalone, no Kit) |
| Solver | TGS, 32 position iterations, dt 1/480 s (`patchbench/physics.py`) | Newton 1.5.0 `SolverMuJoCo` (mujoco-warp 3.11.0), 100 iterations, elliptic cone, 1/480 s split into N substeps |
| Contact | rigid, 0.5 mm contact offset, μ 0.35 | soft: `solref` time constant 2 × substep (the stiffest stable value), `solimp` (0.99, 0.999, 0.2 mm), μ 0.35 |
| Collision | PhysX | MuJoCo-Warp (one point per box–mesh pair), or Newton's pipeline (`--contacts newton`, manifolds, 0.5 mm gap) |

Refactor check: the PhysX probe after moving the metrics into `contact_probe.py` reproduces the pre-refactor script
exactly (0 differing fields over 120 episodes, same day, same machine).

Newton ships its own RJ45 example (`python -m newton.examples contacts_rj45_plug`): SDF meshes, the VBD solver,
a simulated cable, and the plug dragged by hand. Its latch spring (0.15 N·m/rad) is close to our beam estimate
(0.168). It has no scripted insertion or force readout, so it is a qualitative reference only.

## Results

Latch rate over all variants (aligned insertions latch with the signature on every engine and tier):

| engine | loose | medium | spec | explosions | wall s / sim s |
|---|---|---|---|---|---|
| PhysX TGS 32 it | 0.92 | 0.73 | 0.67 | 0 | 4.1 (36 envs, one scene) |
| Newton, MuJoCo collision, 8 substeps | 1.00 | 1.00 | 1.00 | 0 | 5.3 (3 × 12 worlds, sequential) |
| Newton, Newton collision, 8 substeps | 1.00 | 0.95 | 0.87 | 7 | 5.8 |
| Newton, MuJoCo collision, 4 substeps | 1.00 | 1.00 | 0.98 | 2 | 2.8 |
| Newton, Arena USB-C contact settings, 4 substeps | 1.00 | 1.00 | 1.00 | 37 (+45 tunnels) | 2.5 |

Latch rate at 1 mm offset (the only offset outside the 0.5 mm lead-in on medium and spec):

| engine | loose | medium | spec |
|---|---|---|---|
| PhysX | 0.67 | 0.07 | 0.00 |
| Newton, MuJoCo collision, 8 substeps | 1.00 | 1.00 | 1.00 |
| Newton, Newton collision, 8 substeps | 1.00 | 0.80 | 0.47 |

At ≤ 0.5 mm offset the engines agree closely (PhysX loses a few spec/medium seeds at 0.5 mm). Full table:
[COMPARISON.md](COMPARISON.md). Aligned force and latch traces: [overlay_force_vs_depth.png](overlay_force_vs_depth.png).

## Findings

1. **The latch signature is not a PhysX artefact.** On both engines an aligned plug deflects the latch (mean maximum
   deflection, PhysX / Newton at 8 substeps: loose −4.3° / −4.5°, medium −6.6° / −7.2°, spec −7.9° / −8.0°), the axial force rises to a peak, and it drops
   when the latch snaps. Peak axial force (aligned, 5-seed mean): loose 0.49 / 0.46 N, medium 1.37 / 1.47 N,
   spec 2.46 / 2.97 N (PhysX / Newton). Peak force grows as the tier tightens on both engines.

2. **Jamming is where the engines differ, and the cause is static friction.** At 1 mm offset the plug edge lands
   0.25 mm (medium) or 0.39 mm (spec) beyond the lead-in, on the flat jack face. PhysX jams; Newton with MuJoCo
   collision slides the plug sideways into the port every time. A friction sweep on these variants (2 seeds,
   medium + spec, [friction/](friction/)) separates the mechanism from the engine:

   | μ | 0 | 0.35 | 1.0 | 2.0 |
   |---|---|---|---|---|
   | PhysX latched | 4/4 | jams | 0/4 | – |
   | Newton latched | 4/4 | 4/4 | 4/4 | 4/4 |

   With no friction, PhysX also slides the plug in: an edge contact on the slightly rotated plug gives a real
   lateral push. PhysX's Coulomb friction stops that from μ ≈ 0.35. MuJoCo's soft friction does not: in the contact
   dump the plug creeps 0.7 mm sideways in 0.1 s at 2 N normal load, with the friction force pointing along the
   motion. This holds with multi-point contacts (`multiccd`), a pyramidal cone, `impratio` 10 and 100, and Newton's
   collision pipeline (which jams more often: spec 1 mm 0.47). For PatchBench this means the clearance-tier ranking
   (jams rise as clearance shrinks) is much flatter on Newton/MuJoCo than on PhysX.

3. **The thin latch needs much stiffer MuJoCo contacts than Arena's defaults.** Arena's USB-C settings
   (`solimp` 0.95…, `solref` 4 ms) make the 0.3 g latch's contact stiffness about 120 N/m at first touch (MuJoCo's
   impedance scales with effective mass), so the latch sinks through the lip: 37 explosions and 45 tunnels in 180
   runs. Time constant 2 × substep with d₀ = 0.99 fixes it. This only matters for very light parts; the USB-C task
   has none.

4. **The latch chatters on Newton.** During deflection the latch angle reverses 87–167 times per insertion on Newton
   and never on PhysX (aligned, seed 0, all Newton configurations). The axial force dips to −1 to −47 N around the
   snap; PhysX never goes below 0. More substeps do not remove this consistently (loose: −25 N at 8 substeps,
   −5.5 N at 4; medium: −0.9 N vs −47 N). Latch and signature metrics still pass, but a policy reading force or
   Present_Load would see this noise.

5. **Cost.** At comparable fidelity (8 substeps), Newton took 1.6–1.9 wall s per sim s per tier (12 worlds) versus
   PhysX's 4.1 for all 36 envs in one scene: about 3.3k vs 4.2k env-steps/s on this laptop. MuJoCo-Warp needs one
   model per tier because worlds must share geometry, and 12 worlds do not fill the GPU. This is not a scaling
   benchmark.

## Recommendation

Keep PhysX as the benchmark engine. It is the engine where jamming, the effect the clearance tiers measure, behaves
like Coulomb friction. A Newton port of PatchBench (for BAM on Newton) would need either a friction model with real
stiction or an honest note that jam rates are not comparable across engines. The latch signature transfers.

## Reproduce

```bash
cd ~/work/patchbench && source ../IsaacLab-Arena/.venv/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
T=loose,medium,spec; OUT=results/newton_vs_physx
flock -o ~/work/.isaac.lock python scripts/check_contact.py --tiers $T --pos_iters 32 --run_id physx_it32_co0.5 --out $OUT
flock -o ~/work/.isaac.lock python scripts/check_contact_newton.py --tiers $T --substeps 8 --out $OUT
flock -o ~/work/.isaac.lock python scripts/check_contact_newton.py --tiers $T --substeps 8 --contacts newton --out $OUT
flock -o ~/work/.isaac.lock python scripts/check_contact_newton.py --tiers $T --substeps 4 --out $OUT
flock -o ~/work/.isaac.lock python scripts/check_contact_newton.py --tiers $T --substeps 4 --solref 0.004,1.0 \
    --solimp 0.95,0.999,0.0005,0.5,2.0 --run_id nw_s4_arena_usbc_contacts --out $OUT
# friction sweep (medium + spec, 1 mm offset): --friction {0,1,2} --offsets_mm 1.0 --tilts_deg 0 --seeds 2
python scripts/compare_engines.py --plot_first 3 --runs $OUT/physx_it32_co0.5 $OUT/nw_s8_tc0.52_d0.99_ell \
    $OUT/nwc_nw_s8_tc0.52_d0.99_ell $OUT/nw_s4_tc1_d0.99_ell $OUT/nw_s4_arena_usbc_contacts
```
