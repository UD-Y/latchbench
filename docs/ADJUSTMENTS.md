# Adjustments backlog

Known cosmetic or staging issues, deferred on purpose. None of them changes benchmark results. Each entry says where it
was seen, what is wrong, and where a fix would go. Add new items at the bottom; delete an item once it is fixed.

## Newton showcase clip (`results/newton_showcase/frames_hero_1920x1080.mp4`, rendered on Brev 20 Oct 2026)

1. **Robot base floats above the table.** The SO-101 base (right third of the hero shot) hangs a few cm above the
   table plane and casts a shadow under itself. The arm is mounted where the PhysX env frame puts it; the render's
   table is a plane at z = 0 (`scripts/showcase_usd.py`, table), so nothing supports the base.
   Fix: add a pedestal/mount block under the base in `showcase_usd.py`, or lower the table top to the base height,
   or move the camera so the base is out of frame. Visual only; the replayed motion stays the same.

2. **Hero framing crops the scene.** The top of the gripper and the left side of the switch are cut off, and the
   cable leaves the frame at the bottom edge. Fix: camera/look-at/focal values of the `hero` preset in
   `scripts/render_showcase.py` (`PRESETS`). Check with `--frames 60,160` before a full render (~70 min on the L40S
   at 1920x1080, 128 spp).

3. **Plug/cable alignment in the gripper.** The plug sits at the tip of the fixed jaw instead of looking clamped
   between the jaws, and the cable leaves the plug through the jaw with a sharp bend right behind the boot, before it
   drops toward the table. To check: the plug's in-hand pose relative to the jaws (grasp assumption, CLAUDE.md
   section 5) and the cable's kinematic first segments in `scripts/newton_showcase.py`.
   Caution: the in-hand pose is shared with the benchmark (physics, datagen, GR00T data). Change only the showcase's
   visual/cable attachment unless the grasp itself is deliberately revisited, which invalidates the existing data and
   results.

4. **Replicator warning per frame.** `render_showcase.py` logs `Orchestrator failed to schedule a capture after 1000
   app updates` for every frame; the frames are correct, but each one wastes app updates (~18 s/frame alone on the
   L40S). Fix: the capture loop in `render_showcase.py` (timeline / `rep.orchestrator.step` usage).
