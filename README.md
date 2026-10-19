# PatchBench

An Isaac Lab-Arena benchmark: an SO-101 arm inserts an RJ45 plug into a 3-port panel at three clearance tiers
(`loose`, `medium`, `spec`). Success is staged (reach → contact → align → seat → latch) and reported as a funnel per tier.

The headline comparison is three scripted controllers that differ only in what they sense: open-loop,
a spiral search driven by simulated STS3215 servo telemetry (`Present_Load`, from
[so101-contact](https://github.com/UD-Y/so101-contact)), and the same search with perfect wrist force.

Status: under construction (19–21 Oct 2026). Environment notes are in [docs/ENV_AUDIT.md](docs/ENV_AUDIT.md).
