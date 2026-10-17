# Sim-Agent working rules

- The project root is one checkout per machine: `/Users/syun/Sim-Agent` (macOS) and
  `/home/yuu/Documents/Sim-Agent` (this Linux checkout). Do not create a second root on either machine.
- Development environment on the Linux checkout: `uv` (installed at `~/.local/bin/uv`), managed
  Python 3.13 pinned by a machine-local, gitignored `.python-version` (this machine's system Python
  is out of the supported range), and the `.venv/` created by `uv sync`. Run tests with
  `uv run pytest -q` from the project root.
- Use the official OpenArm v2 model; do not substitute another robot.
- The simulation engine is [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world); it is the
  authoritative source for robot/object state. The official `openarm-mujoco` model description is the
  authority for names, keyframe, actuator mapping and grasp geometry, and is attached, never copied.
- Never commit `.env` or expose credentials through logs/APIs.
- Clearly label the deterministic planning fallback and the grasp fallback
  (`kinematic-grasp-hold`: a kinematic hold, not contact-only manipulation).
- Only report features as verified after running the relevant checks.
