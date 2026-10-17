# Sim-Agent working rules

- Keep `/Users/syun/Sim-Agent` as the only project root.
- Use the official OpenArm v2 model; do not substitute another robot.
- The simulation engine is [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world); it is the
  authoritative source for robot/object state. The official `openarm-mujoco` model description is the
  authority for names, keyframe, actuator mapping and grasp geometry, and is attached, never copied.
- Never commit `.env` or expose credentials through logs/APIs.
- Clearly label the deterministic planning fallback and the grasp fallback
  (`kinematic-grasp-hold`: a kinematic hold, not contact-only manipulation).
- Only report features as verified after running the relevant checks.
