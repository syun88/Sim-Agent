# Sim-Agent working rules

- Keep `/Users/syun/Sim-Agent` as the only project root.
- Use the official OpenArm v2 model; do not substitute another robot.
- The simulation engine is [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world); it is the
  authoritative source for robot/object state. The official `openarm-mujoco` model description is the
  authority for names, keyframe, actuator mapping and grasp geometry, and is attached, never copied.
- Never commit `.env` or expose credentials through logs/APIs.
- Clearly label the deterministic planning fallback (used when the AI planner is unavailable).
- Grasping is physical contact: the fingertip pads pinch the object, the grasp is verified from
  finger-object contacts and from the object following the gripper, and success is never faked by
  writing the object's pose (the only object pose writes are scene reset and setup).
- Only report features as verified after running the relevant checks.
