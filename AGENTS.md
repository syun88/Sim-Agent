# Sim-Agent working rules

- Keep `/Users/syun/Sim-Agent` as the only project root.
- Use the official OpenArm v2 model; do not substitute another robot.
- Treat MuJoCo as the authoritative source for robot/object state.
- Never commit `.env` or expose credentials through logs/APIs.
- Clearly label deterministic planning fallback and grasp-weld fallback.
- Only report features as verified after running the relevant checks.
