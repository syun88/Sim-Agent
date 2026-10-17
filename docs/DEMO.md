# Live demo

1. Open `http://127.0.0.1:8000`.
2. Confirm the status badges show the active engine (`MuJoCo` = `Authoritative`, or `Genesis` = `Alternative`), the active grasp mode, `Robot: OpenArm v2`, and `Agent: Idle`.
3. Press **Reset**.
4. Use `Move the red cube into the blue zone.`
5. Press **Execute**.
6. Watch the simulation image, plan, skill events, and verification.
7. After success, press **Random**, then execute the same instruction again.
8. Inspect Execution History.
9. Press **Run recoverable fault demo** to see a real IK failure, bounded replanning, and a second physical attempt.
10. Use `curl http://127.0.0.1:8000/api/memory` to inspect stored experiences.

If ai& credentials are present, the plan provider badge changes from `Fallback` to `ai&`. If Neo4j AuraDB is available, the memory badge changes from `Fallback` to `Connected`.

## Running on the other engine

```bash
SIM_BACKEND=genesis ./run.sh
```

The dashboard then shows `Genesis / Alternative` and `kinematic-grasp-hold`. On that engine the plan
executes but the placement does not reach the zone yet; see
[ARCHITECTURE.md](ARCHITECTURE.md#genesis-world-engine-status). MuJoCo remains the engine used for the
verified demonstration.
