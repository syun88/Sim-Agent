# Live demo

1. Open `http://127.0.0.1:8000`.
2. Confirm status badges show `MuJoCo: Connected`, `Robot: OpenArm v2`, and `Agent: Idle`.
3. Press **Reset**.
4. Use `Move the red cube into the blue zone.`
5. Press **Execute**.
6. Watch the MuJoCo image, plan, skill events, and verification.
7. After success, press **Random**, then execute the same instruction again.
8. Inspect Execution History.
9. Press **Run recoverable fault demo** to see a real IK failure, bounded replanning, and a second physical attempt.
10. Use `curl http://127.0.0.1:8000/api/memory` to inspect stored experiences.

If ai& credentials are present, the plan provider badge changes from `Fallback` to `ai&`. If Neo4j AuraDB is available, the memory badge changes from `Fallback` to `Connected`.
