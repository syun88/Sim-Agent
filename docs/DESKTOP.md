# Desktop app

The native desktop client uses **Dear PyGui**, a GPU-accelerated immediate-mode GUI. It is not Qt and does not open a browser.

## Start

```bash
./run_desktop.sh
```

On macOS you can also double-click `OpenArmAgent.command` in Finder.

This starts the embedded OpenArm v2 MuJoCo backend if it is not already running, then opens the desktop window.

## What it shows

- Live MuJoCo render of OpenArm v2.
- Natural-language task input.
- Agent plan, verification, events, graph memory, and history.
- Stop, reset, randomize, and recoverable fault controls.

## Notes

- If a backend is already listening on port 8000, the desktop app reuses it.
- If no backend is running, it starts one locally and stops only the instance it started.
- The current image pipeline uses a 640×480 MuJoCo render and scales it inside the desktop UI.
