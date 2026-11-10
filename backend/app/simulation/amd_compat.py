"""Make Genesis' CUDA-shaped device queries work on a ROCm build of torch.

`genesis 1.4.4` reads `shared_memory_per_multiprocessor` and `shared_memory_per_block_optin` from
`torch.cuda.get_device_properties(...)`. Those attributes exist only in NVIDIA's build; a ROCm build exposes
`shared_memory_per_block` and the thread/warp counts instead, so the very first scene build dies with
`AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'shared_memory_per_multiprocessor'`
(measured 2026-11-10 on gfx1101 / torch 2.10.0+rocm7.0).

The engine object is a C type, so the attributes cannot be added to it. What can be replaced is the one function
that reads them: this rewrites its source with guarded lookups and re-executes it in the module. The rewrite is
asserted to have happened, so a Genesis upgrade cannot silently leave the shim inert.
"""

from __future__ import annotations

import inspect
import sys
import textwrap

#: What a ROCm build reports instead, and what Genesis needs in its place.
REWRITES = (
    ('device_property.shared_memory_per_multiprocessor',
     "getattr(device_property, 'shared_memory_per_multiprocessor', device_property.shared_memory_per_block)"),
    ('device_property.shared_memory_per_block_optin',
     "getattr(device_property, 'shared_memory_per_block_optin', device_property.shared_memory_per_block)"),
)

_applied: dict | None = None


def apply() -> dict:
    """Patch Genesis when torch is a ROCm build. Returns what happened, for the run log."""
    global _applied
    if _applied is not None:
        return _applied
    import torch

    if not getattr(torch.version, 'hip', None):
        _applied = {'applied': False, 'reason': 'torch is not a ROCm build'}
        return _applied
    try:
        from genesis.utils import misc
        function = misc.get_gpu_shared_tile_sizes
        source = inspect.getsource(function)
    except Exception as error:  # pragma: no cover - only on a Genesis without that function
        _applied = {'applied': False, 'reason': f'{type(error).__name__}: {error}'}
        return _applied

    patched = source
    touched = []
    for needle, replacement in REWRITES:
        if needle in patched:
            patched = patched.replace(needle, replacement)
            touched.append(needle)
    if patched == source:
        _applied = {'applied': False, 'reason': 'no CUDA-only device property found to guard',
                    'looked_for': [needle for needle, _ in REWRITES]}
        return _applied
    namespace = dict(misc.__dict__)
    exec(compile(textwrap.dedent(patched), f'{function.__module__}:{function.__name__}:amd-compat', 'exec'),
         namespace)
    replacement = namespace[function.__name__]
    misc.get_gpu_shared_tile_sizes = replacement
    # Callers did `from genesis.utils.misc import get_gpu_shared_tile_sizes`, so the module attribute alone is not
    # enough: every module that bound the original object is rebound. Measured 2026-11-10: without this the shim
    # looked applied and the solver still crashed on the same line.
    rebound = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        for attribute, value in list(vars(module).items()):
            if value is function:
                setattr(module, attribute, replacement)
                rebound.append(f'{name}.{attribute}')
    _applied = {'applied': True, 'guarded': touched, 'hip': torch.version.hip, 'rebound': rebound}
    return _applied


#: What stops the ROCm path today, measured 2026-11-10 (do not re-derive it by hand):
#:
#: * `torch 2.10.0+rocm7.0` runs the GPU (`hip 7.0.51831`, `cuda.is_available() True`, RX 7800 XT).
#: * `apply()` above is required and sufficient for `genesis 1.4.6` to get past its CUDA-only device query.
#: * `scene.build()` then fails in `genesis/vis/rasterizer.py` -> `pyrender.OffscreenRenderer` ->
#:   `EGLPlatform.init_context()` with "No EGL context could be initialized", **even with no camera and
#:   show_viewer=False**, because Genesis always builds a rasterizer.
#: * The same EGL steps succeed when called directly (`eglInitialize` 1.5, `eglChooseConfig`, `eglBindAPI`,
#:   `eglCreateContext` for 4.1 core, `eglMakeCurrent`), so this is a pyrender/PyOpenGL binding problem: after
#:   Genesis' import, `eglGetDisplay.argtypes` is `(EGLNativeDisplayType_pointer,)` and the later calls pass
#:   integers where opaque pointers are declared ("expected EGLNativeDisplayType instead of int").
#:
#: So the AMD backend is blocked in the rendering layer, not in this project and not in torch. Continuing there
#: means patching pyrender's EGL platform as a whole (not one call), or an upstream fix.
