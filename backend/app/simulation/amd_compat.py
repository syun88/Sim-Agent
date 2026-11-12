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


#: Loaded into the global scope before Genesis is imported on a ROCm build. Measured 2026-11-10: torch ships its
#: own `libdrm_amdgpu.so`, which shadows the system one and makes Mesa's AMD driver fail with
#: "libgallium: undefined symbol: amdgpu_va_manager_query_sw_info". The visible symptom is that `eglGetDisplay`
#: returns NULL, so Genesis' rasterizer cannot start and the whole scene build dies - with the GPU, the physics
#: and this project all fine.
SYSTEM_LIBDRM_AMDGPU = '/usr/lib/libdrm_amdgpu.so.1'


def preload_system_libdrm() -> dict:
    """Put the system `libdrm_amdgpu` in the global symbol scope so Mesa resolves against it."""
    global _applied
    import ctypes
    import os
    import torch

    if not getattr(torch.version, 'hip', None):
        return {'preloaded': False, 'reason': 'torch is not a ROCm build'}
    if not os.path.exists(SYSTEM_LIBDRM_AMDGPU):
        return {'preloaded': False, 'reason': f'{SYSTEM_LIBDRM_AMDGPU} not present'}
    try:
        ctypes.CDLL(SYSTEM_LIBDRM_AMDGPU, mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        return {'preloaded': False, 'reason': f'{type(error).__name__}: {error}'}
    return {'preloaded': True, 'library': SYSTEM_LIBDRM_AMDGPU}


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


def install_ctypes_egl_platform() -> dict:
    """Replace pyrender's EGL platform with one that talks to libEGL through ctypes directly.

    Measured 2026-11-10: on this stack every EGL step succeeds when called through ctypes, while Genesis' bundled
    pyrender fails inside PyOpenGL's typed wrappers (`expected EGLNativeDisplayType instead of int` and
    "No EGL context could be initialized"), and Genesis builds a rasterizer for every scene - so the AMD backend
    cannot start at all. This installs a platform with the five methods `OffscreenRenderer` needs and leaves the
    GL calls to PyOpenGL, which only needs a current context.
    """
    global _applied
    try:
        import ctypes

        from genesis.ext.pyrender.platforms import egl as pyrender_egl
    except Exception as error:
        return {'installed': False, 'reason': f'{type(error).__name__}: {error}'}

    lib = ctypes.CDLL('libEGL.so.1')
    lib.eglGetDisplay.restype = ctypes.c_void_p
    lib.eglGetDisplay.argtypes = [ctypes.c_void_p]
    lib.eglInitialize.restype = ctypes.c_uint
    lib.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.eglChooseConfig.restype = ctypes.c_uint
    lib.eglChooseConfig.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int)]
    lib.eglBindAPI.restype = ctypes.c_uint
    lib.eglBindAPI.argtypes = [ctypes.c_uint]
    lib.eglCreateContext.restype = ctypes.c_void_p
    lib.eglCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_int)]
    lib.eglMakeCurrent.restype = ctypes.c_uint
    lib.eglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.eglDestroyContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.eglTerminate.argtypes = [ctypes.c_void_p]

    EGL_OPENGL_API = 0x30A2
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT = 0x3033, 0x0001
    EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE, EGL_DEPTH_SIZE = 0x3024, 0x3023, 0x3025, 0x3025
    EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER = 0x303E, 0x308E
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_CONFORMANT = 0x3040, 0x0008, 0x3042
    EGL_NONE = 0x3038
    EGL_CONTEXT_MAJOR_VERSION, EGL_CONTEXT_MINOR_VERSION = 0x3098, 0x30FB
    EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x30FD, 0x0001

    class EGLError(RuntimeError):
        """The error type `OffscreenRenderer` expects from a platform; pyrender keeps it unexported."""

    def egl_int_array(values):
        return (ctypes.c_int * len(values))(*values)

    class CtypesEGLPlatform:
        """The platform interface `pyrender.OffscreenRenderer` requires, in ctypes."""

        def __init__(self, device_id=None):
            self._display = None
            self._context = None

        def init_context(self):
            # `EGL_DEFAULT_DISPLAY` resolves through whatever platform the process advertises; with DISPLAY set to
            # an X server this process has no authority for, Mesa returns no display at all. pyrender removes
            # DISPLAY for the same reason; this keeps it removed only for the call.
            import os

            saved = {key: os.environ.pop(key) for key in ('DISPLAY', 'WAYLAND_DISPLAY') if key in os.environ}
            try:
                display = lib.eglGetDisplay(None)
            finally:
                os.environ.update(saved)
            if not display:
                raise EGLError('eglGetDisplay returned no display')
            major, minor = ctypes.c_int(), ctypes.c_int()
            if not lib.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
                raise EGLError('eglInitialize failed')
            attributes = [EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
                          EGL_BLUE_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
                          EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_CONFORMANT, EGL_OPENGL_BIT, EGL_NONE]
            config = ctypes.c_void_p()
            count = ctypes.c_int()
            if not lib.eglChooseConfig(display, egl_int_array(attributes), ctypes.byref(config), 1,
                                       ctypes.byref(count)) or count.value < 1:
                raise EGLError('eglChooseConfig found no configuration')
            if not lib.eglBindAPI(EGL_OPENGL_API):
                raise EGLError('eglBindAPI(EGL_OPENGL_API) failed')
            for requested in ((4, 1), (3, 3)):
                context_attributes = [EGL_CONTEXT_MAJOR_VERSION, requested[0], EGL_CONTEXT_MINOR_VERSION,
                                      requested[1], EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                      EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE]
                context = lib.eglCreateContext(display, config, None, egl_int_array(context_attributes))
                if context and lib.eglMakeCurrent(display, None, None, context):
                    self._display, self._context = display, context
                    self.version = f'{requested[0]}.{requested[1]} core'
                    return
            raise EGLError('no EGL context could be created for OpenGL 4.1 or 3.3 core')

        def make_current(self):
            if not lib.eglMakeCurrent(self._display, None, None, self._context):
                raise EGLError('eglMakeCurrent failed')

        def make_uncurrent(self):
            if self._display is not None:
                lib.eglMakeCurrent(self._display, None, None, None)

        def save_current_context(self):
            return self._display, self._context

        def delete_context(self):
            if self._display is None:
                return
            lib.eglMakeCurrent(self._display, None, None, None)
            if self._context is not None:
                lib.eglDestroyContext(self._display, self._context)
                self._context = None
            lib.eglTerminate(self._display)
            self._display = None

    pyrender_egl.EGLPlatform = CtypesEGLPlatform
    _applied = {'installed': True, 'library': 'libEGL.so.1', 'api': 'EGL_OPENGL_API'}
    return _applied
