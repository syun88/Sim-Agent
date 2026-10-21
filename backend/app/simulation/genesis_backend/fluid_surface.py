"""Smooth rendering of measured SPH particles without a second fluid simulation.

Genesis's macOS splashsurf path spawns a new interpreter per frame. A clean CPU
forkserver retains the imports instead, so a disposable reconstruction job costs
milliseconds and still releases its native heap after each frame. At most one job
is in flight; no queue of old liquid frames can replay after the robot has moved.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np


def reconstruct_surface(points: np.ndarray, radius: float):
    """Reconstruct an appearance mesh; never alter particles or fluid volume."""
    import pysplashsurf

    if not len(points):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32), np.empty((0, 3))
    data, _ = pysplashsurf.reconstruction_pipeline(
        np.ascontiguousarray(points, dtype=np.float32), particle_radius=radius,
        smoothing_length=2., cube_size=.8, iso_surface_threshold=.6,
        mesh_smoothing_weights=True, mesh_smoothing_iters=8,
        normals_smoothing_iters=5, mesh_cleanup=True, compute_normals=True,
        multi_threading=True,
    )
    return (data.mesh.vertices.copy(), data.mesh.triangles.copy(),
            data.point_attributes["normals"].copy())


def _reconstruct_job(connection, sequence, points, radius):
    try:
        started = time.perf_counter()
        arrays = reconstruct_surface(points, radius)
        connection.send((sequence, arrays, time.perf_counter() - started, None))
    except (BrokenPipeError, ConnectionResetError, EOFError):
        # The viewer may close or explicitly reset while this disposable job
        # finishes. A disconnected render consumer is not a physics failure.
        pass
    except Exception as exc:
        try:
            connection.send((sequence, None, 0., f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, ConnectionResetError, EOFError):
            pass
    finally:
        connection.close()


class LiquidSurfaceRenderer:
    """Scene-local adapter using Genesis's normal reconstructed-mesh draw path."""

    def __init__(self, scene, liquid, *, asynchronous=True):
        self.scene, self.liquid = scene, liquid
        self.context = scene.visualizer.context
        self.original_update = self.context.update_sph
        # The server is a separate, clean CPU process. Never fork the Metal/GPU
        # simulation owner. No renderer, Genesis scene, or dynamics runs there.
        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        self.process_context = mp.get_context(method)
        if method == "forkserver":
            self.process_context.set_forkserver_preload(["__main__", "numpy", "pysplashsurf"])
        self.process = self.connection = None
        self.pending_sequence = None
        self.sequence = 0
        self.applied_sequence = -1
        self.last_points = None
        self.last_mesh = None
        self.reconstruction_seconds = 0.
        self.closed = False
        self.asynchronous = asynchronous
        # Override this scene's SPH upload only, without modifying the installed
        # engine or its global functions. on_sph still creates the standard node.
        self.context.update_sph = self.update

    def _start(self, points):
        receive, send = self.process_context.Pipe(duplex=False)
        self.sequence += 1
        self.pending_sequence = self.sequence
        self.process = self.process_context.Process(
            target=_reconstruct_job,
            args=(send, self.sequence, points, self.scene.sph_solver.particle_radius),
            daemon=True,
        )
        self.connection = receive
        self.started = time.monotonic()
        self.process.start()
        send.close()
        self.last_points = points.copy()

    def _receive(self, wait=0.):
        if self.connection is None:
            return
        if not self.connection.poll(wait):
            if not self.process.is_alive():
                raise RuntimeError("Liquid surface reconstruction process exited without a result")
            if time.monotonic() - self.started > 30.:
                raise RuntimeError("Liquid surface reconstruction timed out")
            return
        sequence, arrays, elapsed, error = self.connection.recv()
        self.connection.close()
        self.connection = None
        self.process.join(timeout=.05)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=.1)
        self.process.close()
        self.process = None
        if error:
            raise RuntimeError("Liquid surface reconstruction failed: " + error)
        if sequence != self.pending_sequence or sequence <= self.applied_sequence:
            raise RuntimeError("Out-of-order liquid surface frame")
        self.applied_sequence = sequence
        self.reconstruction_seconds = elapsed
        import trimesh
        import genesis.utils.mesh as mesh_utils
        from genesis.ext import pyrender

        vertices, faces, normals = arrays
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)
        mesh.visual = mesh_utils.surface_uvs_to_trimesh_visual(self.liquid.surface, n_verts=len(vertices))
        self.last_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, envs=0) if len(faces) else None

    def update(self):
        if self.closed:
            return
        from genesis.utils.misc import qd_to_numpy

        solver = self.scene.sph_solver
        if len(solver.entities) != 1 or list(self.context.rendered_envs_idx) != [0]:
            raise RuntimeError("The wine surface renderer requires one liquid and one environment")
        positions = qd_to_numpy(solver.particles_render.pos)
        active = qd_to_numpy(solver.particles_render.active).astype(bool, copy=False)
        start, end = self.liquid.particle_start, self.liquid.particle_end
        points = np.ascontiguousarray(positions[start:end, 0][active[start:end, 0]])
        self._receive()
        if self.process is None and (self.last_points is None or not np.array_equal(points, self.last_points)):
            self._start(points)
        if (self.last_mesh is None or not self.asynchronous) and self.process is not None:
            # Initial build has no previous fluid frame. Later uploads are
            # asynchronous; the motor/physics thread never waits for old jobs.
            self._receive(wait=30.)
        if self.last_mesh is not None:
            self.context.add_dynamic_node(self.liquid, self.last_mesh)

    def close(self):
        self.closed = True
        self.context.update_sph = self.original_update
        self.reset()

    def reset(self):
        """Discard a pending image when the physics scene is explicitly reset."""
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=.2)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.last_points = self.last_mesh = None
