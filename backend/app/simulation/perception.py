"""Inject a perception error on purpose, so "would this survive a real camera?" can be measured.

On the simulator `object_pose` is the truth, so every margin in this project has been calibrated against a
perfect sensor. A real arm hands the planner a biased estimate instead. This module installs that bias:

* `object_pose` (what the skills plan with) returns the truth plus a fixed per-object offset,
* `truth_object_pose` keeps the unbiased answer, and verification uses *that*,
* everything is loud: the offset comes from an environment variable and the run prints it, because a harness
  that quietly lies to its own controller is worse than no harness.

The offset is a fixed vector per object (a calibration bias), not white noise: a bias is what actually breaks
planners, since it does not average out between samples.
"""

from __future__ import annotations

import numpy as np

#: Installed attribute so verification can find the unbiased reader without changing every call site.
TRUTH_ATTRIBUTE = 'truth_object_pose'


def install(env, error_m: float, *, seed: int = 0) -> dict:
    """Make `env.object_pose` biased by `error_m`, keeping the unbiased reader available."""
    if getattr(env, '_perception_bias', None) is not None:
        raise RuntimeError('perception bias is already installed on this body')
    truth = env.object_pose
    biases: dict[str, np.ndarray] = {}

    def bias_for(name: str) -> np.ndarray:
        if name not in biases:
            rng = np.random.default_rng(abs(hash((seed, name))) % (2**32))
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction) or 1.
            biases[name] = direction * float(error_m)
        return biases[name]

    def biased(name: str):
        position, rotation = truth(name)
        return np.asarray(position, float) + bias_for(name), rotation

    env.object_pose = biased
    setattr(env, TRUTH_ATTRIBUTE, truth)
    env._perception_bias = {'error_m': float(error_m), 'seed': int(seed)}
    return {'error_m': float(error_m), 'seed': int(seed),
            'objects': sorted(name for name in (getattr(env.frame(), 'objects', {}) or {}))}


def truth_pose(env, name: str):
    """The unbiased pose, whether or not a bias is installed."""
    reader = getattr(env, TRUTH_ATTRIBUTE, None) or env.object_pose
    return reader(name)


def describe(env) -> dict | None:
    return getattr(env, '_perception_bias', None)


class _TruthView:
    """A view of the body whose pose reads are unbiased, for verification only."""

    def __init__(self, env):
        self._env = env

    def object_pose(self, name):
        return truth_pose(self._env, name)

    def __getattr__(self, item):
        return getattr(self._env, item)


def truth_view(env):
    """`env` itself when nothing is perturbed, otherwise a wrapper whose poses are the truth."""
    return env if describe(env) is None else _TruthView(env)
