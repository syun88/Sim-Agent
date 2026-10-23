"""Reproducible desk sampling, independent of a robot and of live physics."""
import secrets
import numpy as np


MAX_SEED=2**32-1


class SpawnSamplingError(ValueError):
    pass


def resolve_seed(seed=None):
    if seed is None:return secrets.randbits(32)
    if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or not 0<=seed<=MAX_SEED:
        raise ValueError('Randomization seed must be an integer from 0 to 4294967295')
    return int(seed)


def desk_candidates(lower,upper,rng,cells=14):
    """One jittered sample per cell, shuffled across the entire rectangle."""
    lower,upper=np.asarray(lower,float),np.asarray(upper,float)
    if lower.shape!=(2,) or upper.shape!=(2,) or np.any(upper<=lower):
        raise ValueError('No usable desk sampling area')
    indices=np.array([(x,y) for x in range(cells) for y in range(cells)])
    points=lower+(indices+rng.random(indices.shape))/cells*(upper-lower)
    return points[rng.permutation(len(points))]


def changed_position(point,previous,minimum=.04):
    return previous is None or np.linalg.norm(np.asarray(point)[:2]-np.asarray(previous)[:2])>=minimum
