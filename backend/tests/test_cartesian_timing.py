import numpy as np
from backend.app.control.motion import joint_spline
from backend.app.control.path_timing import PathTiming

class CartesianFixture:
    def evaluate(self,q):return np.asarray(q),np.eye(3),None,None

LIMITS={'linear_velocity_m_s':.25,'linear_acceleration_m_s2':1.,
        'angular_velocity_rad_s':2.5,'angular_acceleration_rad_s2':12.,'minimum_duration_s':.2}


def test_short_correction_keeps_acceleration_time_and_dense_sampling_does_not_add_waits():
    kin=CartesianFixture();start=np.zeros(3);end=np.array([.006,0.,0.])
    tiny=PathTiming(kin,None,start,end,np.ones(3),LIMITS).duration
    assert tiny>=.2
    end=np.array([.4,0.,0.])
    coarse=joint_spline(np.linspace(start,end,3),np.ones(3))
    dense=joint_spline(np.linspace(start,end,81),np.ones(3))
    a=PathTiming(kin,coarse,start,end,np.ones(3),LIMITS).duration
    b=PathTiming(kin,dense,start,end,np.ones(3),LIMITS).duration
    assert abs(a-b)<.002
    assert 2.<a<2.5


def test_local_time_profile_respects_velocity_and_acceleration_without_global_corner_wait():
    from backend.app.control.path_timing import PathTiming
    kin=CartesianFixture();points=np.array([[0.,0.,0.],[.15,0.,0.],[.151,.001,0.],[.4,.001,0.]])
    curve=joint_spline(points,np.ones(3));timing=PathTiming(kin,curve,points[0],points[-1],np.ones(3),LIMITS)
    ts=np.linspace(0.,timing.duration,4001)
    phases=np.array([timing.phase(t) for t in ts])
    assert np.all(np.diff(phases[:,0])>=-1e-12)
    velocity=curve.derivative()(phases[:,0])*phases[:,1,None]
    assert np.max(np.linalg.norm(velocity,axis=1))<=.25
    # A long straight portion must continue travelling at the configured rate
    # even though the short corner needs a lower local speed.
    assert np.max(np.linalg.norm(velocity[(phases[:,0]>.1)&(phases[:,0]<.3)],axis=1))>.2
    assert 1.7<timing.duration<4.
    assert phases[-1,0]==1. and phases[-1,1]==0.
