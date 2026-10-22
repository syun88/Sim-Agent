"""Local velocity/acceleration timing of an already collision-checked path.

Forward/backward speed passes slow down the portions that need it; a difficult
corner does not impose its speed on the entire robot motion. Geometry is never
changed by this module.
"""
import numpy as np
from scipy.spatial.transform import Rotation


class PathTiming:
    def __init__(self,kinematics,curve,start,end,joint_velocity,limits):
        grid=np.linspace(0.,1.,257)
        if curve is not None:
            combined=np.unique(np.r_[grid,curve.x,(curve.x[:-1]+curve.x[1:])/2.])
            kept=[0.]
            for s in combined:
                if s-kept[-1]>1e-5 and 1.-s>1e-5:kept.append(float(s))
            grid=np.r_[kept,1.]
        q=np.array([start+(end-start)*s if curve is None else curve(s) for s in grid])
        dq=(np.repeat((end-start)[None,:],len(grid),axis=0) if curve is None else curve.derivative()(grid))
        ddq=np.zeros_like(dq) if curve is None else curve.derivative(2)(grid)
        poses=[kinematics.evaluate(value)[:2] for value in q]
        p=np.array([point for point,rotation in poses]);r=np.array([rotation for point,rotation in poses])
        dp=np.gradient(p,grid,axis=0,edge_order=2);ddp=np.gradient(dp,grid,axis=0,edge_order=2)
        angular=Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).as_rotvec()/np.diff(grid)[:,None]
        omega=np.vstack([angular[0],(angular[:-1]+angular[1:])/2.,angular[-1]])
        domega=np.gradient(omega,grid,axis=0,edge_order=2)
        rate=np.full(len(grid),1000.);accel=np.full(len(grid),1000.)
        def velocity_bound(derivative,bound):
            nonlocal rate
            rate=np.minimum(rate,np.min(np.asarray(bound)*.95/np.maximum(abs(derivative),1e-12),axis=1))
        def acceleration_bound(first,second,bound):
            nonlocal rate,accel
            # Reserve half the acceleration for path curvature and half for
            # increasing/decreasing speed along that path.
            rate=np.minimum(rate,np.sqrt(np.min(np.asarray(bound)*.5/np.maximum(abs(second),1e-12),axis=1)))
            accel=np.minimum(accel,np.min(np.asarray(bound)*.5/np.maximum(abs(first),1e-12),axis=1))
        velocity_bound(dq,joint_velocity)
        acceleration_bound(dq,ddq,limits.get('joint_acceleration_rad_s2',20.))
        velocity_bound(np.linalg.norm(dp,axis=1)[:,None],limits['linear_velocity_m_s'])
        acceleration_bound(np.linalg.norm(dp,axis=1)[:,None],np.linalg.norm(ddp,axis=1)[:,None],limits['linear_acceleration_m_s2'])
        velocity_bound(np.linalg.norm(omega,axis=1)[:,None],limits['angular_velocity_rad_s'])
        acceleration_bound(np.linalg.norm(omega,axis=1)[:,None],np.linalg.norm(domega,axis=1)[:,None],limits['angular_acceleration_rad_s2'])
        speed=rate.copy();speed[0]=speed[-1]=0.
        ds=np.diff(grid);a=np.minimum(accel[:-1],accel[1:])
        for i in range(len(ds)):speed[i+1]=min(speed[i+1],np.sqrt(speed[i]**2+2.*a[i]*ds[i]))
        for i in range(len(ds)-1,-1,-1):speed[i]=min(speed[i],np.sqrt(speed[i+1]**2+2.*a[i]*ds[i]))
        dt=2.*ds/(speed[:-1]+speed[1:])
        stretch=max(1.,limits.get('minimum_duration_s',.2)/sum(dt))
        self.progress=grid;self.speed=speed/stretch
        self.times=np.r_[0.,np.cumsum(dt*stretch)]
        self.duration=float(self.times[-1])

    def phase(self,elapsed):
        if elapsed<=0.:return 0.,0.
        if elapsed>=self.duration:return 1.,0.
        i=int(np.searchsorted(self.times,elapsed,side='right')-1)
        dt=elapsed-self.times[i];width=self.times[i+1]-self.times[i]
        acceleration=(self.speed[i+1]-self.speed[i])/width
        return (float(self.progress[i]+self.speed[i]*dt+.5*acceleration*dt**2),
                float(self.speed[i]+acceleration*dt))
