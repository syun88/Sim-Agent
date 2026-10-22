import numpy as np
from scipy.spatial.transform import Rotation
from backend.app.robots.urdf import rotation_residual_jacobian


def test_rotation_residual_derivative_matches_world_angular_perturbations():
    rng=np.random.default_rng(71)
    for angle in (1e-8,.3,1.5,3.13):
        attitude=Rotation.random(random_state=rng).as_matrix()
        axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
        target=Rotation.from_rotvec(axis*angle).as_matrix()@attitude
        angular=rng.normal(size=(3,7))
        analytic=rotation_residual_jacobian(attitude,target,angular)
        numerical=[]
        for direction in angular.T:
            plus=Rotation.from_rotvec(direction*1e-6).as_matrix()@attitude
            minus=Rotation.from_rotvec(-direction*1e-6).as_matrix()@attitude
            numerical.append((Rotation.from_matrix(target@plus.T).as_rotvec()-Rotation.from_matrix(target@minus.T).as_rotvec())/2e-6)
        np.testing.assert_allclose(analytic,np.array(numerical).T,atol=1e-7)
