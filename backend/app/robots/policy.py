"""SI-unit simulation/control policy for imported robots, independent of names."""
import hashlib
import json

IMPORTED_ROBOT_POLICY={
    'physics_dt_max_s':.005,'physics_dt_per_cube_meter':.175,'physics_dt_min_s':.0005,'physics_substeps':2,
    'arm_response_rad_s':20.,'finger_response_rad_s':20.,'kp_min':10.,'kp_max':500.,
    'grip_weight_margin':8.,'grip_wrench_weight_margin':4.,'friction_assumption':.6,'contact_radius_fraction':.25,
    'surface_floor_m':.0006,'surface_cube_fraction':.025,'surface_exit_scale':1.25,'surface_exit_extra_m':.0002,
    'penetration_floor_m':.001,'penetration_cube_fraction':.05,
    'tool_limits':{'linear_velocity_m_s':.5,'linear_acceleration_m_s2':3.,
        'angular_velocity_rad_s':4.,'angular_acceleration_rad_s2':24.,
        'joint_acceleration_rad_s2':30.,'minimum_duration_s':.2}}


def policy_signature():
    return hashlib.sha256(json.dumps(IMPORTED_ROBOT_POLICY,sort_keys=True).encode()).hexdigest()
