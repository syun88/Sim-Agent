"""A pinch is planned with one jaw frame: the approach picks it, every later step reads it.

Measured 2026-10-24 on a 4-DOF arm: planning the approach with the inferred grasp rotation made a
spawn the sampler had validated unreachable (the frame stopped 6.6mm short of the object, while
the frame the arm's own joints reach solved to 1e-11), and after that was fixed, asking for the
inferred frame again in closing, lifting and holding failed in the same way. These tests pin the
two properties that were missing: the inferred frame is always tried first, so an arm that can
hold it is planned exactly as before, and one frame per pinch is recorded and read back.
"""
import numpy as np

from backend.app.robots.executor import ImportedRobotSkills
from backend.app.robots.urdf import URDFModel, solve_profile_ik

INFERRED = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
ROLLED = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])  # same pad normal, different roll


def pinch_fixture(tmp_path, *, achieved, offset, tolerance=25.):
    source = tmp_path / 'cartesian.urdf'
    text = '<robot name="cartesian"><link name="base"/>'
    parent = 'base'
    for name, axis, lower, upper in [('x', '1 0 0', -.5, .5), ('y', '0 1 0', -.5, .5), ('z', '0 0 1', 0., .8)]:
        child = 'palm' if name == 'z' else 'carriage_' + name
        text += (f'<link name="{child}"/><joint name="{name}" type="prismatic"><parent link="{parent}"/>'
                 f'<child link="{child}"/><axis xyz="{axis}"/><limit lower="{lower}" upper="{upper}" '
                 f'effort="1" velocity="1"/></joint>')
        parent = child
    source.write_text(text + '</robot>')

    class Fixture:
        attitude_tolerance_deg = tolerance

        def __init__(self):
            self.model = URDFModel(source)
            self.profile = {'urdf': str(source), 'root_transform': np.eye(4).tolist(),
                            'chains': {'left': {'joints': ['x', 'y', 'z'], 'end_link': 'palm',
                                                'tool_offset': [0., 0., 0.]}},
                            'home': {'x': 0., 'y': 0., 'z': .3}, 'grippers': {'left': {'joint': 'jaw'}}}
            self.grippers = {'left': {'joint': 'jaw', 'followers': {}, 'open_q': 0., 'closed_q': 1.,
                                      'grasp_rotation': INFERRED, 'pad_region': {'normal': [0., 0., 1.]},
                                      'sweep': [{'q': 0., 'face_a': -.02, 'face_b': .02},
                                                {'q': 1., 'face_a': .02, 'face_b': .06}]}}
            self._spawn_sampler = self
            # A test that expects the inferred frame alone must fail if the free solve never ran.
            self.free_solve_succeeded = solve_profile_ik(
                self.profile, 'left', np.array([.1, .1, .3]), None, model=self.model,
                q_init=np.zeros(3), snapshot={})['success']

        def home_arm_qpos(self, side):
            return np.zeros(3)

        def achieved_attitude(self, arm, q, snapshot):
            return achieved

        def attitude_offset_deg(self, frame, nominal, gripper):
            return offset

    return Fixture(), ImportedRobotSkills(Fixture(), None), np.array([.1, .1, .3])


def test_arm_that_holds_the_inferred_frame_is_planned_with_it_alone(tmp_path):
    env, skills, target = pinch_fixture(tmp_path, achieved=INFERRED.copy(), offset=0.)
    assert env.free_solve_succeeded
    rotations = skills._approach_rotations('left', target, {})
    assert env.free_solve_succeeded
    assert len(rotations) == 1
    assert np.allclose(rotations[0], INFERRED)


def test_achieved_frame_is_only_a_second_candidate_and_the_inferred_one_stays_first(tmp_path):
    env, skills, target = pinch_fixture(tmp_path, achieved=ROLLED, offset=12.)
    assert env.free_solve_succeeded
    rotations = skills._approach_rotations('left', target, {})
    assert env.free_solve_succeeded
    assert len(rotations) == 2
    assert np.allclose(rotations[0], INFERRED)
    assert np.allclose(rotations[1], ROLLED)


def test_frame_beyond_the_samplers_tolerance_is_not_offered(tmp_path):
    env, skills, target = pinch_fixture(tmp_path, achieved=ROLLED, offset=40.)
    assert env.free_solve_succeeded
    assert len(skills._approach_rotations('left', target, {})) == 1


def test_unreachable_target_keeps_the_inferred_frame_alone(tmp_path):
    env, skills, target = pinch_fixture(tmp_path, achieved=ROLLED, offset=12.)
    assert len(skills._approach_rotations('left', np.array([5., 5., 5.]), {})) == 1


def test_attitude_reads_the_frame_the_approach_recorded(tmp_path):
    env, skills, target = pinch_fixture(tmp_path, achieved=ROLLED, offset=12.)
    assert np.allclose(skills._attitude('left'), INFERRED)
    skills._pinch_frames = {'left': ROLLED}
    assert np.allclose(skills._attitude('left'), ROLLED)
    assert np.allclose(skills._nominal_attitude('left'), INFERRED)
