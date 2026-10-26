"""Tests for the analytic gravity-torque margins (planning only).

Run with: uv run pytest -q tests/test_gravity.py
The fixture is a two-link pendulum with hand-computable torques; the
LeKiwi-scale validation (all-zero elbow -3.517 N*m against the engine's
-3.52, tucked-home worst ratio 0.396 against the grid search's 0.39) is
recorded in backend/app/robots/gravity.py.
"""

import numpy as np
import pytest

from backend.app.robots.gravity import gravity_report, gravity_torques
from backend.app.robots.urdf import URDFModel

URDF = """<robot name="pendulum">
  <link name="base"/>
  <link name="upper">
    <inertial><mass value="2"/><origin xyz="0.3 0 0"/></inertial>
  </link>
  <link name="fore">
    <inertial><mass value="1"/><origin xyz="0.2 0 0"/></inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="upper"/>
    <origin xyz="0 0 1"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="3" velocity="1"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper"/><child link="fore"/>
    <origin xyz="0.5 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
</robot>"""


def _model(tmp_path):
    path = tmp_path / "pendulum.urdf"
    path.write_text(URDF)
    return URDFModel(path)


def test_gravity_torques_horizontal_pendulum(tmp_path):
    model = _model(tmp_path)
    torques = gravity_torques(model, ["shoulder", "elbow"], {"shoulder": 0., "elbow": 0.})
    # Upper link: 2 kg with COM 0.3 m out -> 2*9.81*0.3 about the shoulder.
    # Fore link: 1 kg with COM 0.5+0.2 m out -> 1*9.81*0.7 about the
    # shoulder and 1*9.81*0.2 about its own elbow. Both load the shoulder;
    # only the fore link loads the elbow (subtree membership).
    assert torques[0] == pytest.approx(2 * 9.81 * 0.3 + 1 * 9.81 * 0.7)
    assert torques[1] == pytest.approx(1 * 9.81 * 0.2)


def test_gravity_torques_vanish_when_hanging_straight_down(tmp_path):
    model = _model(tmp_path)
    # Rotating the shoulder by +90 deg about +y maps +x to -z: every COM
    # sits on the gravity line through its joint, so no joint is loaded.
    torques = gravity_torques(model, ["shoulder", "elbow"],
                              {"shoulder": np.pi / 2, "elbow": 0.})
    assert torques == pytest.approx([0., 0.], abs=1e-9)


def test_gravity_report_ratios_and_worst_joint(tmp_path):
    model = _model(tmp_path)
    profile = {"chains": {"left": {"joints": ["shoulder", "elbow"]}},
               "joints": [{"name": "shoulder", "effort": 3.},
                          {"name": "elbow", "effort": 1.}],
               "home": {"shoulder": 0., "elbow": 0.}}
    report = gravity_report(profile, model, "left", [0., 0.])
    assert report["ratios"][0] == pytest.approx((2 * 9.81 * 0.3 + 1 * 9.81 * 0.7) / 3.)
    assert report["ratios"][1] == pytest.approx(1 * 9.81 * 0.2 / 1.)
    assert report["worst_joint"] == "shoulder"
    assert report["over_limit"] is True
    # The same posture against generous efforts is within the margin.
    profile["joints"] = [{"name": "shoulder", "effort": 40.},
                         {"name": "elbow", "effort": 10.}]
    report = gravity_report(profile, model, "left", [0., 0.])
    assert report["over_limit"] is False
