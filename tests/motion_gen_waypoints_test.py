import pytest
import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.types import Waypoint


@pytest.fixture(scope="module")
def motion_gen():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        use_cuda_graph=False,
    )
    mg = MotionGen(motion_gen_config)
    return mg


def test_plan_waypoints(motion_gen):
    retract_cfg = motion_gen.get_retract_config()
    start_state = JointState.from_position(retract_cfg.view(1, -1))
    kin_state = motion_gen.compute_kinematics(start_state)
    pose1 = kin_state.ee_pose.clone()
    pose1.position[0, 0] -= 0.1
    pose2 = kin_state.ee_pose.clone()
    pose2.position[0, 1] += 0.1
    wps = [Waypoint(pose1), Waypoint(pose2)]

    result = motion_gen.plan_waypoints(start_state, wps, MotionGenPlanConfig(max_attempts=1))
    assert result.success.item()
    assert len(result.waypoint_plans) == 2
    for wp_plan, target in zip(result.waypoint_plans, wps):
        kin = motion_gen.compute_kinematics(wp_plan[-1].unsqueeze(0))
        assert torch.norm(kin.ee_pos_seq - target.pose.position) < 0.01
