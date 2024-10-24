# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import pickle
from einops import repeat
from force_tool.utils.data_utils import SmartDict
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.sim.spawners import materials
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.screw.screw_env_cfg import (
    BaseNutThreadEnvCfg,
    BaseNutTightenEnvCfg,
)
from omegaconf import OmegaConf
##
# Pre-defined configs
from omni.isaac.lab_assets.kuka import KUKA_VICTOR_LEFT_HIGH_PD_CFG, KUKA_VICTOR_LEFT_CFG  # isort: skip


@configclass
class IKRelKukaNutTightenEnvCfg(BaseNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.act_lows = [-0.001, -0.001, -0.001, -0.1, -0.1, -0.1]
        self.act_highs = [0.001, 0.001, 0.001, 0.1, 0.1, 0.1]
        # Set Kuka as robot

        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = [-0.25, -0.2, -0.8]
        scale = [0.001, 0.001, 0.001, 0.01, 0.01, 0.8]
        
        # # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(command_type="pose",
                                                   use_relative_mode=True, ik_method="dls"),
            scale=scale,
        )
        self.actions.gripper_action = mdp.Robotiq3FingerActionCfg(
            asset_name="robot",
            side="left",
            lows=self.act_lows,
            highs=self.act_highs,
        )

        # self.scene.nut.spawn.activate_contact_sensors = True

        # self.scene.bolt.spawn.activate_contact_sensors = True
        # self.scene.contact_sensor = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Nut/factory_nut",
        #     filter_prim_paths_expr= ["{ENV_REGEX_NS}/Bolt/factory_bolt"],
        #     update_period=0.0,
        # )
        # self.rewards.contact_force_penalty = RewTerm(
        #     func=mdp.contact_forces,
        #     params={"threshold":0, "sensor_cfg": SceneEntityCfg(name="contact_sensor")},
        #     weight=0.01)


def reset_scene_with_grasping(env: ManagerBasedEnv, env_ids: torch.Tensor):
    # standard reset

    # set friction
    # robot = env.unwrapped.scene["robot"]
    # robot_material = robot.root_physx_view.get_material_properties()
    # robot_material[..., 0] = 2
    # robot_material[..., 1] = 2
    # robot.root_physx_view.set_material_properties(robot_material, torch.arange(env.scene.num_envs, device="cpu"))

    mdp.reset_scene_to_default(env, env_ids)

    # cached_env_state = SmartDict(pickle.load(open("data/kuka_nut_thread_pre_grasp.pkl", "rb")))
    # # nut_eulers = torch.zeros(1, 3, device=env.device)
    # # nut_eulers[0, 2] = 0.3
    # # nut_quat = math_utils.quat_from_euler_xyz(nut_eulers[:, 0], nut_eulers[:, 1], nut_eulers[:, 2])
    # # cached_env_state["nut"]["root_state"][0, 3:7] = nut_quat
    # cached_env_state.to_tensor(device=env.device)
    # new_env_state = cached_env_state.apply(lambda x: repeat(x, "1 ... -> n ...", n=env.num_envs).clone())
    # env.unwrapped.write_state(new_env_state)

from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase
class reset_scene_to_grasp_state(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        screw_type = self._env.cfg.scene.screw_type
        # cached_pre_grasp_state = SmartDict(pickle.load(open("data/kuka_nut_thread_pre_grasp.pkl", "rb")))
        cached_pre_grasp_state = pickle.load(open(f"data/kuka_{screw_type}_pre_grasp.pkl", "rb"))
        cached_pre_grasp_state = SmartDict(cached_pre_grasp_state).to_tensor(device=env.device)
        # cached_pre_grasp_state = SmartDict(pickle.load(open(f"data/kuka_{}_pre_grasp.pkl", "rb")))
        self.cached_pre_grasp_state = cached_pre_grasp_state.apply(lambda x: repeat(x, "1 ... -> n ...", n=env.num_envs).clone())

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor):
        env.unwrapped.write_state(self.cached_pre_grasp_state[env_ids].clone(), env_ids)

@configclass
class EventCfg:
    """Configuration for events."""
    reset_default = EventTerm(
        # func=reset_scene_with_grasping,
        func=reset_scene_to_grasp_state,
        mode="reset",
    )


@configclass
class IKRelKukaNutThreadEnv(BaseNutThreadEnvCfg):
    """Configuration for the IK-based relative Kuka nut threading environment."""

    def get_default_env_params(self):
        super().get_default_env_params()
        self.env_params.sim.dt = self.env_params.sim.get("dt", 1.0 / 120.0)
        self.env_params.scene.robot = self.env_params.scene.get("robot", OmegaConf.create())
        # self.pre_grasp_path
        robot_params = self.env_params.scene.robot
        robot_params["collision_approximation"] = robot_params.get("collision_approximation", "sdf")
        robot_params["contact_offset"] = robot_params.get("contact_offset", 0.001)
        robot_params["rest_offset"] = robot_params.get("rest_offset", 0.00)
        robot_params["max_depenetration_velocity"] = robot_params.get("max_depenetration_velocity", 0.5)
        robot_params["sleep_threshold"] = robot_params.get("sleep_threshold", None)
        robot_params["stabilization_threshold"] = robot_params.get("stabilization_threshold", None)
        robot_params["static_friction"] = robot_params.get("static_friction", 1)
        robot_params["dynamic_friction"] = robot_params.get("dynamic_friction", 1)
        robot_params["compliant_contact_stiffness"] = robot_params.get("compliant_contact_stiffness", 0.)
        robot_params["compliant_contact_damping"] = robot_params.get("compliant_contact_damping", 0.)

        # By default use the default params in USD
        nut_params = self.env_params.scene.nut
        nut_params["max_depenetration_velocity"] = nut_params.get("max_depenetration_velocity", None)
        nut_params["sleep_threshold"] = nut_params.get("sleep_threshold", None)
        nut_params["stabilization_threshold"] = nut_params.get("stabilization_threshold", None)
        nut_params["linear_damping"] = nut_params.get("linear_damping", None)
        nut_params["angular_damping"] = nut_params.get("angular_damping", None)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.events = EventCfg()
        self.act_lows = [-0.0001, -0.0001, -0.015, -0.01, -0.01, -0.8]
        self.act_highs = [0.0001, 0.0001, 0.015, 0.01, 0.01, 0.]
        scale = [0.01, 0.01, 0.01, 0.01, 0.01, 0.8]
        self.sim.dt = self.env_params.sim.dt

        # self.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
        #     contact_offset=0.002, rest_offset=0.001)
        robot_params = self.env_params.scene.robot
        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if robot_params.collision_approximation == "convexHull":
            self.scene.robot.spawn.usd_path = "assets/victor/victor_left_arm_with_gripper_v2/victor_left_arm_with_gripper_v2.usd"
        self.scene.robot.init_state.pos = [-0.15, -0.5, -0.8]

        self.scene.robot.spawn.collision_props.contact_offset = robot_params.contact_offset
        self.scene.robot.spawn.collision_props.rest_offset = robot_params.rest_offset
        self.scene.robot.spawn.rigid_props.max_depenetration_velocity = robot_params.max_depenetration_velocity
        self.scene.robot.spawn.rigid_props.sleep_threshold = robot_params.sleep_threshold
        self.scene.robot.spawn.rigid_props.stabilization_threshold = robot_params.stabilization_threshold

        self.scene.robot.spawn.physics_material = materials.RigidBodyMaterialCfg(
            static_friction=robot_params.static_friction,
            dynamic_friction=robot_params.dynamic_friction,
            compliant_contact_stiffness=robot_params.compliant_contact_stiffness,
            compliant_contact_damping=robot_params.compliant_contact_damping,
        )

        nut_params = self.env_params.scene.nut
        self.scene.nut.spawn.rigid_props.max_depenetration_velocity = nut_params.max_depenetration_velocity
        self.scene.nut.spawn.rigid_props.sleep_threshold = nut_params.sleep_threshold
        self.scene.nut.spawn.rigid_props.stabilization_threshold = nut_params.stabilization_threshold
        self.scene.nut.spawn.rigid_props.linear_damping = nut_params.linear_damping
        self.scene.nut.spawn.rigid_props.angular_damping = nut_params.angular_damping

        # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=scale,
        )
        #     arm_joint_angles = [
        #   1.4693e+00, -4.3030e-01,  2.2680e+00,  1.5199e+00, -2.1248e+00,
        #       1.0958e+00,  3.9552e-01,
        #     ]
        arm_joint_angles = [
            1.4464e00,
            -4.6657e-01,
            2.2600e00,
            1.5216e00,
            -2.1492e00,
            1.1364e00,
            4.0521e-01,
            8.8714e-02,
            -8.8715e-02,
            7.3445e-01,
            7.3446e-01,
            7.3445e-01,
            -9.1062e-12,
            1.3638e-10,
            2.5490e-10,
            -7.3443e-01,
            -7.3443e-01,
            -7.3443e-01,
        ]
        ori_init_joints = self.scene.robot.init_state.joint_pos

        for key, value in zip(ori_init_joints.keys(), arm_joint_angles):
            if "arm" in key:
                ori_init_joints[key] = value
        self.scene.robot.init_state.joint_pos = ori_init_joints

        self.gripper_act_lows = [-0.005, -0.005]
        self.gripper_act_highs = [0.005, 0.005]
        self.actions.gripper_action = mdp.Robotiq3FingerActionCfg(
            asset_name="robot",
            side="left",
            lows=self.gripper_act_lows,
            highs=self.gripper_act_highs,
            use_relative_mode=True,
            is_accumulate_action=True,
        )
        self.viewer.eye = (0.3, 0, 0.15)
        # self.scene.bolt.spawn.activate_contact_sensors = True
        self.scene.nut.spawn.activate_contact_sensors = True

        # Only contact with the finger tips
        # gripper_path_regex = "{ENV_REGEX_NS}/Robot/.*finger.*_link_3"
        # gripper_prim_paths = sim_utils.find_matching_prims(gripper_path_regex)
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Nut/factory_nut",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*finger.*_link_3"],
            update_period=0.0,
            max_contact_data_count=512,
        )
        # self.rewards.contact_force_penalty = RewTerm(
        #     func=mdp.contact_forces,
        #     params={"threshold":0, "sensor_cfg": SceneEntityCfg(name="contact_sensor")},
        #     weight=0.01)

# @configclass
# class RelFloatNutTightenEnvCfg_PLAY(RelFloatNutTightenEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()
#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         # disable randomization for play
#         self.observations.policy.enable_corruption = False
