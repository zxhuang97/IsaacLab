# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pickle
import torch
from typing import Literal, Sequence
import copy

from numba.core import event
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp import robot_tool_pose
import omni.physx.scripts.utils as physx_utils
from einops import repeat
from force_tool.utils.data_utils import SmartDict, read_h5_dict
from force_tool.utils.curobo_utils import CuRoboArm
from omegaconf import OmegaConf
from pxr import Usd, UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject

from omni.isaac.lab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    OperationalSpaceControllerActionCfg,
)
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ManagerTermBase
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import TiledCameraCfg


import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.screw.screw_env_cfg import (
    BaseNutThreadEnvCfg,
    BaseNutTightenEnvCfg,
)
from omni.isaac.lab.utils.noise import GaussianNoiseCfg, RadialNoiseCfg
from omni.isaac.lab.utils.modifiers import NoiseModifierCfg
from curobo.types.math import Pose
from curobo.types.base import TensorDeviceType
import time
import numpy as np

##
# Pre-defined configs
from omni.isaac.lab_assets.kuka import KUKA_VICTOR_LEFT_HIGH_PD_CFG


@configclass
class IKRelKukaNutTightenEnvCfg(BaseNutTightenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.act_lows = [-0.001, -0.001, -0.001, -0.1, -0.1, -0.1]
        self.act_highs = [0.001, 0.001, 0.001, 0.1, 0.1, 0.1]

        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = [-0.25, -0.2, -0.8]
        scale = [0.001, 0.001, 0.001, 0.01, 0.01, 0.8]

        # # override actions
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["victor_left_arm_joint.*"],
            body_name="victor_left_tool0",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=scale,
        )
        self.actions.gripper_action = mdp.Robotiq3FingerActionCfg(
            asset_name="robot",
            side="left",
            lows=self.act_lows,
            highs=self.act_highs,
        )


class GraspResetEventTermCfg(EventTerm):
    def __init__(
        self,
        reset_target: Literal[
            "pre_grasp", "grasp", "mate", "rigid_grasp", "rigid_grasp_open_align"
        ] = "grasp",
        nut_rel_pose: tuple[float, float, float, float] | None = None,
        reset_range_scale: float = 1.0,
        reset_joint_std: float = 0.0,
        reset_randomize_mode: Literal["task", "joint", None] = "task",
        reset_use_adr: bool = False,
        reset_close_gripper: Literal["adaptive", "close", None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reset_target = reset_target
        self.nut_rel_pose = nut_rel_pose
        self.reset_range_scale = reset_range_scale
        self.reset_joint_std = reset_joint_std
        self.reset_randomize_mode = reset_randomize_mode
        self.reset_use_adr = reset_use_adr
        self.reset_close_gripper = reset_close_gripper

class reset_scene_to_grasp_state(ManagerTermBase):
    def __init__(self, cfg: GraspResetEventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        screw_type = env.cfg.scene.screw_type
        col_approx = env.cfg.params.scene.robot.collision_approximation
        cached_state = pickle.load(
            open(f"cached/{col_approx}/{screw_type}/kuka_{cfg.reset_target}.pkl", "rb")
        )
        cached_state = SmartDict(cached_state).to_tensor(device=env.device)
        self.cached_state = cached_state.apply(
            lambda x: repeat(x, "1 ... -> n ...", n=env.num_envs).clone()
        )

        # randomization parameters
        self.tensor_args = TensorDeviceType(device=env.device)
        self.robot_base_pose = Pose.from_list([-0.15, -0.5, -0.8, 1, 0, 0, 0], self.tensor_args)
        self.curobo_arm = CuRoboArm(
            "victor_left.yml",
            external_asset_path=os.path.abspath("assets/victor"),
            base_pose=self.robot_base_pose,
            num_ik_seeds=10,
            device=env.device,
        )
        self.curobo_arm.update_world()
        self.reset_randomize_mode = cfg.reset_randomize_mode
        self.reset_trans_low = (
            torch.tensor([-0.03, -0.03, -0.0], device=env.device) * cfg.reset_range_scale
        )
        self.reset_trans_high = (
            torch.tensor([0.03, 0.03, 0.04], device=env.device) * cfg.reset_range_scale
        )
        self.reset_rot_std = 0.2 * cfg.reset_range_scale
        self.reset_joint_std = cfg.reset_joint_std
        self.reset_use_adr = cfg.reset_use_adr
        self.reset_close_gripper = cfg.reset_close_gripper
        self.nut_rel_pose = cfg.nut_rel_pose
        self.rand_init_robot_joint = None
        self.rand_init_nut_state = None
        self.num_buckets = int(2e3)
        # self.num_buckets = 1
        self.bucket_update_freq = 4
        self.gripper_action = mdp.Robotiq3FingerAction(mdp.Robotiq3FingerActionCfg(
            asset_name="robot",
            side="left",
            lows=[-0.005, -0.005],
            highs=[0.005, 0.005],
            use_relative_mode=True,
            is_accumulate_action=False,
            keep_grasp_state=False
        ), env)
        # self.update_random_initializations(env)

    def update_random_initializations(self, env: ManagerBasedEnv):
        cached_state = self.cached_state[0:1].clone()
        B = self.num_buckets
        noise_scale = 1.0
        if self.reset_use_adr:
            # step a: activate noise
            # step b: maximize noise
            raise NotImplementedError
        full_joint_state = cached_state["robot"]["joint_state"]["position"]
        full_nut_state = cached_state["nut"]["root_state"]

        if self.reset_randomize_mode == "task":
            arm_state = full_joint_state[:, :7]
            default_tool_pose = self.curobo_arm.forward_kinematics(arm_state.clone()).ee_pose
            nut_rel_pose = Pose.from_list(self.nut_rel_pose, self.tensor_args)
            default_nut_pose = default_tool_pose.multiply(nut_rel_pose)
            default_nut_pose = default_nut_pose.repeat(B)
            delta_trans = (
                torch.rand((B, 3), device=env.device) * (self.reset_trans_high - self.reset_trans_low)
                + self.reset_trans_low
            )
            delta_trans *= noise_scale

            delta_rot = (
                2 * torch.rand((B, 3), device=env.device) * self.reset_rot_std - self.reset_rot_std
            )
            delta_quat = math_utils.quat_from_euler_xyz(
                delta_rot[:, 0], delta_rot[:, 1], delta_rot[:, 2]
            )

            # [-0.15, -0.5, -0.8]
            delta_pose = Pose(position=torch.zeros((B, 3), device=env.device), quaternion=delta_quat)
            randomized_nut_pose = default_nut_pose.multiply(delta_pose)
            randomized_nut_pose.position += delta_trans
            randomized_tool_pose = randomized_nut_pose.multiply(nut_rel_pose)

            # this should give you randomized_nut_pose.
            # print(math_utils.combine_frame_transforms(randomized_tool_pose.position[:1], randomized_tool_pose.quaternion[:1], nut_rel_pose.position, nut_rel_pose.quaternion))

            ik_result = self.curobo_arm.compute_ik(randomized_tool_pose)
            randomized_joint_state = full_joint_state.repeat(B, 1).contiguous()
            randomized_joint_state[:, :7] = ik_result.solution.squeeze(1)
            randomized_nut_state = full_nut_state.repeat(B, 1).contiguous()
            if self.reset_close_gripper is not None:
                cur_gripper_joint = full_joint_state[:, self.gripper_action._joint_ids]
                target_gripper_joint = torch.zeros(B, 11, device=env.device, dtype=torch.float32)
                cur_finger_open = mdp.inverse_compute_finger_angles_jit(cur_gripper_joint)[:, 0]
                cur_finger_scissor = mdp.inverse_compute_scissor_angle_jit(cur_gripper_joint[:, -2:])[:, 0]
                close_finger_open = torch.tensor(0.32, device=env.device)
                close_finger_scissor = torch.tensor(0.275, device=env.device)
                if self.reset_close_gripper == "close":
                    tgt_finger_open = close_finger_open
                    tgt_finger_scissor = close_finger_scissor
                elif self.reset_close_gripper == "adaptive":
                    global_step = env._sim_step_counter // env.cfg.decimation
                    ratio = global_step / 80 * 600 # fully close at 600 iters
                    tgt_finger_open = cur_finger_open + (close_finger_open - cur_finger_open) * ratio
                    tgt_finger_scissor = cur_finger_scissor + (close_finger_scissor - cur_finger_scissor) * ratio
                tgt_finger_open = tgt_finger_open.repeat(B)
                tgt_finger_scissor = tgt_finger_scissor.repeat(B)
                mdp.compute_finger_angles_jit(tgt_finger_open, target_gripper_joint)
                target_gripper_joint[:, 3:6] = target_gripper_joint[:, 0:3]
                target_gripper_joint[:, 6:9] = target_gripper_joint[:, 0:3]
                mdp.compute_scissor_angle_jit(tgt_finger_scissor, target_gripper_joint[:, 9:11])
                randomized_joint_state[:, self.gripper_action._joint_ids] = target_gripper_joint
            nut_pose_w = self.robot_base_pose.repeat(B).multiply(randomized_nut_pose)
            randomized_nut_state[:, :3] = nut_pose_w.position
            randomized_nut_state[:, 3:7] = nut_pose_w.quaternion
        elif self.reset_randomize_mode == "joint":
            raise NotImplementedError
            randomized_joint_state = torch.randn_like(full_joint_state) * self.reset_joint_std + full_joint_state
        else:  # no randomization
            randomized_joint_state = full_joint_state.repeat(B, 1).contiguous()
        # self.rand_init_configurations = randomized_joint_state.detach().cpu().numpy()
        self.rand_init_robot_joint = randomized_joint_state
        self.rand_init_nut_state = randomized_nut_state

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor):
        cached_state = self.cached_state[env_ids].clone()
        global_step = env._sim_step_counter // env.cfg.decimation
        if global_step % int(self.bucket_update_freq * env.unwrapped.max_episode_length) == 0:
            with torch.inference_mode(False):
                self.update_random_initializations(env)
        if self.reset_randomize_mode is not None:
            # draw random initializations
            select = torch.randint(0, self.num_buckets, (env_ids.shape[0],), device=env.device)
            randomized_joint_state = self.rand_init_robot_joint[select].clone()
            # select = np.random.choice(self.num_buckets, env_ids.shape[0], replace=True)
            # randomized_joint_state = self.rand_init_configurations[select].copy()
            # randomized_joint_state = torch.tensor(randomized_joint_state, device=env.device)
            cached_state["robot"]["joint_state"]["position"] = randomized_joint_state
            cached_state["robot"]["joint_state"]["position_target"] = randomized_joint_state
            cached_state["nut"]["root_state"] = self.rand_init_nut_state[select].clone()

        env.unwrapped.write_state(cached_state, env_ids)


class DTWReferenceTrajRewardCfg(RewTerm):
    def __init__(self, his_traj_len: int = 10, soft_dtw_gamma: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.his_traj_len = his_traj_len
        self.soft_dtw_gamma = soft_dtw_gamma


class DTWReferenceTrajReward(ManagerTermBase):
    def __init__(self, cfg: DTWReferenceTrajRewardCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        screw_type = env.cfg.scene.screw_type
        col_approx = env.cfg.params.scene.robot.collision_approximation
        ref_traj_path = f"cached/{col_approx}/{screw_type}/kuka_disassembly.h5"
        ref_traj = read_h5_dict(ref_traj_path)
        self.nut_ref_pos_traj = torch.tensor(ref_traj["pos"], device=env.device).flip(1)
        self.nut_ref_quat_traj = torch.tensor(ref_traj["quat"], device=env.device).flip(1)
        self.nut_traj_his = torch.zeros((env.num_envs, cfg.his_traj_len, 3), device=env.device)
        self.soft_dtw_criterion = mdp.SoftDTW(use_cuda=True, gamma=cfg.soft_dtw_gamma)

    def reset(self, env_ids: torch.Tensor):
        scene = self._env.unwrapped.scene
        nut_frame = scene["nut_frame"]
        nut_cur_pos = nut_frame.data.target_pos_w - scene.env_origins[:, None]
        self.nut_traj_his[env_ids] = nut_cur_pos[env_ids]

    def __call__(self, env: ManagerBasedEnv):
        nut_frame = env.unwrapped.scene["nut_frame"]
        cur_nut_pos = nut_frame.data.target_pos_w[:, 0] - env.unwrapped.scene.env_origins

        imitation_rwd, new_nut_traj_his = mdp.get_imitation_reward_from_dtw(
            self.nut_ref_pos_traj, cur_nut_pos, self.nut_traj_his, self.soft_dtw_criterion, env.device
        )
        self.nut_traj_his = new_nut_traj_his
        return imitation_rwd


def spawn_nut_with_rigid_grasp(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    stage = stage_utils.get_current_stage()
    tool_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/victor_left_tool0")
    xfCache = UsdGeom.XformCache()
    tool_pose = xfCache.GetLocalToWorldTransform(tool_prim)
    tool_pos = tuple(tool_pose.ExtractTranslation())
    tool_pos = torch.tensor(tool_pos)[None]
    tool_quat = tool_pose.ExtractRotationQuat()
    tool_quat = [
        tool_quat.real,
        tool_quat.imaginary[0],
        tool_quat.imaginary[1],
        tool_quat.imaginary[2],
    ]
    tool_quat = torch.tensor(tool_quat)[None]

    grasp_rel_pos = torch.tensor(translation)[None]
    grasp_rel_quat = torch.tensor(orientation)[None]

    nut_pos, nut_quat = math_utils.combine_frame_transforms(
        tool_pos, tool_quat, grasp_rel_pos, grasp_rel_quat
    )
    #    nut.init_state.pos = (0, 0, 0.02)
    #     nut.init_state.rot = (0, 1, 0, 0)
    nut_prim = sim_utils.spawn_from_usd(prim_path, cfg, nut_pos[0], nut_quat[0])
    return nut_prim


def create_fixed_joint(env: ManagerBasedEnv, env_ids: torch.Tensor):
    stage = stage_utils.get_current_stage()
    for i in range(env.num_envs):
        child_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/victor_left_tool0")
        parent_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Nut/factory_nut")
        physx_utils.createJoint(stage, "Fixed", child_prim, parent_prim)
        # physx_utils.createJoint(stage, "Fixed", parent_prim, child_prim)


@configclass
class EventCfg:
    """Configuration for events."""


def terminate_if_nut_fallen(env):
    # relative pose between gripper and nut
    nut_root_pose = env.unwrapped.scene["nut"].read_root_state_from_sim()
    gripper_state_w = robot_tool_pose(env)
    # relative_pos = nut_root_pose[:, :3] - gripper_state_w[:, :3]
    relative_pos, relative_quat = math_utils.subtract_frame_transforms(
        nut_root_pose[:, :3], nut_root_pose[:, 3:7], gripper_state_w[:, :3], gripper_state_w[:, 3:7]
    )
    ideal_relative_pose = torch.tensor([[0.0018, 0.9668, -0.2547, -0.0181]], device=env.device)
    ideal_relative_pose = ideal_relative_pose.repeat(relative_pos.shape[0], 1)
    quat_dis = math_utils.quat_error_magnitude(relative_quat, ideal_relative_pose)
    dis = torch.norm(relative_pos, dim=-1)
    return torch.logical_or(dis > 0.03, quat_dis > 0.3)


def terminate_if_far_from_bolt(env):
    diff = mdp.rel_nut_bolt_tip_distance(env)
    return torch.norm(diff, dim=-1) > 0.05


def initialize_contact_properties(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    contact_offset: float,
    rest_offset: float,
):
    asset: RigidObject | Articulation = env.unwrapped.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    # Note: shape of contact_offset and bodies are not matched
    # since there're several virtual body in kuka.
    # if asset_cfg.body_ids == slice(None):
    #     body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    # else:
    #     body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    cur_contact_offset = asset.root_physx_view.get_contact_offsets()
    cur_contact_offset[env_ids] = contact_offset
    asset.root_physx_view.set_contact_offsets(cur_contact_offset, env_ids)
    cur_rest_offset = asset.root_physx_view.get_rest_offset()
    cur_rest_offset[env_ids] = rest_offset
    asset.root_physx_view.set_rest_offset(cur_rest_offset, env_ids)


# curriculum
def modify_noise_scale(env: ManagerBasedEnv, env_ids: torch.Tensor, begin_steps: int, end_steps: int):
    modifier = env.observation_manager.get_term_cfg("policy", "nut_pos").modifiers[0].func
    # linear
    scale = max(0.0, env.common_step_counter - begin_steps) / (begin_steps - end_steps)
    scale = min(1.0, scale)
    modifier.noise_scale = scale


@configclass
class IKRelKukaNutThreadEnvCfg(BaseNutThreadEnvCfg):
    """Configuration for the IK-based relative Kuka nut threading environment."""

    def get_default_env_params(self):
        super().get_default_env_params()
        self.params.sim.dt = self.params.sim.get("dt", 1.0 / 120.0)
        self.params.use_factory_params = self.params.get("use_factory_params", False)
        self.params.scene.robot = self.params.scene.get("robot", OmegaConf.create())
        self.params.agent = self.params.get("agent", OmegaConf.create())
        # self.pre_grasp_path
        robot_params = self.params.scene.robot
        robot_params.collision_approximation = robot_params.get(
            "collision_approximation", "convexHull2"
        )
        robot_params.contact_offset = robot_params.get("contact_offset", 0.002)
        robot_params.rest_offset = robot_params.get("rest_offset", 0.001)
        robot_params.max_depenetration_velocity = robot_params.get("max_depenetration_velocity", 0.5)
        robot_params.sleep_threshold = robot_params.get("sleep_threshold", None)
        robot_params.stabilization_threshold = robot_params.get("stabilization_threshold", None)
        robot_params.static_friction = robot_params.get("static_friction", 3)
        robot_params.dynamic_friction = robot_params.get("dynamic_friction", 3)
        robot_params.compliant_contact_stiffness = robot_params.get("compliant_contact_stiffness", 0.0)
        robot_params.compliant_contact_damping = robot_params.get("compliant_contact_damping", 0.0)
        robot_params.arm_stiffness = robot_params.get("arm_stiffness", 300.0)
        robot_params.arm_damping = robot_params.get("arm_damping", 50.0)
        robot_params.gripper_stiffness = robot_params.get("gripper_stiffness", 100)
        robot_params.gripper_damping = robot_params.get("gripper_damping", 1e2)
        robot_params.gripper_effort_limit = robot_params.get("gripper_effort_limit", 200.0)

        nut_params = self.params.scene.nut
        nut_params.rigid_grasp = nut_params.get("rigid_grasp", True)

        action_params = self.params.actions
        action_params.controller_type = action_params.get("controller_type", "ik")
        action_params.ik_lambda = action_params.get("ik_lambda", 0.001)
        action_params.keep_grasp_state = action_params.get("keep_grasp_state", True)
        action_params.uni_rotate = action_params.get("uni_rotate", True)

        obs_params = self.params.observations
        obs_params.use_tiled_camera = obs_params.get("use_tiled_camera", False)
        obs_params.history_length = obs_params.get("history_length", 1)
        obs_params.flatten_history_dim = obs_params.get("flatten_history_dim", True)
        obs_params.include_action = obs_params.get("include_action", True)
        obs_params.include_wrench = obs_params.get("include_wrench", True)
        obs_params.wrench_target_body = obs_params.get("wrench_target_body", "victor_left_tool0")
        obs_params.include_tool = obs_params.get("include_tool", False)
        obs_params.nut_pos = obs_params.get("nut_pos", OmegaConf.create())
        obs_params.nut_pos.noise_std = obs_params.nut_pos.get("noise_std", 0.0)
        obs_params.nut_pos.bias_mean = obs_params.nut_pos.get("bias_mean", 0.0)
        obs_params.nut_pos.bias_std = obs_params.nut_pos.get("bias_std", 0.0)
        obs_params.critic_privil_obs = obs_params.get("critic_privil_obs", False)
        obs_params.use_obs_camera = obs_params.get("use_obs_camera", False)
        obs_params.obs_camera_type = obs_params.get("obs_camera_type", ["distance_to_image_plane"])

        rewards_params = self.params.rewards
        rewards_params.dtw_ref_traj_w = rewards_params.get("dtw_ref_traj_w", 0.0)
        rewards_params.coarse_nut_w = rewards_params.get("coarse_nut_w", 1)
        rewards_params.fine_nut_w = rewards_params.get("fine_nut_w", 2.0)
        rewards_params.upright_reward_w = rewards_params.get("upright_reward_w", 0.3)
        rewards_params.success_w = rewards_params.get("success_w", 1.0)
        rewards_params.action_rate_w = rewards_params.get("action_rate_w", -0.0)
        rewards_params.action_l2_w = rewards_params.get("action_l2_w", -0.0)
        rewards_params.contact_force_penalty_w = rewards_params.get("contact_force_penalty_w", -0.01)
        rewards_params.incoming_wrench_mag_w = rewards_params.get("incoming_wrench_mag_w", 0.0)
        termination_params = self.params.terminations
        termination_params.far_from_bolt = termination_params.get("far_from_bolt", False)
        termination_params.nut_fallen = termination_params.get("nut_fallen", False)

        events_params = self.params.events
        events_params.reset_target = events_params.get("reset_target", "rigid_grasp_open_align")
        events_params.reset_range_scale = events_params.get("reset_range_scale", 1.0)
        events_params.reset_joint_std = events_params.get("reset_joint_std", 0.0)
        events_params.reset_randomize_mode = events_params.get("reset_randomize_mode", "task")
        events_params.reset_use_adr = events_params.get("reset_use_adr", False)
        events_params.reset_close_gripper = events_params.get("reset_close_gripper", None)

        curri_params = self.params.curriculum
        curri_params.use_obs_noise_curri = curri_params.get("use_obs_noise_curri", False)
        curri_params.use_contact_force_curri = curri_params.get("use_contact_force_curri", False)
        

        agent_params = self.params.agent
        agent_params.policy = agent_params.get("policy", OmegaConf.create())
        agent_params.policy.class_name = agent_params.get("class_name", "ActorCritic")

    def __post_init__(self):
        super().__post_init__()
        # robot
        self.scene.robot = KUKA_VICTOR_LEFT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot = self.scene.robot
        nut = self.scene.nut
        bolt = self.scene.bolt
        robot_params = self.params.scene.robot
        if robot_params.collision_approximation == "convexHull":
            robot.spawn.usd_path = (
                "assets/victor/victor_left_arm_with_gripper_v2/victor_left_arm_with_gripper_v2.usd"
            )
        elif robot_params.collision_approximation == "convexHull2":
            robot.spawn.usd_path = "assets/victor/victor_left_arm/victor_left_arm.usd"
            # robot.spawn.usd_path = "assets/victor/victor_left_arm_v2/victor_left_arm_v2.usd"
        robot.init_state.pos = [-0.15, -0.5, -0.8]

        # robot.spawn.collision_props.contact_offset = robot_params.contact_offset
        # robot.spawn.collision_props.rest_offset = robot_params.rest_offset
        robot.spawn.rigid_props.max_depenetration_velocity = robot_params.max_depenetration_velocity
        robot.spawn.rigid_props.sleep_threshold = robot_params.sleep_threshold
        robot.spawn.rigid_props.stabilization_threshold = robot_params.stabilization_threshold

        if self.params.use_factory_params:
            self.sim = SimulationCfg(
                dt=1 / 120,
                physx=PhysxCfg(
                    solver_type=1,
                    max_position_iteration_count=192,  # Important to avoid interpenetration.
                    max_velocity_iteration_count=1,
                    bounce_threshold_velocity=0.2,
                    friction_offset_threshold=0.01,
                    friction_correlation_distance=0.00625,
                    gpu_max_rigid_contact_count=2**23,
                    gpu_max_rigid_patch_count=2**23,
                    gpu_max_num_partitions=1,  # Important for stable simulation.
                ),
                physics_material=RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
            )
            robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            )
            robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            )

            nut.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            )
            nut.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.05)
            nut.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            )

            bolt.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            )
            bolt.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.05)
            bolt.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            )

        robot.actuators["victor_left_arm"].stiffness = robot_params.arm_stiffness
        robot.actuators["victor_left_arm"].damping = robot_params.arm_damping
        robot.actuators["victor_left_gripper"].velocity_limit = 1
        robot.actuators["victor_left_gripper"].effort_limit = robot_params.gripper_effort_limit
        robot.actuators["victor_left_gripper"].stiffness = robot_params.gripper_stiffness
        robot.actuators["victor_left_gripper"].damping = robot_params.gripper_damping

        # action
        action_params = self.params.actions
        arm_lows = [-0.003, -0.003, -0.003, -0.02, -0.02, -0.4]
        arm_highs = [0.003, 0.003, 0.003, 0.02, 0.02, 0.4]
        scale = [0.003, 0.003, 0.003, 0.02, 0.02, 0.4]

        # arm_lows = [-0.002, -0.002, -0.002, -0.01, -0.01, -0.4]
        # arm_highs = [0.002, 0.002, 0.002, 0.01, 0.01, 0.4]
        # scale = [0.002, 0.002, 0.002, 0.01, 0.01, 0.4]
        if (
            self.params.events.reset_target == "rigid_grasp_open_tilt"
            or self.params.events.reset_joint_std > 0
        ):
            arm_lows = [-0.002, -0.002, -0.002, -0.01, -0.01, -0.5]
            arm_highs = [0.002, 0.002, 0.002, 0.01, 0.01, 0.5]
            scale = [0.002, 0.002, 0.002, 0.01, 0.01, 0.5]

        if action_params.uni_rotate:
            arm_highs[5] = 0.0

        if action_params.controller_type == "ik":
            self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["victor_left_arm_joint.*"],
                body_name="victor_left_tool0",
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=True,
                    ik_method="dls",
                    ik_params={"lambda_val": action_params.ik_lambda},
                ),
                lows=arm_lows,
                highs=arm_highs,
                scale=scale,
            )
        elif action_params.controller_type == "osc":
            self.scene.robot.actuators["victor_left_arm"].stiffness = 0.0
            self.scene.robot.actuators["victor_left_arm"].damping = 0.0
            self.actions.arm_action = OperationalSpaceControllerActionCfg(
                asset_name="robot",
                joint_names=["victor_left_arm_joint.*"],
                body_name="victor_left_tool0",
                controller_cfg=OperationalSpaceControllerCfg(
                    target_types=["pose_rel"],
                    impedance_mode="fixed",
                    inertial_dynamics_decoupling=True,
                    partial_inertial_dynamics_decoupling=False,
                    gravity_compensation=False,
                    motion_stiffness_task=[100, 100, 100, 100, 100, 100],
                    motion_damping_ratio_task=1.0,
                    nullspace_control="position",
                ),
                nullspace_joint_pos_target="center",
                position_scale=0.002,
                orientation_scale=0.2,
                stiffness_scale=100.0,
            )
        # self.gripper_act_lows = [-0.005, -0.005]
        # self.gripper_act_highs = [0.005, 0.005]
        # self.actions.gripper_action = mdp.Robotiq3FingerActionCfg(
        #     asset_name="robot",
        #     side="left",
        #     lows=self.gripper_act_lows,
        #     highs=self.gripper_act_highs,
        #     use_relative_mode=True,
        #     is_accumulate_action=False,
        #     keep_grasp_state=action_params.keep_grasp_state
        # )

        nut.init_state.pos = (0, 0, 0.02)
        nut.init_state.rot = (0, 1, 0, 0)
        nut_params = self.params.scene.nut
        if nut_params.rigid_grasp:
            self.scene.nut.spawn.func = spawn_nut_with_rigid_grasp

        # observations
        obs_params = self.params.observations
        self.observations.policy.history_length = obs_params.history_length

        if obs_params.critic_privil_obs:
            self.observations.critic = self.observations.CriticCfg()
            self.observations.critic.history_length = obs_params.history_length     # Add the critic length

        # Add auxiliary task config
        agent_class = self.params.agent.policy.class_name
        aux_task_obs_dict = {
            "ActorCriticNutPose": self.observations.AuxNutPoseCfg,
            "ActorCriticNutDist": self.observations.AuxNutPoseCfg,
            "ActorCriticNutPos": self.observations.AuxNutPosCfg,
            # "ActorCriticRMA": self.observations.RMAPrivilCfg,
        }
        if agent_class in aux_task_obs_dict.keys():
            self.observations.aux_observations = aux_task_obs_dict[agent_class]()

        self.observations.policy.flatten_history_dim = obs_params.flatten_history_dim
        if obs_params.include_wrench:
            self.observations.policy.wrist_wrench = ObsTerm(
                func=mdp.body_incoming_wrench,
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=[obs_params.wrench_target_body])
                },
                scale=1,
            )
            if obs_params.critic_privil_obs:
                self.observations.critic.wrist_wrench = ObsTerm(
                    func=mdp.body_incoming_wrench,
                    params={"asset_cfg": SceneEntityCfg("robot", body_names=[obs_params.wrench_target_body])},
                    scale=1,
                )
            # if agent_class.startswith("ActorCriticRMA"):
            #     self.observations.aux_observations.wrist_wrench = ObsTerm(
            #         func=mdp.body_incoming_wrench,
            #         params={"asset_cfg": SceneEntityCfg("robot", body_names=[obs_params.wrench_target_body])},
            #         scale=1,
            #     )
        if obs_params.include_tool:
            self.observations.policy.tool_pose = ObsTerm(
                func=robot_tool_pose
            )
            if obs_params.critic_privil_obs:
                self.observations.critic.tool_pose = ObsTerm(
                    func=robot_tool_pose
                )
            # if agent_class.startswith("ActorCriticRMA"):
            #     self.observations.aux_observations.tool_pose = ObsTerm(
            #         func=robot_tool_pose
            #     )
        if obs_params.include_action:
            self.observations.policy.last_action = ObsTerm(
                func=mdp.last_action,
                params={"action_name": "arm_action"},
                scale=1,
            )
        # observation space for oracle - noise-free
        if obs_params.critic_privil_obs:
            self.observations.critic = copy.deepcopy(self.observations.policy)
        self.observations.policy.nut_pos.modifiers = [
            NoiseModifierCfg(
                noise_cfg=GaussianNoiseCfg(
                    mean=0.0, std=obs_params.nut_pos.noise_std, operation="add"
                ),
                bias_noise_cfg=GaussianNoiseCfg(
                    mean=0.0, std=obs_params.nut_pos.bias_std, operation="abs"
                ),
            )
        ]
        if obs_params.critic_privil_obs:
            self.observations.critic.last_action = ObsTerm(
                func=mdp.last_action,
                params={"action_name": "arm_action"},
                scale=1,
            )
        # if agent_class.startswith("ActorCriticRMA"):
        #     self.observations.aux_observations.last_action = ObsTerm(
        #         func=mdp.last_action,
        #         params={"action_name": "arm_action"},
        #         scale=1,
        #     )

        self.observations.policy.nut_pos.modifiers = [NoiseModifierCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=obs_params.nut_pos.noise_std, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=obs_params.nut_pos.bias_std, operation="abs"),
            # bias_noise_cfg=RadialNoiseCfg(mean=obs_params.nut_pos.bias_mean),
        )]

        # bolt : (0.63, 0.0, 0.0)
        # for debug only
        if obs_params.use_tiled_camera:
            self.scene.tiled_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Camera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(1.0, 0.1, 0.12),
                    rot=[0.4497752, 0.4401843, 0.5533875, 0.545621],
                    convention="opengl",
                ),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.0001, 20.0),
                ),
                width=720,
                height=720,
            )
        if obs_params.use_obs_camera:
            self.scene.obs_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/DepthCamera",
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(1.0, 0.1, 0.12),
                    rot=[0.4402, -0.4498, -0.5456, 0.5534],
                    convention="ros",
                ),
                data_types=obs_params.obs_camera_type,
                spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.0001, 0.5)),
                width=200,
                height=200,
            )
            # -90 78 179
            # -30 71 121

        # events
        event_params = self.params.events
        self.events = EventCfg()
        self.events.set_robot_collision = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (robot_params.static_friction, robot_params.static_friction),
                "dynamic_friction_range": (
                    robot_params.dynamic_friction,
                    robot_params.dynamic_friction,
                ),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
            },
        )
        if nut_params.rigid_grasp:
            self.events.set_robot_properties = EventTerm(
                func=create_fixed_joint,
                mode="startup",
            )
        # self.events.set_robot_properties = EventTerm(
        #     func=initialize_contact_properties,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
        #         "contact_offset": robot_params.contact_offset,
        #         "rest_offset": robot_params.rest_offset},
        #     mode="startup",
        # )
        # this is for randomize alongthe nut_pose frame
        nut_rel_pos = np.array(nut.init_state.pos) - np.array(
            self.scene.nut_frame.target_frames[0].offset.pos
        )
        nut_rel_pos = nut.init_state.pos
        self.events.reset_default = GraspResetEventTermCfg(
            func=reset_scene_to_grasp_state,
            mode="reset",
            nut_rel_pose=np.concatenate([nut_rel_pos, nut.init_state.rot]).tolist(),
            reset_target=event_params.reset_target,
            reset_range_scale=event_params.reset_range_scale,
            reset_randomize_mode=event_params.reset_randomize_mode,
            reset_joint_std=event_params.reset_joint_std,
            reset_use_adr=event_params.reset_use_adr,
            reset_close_gripper=event_params.reset_close_gripper,
        )

        # terminations
        termination_params = self.params.terminations
        if termination_params.nut_fallen:
            self.terminations.nut_fallen = DoneTerm(func=terminate_if_nut_fallen)
        if termination_params.far_from_bolt:
            self.terminations.far_from_bolt = DoneTerm(func=terminate_if_far_from_bolt)
        self.scene.nut.spawn.activate_contact_sensors = True

        # rewards
        rewards_params = self.params.rewards
        self.rewards.coarse_nut.weight = rewards_params.coarse_nut_w
        self.rewards.fine_nut.weight = rewards_params.fine_nut_w
        self.rewards.upright_reward.weight = rewards_params.upright_reward_w
        self.rewards.success.weight = rewards_params.success_w
        self.rewards.action_rate.weight = rewards_params.action_rate_w
        self.rewards.action_l2.weight = rewards_params.action_l2_w

        self.rewards.incoming_wrench_mag = RewTerm(
            func=mdp.incoming_wrench_mag,
            params={"asset_cfg": self.observations.policy.wrist_wrench.params["asset_cfg"]},
            weight=rewards_params.incoming_wrench_mag_w,
        )
        if rewards_params.dtw_ref_traj_w > 0:
            self.rewards.dtw_ref_traj = DTWReferenceTrajRewardCfg(
                his_traj_len=10,
                func=DTWReferenceTrajReward,
                weight=rewards_params.dtw_ref_traj_w,
            )
        self.rewards.contact_force_penalty = RewTerm(
            func=mdp.contact_forces,
            params={"threshold": 0, "sensor_cfg": SceneEntityCfg(name="contact_sensor")},
            weight=rewards_params.contact_force_penalty_w,
        )
        self.viewer.eye = (0.3, 0, 0.15)

        # curriculum
        curri_params = self.params.curriculum
        if curri_params.use_obs_noise_curri:
            self.curriculum.modify_nut_pos_noise = CurrTerm(
                func=modify_noise_scale,
                params={"begin_steps": 500 * 32, "end_steps": 2000 * 32},
            )
        if curri_params.use_contact_force_curri:
            self.curriculum.modify_contact_force_penalty = CurrTerm(
                func=mdp.modify_reward_weight,
                params={"term_name": "contact_force_penalty", "weight": -0.1, "num_steps": 800 * 32},
            )
