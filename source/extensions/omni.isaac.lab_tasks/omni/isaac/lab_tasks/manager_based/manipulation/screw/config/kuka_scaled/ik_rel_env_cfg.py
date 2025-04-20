# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
import torch
import numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.lab.utils.math as math_utils
import omni.physx.scripts.utils as physx_utils
from typing import Literal
from omni.isaac.lab.managers import EventTermCfg
from omni.isaac.lab.envs import ManagerBasedEnv
from pxr import Usd, UsdGeom
from curobo.types.math import Pose
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from curobo.types.base import TensorDeviceType

import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.manager_based.manipulation.screw.screw_env_cfg import (
    nut_upright_reward_forge,
    asset_factory
)
from omni.isaac.lab_tasks.manager_based.manipulation.screw.config.kuka.ik_rel_env_cfg import (
    IKRelKukaNutThreadEnvCfg,
    DTWReferenceTrajRewardCfg,
    DTWReferenceTrajReward,
    reset_scene_to_grasp_state,
)

def spawn_nut_with_rigid_grasp_scaled(
        prim_path: str,
        cfg: sim_utils.UsdFileCfg,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        index: int=0,
    ) -> Usd.Prim:
    stage = stage_utils.get_current_stage()
    tool_prim = stage.GetPrimAtPath(f"/World/envs/env_{index}/Robot/victor_left_tool0")
    xfCache = UsdGeom.XformCache()
    tool_pose = xfCache.GetLocalToWorldTransform(tool_prim)
    tool_pos = tuple(tool_pose.ExtractTranslation())
    tool_pos = torch.tensor(tool_pos)[None]
    tool_quat = tool_pose.ExtractRotationQuat()
    tool_quat = [tool_quat.real, tool_quat.imaginary[0], tool_quat.imaginary[1], tool_quat.imaginary[2]]
    tool_quat = torch.tensor(tool_quat)[None]

    origin_prim = stage.GetPrimAtPath(f"/World/envs/env_{index}/Origin")
    xfCache = UsdGeom.XformCache()
    origin_pose = xfCache.GetLocalToWorldTransform(origin_prim)
    origin_pos = torch.tensor(tuple(origin_pose.ExtractTranslation()))

    grasp_rel_pos = torch.tensor(translation)[None]
    grasp_rel_quat = torch.tensor(orientation)[None]

    # nut_init_pos = init_pos_repeated - self.nut_orig_offset
    # 0.02 - origin offset

    # Radomize in hand pose
    # rand_delta_pos, rand_delta_quat = some_sampling()
    # grasp_rel_pos, grasl_rel_quat = math_utils.combine_frame_transforms(
    #     gras_rel_pos, grasl_rel_quat, rand_delta_pos, rand_delta_quat
    # )
    # ...existing code...

    nut_pos, nut_quat = math_utils.combine_frame_transforms(tool_pos, tool_quat, grasp_rel_pos, grasp_rel_quat)

    nut_prim = sim_utils.spawn_from_usd(prim_path, cfg, nut_pos[0]-origin_pos, nut_quat[0])
    # nut_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation)
    return nut_prim

def get_prim_pos(prim):
    if isinstance(prim, str):
        stage = stage_utils.get_current_stage()
        prim = stage.GetPrimAtPath(prim)
    xfCache = UsdGeom.XformCache()
    pose = xfCache.GetLocalToWorldTransform(prim)
    pos = tuple(pose.ExtractTranslation())
    pos = torch.tensor(pos)[None]
    return pos

def create_fixed_joint_scaled(env: ManagerBasedEnv, env_ids: torch.Tensor):
    for i in range(env.num_envs):
        stage = stage_utils.get_current_stage()
        child_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/victor_left_tool0")
        parent_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Nut/factory_nut")
        # Create fixed joint
        physx_utils.createJoint(stage, "Fixed", child_prim, parent_prim)
        # physx_utils.createJoint(stage, "Fixed", parent_prim, child_prim)

def get_env_scales(env):
    return env.cfg.asset_scale_samples.reshape(-1,1)
        
# Do a scaled version of Grasp Reset
class GraspResetEventTermScaledCfg(EventTermCfg):
    def __init__(
        self,
        reset_target: Literal["pre_grasp", "grasp", "mate", "rigid_grasp", "rigid_grasp_open_align"] = "grasp",
        nut_rel_pose: torch.Tensor = None,
        reset_range_scale: float = 1.0,
        reset_joint_std: float = 0.0,
        reset_randomize_mode: Literal["task", "joint", None] = "task",
        reset_use_adr: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reset_target = reset_target
        self.nut_rel_pose = nut_rel_pose
        self.reset_range_scale = reset_range_scale
        self.reset_joint_std = reset_joint_std
        self.reset_randomize_mode = reset_randomize_mode
        self.reset_use_adr = reset_use_adr

class reset_scene_to_grasp_state_scaled(reset_scene_to_grasp_state):
    def __init__(self, cfg: GraspResetEventTermScaledCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.num_buckets = 10
        self.bucket_update_freq = 2
        self.max_ik_batch_size = 10240

    def update_random_initializations(self, env:ManagerBasedEnv):
        cached_state = self.cached_state[0:1].clone()
        B = self.num_buckets
        noise_scale = 1.
        if self.reset_use_adr:
            # step a: activate noise
            # step b: maximize noise
            raise NotImplementedError
        
        if self.reset_randomize_mode == "task":
            arm_state = cached_state["robot"]["joint_state"]["position"][:, :7]
            num_envs = env.num_envs
            scales = env.cfg.asset_scale_samples

            # Vectorize this!!!!!!
            # Compute scaled tool position
            default_tool_pose = self.curobo_arm.forward_kinematics(arm_state.clone()).ee_pose
            default_tool_pose = default_tool_pose.repeat(num_envs)

            # Compute nut relative position scaled
            # annoyingly difficult to initialize
            rel_pose_list = self.nut_rel_pose.tolist()
            nut_rel_pose = Pose.from_batch_list(rel_pose_list, self.tensor_args)
            # nut_rel_pose = Pose(torch.from_numpy(self.nut_rel_pose))
            default_nut_pose = default_tool_pose.multiply(nut_rel_pose)
            default_nut_pose = default_nut_pose.repeat(B)

            # Compute scaled obs bias
            scales = scales.repeat(B)
            low = self.reset_trans_low.clone().reshape(1,-1)
            low = low.repeat(B*num_envs, 1)
            # Add z to low
            delta_z = env.cfg.base_bolt_height * torch.clip(scales-1, 0.0, 10.0).reshape(-1,1)
            low[:,2] += delta_z[:,0]
            rand_range = (self.reset_trans_high-self.reset_trans_low).reshape(1,-1)

            # ONLY FOR TESTING
            # low[:,:2] = 0.0
            # rand_range = torch.zeros_like(rand_range)
            # self.reset_rot_std = 0.0
            delta_trans = torch.rand((B*num_envs, 3), device=env.device) * rand_range + low
            delta_trans *= noise_scale
            
            delta_rot = 2 * torch.rand((B*num_envs, 3), device=env.device) * self.reset_rot_std - self.reset_rot_std
            delta_quat = math_utils.quat_from_euler_xyz(delta_rot[:, 0], delta_rot[:, 1], delta_rot[:, 2])

            # Add together
            delta_pose = Pose(position=torch.zeros((B*num_envs, 3), device=env.device), quaternion=delta_quat)
            randomized_nut_pose = default_nut_pose.multiply(delta_pose)
            randomized_nut_pose.position += delta_trans
            nut_rel_pose = nut_rel_pose.repeat(B)

            randomized_tool_pose = randomized_nut_pose.multiply(nut_rel_pose.inverse())
            
            # Do IK
            ik_results = []
            # This is just to do batch IK in case of CUDA out of memory error for large bucket size
            for iter in range(0,B*num_envs, self.max_ik_batch_size):
                end_idx = min(iter+self.max_ik_batch_size, B*num_envs)
                ik_result = self.curobo_arm.compute_ik(
                    randomized_tool_pose[iter:iter+end_idx]
                )
                ik_results.append(ik_result.solution.squeeze(1))
            randomized_joint_state = torch.cat(ik_results, dim=0)
            randomized_joint_state = randomized_joint_state.reshape(B, num_envs, -1)
            # randomized_joint_state = arm_state.repeat(B*num_envs, 1)
        
        elif self.reset_randomize_mode == "joint":
            randomized_joint_state = torch.randn_like(arm_state) * self.reset_joint_std + arm_state
        else:
            arm_state = cached_state["robot"]["joint_state"]["position"][:, :7].repeat(B, 1)
            randomized_joint_state = arm_state
        self.rand_init_configurations = randomized_joint_state.detach().cpu().numpy()

    def sample_reset_poses(self, env_ids, device):
        B = self.num_buckets
        # Pick Indices
        select = np.random.choice(B)
        randomized_joint_state = self.rand_init_configurations[select]
        randomized_joint_state = torch.tensor(randomized_joint_state, device=device)
        return randomized_joint_state[env_ids]

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor):
        cached_state = self.cached_state[env_ids].clone()
        global_step = env._sim_step_counter // env.cfg.decimation
        if global_step % int(self.bucket_update_freq * env.unwrapped.max_episode_length) == 0:
            with torch.inference_mode(False):
                self.update_random_initializations(env)
        if self.reset_randomize_mode is not None:
            randomized_joint_state = self.sample_reset_poses(env_ids, env.device)
            cached_state["robot"]["joint_state"]["position"][:, :7] = randomized_joint_state
            cached_state["robot"]["joint_state"]["position_target"][:, :7] = randomized_joint_state
            # To prevent nut reset to within bolt mesh, we also overwrite default reset nut state
            safe_nut_z = env.cfg.base_bolt_height * torch.max(env.cfg.asset_scale_samples) * 1.2     # 20% more than max screw height scaled
            cached_state["nut"]["root_state"][:, 2] = safe_nut_z
        env.unwrapped.write_state(cached_state, env_ids)


# Do a scaled version of the DTW reward
class DTWReferenceTrajRewardScaled(DTWReferenceTrajReward):
    def __init__(self, cfg: DTWReferenceTrajRewardCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def reset(self, env_ids: torch.Tensor):
        scene = self._env.unwrapped.scene

        # Get this in a different way, and compare results
        OLD_nut_frame = scene["nut_frame"]
        OLD_nut_cur_pos = OLD_nut_frame.data.target_pos_w - scene.env_origins[:, None]
        
        nut = scene["nut"]
        nut_cur_pos = nut.data.root_state_w[...,:3] - scene.env_origins[:,None]
        # Asset
        offset_tensor = torch.tensor(scene["nut_frame"].offset.pos).reshape(1,3).to(self._env.device)
        assert ((nut_cur_pos-OLD_nut_cur_pos-offset_tensor) == 0).all(), "Nut cur pos is not equal to the old nut cur pos"

        self.nut_traj_his[env_ids] = nut_cur_pos[env_ids]

    def __call__(self, env: ManagerBasedEnv):
        # nut_frame = env.unwrapped.scene["nut_frame"]
        # cur_nut_pos = nut_frame.data.target_pos_w[:, 0] - env.unwrapped.scene.env_origins
        scene = self._env.unwrapped.scene
        nut = scene["nut"]
        cur_nut_pos = nut.data.root_state_w[...,:3] - scene.env_origins[:,None]

        imitation_rwd, new_nut_traj_his = mdp.get_imitation_reward_from_dtw(
            self.nut_ref_pos_traj, cur_nut_pos, self.nut_traj_his, self.soft_dtw_criterion, env.device
        )
        self.nut_traj_his = new_nut_traj_his
        return imitation_rwd


@configclass
class IKRelKukaNutThreadScaledEnvCfg(IKRelKukaNutThreadEnvCfg):
    """Configuration for the IK-based relative Kuka nut threading environment."""
    # DO NOT OVERWRITE REWARDS
    # rewards: NutThreadScaledRewardsCfg = NutThreadScaledRewardsCfg()
    
    def get_default_env_params(self):
        super().get_default_env_params()
        
        events_params = self.params.events
        events_params.reset_scale_method = events_params.get("reset_scale_method", "none")
        if events_params.reset_scale_method not in ["none", "uniform", "gaussian"]:
            raise ValueError(f"Invalid reset_scale_method: {events_params.reset_scale_method}. Must be 'none', 'uniform' or 'gaussian'.")
        events_params.reset_scale_range = events_params.get("reset_scale_range", (0.8, 1.2))
        events_params.reference_nut_part = events_params.get("reference_nut_part", "center")
        if events_params.reference_nut_part not in ["center", "bottom"]:
            raise ValueError(f"Invalid reference_nut_part: {events_params.reference_nut_part}. Must be 'center' or 'bottom'.")

        events_params.in_hand_rand_pos_range = events_params.get("in_hand_rand_pos_range", (0.01, 0.01, 0.0))
        events_params.in_hand_rand_rot_std = events_params.get("in_hand_rand_rot_std", (0.01, 0.01, 0.01))

        obs_params = self.params.observations
        obs_params.include_relative = obs_params.get("include_relative", True)

        # Add whether scale is observed
        obs_params = self.params.observations
        obs_params.include_scale = obs_params.get("include_scale", False)

    def randomize_scales(self):
        method = self.params.events.reset_scale_method
        rand_range = self.params.events.reset_scale_range
        assert len(rand_range) == 2

        # Sample and randomize
        if method.lower() == "uniform":
            upper, lower = rand_range
            scale = torch.rand((self.params.num_envs,))
            asset_scale_samples = scale * (upper - lower) + lower
            self.asset_scale_samples = asset_scale_samples.to(self.device)
        elif method.lower() == "gaussian":
            mean, std = rand_range
            self.asset_scale_samples = torch.normal(
                mean=mean, std=std, size=(self.params.num_envs,)
            ).to(self.device)
        # elif self.params.events.reset_scale_method == "choice":

        else:
            self.asset_scale_samples = None
        # Debug scale=1
        # self.asset_scale_samples = torch.ones_like(self.asset_scale_samples)


    def multiplicate_assets(self, articulation_cfg, new_spawn_func=None):
        """Make fixed and held assets multiplicative."""
        # Make a list for easy indexing
        asset_scale_samples = self.asset_scale_samples.cpu().tolist()
        
        # If no
        if asset_scale_samples is None:
            return articulation_cfg
        assert len(asset_scale_samples) == self.params.num_envs

        spawn_cfg = articulation_cfg.spawn
        asset_cfgs = []
        for scale in asset_scale_samples:
            scaled_spawn_cfg = deepcopy(spawn_cfg)
            scaled_spawn_cfg.scale = (scale, scale, scale)
            # Reset the spawn function
            if new_spawn_func is not None:
                scaled_spawn_cfg.func = new_spawn_func
            asset_cfgs.append(scaled_spawn_cfg)
        assert len(asset_cfgs) == self.params.num_envs
        if spawn_cfg.activate_contact_sensors:
            assert sum([1 if cfg.activate_contact_sensors else 0 for cfg in asset_cfgs]) == self.params.num_envs

        multi_asset_spawn_cfg = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=asset_cfgs,
            random_choice=False,
            activate_contact_sensors=True,
        )
        articulation_cfg.spawn = multi_asset_spawn_cfg
        return articulation_cfg

    def __post_init__(self):
        self.replicate_physics = False
        super().__post_init__()

        # Override with scaled multi-instance assets
        self.get_default_env_params()
        
        # Randomize scales
        self.randomize_scales()
        num_envs = self.params.num_envs
        nut_pos_repeated = torch.tensor(
            self.scene.nut.init_state.pos, device=self.device, dtype=torch.float32
        ).expand(num_envs, -1)   # or .repeat(self.num_envs, 1)
        nut_rot_repeated = torch.tensor(
            self.scene.nut.init_state.rot, device=self.device, dtype=torch.float32
        ).expand(num_envs, -1)
        self.scene.nut.init_state.pos = nut_pos_repeated
        self.scene.nut.init_state.rot = nut_rot_repeated

        self.scene.nut = self.multiplicate_assets(self.scene.nut, spawn_nut_with_rigid_grasp_scaled)
        self.scene.bolt = self.multiplicate_assets(self.scene.bolt)

        # Cache the size of the bolt
        screw_dict = asset_factory[self.params.scene.screw_type]
        # 1.15 seems to work well
        self.base_bolt_height = screw_dict["bolt_tip_offset"].pos[2]

        # Override the create_fixed_joint function for scaled environment
        nut_params = self.params.scene.nut
        if nut_params.rigid_grasp:
            self.events.set_robot_properties = EventTermCfg(
                func=create_fixed_joint_scaled,
                mode="startup",
            )

        # For randomizing scales, we need to set the scaled offsets differently for each env
        # Thus, computing rewards cannot depend on nut_frame or bolt_frame
        # We keep a separate copy of the offset specific for each env

        # Overwrite the 
        frame_offset = torch.tensor(
            self.scene.screw_dict["nut_frame_offset"].pos,
            device=self.device, dtype=torch.float32
        ).reshape(1,3)
        # Since origin of nut is below the body of the nut, we need to compensate
        # So that the bottom surface of nut is aligned relative to gripper for various sizes of nut. 
        origin_offset = torch.tensor(
            self.scene.screw_dict["nut_origin_bottom_offset"].pos,
            device=self.device, dtype=torch.float32
        ).reshape(1,3)

        scales = self.asset_scale_samples.reshape(-1,1)
        self.nut_orig_offset = origin_offset * (scales-1)

        # Override reset term
        nut = self.scene.nut
        init_pos_repeated = nut.init_state.pos
        init_rot_repeated = nut.init_state.rot

        # Compute the compensated init pos
        nut_init_pos = init_pos_repeated + self.nut_orig_offset
        nut.init_state.pos = nut_init_pos

        # Re-compute the relative position for reset frame
        nut_rel_pos = init_pos_repeated - frame_offset - self.nut_orig_offset
        nut_rel_pose = torch.cat([nut_rel_pos, init_rot_repeated], dim=-1).cpu().numpy()

        # Compute scaled bolt tip offset
        bolt_tip_offset_pos = self.scene.bolt_frame.target_frames[0].offset.pos
        self.scaled_bolt_tip_offset = torch.tensor(
            bolt_tip_offset_pos, device=self.device
        ).reshape(1,3) * self.asset_scale_samples.reshape(-1,1)

        # Compute scaled nut origin to center offset
        nut_center_offset_pos = self.scene.nut_frame.target_frames[0].offset.pos
        self.scaled_nut_center_offset = torch.tensor(
            nut_center_offset_pos, device=self.device
        ).reshape(1,3) * self.asset_scale_samples.reshape(-1,1)

        # Optimize for frame and bolt markers
        # Make sure they align with first env
        for marker_cfg in [
            self.scene.nut_frame,
            self.scene.nut_frame_plate, 
            self.scene.bolt_frame
        ]:
            offset_pos = marker_cfg.target_frames[0].offset.pos
            scaled_offset_pos = np.array(offset_pos) * self.asset_scale_samples[0].cpu().numpy()
            scaled_offset_pos = tuple(scaled_offset_pos.tolist())
            marker_cfg.target_frames[0].offset = OffsetCfg(pos=scaled_offset_pos)

        # Also compute scaled offset for distance computation
        nut_bottom_offset_pos = screw_dict["nut_origin_bottom_offset"].pos
        self.scaled_nut_bottom_offset = torch.tensor(
            nut_bottom_offset_pos, device=self.device
        ).reshape(1,3) * self.asset_scale_samples.reshape(-1,1)
        
        # Configure in hand randomization of things
        events_params = self.params.events
        # Configure if we want to use the bottom of the nut as reference
        if events_params.reference_nut_part == "bottom":
            for marker_cfg in [
                self.scene.nut_frame,
                self.scene.nut_frame_plate,
            ]:
                offset_pos = screw_dict["nut_origin_bottom_offset"].pos
                scaled_offset_pos = np.array(offset_pos) * self.asset_scale_samples[0].cpu().numpy()
                # scaled_offset_pos = np.zeros_like(scaled_offset_pos)
                scaled_offset_pos = tuple(scaled_offset_pos.tolist())
                marker_cfg.target_frames[0].offset = OffsetCfg(pos=scaled_offset_pos)


        # ================== TESTING =========================
        # Randomize in-hand grasp pose
        # rand_delta_pos, rand_delta_quat = some_sampling()
        # grasp_rel_pos, grasl_rel_quat = math_utils.combine_frame_transforms(
        #     gras_rel_pos, grasl_rel_quat, rand_delta_pos, rand_delta_quat
        # )
        # ...existing code...
        # Compute scaled obs bias
        # First compute some transformations
        tensor_args = TensorDeviceType(device=self.device)

        # A: relative pose of the nut origin to the hand
        A = Pose.from_batch_list(
            torch.cat([
                nut.init_state.pos,
                nut.init_state.rot
            ], dim=-1).cpu().numpy().tolist(),
            tensor_args
        )

        # B: relative pose of the center of nut to nut origin
        # Compute relative transformations
        self.upright_relative_rot = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]]
        ).repeat(self.params.num_envs, 1).to(self.device)
        B = Pose.from_batch_list(
            torch.cat([
                self.scaled_nut_center_offset,
                self.upright_relative_rot
            ], dim=-1).cpu().numpy().tolist(),
            tensor_args
        )

        # PNC: pose of center of nut relative to hand
        PNC = A.multiply(B)

        # --------------- Sample
        low_pos = -torch.tensor(events_params.in_hand_rand_pos_range, device=self.device)/2.0
        # in x,y axis sample +- from current, while in z axis only sample up to prevent init collision
        low_pos[2] += events_params.in_hand_rand_pos_range[2]/2.0
        range_pos = torch.tensor(events_params.in_hand_rand_pos_range, device=self.device)
        euler_rot_std = torch.tensor(events_params.in_hand_rand_rot_std, device=self.device)
        noise_scale = 1.0
        # Randomize position
        in_hand_delta_trans = torch.rand((num_envs, 3), device=self.device) * range_pos + low_pos
        in_hand_delta_trans *= noise_scale
        # Randomize rotation
        in_hand_delta_rot = 2*torch.rand((num_envs, 3), device=self.device) * euler_rot_std - euler_rot_std
        in_hand_delta_quat = math_utils.quat_from_euler_xyz(
            in_hand_delta_rot[:, 0], in_hand_delta_rot[:, 1], in_hand_delta_rot[:, 2]
        )
        # in_hand_delta_trans = torch.zeros_like(in_hand_delta_trans)
        # in_hand_delta_quat = torch.tensor(
        #     [[1.0, 0.0, 0.0, 0.0]]
        # ).repeat(self.params.num_envs, 1).to(self.device)
        # ---------- End of Sample

        # R: random transforms of the center of nut
        R= Pose.from_batch_list(
            torch.cat([
                in_hand_delta_trans,
                in_hand_delta_quat
            ], dim=-1).cpu().numpy().tolist(),
            tensor_args
        )
        # PNC_R: randomized pose of center of nut relative to hand
        PNC_R = PNC.multiply(R)
        # PNC_R = PNC

        # A_R: randomized relative pose of the nut origin to the hand
        A_R = PNC_R.multiply(B.inverse())

        # Finally, set the initial state of the nut with A_R
        nut.init_state.pos = A_R.position
        nut.init_state.rot = A_R.quaternion

        # Combine delta transforms with original pose
        # nut_init_pos, nut_init_quat = math_utils.combine_frame_transforms(
        #     nut_init_pos, init_rot_repeated, in_hand_delta_trans, in_hand_delta_quat
        # )

        # nut.init_state.pos = nut_init_pos
        # nut.init_state.rot = nut_init_quat

        # ================== TESTING =========================

        # Pass scale as observation
        obs_params = self.params.observations
        if obs_params.include_scale:
            self.observations.policy.asset_scale = ObsTerm(
                func=get_env_scales,
                scale=1,
            )

        self.events.reset_default = GraspResetEventTermScaledCfg(
            func=reset_scene_to_grasp_state_scaled,
            mode="reset",
            nut_rel_pose=nut_rel_pose,
            reset_target=events_params.reset_target,
            reset_range_scale=events_params.reset_range_scale,
            reset_randomize_mode=events_params.reset_randomize_mode,
            reset_joint_std=events_params.reset_joint_std,
            reset_use_adr=events_params.reset_use_adr,
        )

        # Nut Frame is used in nut_upright_reward_forge(), but only quat data
        # So no update is needed there

        # Update the DTWReferenceTrajReward
        rewards_params = self.params.rewards
        if rewards_params.dtw_ref_traj_w > 0:
            self.rewards.dtw_ref_traj = DTWReferenceTrajRewardCfg(
                his_traj_len=10,
                func=DTWReferenceTrajRewardScaled,
                weight=rewards_params.dtw_ref_traj_w,
            )
    
        if obs_params.include_relative:
            self.observations.policy.nut_bolt_relative = ObsTerm(
                func=mdp.rel_nut_bolt_distance,
                params={"bolt_part_name": "bolt_tip"},
                scale=1,
            )