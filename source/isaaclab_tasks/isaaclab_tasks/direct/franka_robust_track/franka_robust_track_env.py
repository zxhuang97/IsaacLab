# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import FRAME_MARKER_CFG, SPHERE_MARKER_CFG, VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils

from isaaclab_tasks.direct.factory import factory_control, factory_utils

from .franka_robust_track_env_cfg import FrankaRobustTrackEnvCfg


class FrankaRobustTrackEnv(DirectRLEnv):
    """Robot-only Franka direct env for robust Cartesian command tracking."""

    cfg: FrankaRobustTrackEnvCfg

    def __init__(self, cfg: FrankaRobustTrackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Observation history: only the proprioceptive part is stacked over the last
        # `obs_history_length` frames (concatenated oldest->newest) so the network
        # can infer velocities/dynamics from the past. The lookahead future errors
        # and the critic's privileged dims are forward-looking / episode-constant, so
        # they are appended once from the current frame rather than duplicated across
        # the history. H = 1 keeps the single-step observation.
        self.obs_history_length = max(1, int(self.cfg.obs_history_length))
        self.proprio_dim = 20
        self.future_dim = 6 * self.cfg.tracking.num_future_steps
        self.privileged_dim = 23
        self.proprio_history = torch.zeros(
            (self.num_envs, self.obs_history_length, self.proprio_dim), device=self.device
        )

        self.command_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Nominal reset EE pose (from `ctrl.reset_joints`), captured at reset and
        # used as the center of the randomized start-pose sampling box.
        self.nominal_start_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.nominal_start_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # cuRobo IK runs in a persistent subprocess (avoids a warp/PhysX GPU clash);
        # the worker + cached kinematic frames are set up lazily on first reset.
        self._ik_proc = None
        self._ik_worker_mod = None
        self._kin_frames_cached = False
        # Constant transform of `panda_hand` expressed in the fingertip frame.
        self.hand_in_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_in_fingertip_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # World pose of the `panda_link0` base (fixed per env).
        self.base_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Reference-trajectory parameters, sampled per env at reset.
        self.traj_start_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_start_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.traj_dir = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_speed = torch.zeros((self.num_envs, 1), device=self.device)
        self.traj_length = torch.zeros((self.num_envs, 1), device=self.device)
        # Circle-mode params: orthonormal plane basis (u, v), radius, signed omega.
        self.traj_u = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_u[:, 0] = 1.0
        self.traj_v = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_v[:, 1] = 1.0
        self.traj_radius = torch.zeros((self.num_envs, 1), device=self.device)
        self.traj_omega = torch.zeros((self.num_envs, 1), device=self.device)
        self.traj_rot_axis = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_rot_axis[:, 2] = 1.0
        self.traj_rot_speed = torch.zeros((self.num_envs, 1), device=self.device)
        self.traj_rot_angle = torch.zeros((self.num_envs, 1), device=self.device)

        # Discretized reference trajectory: at reset the analytic curve is sampled
        # onto a fixed grid of control steps (one waypoint per env step) and cached
        # here. Runtime command/lookahead just index into these buffers by step,
        # so the policy tracks a discrete pose sequence. The grid is extended past
        # the episode so the furthest lookahead sample is always in-range.
        self._traj_len = self.max_episode_length + (self.cfg.tracking.num_future_steps - 1)
        self.traj_pos_buf = torch.zeros((self.num_envs, self._traj_len, 3), device=self.device)
        self.traj_quat_buf = torch.zeros((self.num_envs, self._traj_len, 4), device=self.device)
        self.traj_quat_buf[..., 0] = 1.0
        self._env_arange = torch.arange(self.num_envs, device=self.device)

        self.fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.fingertip_midpoint_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_jacobian = torch.zeros((self.num_envs, 6, 7), device=self.device)
        self.arm_mass_matrix = torch.eye(7, device=self.device).repeat(self.num_envs, 1, 1)
        self.joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.joint_torque = torch.zeros_like(self.joint_pos)
        self.applied_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(self.num_envs, 1)
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(self.num_envs, 1)
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            self.num_envs, 1
        )
        self.task_prop_gains = self.default_gains.clone()
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.task_prop_gains)

        self.payload_mass = torch.zeros((self.num_envs, 1), device=self.device)
        self.payload_com = torch.zeros((self.num_envs, 3), device=self.device)
        self.gravity_vec = torch.tensor(self.cfg.sim.gravity, device=self.device).repeat(self.num_envs, 1)
        self.link_mass_scales = torch.ones((self.num_envs, self._robot.num_bodies), device=self.device)
        self.joint_friction = torch.zeros((self.num_envs, 7), device=self.device)

        # Validate the mass-matrix reconstruction against PhysX once at startup (nominal parameters)
        # and once more after the first dynamics randomization (payload merged, link masses scaled).
        # PhysX refreshes its generalized mass matrix only when the simulation steps, so checks run
        # at least two sim steps after the last mass-property write.
        self._mass_matrix_checks_pending = 1
        self._dynamics_write_sim_step = 0
        self._randomized_mass_matrix_validated = False

        self._resolve_robot_indices()
        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._cache_default_dynamics()
        self._compute_intermediate_values()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["pos_track", "rot_track", "ee_vel", "action_rate", "joint_vel", "joint_limit"]
        }

        # Per-episode-step tracking-error profile: bin the pos/rot tracking error
        # by the step index within the episode, then periodically emit a mean±std
        # line plot to wandb. All envs step in lockstep (fixed-length episodes, no
        # early termination), so each bin collects one num_envs-sized sample per
        # episode; accumulating over several episodes yields the spread. This tests
        # whether the error drops in later stages as the policy gains more context.
        self._perstep_num_bins = int(self.max_episode_length)
        self._perstep_pos_sum = torch.zeros(self._perstep_num_bins, device=self.device)
        self._perstep_pos_sqsum = torch.zeros(self._perstep_num_bins, device=self.device)
        self._perstep_rot_sum = torch.zeros(self._perstep_num_bins, device=self.device)
        self._perstep_rot_sqsum = torch.zeros(self._perstep_num_bins, device=self.device)
        self._perstep_count = torch.zeros(self._perstep_num_bins, device=self.device)
        self._perstep_log_every = int(self.cfg.log_perstep_error_episodes) * self._perstep_num_bins
        self._perstep_step_counter = 0

        # Latest per-step tracking error of env 0, cached for the video overlay.
        self._vis_pos_error_norm = None
        self._vis_rot_error_norm = None

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _resolve_robot_indices(self):
        body_names = self._robot.body_names
        fingertip_candidates = ["panda_fingertip_centered", "force_sensor", "panda_hand"]
        for body_name in fingertip_candidates:
            if body_name in body_names:
                self.fingertip_body_idx = body_names.index(body_name)
                break
        else:
            raise RuntimeError(f"Could not find a Franka EE body in body names: {body_names}")

        payload_body_name = self.cfg.randomization.payload_body_name
        if payload_body_name in body_names:
            self.payload_body_idx = body_names.index(payload_body_name)
        else:
            self.payload_body_idx = self.fingertip_body_idx

        # cuRobo IK solves for `panda_hand` in the `panda_link0` base frame; cache both
        # body indices so reset targets (defined at the fingertip) can be converted.
        self.hand_body_idx = body_names.index("panda_hand")
        self.base_body_idx = body_names.index("panda_link0")

        self.arm_joint_ids = list(range(7))
        joint_names = list(self._robot.joint_names)
        self.arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        self.isaac_arm_joint_idx = [joint_names.index(name) for name in self.arm_joint_names]

    def _cache_default_dynamics(self):
        self.default_masses = self._robot.root_physx_view.get_masses().clone()
        self.default_inertias = self._robot.root_physx_view.get_inertias().clone()
        self.default_coms = self._robot.root_physx_view.get_coms().clone()

        # Per-link nominal inertial parameters for the controller-side mass-matrix reconstruction.
        # Body 0 (fixed base) is dropped to match the PhysX jacobian rows.
        num_links = self._robot.num_bodies - 1
        self.nominal_link_masses = self.default_masses[:, 1:].to(self.device)
        self.nominal_link_inertias = self.default_inertias[:, 1:].to(self.device).view(self.num_envs, num_links, 3, 3)
        # Link-frame offset from the current link CoM (the PhysX jacobian reference point) to the
        # nominal link CoM. Nonzero only for the payload body after a payload merge.
        self.nominal_com_offsets = torch.zeros((self.num_envs, num_links, 3), device=self.device)
        self.arm_armature = self._robot.root_physx_view.get_dof_armatures().to(self.device)[:, 0:7]

    def _compute_intermediate_values(self):
        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()
        self.fingertip_midpoint_jacobian = jacobians[:, self.fingertip_body_idx - 1, 0:6, 0:7]
        # The controller runs on the nominal model: rebuild the mass matrix from the default inertial
        # parameters instead of querying PhysX, which would expose the payload and link-mass scales.
        self.arm_mass_matrix = self._compute_arm_mass_matrix(
            jacobians, self.nominal_link_masses, self.nominal_link_inertias, self.nominal_com_offsets
        )
        if self._mass_matrix_checks_pending > 0 and self._sim_step_counter >= self._dynamics_write_sim_step + 2:
            self._validate_arm_mass_matrix(jacobians)
            self._mass_matrix_checks_pending -= 1
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

    def _compute_arm_mass_matrix(
        self,
        jacobians: torch.Tensor,
        link_masses: torch.Tensor,
        link_inertias: torch.Tensor,
        com_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Rebuild the 7x7 arm mass matrix from per-link jacobians and inertial parameters.

        M(q) = sum_l m_l J_vc^T J_vc + J_w^T (R I R^T) J_w + diag(armature). PhysX link jacobians are
        referenced at each link's *current* CoM (world axes) and inertias are link-frame tensors about
        the CoM (verified against `get_generalized_mass_matrices`). `com_offsets` is the link-frame
        offset from the current CoM to the CoM that `link_masses`/`link_inertias` refer to — nonzero
        only for the payload body when reconstructing the nominal model after a payload merge.

        Evaluated with the cached nominal parameters this gives the mass matrix of the unrandomized
        robot, independent of the payload and link-mass randomization in PhysX.
        """
        num_links = self._robot.num_bodies - 1
        J_v = jacobians[:, :, 0:3, 0:7]
        J_w = jacobians[:, :, 3:6, 0:7]

        link_quat_w = self._robot.data.body_quat_w[:, 1:]
        R = math_utils.matrix_from_quat(link_quat_w.reshape(-1, 4)).view(self.num_envs, num_links, 3, 3)
        offset_w = (R @ com_offsets.unsqueeze(-1)).squeeze(-1)
        offset_skew = math_utils.skew_symmetric_matrix(offset_w.reshape(-1, 3)).view(
            self.num_envs, num_links, 3, 3
        )
        J_vc = J_v - offset_skew @ J_w

        inertia_w = R @ link_inertias @ R.transpose(-1, -2)
        mass_term = torch.einsum("nl,nlai,nlaj->nij", link_masses, J_vc, J_vc)
        rot_term = torch.einsum("nlai,nlab,nlbj->nij", J_w, inertia_w, J_w)
        return mass_term + rot_term + torch.diag_embed(self.arm_armature)

    def _validate_arm_mass_matrix(self, jacobians: torch.Tensor):
        """Check the jacobian-based reconstruction against PhysX using the current inertial parameters.

        Rebuilding with the *current* (randomized, payload-merged) parameters must reproduce PhysX's own
        generalized mass matrix; this pins down the jacobian reference conventions and the payload merge.
        Only valid once the simulation has stepped after the last dynamics write, since
        `get_generalized_mass_matrices` does not refresh until then.
        """
        num_links = self._robot.num_bodies - 1
        masses = self._robot.root_physx_view.get_masses().to(self.device)[:, 1:]
        inertias = (
            self._robot.root_physx_view.get_inertias().to(self.device)[:, 1:].view(self.num_envs, num_links, 3, 3)
        )
        zero_offsets = torch.zeros((self.num_envs, num_links, 3), device=self.device)
        reconstructed = self._compute_arm_mass_matrix(jacobians, masses, inertias, zero_offsets)
        physx_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        error = (reconstructed - physx_matrix).abs().max().item()
        assert error < 0.01, (
            f"Arm mass-matrix reconstruction deviates from PhysX (max abs error {error:.4f}). "
            "Likely a convention mismatch: expected jacobians referenced at the link CoM and inertia "
            "tensors expressed in the link frame about the CoM."
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = self.cfg.ctrl.ema_factor * actions.clone().to(self.device).clamp(-1.0, 1.0) + (
            1.0 - self.cfg.ctrl.ema_factor
        ) * self.actions

    def _apply_action(self):
        self._compute_intermediate_values()
        self._apply_payload_gravity()
        target_pos, target_quat = self._get_action_target_pose()

        if self.cfg.ctrl.backend == "factory_osc":
            self._apply_factory_osc(target_pos, target_quat)
        elif self.cfg.ctrl.backend == "dls_ik":
            self._apply_dls_ik(target_pos, target_quat)
        else:
            raise ValueError(f"Unsupported Franka robust track controller backend: {self.cfg.ctrl.backend}")

    def _apply_payload_gravity(self):
        """Apply the payload weight as an external wrench on the payload body.

        Robot gravity is disabled (nominal gravity compensation is assumed perfect), so the payload
        shows up exactly as its uncompensated weight `m * g`, acting at the payload CoM. The force is
        rotated into the link frame each step since the wrench buffers are consumed in that frame.
        """
        payload_quat_w = self._robot.data.body_quat_w[:, self.payload_body_idx]
        force_w = self.payload_mass * self.gravity_vec
        force_b = math_utils.quat_apply_inverse(payload_quat_w, force_w)
        self._robot.set_external_force_and_torque(
            forces=force_b.unsqueeze(1),
            torques=torch.zeros((self.num_envs, 1, 3), device=self.device),
            positions=self.payload_com.unsqueeze(1),
            body_ids=[self.payload_body_idx],
        )

    def _get_action_target_pose(self):
        pos_actions = self.actions[:, 0:3] * self.pos_threshold
        rot_actions = self.actions[:, 3:6] * self.rot_threshold

        target_pos = self.fingertip_midpoint_pos + pos_actions

        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / torch.clamp(angle.unsqueeze(-1), min=1.0e-6)
        delta_quat = torch_utils.quat_from_angle_axis(angle, axis)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        delta_quat = torch.where(angle.unsqueeze(-1) > 1.0e-6, delta_quat, identity_quat)
        target_quat = torch_utils.quat_mul(delta_quat, self.fingertip_midpoint_quat)
        return target_pos, target_quat

    def _apply_factory_osc(self, target_pos: torch.Tensor, target_quat: torch.Tensor):
        self.joint_torque, self.applied_wrench = factory_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )
        self.ctrl_target_joint_pos[:, 7:] = 0.04
        self.joint_torque[:, 7:] = 0.0
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _apply_dls_ik(self, target_pos: torch.Tensor, target_quat: torch.Tensor):
        pos_error, axis_angle_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        delta_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
        delta_joint_pos = factory_control.get_delta_dof_pos(
            delta_pose=delta_pose,
            ik_method="dls",
            jacobian=self.fingertip_midpoint_jacobian,
            device=self.device,
        )
        self.ctrl_target_joint_pos[:] = self.joint_pos
        self.ctrl_target_joint_pos[:, 0:7] = self.joint_pos[:, 0:7] + delta_joint_pos[:, 0:7]
        self.ctrl_target_joint_pos[:, 7:] = 0.04

        self.joint_torque.zero_()
        joint_error = self.ctrl_target_joint_pos[:, 0:7] - self.joint_pos[:, 0:7]
        joint_torque = self.cfg.ctrl.joint_pos_kp * joint_error - self.cfg.ctrl.joint_pos_kd * self.joint_vel[:, 0:7]
        self.joint_torque[:, 0:7] = torch.clamp(joint_torque, min=-100.0, max=100.0)
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _compute_obs_parts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the current-frame observation parts.

        Returns (proprio, future_errors, privileged): `proprio` is the part stacked
        over history, `future_errors` is the lookahead reference (policy + critic),
        and `privileged` is the critic-only privileged information.
        """
        self._compute_intermediate_values()
        self._update_command()
        future_errors = self._future_command_errors()
        proprio = torch.cat(
            [
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
                self.joint_pos[:, 0:7],
                self.actions,
            ],
            dim=-1,
        )
        privileged = torch.cat(
            [
                self.payload_mass,
                self.payload_com,
                self.joint_friction,
                self.task_prop_gains,
                self.pos_threshold,
                self.rot_threshold,
            ],
            dim=-1,
        )
        return proprio, future_errors, privileged

    def _get_observations(self) -> dict:
        proprio, future_errors, privileged = self._compute_obs_parts()
        if self.obs_history_length == 1:
            proprio_stacked = proprio
        else:
            # Shift the proprio history left by one frame and append the newest
            # frame, then flatten oldest->newest into a single vector per env.
            self.proprio_history = torch.roll(self.proprio_history, shifts=-1, dims=1)
            self.proprio_history[:, -1] = proprio
            proprio_stacked = self.proprio_history.reshape(self.num_envs, -1)

        policy_obs = torch.cat([proprio_stacked, future_errors], dim=-1)
        critic_obs = torch.cat([proprio_stacked, future_errors, privileged], dim=-1)
        return {"policy": policy_obs, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        self._update_command()
        pos_error, rot_error = self._command_errors()
        pos_error_norm = torch.linalg.norm(pos_error, dim=-1)
        rot_error_norm = torch.linalg.norm(rot_error, dim=-1)
        ee_vel_norm = torch.linalg.norm(self.fingertip_midpoint_linvel, dim=-1) + 0.1 * torch.linalg.norm(
            self.fingertip_midpoint_angvel, dim=-1
        )
        action_rate = torch.linalg.norm(self.actions - self.prev_actions, dim=-1)
        joint_vel_norm = torch.linalg.norm(self.joint_vel[:, 0:7], dim=-1)
        joint_limit_penalty = self._joint_limit_penalty()

        rewards = {
            "pos_track": torch.exp(-pos_error_norm / self.cfg.reward.pos_error_temp) * self.cfg.reward.pos_error_scale,
            "rot_track": torch.exp(-rot_error_norm / self.cfg.reward.rot_error_temp) * self.cfg.reward.rot_error_scale,
            "ee_vel": ee_vel_norm * self.cfg.reward.ee_vel_scale,
            "action_rate": action_rate * self.cfg.reward.action_rate_scale,
            "joint_vel": joint_vel_norm * self.cfg.reward.joint_vel_scale,
            "joint_limit": joint_limit_penalty * self.cfg.reward.joint_limit_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0) * self.step_dt
        for key, value in rewards.items():
            self._episode_sums[key] += value * self.step_dt

        successes = torch.logical_and(
            pos_error_norm < self.cfg.reward.success_pos_threshold,
            rot_error_norm < self.cfg.reward.success_rot_threshold,
        )
        self.extras["curr_successes"] = successes.float().mean()
        self.extras["tracking_pos_error"] = pos_error_norm.mean()
        self.extras["tracking_rot_error"] = rot_error_norm.mean()
        self._accumulate_perstep_error(pos_error_norm.detach(), rot_error_norm.detach())
        self._vis_pos_error_norm = pos_error_norm.detach()
        self._vis_rot_error_norm = rot_error_norm.detach()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._log_reset_metrics(env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Move to the nominal reset joints and record the resulting EE pose as the
        # center of the start-pose sampling box.
        self._set_franka_to_reset_pose(env_ids)
        self.step_sim_no_action()
        self.nominal_start_pos[env_ids] = self.fingertip_midpoint_pos[env_ids].clone()
        self.nominal_start_quat[env_ids] = self.fingertip_midpoint_quat[env_ids].clone()
        self._cache_kinematic_frames()

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._randomize_dynamics(env_ids)
        self._randomize_controller(env_ids)
        self._sample_reachable_start_and_trajectory(env_ids)
        self._update_command()

        # Seed the proprio history for the reset envs with their current frame so
        # the stacked observation does not mix in stale frames from the prior
        # episode. The subsequent `_get_observations` rolls in the same frame again.
        if self.obs_history_length > 1:
            proprio, _, _ = self._compute_obs_parts()
            self.proprio_history[env_ids] = proprio[env_ids].unsqueeze(1)

    def _set_franka_to_reset_pose(self, env_ids: torch.Tensor):
        """Teleport the arm to the nominal `ctrl.reset_joints` configuration."""
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, 0:7] = torch.tensor(self.cfg.ctrl.reset_joints, device=self.device).unsqueeze(0)
        if joint_pos.shape[1] > 7:
            joint_pos[:, 7:] = 0.04
        joint_vel = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids] = joint_pos
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.set_joint_effort_target(torch.zeros_like(joint_pos), env_ids=env_ids)

    def step_sim_no_action(self):
        """Advance the sim one physics step without applying a policy action.

        Used only during resets, where all environments are stepped together while
        joint states are written directly (teleported) for the IK solve.
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values()

    def _cache_kinematic_frames(self):
        """Cache the constant fingertip->hand and world->base transforms once.

        cuRobo IK targets `panda_hand` in the `panda_link0` base frame, while the
        env's reference poses are defined at the fingertip. Both the fingertip->hand
        offset and the base pose are fixed, so they are captured a single time after
        the first reset-pose step.
        """
        if self._kin_frames_cached:
            return
        fp_pos_w = self._robot.data.body_pos_w[:, self.fingertip_body_idx]
        fp_quat_w = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        hand_pos_w = self._robot.data.body_pos_w[:, self.hand_body_idx]
        hand_quat_w = self._robot.data.body_quat_w[:, self.hand_body_idx]
        self.hand_in_fingertip_pos[:], self.hand_in_fingertip_quat[:] = math_utils.subtract_frame_transforms(
            fp_pos_w, fp_quat_w, hand_pos_w, hand_quat_w
        )
        self.base_pos_w[:] = self._robot.data.body_pos_w[:, self.base_body_idx]
        self.base_quat_w[:] = self._robot.data.body_quat_w[:, self.base_body_idx]
        self._kin_frames_cached = True

    def _ensure_ik_worker(self):
        """Launch (once) the persistent cuRobo IK subprocess.

        cuRobo IK is run out-of-process because its warp kernels clash with Isaac
        Sim's GPU pipeline in-process (illegal memory access). The worker is a
        clean interpreter with standalone warp; it builds the solver lazily, so the
        first solve request blocks until the solver is ready.
        """
        if self._ik_proc is not None:
            return

        import json
        import os
        import subprocess
        import sys

        from force_tool.utils import curobo_ik_worker

        cfg = self.cfg.init
        # cuRobo/warp default to cuda:0 for the solver and kernel launches, so pin
        # the worker to our physics GPU via CUDA_VISIBLE_DEVICES and address it as
        # cuda:0 inside the worker. This avoids the device mismatch (e.g. goal
        # tensors on cuda:1 while kernels launch on cuda:0) when self.device != cuda:0.
        device = torch.device(self.device)
        gpu_index = device.index if device.index is not None else 0
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        worker_cfg = {
            "robot_cfg": cfg.ik_robot_cfg,
            "num_seeds": cfg.ik_num_seeds,
            "max_batch_size": self.num_envs * max(cfg.reach_check_waypoints, cfg.reach_oversample),
            "ik_batch_size": cfg.ik_batch_size,
            "position_tolerance": cfg.reach_pos_tol,
            "orientation_tolerance": cfg.reach_rot_tol,
            "use_cuda_graph": cfg.ik_use_cuda_graph,
            "device": "cuda:0",
        }
        self._ik_worker_mod = curobo_ik_worker
        self._ik_proc = subprocess.Popen(
            [sys.executable, curobo_ik_worker.__file__, json.dumps(worker_cfg)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=worker_env,
        )

    def _solve_curobo_ik(
        self,
        fingertip_pos: torch.Tensor,
        fingertip_quat: torch.Tensor,
        env_ids: torch.Tensor,
        seed_arm_q: torch.Tensor | None = None,
    ):
        """Solve cuRobo IK (in the worker) for fingertip targets.

        `fingertip_pos`/`fingertip_quat` are (n, W, ...) env-local targets for the
        envs in `env_ids`. Targets are converted fingertip->hand->base frame here,
        sent to the worker as a flat batch of n*W problems, and the returned
        per-target success and arm joints are reshaped back. `seed_arm_q`, if given,
        is a (n, W, 7) tensor of initial joint seeds (panda_joint1..7 order) passed
        to the solver so the returned solution stays near it. Returns
        (success (n, W) bool, arm_q (n, W, 7)).
        """
        self._ensure_ik_worker()
        n_envs, num_wp = fingertip_pos.shape[0], fingertip_pos.shape[1]

        env_origins = self.scene.env_origins[env_ids].unsqueeze(1)  # (n, 1, 3)
        base_pos = self.base_pos_w[env_ids].unsqueeze(1).expand(-1, num_wp, -1)
        base_quat = self.base_quat_w[env_ids].unsqueeze(1).expand(-1, num_wp, -1)
        off_pos = self.hand_in_fingertip_pos[env_ids].unsqueeze(1).expand(-1, num_wp, -1)
        off_quat = self.hand_in_fingertip_quat[env_ids].unsqueeze(1).expand(-1, num_wp, -1)

        # Fingertip target: env-local -> world (env frame is a pure translation).
        fp_pos_w = (fingertip_pos + env_origins).reshape(-1, 3)
        fp_quat_w = fingertip_quat.reshape(-1, 4)
        # Fingertip -> hand (constant offset), then world -> base frame.
        hand_pos_w, hand_quat_w = math_utils.combine_frame_transforms(
            fp_pos_w, fp_quat_w, off_pos.reshape(-1, 3), off_quat.reshape(-1, 4)
        )
        hand_pos_b, hand_quat_b = math_utils.subtract_frame_transforms(
            base_pos.reshape(-1, 3), base_quat.reshape(-1, 4), hand_pos_w, hand_quat_w
        )

        seed_np = None if seed_arm_q is None else seed_arm_q.reshape(-1, 7).detach().cpu().numpy()
        self._ik_worker_mod.send_msg(
            self._ik_proc.stdin,
            (hand_pos_b.detach().cpu().numpy(), hand_quat_b.detach().cpu().numpy(), seed_np),
        )
        response = self._ik_worker_mod.recv_msg(self._ik_proc.stdout)
        if response is None:
            raise RuntimeError(
                f"cuRobo IK worker exited unexpectedly (return code {self._ik_proc.poll()}); see its stderr above."
            )
        success_np, arm_q_np = response

        success = torch.as_tensor(success_np, device=self.device).view(n_envs, num_wp)
        arm_q = torch.as_tensor(arm_q_np, device=self.device, dtype=torch.float32).view(n_envs, num_wp, 7)
        return success, arm_q

    def close(self):
        """Tear down the cuRobo IK worker before closing the sim."""
        import subprocess

        if self._ik_proc is not None:
            if self._ik_proc.poll() is None:
                self._ik_proc.stdin.close()
                try:
                    self._ik_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._ik_proc.kill()
            self._ik_proc = None
        super().close()

    def _sample_reachable_start_and_trajectory(self, env_ids: torch.Tensor):
        """Sample a start pose + trajectory per env, keeping only continuously reachable ones.

        Because the reachability + continuity check rejects many samples, each env
        is *oversampled*: `init.reach_oversample` independent candidate trajectories
        are drawn per pending env and evaluated together. For every candidate we
        walk the `reach_check_waypoints` waypoints (spanning the episode) from first
        to last, solving cuRobo IK one waypoint at a time, seeding each solve after
        the first with the previous waypoint's joint solution so the solutions stay
        on a single IK branch. A candidate is accepted only if every waypoint is
        reachable (IK success within `reach_pos_tol`/`reach_rot_tol`) *and*
        consecutive joint solutions never jump by more than
        `reach_joint_diff_threshold` on any joint. The first accepted candidate is
        kept per env; envs with no accepted candidate are re-oversampled (up to
        `init.max_reach_attempts`). Finally every reset env is placed at the cuRobo
        joint solution for its start pose (t=0).
        """
        cfg = self.cfg.init
        num_wp = cfg.reach_check_waypoints
        k = max(1, int(cfg.reach_oversample))
        # cuRobo arm-joint solution for each env's start pose; filled during the check.
        start_arm_q = self.joint_pos[:, 0:7].clone()
        waypoint_times = torch.linspace(0.0, self.max_episode_length_s, num_wp, device=self.device)

        bad_envs = env_ids.clone()
        attempt = 0
        while True:
            n_bad = len(bad_envs)
            # Draw `k` candidate trajectories per pending env (env-major flatten:
            # candidate index c = b * k + j maps to env bad_envs[b]).
            params = self._sample_candidate_params(bad_envs, k)
            cand_env_ids = bad_envs.repeat_interleave(k)

            # Walk the waypoints in order, seeding each solve with the previous
            # solution and rejecting candidates whose joints jump between waypoints.
            accept = torch.ones(n_bad * k, dtype=torch.bool, device=self.device)
            cand_start_q = torch.zeros((n_bad * k, 7), device=self.device)
            prev_q = None
            for i in range(num_wp):
                elapsed_time = torch.full((n_bad * k, 1), waypoint_times[i].item(), device=self.device)
                pos_i, quat_i = self._eval_pose_from_params(params, elapsed_time)
                seed = None if prev_q is None else prev_q.unsqueeze(1)
                success, arm_q = self._solve_curobo_ik(
                    pos_i.unsqueeze(1), quat_i.unsqueeze(1), cand_env_ids, seed_arm_q=seed
                )
                success = success[:, 0]
                arm_q = arm_q[:, 0]
                accept &= success
                if prev_q is not None:
                    joint_diff = (arm_q - prev_q).abs().amax(dim=-1)
                    accept &= joint_diff <= cfg.reach_joint_diff_threshold
                if i == 0:
                    cand_start_q = arm_q.clone()
                prev_q = arm_q

            # Pick the first accepted candidate per env (fallback to candidate 0 so
            # envs that never pass still get a fresh, if invalid, trajectory).
            accept_bk = accept.view(n_bad, k)
            any_ok = accept_bk.any(dim=1)
            sel_j = torch.argmax(accept_bk.int(), dim=1)
            sel_flat = torch.arange(n_bad, device=self.device) * k + sel_j
            self._assign_params_to_envs(bad_envs, params, sel_flat)
            start_arm_q[bad_envs] = cand_start_q[sel_flat]

            bad_envs = bad_envs[(~any_ok).nonzero(as_tuple=False).squeeze(-1)]
            attempt += 1
            if bad_envs.shape[0] == 0 or attempt >= cfg.max_reach_attempts:
                break

        # Discretize the (now-final) analytic trajectory onto the per-step grid.
        self._build_trajectory_buffer(env_ids)

        # Place every reset env at the cuRobo joint solution that reaches its start pose.
        self.joint_pos[env_ids, 0:7] = start_arm_q[env_ids]
        if self.joint_pos.shape[1] > 7:
            self.joint_pos[env_ids, 7:] = 0.04
        self.joint_vel[env_ids] = 0.0
        self.ctrl_target_joint_pos[env_ids] = self.joint_pos[env_ids]
        self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(torch.zeros_like(self.joint_pos))
        self.step_sim_no_action()
        self.extras["log"]["Init/unreachable_envs"] = torch.tensor(float(bad_envs.shape[0]), device=self.device)

    def _sample_candidate_params(self, env_ids: torch.Tensor, num_candidates: int) -> dict:
        """Sample `num_candidates` start-pose + trajectory param sets per env.

        Returns a dict of tensors, each with leading dim ``len(env_ids) *
        num_candidates`` flattened env-major (candidate c = b * k + j belongs to
        env ``env_ids[b]``). Env-specific centers (`nominal_start_pos/quat`) are
        broadcast across the `num_candidates` candidates of each env. Trajectory
        params for the inactive mode are filled with harmless defaults so
        `_assign_params_to_envs` can copy every key uniformly.
        """
        cfg = self.cfg.init
        tcfg = self.cfg.tracking
        device = self.device
        m = len(env_ids) * num_candidates
        env_rep = env_ids.repeat_interleave(num_candidates)

        if cfg.start_pos_box is not None:
            # Absolute box in the base/env-local frame: uniform in [min, max] per axis.
            box = torch.tensor(cfg.start_pos_box, device=device)  # (3, 2)
            lo, hi = box[:, 0], box[:, 1]
            start_pos = lo + (hi - lo) * torch.rand((m, 3), device=device)
        else:
            pos_noise = torch.tensor(cfg.start_pos_noise, device=device)
            offset = (2.0 * torch.rand((m, 3), device=device) - 1.0) * pos_noise
            start_pos = self.nominal_start_pos[env_rep] + offset

        axis = torch.randn((m, 3), device=device)
        axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-6)
        angle = cfg.start_rot_noise * torch.rand(m, device=device)
        delta_quat = torch_utils.quat_from_angle_axis(angle, axis)
        start_quat = torch_utils.quat_mul(delta_quat, self.nominal_start_quat[env_rep])

        params = {
            "start_pos": start_pos,
            "start_quat": start_quat,
            "dir": torch.zeros((m, 3), device=device),
            "speed": torch.zeros((m, 1), device=device),
            "length": torch.zeros((m, 1), device=device),
            "u": torch.zeros((m, 3), device=device),
            "v": torch.zeros((m, 3), device=device),
            "radius": torch.zeros((m, 1), device=device),
            "omega": torch.zeros((m, 1), device=device),
        }
        params["u"][:, 0] = 1.0
        params["v"][:, 1] = 1.0

        if tcfg.mode == "line":
            direction = torch.randn((m, 3), device=device)
            direction = direction / torch.clamp(torch.linalg.norm(direction, dim=-1, keepdim=True), min=1.0e-6)
            params["dir"] = direction
            speed_lo, speed_hi = tcfg.line_speed_range
            params["speed"] = speed_lo + (speed_hi - speed_lo) * torch.rand((m, 1), device=device)
            length_lo, length_hi = tcfg.line_length_range
            params["length"] = length_lo + (length_hi - length_lo) * torch.rand((m, 1), device=device)
        elif tcfg.mode == "circle":
            # Random orthonormal plane basis (u, v) via Gram-Schmidt.
            u = torch.randn((m, 3), device=device)
            u = u / torch.clamp(torch.linalg.norm(u, dim=-1, keepdim=True), min=1.0e-6)
            v = torch.randn((m, 3), device=device)
            v = v - (v * u).sum(dim=-1, keepdim=True) * u
            v = v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1.0e-6)
            params["u"] = u
            params["v"] = v
            radius_lo, radius_hi = tcfg.circle_radius_range
            params["radius"] = radius_lo + (radius_hi - radius_lo) * torch.rand((m, 1), device=device)
            speed_lo, speed_hi = tcfg.circle_speed_range
            omega = speed_lo + (speed_hi - speed_lo) * torch.rand((m, 1), device=device)
            direction_sign = torch.where(
                torch.rand((m, 1), device=device) < 0.5,
                -torch.ones((m, 1), device=device),
                torch.ones((m, 1), device=device),
            )
            params["omega"] = omega * direction_sign
        else:
            raise ValueError(f"Unsupported tracking mode: {tcfg.mode}")

        rot_axis = torch.randn((m, 3), device=device)
        rot_axis = rot_axis / torch.clamp(torch.linalg.norm(rot_axis, dim=-1, keepdim=True), min=1.0e-6)
        params["rot_axis"] = rot_axis
        rot_speed_lo, rot_speed_hi = tcfg.rot_speed_range
        params["rot_speed"] = rot_speed_lo + (rot_speed_hi - rot_speed_lo) * torch.rand((m, 1), device=device)
        rot_angle_lo, rot_angle_hi = tcfg.rot_angle_range
        params["rot_angle"] = rot_angle_lo + (rot_angle_hi - rot_angle_lo) * torch.rand((m, 1), device=device)
        return params

    _TRAJ_PARAM_BUFFERS = {
        "start_pos": "traj_start_pos",
        "start_quat": "traj_start_quat",
        "dir": "traj_dir",
        "speed": "traj_speed",
        "length": "traj_length",
        "u": "traj_u",
        "v": "traj_v",
        "radius": "traj_radius",
        "omega": "traj_omega",
        "rot_axis": "traj_rot_axis",
        "rot_speed": "traj_rot_speed",
        "rot_angle": "traj_rot_angle",
    }

    def _assign_params_to_envs(self, env_ids: torch.Tensor, params: dict, cand_idx: torch.Tensor):
        """Copy the selected candidate params (`cand_idx`) into the per-env traj buffers."""
        for key, buf_name in self._TRAJ_PARAM_BUFFERS.items():
            getattr(self, buf_name)[env_ids] = params[key][cand_idx]

    def _eval_pose_from_params(self, params: dict, elapsed_time: torch.Tensor):
        """Evaluate the reference pose at `elapsed_time` for the given trajectory params.

        `params` maps the trajectory keys to tensors with a shared leading dim `M`;
        `elapsed_time` is (M, 1). The trajectory is anchored at `start_pos`/
        `start_quat`, so all modes pass through it at ``elapsed_time == 0``.
        """
        mode = self.cfg.tracking.mode
        start_pos = params["start_pos"]
        start_quat = params["start_quat"]
        if mode == "line":
            displacement = torch.minimum(params["speed"] * elapsed_time, params["length"])
            pos = start_pos + params["dir"] * displacement
        elif mode == "circle":
            theta = params["omega"] * elapsed_time
            pos = start_pos + params["radius"] * (
                (torch.cos(theta) - 1.0) * params["u"] + torch.sin(theta) * params["v"]
            )
        else:
            raise ValueError(f"Unsupported tracking mode: {mode}")

        sweep_angle = torch.minimum(params["rot_speed"] * elapsed_time, params["rot_angle"]).squeeze(-1)
        delta_quat = torch_utils.quat_from_angle_axis(sweep_angle, params["rot_axis"])
        quat = torch_utils.quat_mul(delta_quat, start_quat)
        return pos, quat

    def _command_pose_at(self, elapsed_time: torch.Tensor):
        """Evaluate the reference pose at `elapsed_time` (shape (num_envs, 1))."""
        params = {key: getattr(self, buf_name) for key, buf_name in self._TRAJ_PARAM_BUFFERS.items()}
        return self._eval_pose_from_params(params, elapsed_time)

    def _build_trajectory_buffer(self, env_ids: torch.Tensor):
        """Sample the analytic trajectory onto the fixed per-step grid for `env_ids`.

        The grid times are `arange(self._traj_len) * step_dt`, i.e. one waypoint
        per control step, extended past the episode so the furthest lookahead is
        always in-range. After this call the runtime command and lookahead read
        straight out of `traj_pos_buf`/`traj_quat_buf` by integer step index.
        """
        for i in range(self._traj_len):
            elapsed_time = torch.full((self.num_envs, 1), i * self.step_dt, device=self.device)
            pos_i, quat_i = self._command_pose_at(elapsed_time)
            self.traj_pos_buf[env_ids, i] = pos_i[env_ids]
            self.traj_quat_buf[env_ids, i] = quat_i[env_ids]

    def _traj_pose_at_index(self, idx: torch.Tensor):
        """Gather the discretized reference pose at per-env step index `idx` (num_envs,)."""
        idx = idx.clamp(0, self._traj_len - 1)
        return self.traj_pos_buf[self._env_arange, idx], self.traj_quat_buf[self._env_arange, idx]

    def _update_command(self):
        """Set the current reference pose to the discretized waypoint at the current step."""
        self.command_pos[:], self.command_quat[:] = self._traj_pose_at_index(self.episode_length_buf)

    def _future_command_errors(self) -> torch.Tensor:
        """Errors to `num_future_steps` lookahead reference poses (index 0 = now).

        Lookahead poses are the next discretized waypoints (one per step) starting
        from the current step. Returns a (num_envs, 6 * num_future_steps) tensor of
        stacked (pos_error, axis_angle_error) pairs, letting the policy infer the
        reference velocity from the future pose sequence.
        """
        num_steps = self.cfg.tracking.num_future_steps
        base_idx = self.episode_length_buf

        errors = []
        for i in range(num_steps):
            target_pos, target_quat = self._traj_pose_at_index(base_idx + i)
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )
            errors.append(pos_error)
            errors.append(axis_angle_error)
        return torch.cat(errors, dim=-1)

    def _randomize_controller(self, env_ids: torch.Tensor):
        gains = self.default_gains[env_ids].clone()
        noise = torch.tensor(self.cfg.ctrl.task_prop_gains_noise_level, device=self.device).unsqueeze(0)
        if torch.any(noise > 0):
            gains = gains * (1.0 + (2.0 * torch.rand_like(gains) - 1.0) * noise)
        self.task_prop_gains[env_ids] = gains
        self.task_deriv_gains[env_ids] = factory_utils.get_deriv_gains(gains)

    def _randomize_dynamics(self, env_ids: torch.Tensor):
        env_ids_cpu = env_ids.cpu()
        masses = self.default_masses.clone()
        inertias = self.default_inertias.clone()
        coms = self.default_coms.clone()

        rand_cfg = self.cfg.randomization
        mass_device = masses.device
        default_masses = self.default_masses.to(mass_device)
        default_inertias = self.default_inertias.to(mass_device)

        if rand_cfg.enable_link_mass:
            lower, upper = rand_cfg.link_mass_scale_range
            scales = lower + (upper - lower) * torch.rand(
                (len(env_ids), self._robot.num_bodies), device=mass_device
            )
        else:
            scales = torch.ones((len(env_ids), self._robot.num_bodies), device=mass_device)

        masses[env_ids_cpu] = default_masses[env_ids_cpu] * scales
        inertias[env_ids_cpu] = default_inertias[env_ids_cpu] * scales.unsqueeze(-1)

        # The payload is merged into the payload body as a rigid point mass: combined mass, mass-weighted
        # CoM, and parallel-axis inertia update. The simulated dynamics are then exactly "nominal robot +
        # point mass", while its weight is applied separately as an external wrench (robot gravity is
        # disabled, see `_apply_payload_gravity`). The controller stays blind to all of this because it
        # receives the reconstructed nominal mass matrix (see `_compute_arm_mass_matrix`).
        if rand_cfg.enable_payload:
            payload_mass_lower, payload_mass_upper = rand_cfg.payload_mass_range
            payload_mass = payload_mass_lower + (payload_mass_upper - payload_mass_lower) * torch.rand(
                (len(env_ids), 1), device=self.device
            )
            payload_com_ranges = torch.tensor(rand_cfg.payload_com_range, device=self.device)
            payload_com = payload_com_ranges[:, 0] + (
                payload_com_ranges[:, 1] - payload_com_ranges[:, 0]
            ) * torch.rand((len(env_ids), 3), device=self.device)
            self.payload_mass[env_ids] = payload_mass
            self.payload_com[env_ids] = payload_com

            body_idx = self.payload_body_idx
            m_p = payload_mass.to(mass_device)
            p = payload_com.to(mass_device)
            m_b = masses[env_ids_cpu, body_idx].unsqueeze(-1)
            c_b = coms[env_ids_cpu, body_idx, 0:3]
            m_new = m_b + m_p
            c_new = (m_b * c_b + m_p * p) / m_new
            d_b = c_b - c_new
            d_p = p - c_new
            eye = torch.eye(3, device=mass_device).unsqueeze(0)
            inertia_b = inertias[env_ids_cpu, body_idx].view(-1, 3, 3)
            shift_b = m_b.unsqueeze(-1) * (
                (d_b * d_b).sum(-1)[:, None, None] * eye - d_b.unsqueeze(-1) * d_b.unsqueeze(-2)
            )
            shift_p = m_p.unsqueeze(-1) * (
                (d_p * d_p).sum(-1)[:, None, None] * eye - d_p.unsqueeze(-1) * d_p.unsqueeze(-2)
            )
            masses[env_ids_cpu, body_idx] = m_new.squeeze(-1)
            coms[env_ids_cpu, body_idx, 0:3] = c_new
            inertias[env_ids_cpu, body_idx] = (inertia_b + shift_b + shift_p).view(-1, 9)
            # PhysX jacobians are referenced at the current (merged) CoM; record the link-frame offset
            # back to the nominal CoM for the controller-side mass-matrix reconstruction.
            self.nominal_com_offsets[env_ids, body_idx - 1] = (c_b - c_new).to(self.device)
        else:
            self.payload_mass[env_ids] = 0.0
            self.payload_com[env_ids] = 0.0
            self.nominal_com_offsets[env_ids, self.payload_body_idx - 1] = 0.0

        self._robot.root_physx_view.set_masses(masses, env_ids_cpu)
        self._robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
        self._robot.root_physx_view.set_coms(coms, env_ids_cpu)

        self._dynamics_write_sim_step = self._sim_step_counter
        if not self._randomized_mass_matrix_validated:
            self._mass_matrix_checks_pending += 1
            self._randomized_mass_matrix_validated = True

        self.link_mass_scales[env_ids] = scales.to(self.device)

        if rand_cfg.enable_joint_friction:
            lower, upper = rand_cfg.joint_friction_range
            joint_friction = lower + (upper - lower) * torch.rand((len(env_ids), 7), device=self.device)
        else:
            joint_friction = torch.zeros((len(env_ids), 7), device=self.device)
        self.joint_friction[env_ids] = joint_friction
        self._robot.write_joint_friction_coefficient_to_sim(
            joint_friction,
            joint_ids=self.arm_joint_ids,
            env_ids=env_ids,
        )

    def render(self, recompute: bool = False):
        """Render and overlay env-0 per-step tracking error on the top-right corner."""
        frame = super().render(recompute)
        if frame is None or self.render_mode != "rgb_array":
            return frame
        if self._vis_pos_error_norm is None or self._vis_rot_error_norm is None:
            return frame

        import cv2

        pos_err_mm = float(self._vis_pos_error_norm[0]) * 1000.0
        rot_err_deg = math.degrees(float(self._vis_rot_error_norm[0]))
        lines = [
            f"trans err: {pos_err_mm:6.1f} mm",
            f"rot err:   {rot_err_deg:6.2f} deg",
        ]

        # The annotator returns an RGBA buffer; draw on a contiguous RGB copy.
        frame = np.ascontiguousarray(frame[:, :, :3])
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        pad = 6
        line_h = 22
        text_w = max(cv2.getTextSize(t, font, scale, thickness)[0][0] for t in lines)
        box_w = text_w + 2 * pad
        box_h = line_h * len(lines) + pad
        x0 = w - box_w - 10
        y0 = 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0.0)
        for i, text in enumerate(lines):
            y = y0 + pad + (i + 1) * line_h - 6
            cv2.putText(frame, text, (x0 + pad, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
        return frame

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "command_pose_visualizer"):
                frame_cfg = FRAME_MARKER_CFG.copy()
                frame_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
                frame_cfg.prim_path = "/Visuals/Command/pose"
                self.command_pose_visualizer = VisualizationMarkers(frame_cfg)

                # Current end-effector pose (smaller frame) so the gap to the
                # command frame shows the live tracking error.
                ee_cfg = FRAME_MARKER_CFG.copy()
                ee_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
                ee_cfg.prim_path = "/Visuals/EndEffector/pose"
                self.ee_pose_visualizer = VisualizationMarkers(ee_cfg)

                path_cfg = SPHERE_MARKER_CFG.copy()
                path_cfg.markers["sphere"].radius = 0.004
                path_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                path_cfg.prim_path = "/Visuals/Command/traj_path"
                self.traj_path_visualizer = VisualizationMarkers(path_cfg)

                future_cfg = SPHERE_MARKER_CFG.copy()
                future_cfg.markers["sphere"].radius = 0.008
                future_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 0.4, 1.0)
                future_cfg.prim_path = "/Visuals/Command/future_targets"
                self.future_target_visualizer = VisualizationMarkers(future_cfg)
            self.command_pose_visualizer.set_visibility(True)
            self.ee_pose_visualizer.set_visibility(True)
            self.traj_path_visualizer.set_visibility(True)
            self.future_target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "command_pose_visualizer"):
                self.command_pose_visualizer.set_visibility(False)
                self.ee_pose_visualizer.set_visibility(False)
                self.traj_path_visualizer.set_visibility(False)
                self.future_target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        env_origins = self.scene.env_origins

        # Current reference pose (frame marker).
        self.command_pose_visualizer.visualize(self.command_pos + env_origins, self.command_quat)

        # Current end-effector pose (frame marker); offset from the command
        # frame is the live tracking error.
        self.ee_pose_visualizer.visualize(self.fingertip_midpoint_pos + env_origins, self.fingertip_midpoint_quat)

        # Full discretized reference path over the episode (strided waypoints).
        num_samples = min(self.cfg.debug_vis_path_samples, self.max_episode_length)
        sample_idx = torch.linspace(0, self.max_episode_length - 1, num_samples, device=self.device).long()
        path_points = []
        for idx in sample_idx:
            path_points.append(self.traj_pos_buf[:, idx] + env_origins)
        self.traj_path_visualizer.visualize(torch.cat(path_points, dim=0))

        # Lookahead targets fed to the policy.
        base_idx = self.episode_length_buf
        future_points = []
        for i in range(self.cfg.tracking.num_future_steps):
            pos, _ = self._traj_pose_at_index(base_idx + i)
            future_points.append(pos + env_origins)
        self.future_target_visualizer.visualize(torch.cat(future_points, dim=0))

    def _command_errors(self):
        pos_error, rot_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=self.command_pos,
            ctrl_target_fingertip_midpoint_quat=self.command_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        return pos_error, rot_error

    def _accumulate_perstep_error(self, pos_error_norm: torch.Tensor, rot_error_norm: torch.Tensor):
        """Bin the per-step tracking error by episode step and log a profile periodically."""
        if self._perstep_log_every <= 0:
            return
        step_idx = self.episode_length_buf.clamp(0, self._perstep_num_bins - 1)
        self._perstep_pos_sum.index_add_(0, step_idx, pos_error_norm)
        self._perstep_pos_sqsum.index_add_(0, step_idx, pos_error_norm.square())
        self._perstep_rot_sum.index_add_(0, step_idx, rot_error_norm)
        self._perstep_rot_sqsum.index_add_(0, step_idx, rot_error_norm.square())
        self._perstep_count.index_add_(0, step_idx, torch.ones_like(pos_error_norm))
        self._perstep_step_counter += 1
        if self._perstep_step_counter >= self._perstep_log_every:
            self._log_perstep_error_profile()
            self._reset_perstep_error()

    def _reset_perstep_error(self):
        self._perstep_pos_sum.zero_()
        self._perstep_pos_sqsum.zero_()
        self._perstep_rot_sum.zero_()
        self._perstep_rot_sqsum.zero_()
        self._perstep_count.zero_()
        self._perstep_step_counter = 0

    def _log_perstep_error_profile(self):
        """Log a mean±std line plot of the per-episode-step tracking error to wandb."""
        import wandb

        if wandb.run is None:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = self._perstep_count > 0
        if not torch.any(valid):
            return
        count_safe = self._perstep_count.clamp(min=1.0)
        pos_mean = self._perstep_pos_sum / count_safe
        pos_std = (self._perstep_pos_sqsum / count_safe - pos_mean.square()).clamp(min=0.0).sqrt()
        rot_mean = self._perstep_rot_sum / count_safe
        rot_std = (self._perstep_rot_sqsum / count_safe - rot_mean.square()).clamp(min=0.0).sqrt()

        steps = torch.arange(self._perstep_num_bins, device=self.device)[valid]
        time_s = (steps.float() * self.step_dt).cpu().numpy()
        pos_mean_np = pos_mean[valid].cpu().numpy()
        pos_std_np = pos_std[valid].cpu().numpy()
        rot_mean_np = rot_mean[valid].cpu().numpy()
        rot_std_np = rot_std[valid].cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(time_s, pos_mean_np, color="C0", label="mean")
        axes[0].fill_between(
            time_s, pos_mean_np - pos_std_np, pos_mean_np + pos_std_np, color="C0", alpha=0.25, label="±1 std"
        )
        axes[0].set_xlabel("time in episode (s)")
        axes[0].set_ylabel("position error (m)")
        axes[0].set_title("Per-step position tracking error")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_s, rot_mean_np, color="C1", label="mean")
        axes[1].fill_between(
            time_s, rot_mean_np - rot_std_np, rot_mean_np + rot_std_np, color="C1", alpha=0.25, label="±1 std"
        )
        axes[1].set_xlabel("time in episode (s)")
        axes[1].set_ylabel("orientation error (rad)")
        axes[1].set_title("Per-step orientation tracking error")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        wandb.log({"tracking/perstep_error_profile": wandb.Image(fig)})
        plt.close(fig)

    def _joint_limit_penalty(self):
        soft_limit = 0.95 * math.pi
        excess = torch.relu(torch.abs(self.joint_pos[:, 0:7]) - soft_limit)
        return torch.sum(excess, dim=-1)

    def _log_reset_metrics(self, env_ids: torch.Tensor):
        self.extras["log"] = {}
        if not hasattr(self, "_episode_sums"):
            return
        for key, episodic_sum in self._episode_sums.items():
            self.extras["log"][f"Episode_Reward/{key}"] = torch.mean(episodic_sum[env_ids]) / self.max_episode_length_s
            episodic_sum[env_ids] = 0.0
        self.extras["log"]["Dynamics/payload_mass"] = self.payload_mass[env_ids].mean()
        self.extras["log"]["Dynamics/joint_friction"] = self.joint_friction[env_ids].mean()
