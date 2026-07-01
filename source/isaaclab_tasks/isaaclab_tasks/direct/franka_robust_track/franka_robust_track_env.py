# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
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

        self.command_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

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
        self.link_mass_scales = torch.ones((self.num_envs, self._robot.num_bodies), device=self.device)
        self.joint_friction = torch.zeros((self.num_envs, 7), device=self.device)

        self._resolve_robot_indices()
        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._cache_default_dynamics()
        self._compute_intermediate_values()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["pos_track", "rot_track", "ee_vel", "action_rate", "joint_vel", "joint_limit"]
        }

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

        self.arm_joint_ids = list(range(7))

    def _cache_default_dynamics(self):
        self.default_masses = self._robot.root_physx_view.get_masses().clone()
        self.default_inertias = self._robot.root_physx_view.get_inertias().clone()
        self.default_coms = self._robot.root_physx_view.get_coms().clone()

    def _compute_intermediate_values(self):
        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()
        self.fingertip_midpoint_jacobian = jacobians[:, self.fingertip_body_idx - 1, 0:6, 0:7]
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = self.cfg.ctrl.ema_factor * actions.clone().to(self.device).clamp(-1.0, 1.0) + (
            1.0 - self.cfg.ctrl.ema_factor
        ) * self.actions

    def _apply_action(self):
        self._compute_intermediate_values()
        target_pos, target_quat = self._get_action_target_pose()

        if self.cfg.ctrl.backend == "factory_osc":
            self._apply_factory_osc(target_pos, target_quat)
        elif self.cfg.ctrl.backend == "dls_ik":
            self._apply_dls_ik(target_pos, target_quat)
        else:
            raise ValueError(f"Unsupported Franka robust track controller backend: {self.cfg.ctrl.backend}")

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

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        self._update_command()
        future_errors = self._future_command_errors()
        policy_obs = torch.cat(
            [
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
                self.fingertip_midpoint_linvel,
                self.fingertip_midpoint_angvel,
                self.joint_pos[:, 0:7],
                self.joint_vel[:, 0:7],
                future_errors,
                self.actions,
            ],
            dim=-1,
        )
        critic_obs = torch.cat(
            [
                policy_obs,
                self.payload_mass,
                self.payload_com,
                self.joint_friction,
                self.task_prop_gains,
                self.pos_threshold,
                self.rot_threshold,
            ],
            dim=-1,
        )
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
            "pos_track": (1.0 - torch.tanh(pos_error_norm / 0.05)) * self.cfg.reward.pos_error_scale,
            "rot_track": (1.0 - torch.tanh(rot_error_norm / 0.35)) * self.cfg.reward.rot_error_scale,
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

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, 0:7] = torch.tensor(self.cfg.ctrl.reset_joints, device=self.device).unsqueeze(0)
        if joint_pos.shape[1] > 7:
            joint_pos[:, 7:] = 0.04
        joint_vel = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids] = joint_pos
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.set_joint_effort_target(torch.zeros_like(joint_pos), env_ids=env_ids)

        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values()

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._randomize_dynamics(env_ids)
        self._randomize_controller(env_ids)
        self._sample_trajectory(env_ids)
        self._update_command()

    def _sample_trajectory(self, env_ids: torch.Tensor):
        """Sample per-env reference trajectory parameters at reset.

        The trajectory anchors at the current fingertip pose and follows the
        configured `tracking.mode` for the whole episode (no resampling).
        """
        n_envs = len(env_ids)
        cfg = self.cfg.tracking

        self.traj_start_pos[env_ids] = self.fingertip_midpoint_pos[env_ids].clone()
        self.traj_start_quat[env_ids] = self.fingertip_midpoint_quat[env_ids].clone()

        if cfg.mode == "line":
            direction = torch.randn((n_envs, 3), device=self.device)
            direction = direction / torch.clamp(torch.linalg.norm(direction, dim=-1, keepdim=True), min=1.0e-6)
            self.traj_dir[env_ids] = direction

            speed_lo, speed_hi = cfg.line_speed_range
            self.traj_speed[env_ids] = speed_lo + (speed_hi - speed_lo) * torch.rand((n_envs, 1), device=self.device)
            length_lo, length_hi = cfg.line_length_range
            self.traj_length[env_ids] = length_lo + (length_hi - length_lo) * torch.rand(
                (n_envs, 1), device=self.device
            )
        elif cfg.mode == "circle":
            # Random orthonormal plane basis (u, v) via Gram-Schmidt.
            u = torch.randn((n_envs, 3), device=self.device)
            u = u / torch.clamp(torch.linalg.norm(u, dim=-1, keepdim=True), min=1.0e-6)
            v = torch.randn((n_envs, 3), device=self.device)
            v = v - (v * u).sum(dim=-1, keepdim=True) * u
            v = v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1.0e-6)
            self.traj_u[env_ids] = u
            self.traj_v[env_ids] = v

            radius_lo, radius_hi = cfg.circle_radius_range
            self.traj_radius[env_ids] = radius_lo + (radius_hi - radius_lo) * torch.rand(
                (n_envs, 1), device=self.device
            )
            speed_lo, speed_hi = cfg.circle_speed_range
            omega = speed_lo + (speed_hi - speed_lo) * torch.rand((n_envs, 1), device=self.device)
            direction_sign = torch.where(
                torch.rand((n_envs, 1), device=self.device) < 0.5,
                -torch.ones((n_envs, 1), device=self.device),
                torch.ones((n_envs, 1), device=self.device),
            )
            self.traj_omega[env_ids] = omega * direction_sign
        else:
            raise ValueError(f"Unsupported tracking mode: {cfg.mode}")

        rot_axis = torch.randn((n_envs, 3), device=self.device)
        rot_axis = rot_axis / torch.clamp(torch.linalg.norm(rot_axis, dim=-1, keepdim=True), min=1.0e-6)
        self.traj_rot_axis[env_ids] = rot_axis
        rot_speed_lo, rot_speed_hi = cfg.rot_speed_range
        self.traj_rot_speed[env_ids] = rot_speed_lo + (rot_speed_hi - rot_speed_lo) * torch.rand(
            (n_envs, 1), device=self.device
        )
        rot_angle_lo, rot_angle_hi = cfg.rot_angle_range
        self.traj_rot_angle[env_ids] = rot_angle_lo + (rot_angle_hi - rot_angle_lo) * torch.rand(
            (n_envs, 1), device=self.device
        )

    def _command_pose_at(self, elapsed_time: torch.Tensor):
        """Evaluate the reference pose at `elapsed_time` (shape (num_envs, 1)).

        The trajectory is anchored at the reset fingertip pose, so all modes pass
        through `traj_start_pos`/`traj_start_quat` at ``elapsed_time == 0``.
        """
        mode = self.cfg.tracking.mode
        if mode == "line":
            displacement = torch.minimum(self.traj_speed * elapsed_time, self.traj_length)
            pos = self.traj_start_pos + self.traj_dir * displacement
        elif mode == "circle":
            theta = self.traj_omega * elapsed_time
            pos = self.traj_start_pos + self.traj_radius * (
                (torch.cos(theta) - 1.0) * self.traj_u + torch.sin(theta) * self.traj_v
            )
        else:
            raise ValueError(f"Unsupported tracking mode: {mode}")

        sweep_angle = torch.minimum(self.traj_rot_speed * elapsed_time, self.traj_rot_angle).squeeze(-1)
        delta_quat = torch_utils.quat_from_angle_axis(sweep_angle, self.traj_rot_axis)
        quat = torch_utils.quat_mul(delta_quat, self.traj_start_quat)
        return pos, quat

    def _update_command(self):
        """Advance the current reference pose to the current episode time."""
        elapsed_time = (self.episode_length_buf.to(torch.float) * self.step_dt).unsqueeze(-1)
        self.command_pos[:], self.command_quat[:] = self._command_pose_at(elapsed_time)

    def _future_command_errors(self) -> torch.Tensor:
        """Errors to `num_future_steps` lookahead reference poses (index 0 = now).

        Returns a (num_envs, 6 * num_future_steps) tensor of stacked
        (pos_error, axis_angle_error) pairs, letting the policy infer the
        reference velocity from the future pose sequence.
        """
        num_steps = self.cfg.tracking.num_future_steps
        step_dt = self.cfg.tracking.future_step_dt
        base_time = self.episode_length_buf.to(torch.float) * self.step_dt

        errors = []
        for i in range(num_steps):
            elapsed_time = (base_time + i * step_dt).unsqueeze(-1)
            target_pos, target_quat = self._command_pose_at(elapsed_time)
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
        default_coms = self.default_coms.to(mass_device)

        if rand_cfg.enable_link_mass:
            lower, upper = rand_cfg.link_mass_scale_range
            scales = lower + (upper - lower) * torch.rand(
                (len(env_ids), self._robot.num_bodies), device=mass_device
            )
        else:
            scales = torch.ones((len(env_ids), self._robot.num_bodies), device=mass_device)

        masses[env_ids_cpu] = default_masses[env_ids_cpu] * scales
        inertias[env_ids_cpu] = default_inertias[env_ids_cpu] * scales.unsqueeze(-1)
        coms[env_ids_cpu] = default_coms[env_ids_cpu]

        if rand_cfg.enable_payload:
            payload_mass_lower, payload_mass_upper = rand_cfg.payload_mass_range
            payload_mass = payload_mass_lower + (payload_mass_upper - payload_mass_lower) * torch.rand(
                (len(env_ids),), device=mass_device
            )
            payload_com_ranges = torch.tensor(rand_cfg.payload_com_range, device=mass_device)
            payload_com = payload_com_ranges[:, 0] + (
                payload_com_ranges[:, 1] - payload_com_ranges[:, 0]
            ) * torch.rand((len(env_ids), 3), device=mass_device)

            body_idx = self.payload_body_idx
            body_mass = masses[env_ids_cpu, body_idx]
            base_com = default_coms[env_ids_cpu, body_idx, 0:3]
            new_mass = body_mass + payload_mass
            new_com = (body_mass.unsqueeze(-1) * base_com + payload_mass.unsqueeze(-1) * payload_com) / torch.clamp(
                new_mass.unsqueeze(-1), min=1.0e-6
            )
            masses[env_ids_cpu, body_idx] = new_mass
            coms[env_ids_cpu, body_idx, 0:3] = new_com
            self.payload_mass[env_ids] = payload_mass.to(self.device).unsqueeze(-1)
            self.payload_com[env_ids] = payload_com.to(self.device)
        else:
            self.payload_mass[env_ids] = 0.0
            self.payload_com[env_ids] = 0.0

        self._robot.root_physx_view.set_masses(masses, env_ids_cpu)
        self._robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
        self._robot.root_physx_view.set_coms(coms, env_ids_cpu)

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
