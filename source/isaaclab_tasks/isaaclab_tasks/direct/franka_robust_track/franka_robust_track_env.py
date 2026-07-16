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


_AUTO_TOOL_BODY_CANDIDATES = ("panda_fingertip_centered", "force_sensor", "panda_hand")
_IDENTITY_TOOL_OFFSET = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

# Virtual frames are (parent body, parent->tool position, parent->tool quaternion).
# From panda_arm_wsg_horizontal.urdf, hand->fr3_wsg_tcp is z=0.271 m,
# yaw=+45 degrees. These frames are inserted into cuRobo's tree as fixed links.
_VIRTUAL_TOOL_FRAMES = {
    "fr3_wsg_tcp": (
        "panda_hand",
        (0.0, 0.0, 0.271),
        (0.9238795325, 0.0, 0.0, 0.3826834324),
    ),
}


class FrankaRobustTrackEnv(DirectRLEnv):
    """Robot-only Franka direct env for robust Cartesian command tracking."""

    cfg: FrankaRobustTrackEnvCfg

    def __init__(self, cfg: FrankaRobustTrackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.current_action_rep = str(getattr(self.cfg.ctrl, "action_rep", "delta_ee_pose"))
        self.delta_target_mode = str(getattr(self.cfg.ctrl, "delta_target_mode", "per_physics_step"))
        if self.delta_target_mode not in ("per_physics_step", "per_control_step"):
            raise ValueError(
                "ctrl.delta_target_mode must be 'per_physics_step' or 'per_control_step', "
                f"got {self.delta_target_mode!r}."
            )
        self._action_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._action_target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )
        self.packet_stiffness_override = None
        self.packet_deriv_override = None
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        # Observation history: only the proprioceptive part is stacked over the last
        # `obs_history_length` frames (concatenated oldest->newest) so the network
        # can infer velocities/dynamics from the past. The lookahead future errors
        # controller gains, and the critic's privileged dims are forward-looking /
        # episode-constant, so they are appended once from the current frame rather
        # than duplicated across the history. H = 1 keeps the single-step observation.
        self.obs_history_length = max(1, int(self.cfg.obs_history_length))
        # ee_pos(3)+ee_quat(4)+joint_pos(7) = 14, plus the (possibly gain-augmented)
        # action vector fed back into the proprio observation.
        self.action_dim = self.actions.shape[-1]
        self.proprio_dim = 14 + self.action_dim
        self.future_dim = 6 * self.cfg.tracking.num_future_steps
        # Force-tracking add-on: (current + lookahead) target-wrench block appended
        # to the policy/critic observation when enabled.
        self.enable_force = bool(self.cfg.tracking.enable_force)
        self.force_dim = 3 * self.cfg.tracking.num_future_steps if self.enable_force else 0
        self._force_mag_max = max(float(self.cfg.tracking.force_mag_range[1]), 1.0e-6)
        self.controller_context_dim = 6
        self.privileged_dim = 24
        self.proprio_history = torch.zeros(
            (self.num_envs, self.obs_history_length, self.proprio_dim), device=self.device
        )

        self.command_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Nominal reset EE pose (from `ctrl.reset_joints`), captured at reset and
        # used as the center of the randomized start-pose sampling box.
        self.nominal_start_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.nominal_start_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Per-env arm joint configuration the env resets into (the cuRobo IK solution
        # for the sampled start pose). Used as the OSC nullspace posture anchor so the
        # redundancy resolution biases toward each env's own reachable start config
        # rather than a single global `ctrl.default_dof_pos_tensor`.
        self.nominal_start_joint_pos = torch.tensor(
            self.cfg.ctrl.reset_joints, device=self.device
        ).repeat(self.num_envs, 1)

        # cuRobo IK runs in a persistent subprocess (avoids a warp/PhysX GPU clash);
        # the worker + cached kinematic frames are set up lazily on first reset.
        self._ik_proc = None
        self._ik_worker_mod = None
        self._kin_frames_cached = False
        # Constant transforms between cuRobo's `panda_hand` IK frame and the
        # configured controller/dataset tool frame.
        self.hand_in_tool_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_in_tool_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.tool_in_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.tool_in_hand_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
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
        # Discretized target-wrench sequence (world/base-frame 3D force per step),
        # sampled per env at reset when force tracking is enabled.
        self.traj_wrench_buf = torch.zeros((self.num_envs, self._traj_len, 3), device=self.device)
        # Per-step contact flag (True where a contact band is active), used to split
        # tracking metrics into free-space vs contact phases.
        self.traj_contact_buf = torch.zeros((self.num_envs, self._traj_len), dtype=torch.bool, device=self.device)
        # Current-step target wrench (world frame), used for the applied external
        # force and the debug visualization.
        self.target_wrench = torch.zeros((self.num_envs, 3), device=self.device)
        # Whether a contact band is active this step (for free/contact metric split).
        self.current_contact = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._env_arange = torch.arange(self.num_envs, device=self.device)

        # Dataset-reference mode: per-env index of the demonstration trajectory
        # currently being tracked (into `self._ds_pos`/`_ds_quat`/`_ds_wrench`),
        # plus the sampled smooth-warp vectors. A zero warp vector leaves a sampled
        # demonstration unchanged.
        self.traj_ds_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.traj_ds_warp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_ds_warp_rot = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_ds_warp_pos_u = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_ds_warp_rot_u = torch.zeros((self.num_envs, 3), device=self.device)
        self.traj_ds_warp_phase = torch.zeros((self.num_envs, 1), device=self.device)
        self.traj_ds_warp_cycles = torch.zeros((self.num_envs, 1), device=self.device)
        self._ds_pos = None
        self._ds_quat = None
        self._ds_wrench = None
        self._ds_num = 0
        if self.cfg.tracking.mode == "dataset":
            self._load_dataset_trajectories()

        self.fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.fingertip_midpoint_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        # Previous-step selected-tool velocity, used to penalize the step-to-step
        # velocity change (acceleration) as a smoothness regularizer. Legacy names
        # are retained because these tensors also feed existing checkpoints/code.
        self.prev_fingertip_midpoint_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_midpoint_angvel = torch.zeros((self.num_envs, 3), device=self.device)
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
        # Bounds the policy-scheduled proportional gains are mapped into when
        # ctrl.control_gains is enabled (normalized action [-1, 1] -> [min, max]).
        self.gain_min = torch.tensor(self.cfg.ctrl.task_prop_gains_min, device=self.device).repeat(self.num_envs, 1)
        self.gain_max = torch.tensor(self.cfg.ctrl.task_prop_gains_max, device=self.device).repeat(self.num_envs, 1)

        self.payload_mass = torch.zeros((self.num_envs, 1), device=self.device)
        self.payload_com = torch.zeros((self.num_envs, 3), device=self.device)
        self.gravity_vec = torch.tensor(self.cfg.sim.gravity, device=self.device).repeat(self.num_envs, 1)
        self.link_mass_scales = torch.ones((self.num_envs, self._robot.num_bodies), device=self.device)
        self.joint_friction = torch.zeros((self.num_envs, 7), device=self.device)
        # Per-joint reflected rotor inertia written to PhysX (domain randomization).
        # Cached for the critic's privileged observation and logging; the OSC
        # controller stays blind to it (see `arm_armature`).
        self.joint_armature = torch.zeros((self.num_envs, 7), device=self.device)

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
            for key in [
                "pos_track",
                "rot_track",
                "ee_vel",
                "ee_accel",
                "action_rate",
                "gain_rate",
                "joint_vel",
                "joint_limit",
            ]
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

        # Policy-scheduled PD gain statistics: when the policy controls the gains,
        # accumulate the per-step task_prop_gains (num_envs x 6) into running
        # sum/sqsum (for mean/std) and per-dim histogram bins over [gain_min,
        # gain_max], then periodically emit scalars + a histogram figure to wandb.
        # Reuses the per-step-error logging cadence and is a no-op otherwise.
        self._gain_stats_num_dims = 6
        self._gain_hist_num_bins = 40
        self._gain_sum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_sqsum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_count = torch.zeros((), device=self.device)
        self._gain_hist = torch.zeros(
            (self._gain_stats_num_dims, self._gain_hist_num_bins), device=self.device
        )
        self._gain_step_counter = 0
        # Same gain sum/sqsum, but split by whether a contact band is active this
        # step, so the scheduled-gain mean/std can be compared between free-space
        # and contact phases (only populated when force tracking is enabled).
        self._gain_free_sum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_free_sqsum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_free_count = torch.zeros((), device=self.device)
        self._gain_contact_sum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_contact_sqsum = torch.zeros(self._gain_stats_num_dims, device=self.device)
        self._gain_contact_count = torch.zeros((), device=self.device)

        # Running rollout mean of the batch tracking error. The rl_games observer
        # only logs the *last* step's `tracking_*_error` per epoch; with synchronized
        # fixed-length episodes that samples a single episode phase and aliases the
        # per-episode reference transient into a sawtooth. Averaging over a full
        # episode-length window (aligned with the synchronized resets) removes the
        # phase dependence so the logged scalar is smooth.
        self._track_err_window = int(self.max_episode_length)
        self._track_err_pos_sum = torch.zeros((), device=self.device)
        self._track_err_rot_sum = torch.zeros((), device=self.device)
        self._track_err_count = 0

        # Same running-window mean, but split by whether a contact band is active
        # this step (free-space vs contact phase). Sums accumulate per-env errors
        # and successes with per-category counts so the logged scalar is the mean
        # over all (env, step) samples of that category in the window. Reset in
        # lockstep with the aggregate window above.
        self._track_free_pos_sum = torch.zeros((), device=self.device)
        self._track_free_rot_sum = torch.zeros((), device=self.device)
        self._track_free_succ_sum = torch.zeros((), device=self.device)
        self._track_free_count = torch.zeros((), device=self.device)
        self._track_contact_pos_sum = torch.zeros((), device=self.device)
        self._track_contact_rot_sum = torch.zeros((), device=self.device)
        self._track_contact_succ_sum = torch.zeros((), device=self.device)
        self._track_contact_count = torch.zeros((), device=self.device)

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
        tool_frame_name = str(getattr(self.cfg, "tool_frame", "auto")).strip() or "auto"
        if tool_frame_name == "auto":
            tool_frame_name = next(
                (name for name in _AUTO_TOOL_BODY_CANDIDATES if name in body_names), None
            )
        if tool_frame_name is None:
            raise RuntimeError(f"Could not find a default Franka tool body in: {body_names}")

        virtual_spec = _VIRTUAL_TOOL_FRAMES.get(tool_frame_name)
        if virtual_spec is None:
            tool_body_name = tool_frame_name
            tool_offset_pos, tool_offset_quat = _IDENTITY_TOOL_OFFSET
        else:
            tool_body_name, tool_offset_pos, tool_offset_quat = virtual_spec

        if tool_body_name not in body_names:
            raise RuntimeError(
                f"Tool frame {tool_frame_name!r} requires body {tool_body_name!r}; "
                f"available bodies: {body_names}."
            )

        self.tool_frame_name = tool_frame_name
        self.tool_body_name = tool_body_name
        self.curobo_uses_selected_tool_frame = virtual_spec is not None
        self.tool_body_idx = body_names.index(tool_body_name)
        self.tool_pos_in_body = torch.tensor(tool_offset_pos, device=self.device).repeat(self.num_envs, 1)
        self.tool_quat_in_body = torch.tensor(tool_offset_quat, device=self.device).repeat(self.num_envs, 1)
        # Backward-compatible alias; for a virtual frame this is its parent body.
        self.fingertip_body_idx = self.tool_body_idx

        payload_body_name = self.cfg.randomization.payload_body_name
        self.payload_body_idx = (
            body_names.index(payload_body_name) if payload_body_name in body_names else self.tool_body_idx
        )

        # Needed for base-frame IK and the legacy tool->panda_hand conversion path.
        self.hand_body_idx = body_names.index("panda_hand")
        self.base_body_idx = body_names.index("panda_link0")
        if self.tool_body_idx == self.base_body_idx:
            raise RuntimeError("tool_frame must name a non-base rigid body with a PhysX Jacobian.")

        if self.enable_force and "force_sensor" not in body_names:
            raise RuntimeError(
                f"tracking.enable_force is set but no 'force_sensor' body exists in {body_names}."
            )
        force_body_name = "force_sensor" if "force_sensor" in body_names else tool_body_name
        self.force_sensor_body_idx = body_names.index(force_body_name)

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
        # Nominal (config-default) armature the controller's reconstructed mass matrix uses. Cached
        # once so the controller stays blind to the per-env armature randomization written to PhysX
        # in `_randomize_dynamics` (that mismatch is a disturbance the policy must reject).
        self.arm_armature = self._robot.root_physx_view.get_dof_armatures().to(self.device)[:, 0:7]

    def _compute_intermediate_values(self):
        # Legacy tensor names are retained for checkpoint/downstream compatibility,
        # but they now contain the configured tool-frame state.
        body_pos_w = self._robot.data.body_pos_w[:, self.tool_body_idx]
        body_quat_w = self._robot.data.body_quat_w[:, self.tool_body_idx]
        tool_pos_w, tool_quat_w = math_utils.combine_frame_transforms(
            body_pos_w, body_quat_w, self.tool_pos_in_body, self.tool_quat_in_body
        )
        body_com_pos_w = self._robot.data.body_com_pos_w[:, self.tool_body_idx]
        body_com_linvel_w = self._robot.data.body_com_lin_vel_w[:, self.tool_body_idx]
        body_angvel_w = self._robot.data.body_com_ang_vel_w[:, self.tool_body_idx]
        com_to_tool_w = tool_pos_w - body_com_pos_w

        self.fingertip_midpoint_pos = tool_pos_w - self.scene.env_origins
        self.fingertip_midpoint_quat = tool_quat_w
        self.fingertip_midpoint_linvel = body_com_linvel_w + torch.linalg.cross(
            body_angvel_w, com_to_tool_w, dim=-1
        )
        self.fingertip_midpoint_angvel = body_angvel_w

        jacobians = self._robot.root_physx_view.get_jacobians()
        tool_jacobian = jacobians[:, self.tool_body_idx - 1, 0:6, 0:7].clone()
        angular_cols = tool_jacobian[:, 3:6, :].transpose(1, 2)
        linear_shift = torch.linalg.cross(
            angular_cols, com_to_tool_w.unsqueeze(1).expand_as(angular_cols), dim=-1
        ).transpose(1, 2)
        tool_jacobian[:, 0:3, :] += linear_shift
        self.fingertip_midpoint_jacobian = tool_jacobian
        if self.cfg.ctrl.use_gt_mass_matrix:
            # Ground-truth controller: use PhysX's own generalized mass matrix, which reflects the
            # payload merge and link-mass scaling (a perfectly-modeled controller).
            self.arm_mass_matrix = (
                self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7].to(self.device)
            )
        else:
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
        armature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rebuild the 7x7 arm mass matrix from per-link jacobians and inertial parameters.

        M(q) = sum_l m_l J_vc^T J_vc + J_w^T (R I R^T) J_w + diag(armature). PhysX link jacobians are
        referenced at each link's *current* CoM (world axes) and inertias are link-frame tensors about
        the CoM (verified against `get_generalized_mass_matrices`). `com_offsets` is the link-frame
        offset from the current CoM to the CoM that `link_masses`/`link_inertias` refer to — nonzero
        only for the payload body when reconstructing the nominal model after a payload merge.

        Evaluated with the cached nominal parameters this gives the mass matrix of the unrandomized
        robot, independent of the payload and link-mass randomization in PhysX.

        `armature` is the per-joint reflected rotor inertia added to the diagonal; it defaults to the
        nominal `self.arm_armature` (what the controller uses, keeping it blind to the sim-side armature
        randomization). This is a controller-side modeling term only: PhysX's `get_generalized_mass_matrices`
        is the pure rigid-body inertia and excludes the DOF armature, so the validation path passes a zero
        armature to compare the reconstruction against PhysX on equal footing.
        """
        if armature is None:
            armature = self.arm_armature
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
        return mass_term + rot_term + torch.diag_embed(armature)

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
        # PhysX's generalized mass matrix is the pure rigid-body joint-space inertia and does NOT
        # include the DOF armature written via `write_joint_armature_to_sim` (armature is applied
        # separately by the solver). Reconstruct without any armature so the comparison isolates the
        # jacobian/inertia conventions rather than tripping on the randomized armature diagonal.
        zero_armature = torch.zeros((self.num_envs, 7), device=self.device)
        reconstructed = self._compute_arm_mass_matrix(
            jacobians, masses, inertias, zero_offsets, armature=zero_armature
        )
        physx_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        error = (reconstructed - physx_matrix).abs().max().item()
        assert error < 0.01, (
            f"Arm mass-matrix reconstruction deviates from PhysX (max abs error {error:.4f}). "
            "Likely a convention mismatch: expected jacobians referenced at the link CoM and inertia "
            "tensors expressed in the link frame about the CoM."
        )

    def _parse_action_packet(self, action_or_packet):
        """Return (action, action_rep, compliance) for tensor or dict step input."""
        if not isinstance(action_or_packet, dict):
            return action_or_packet, str(getattr(self.cfg.ctrl, "action_rep", "delta_ee_pose")), {}
        if "action" not in action_or_packet:
            raise KeyError("Action packet must contain an 'action' tensor.")
        action = action_or_packet["action"]
        action_rep = str(action_or_packet.get("action_rep", getattr(self.cfg.ctrl, "action_rep", "delta_ee_pose")))
        compliance = action_or_packet.get("compliance", None) or {}
        return action, action_rep, compliance

    def _fit_action_dim(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] < self.actions.shape[-1]:
            pad = torch.zeros(
                actions.shape[0],
                self.actions.shape[-1] - actions.shape[-1],
                device=self.device,
                dtype=actions.dtype,
            )
            actions = torch.cat((actions, pad), dim=-1)
        elif actions.shape[-1] > self.actions.shape[-1]:
            actions = actions[:, : self.actions.shape[-1]]
        return actions

    def _delta_action_from_abs_ee_pose(self, target_pose: torch.Tensor) -> torch.Tensor:
        if target_pose.shape[-1] < 7:
            raise ValueError(f"action_rep='abs_ee_pose' requires at least 7 dims, got {tuple(target_pose.shape)}")
        target_pos = target_pose[:, 0:3]
        target_quat = target_pose[:, 3:7]
        target_quat = target_quat / torch.linalg.norm(target_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
        pos_error, axis_angle_error = factory_control.get_pose_error(
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_quat,
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        pose_action = torch.cat(
            (pos_error / self.pos_threshold, axis_angle_error / self.rot_threshold),
            dim=-1,
        )
        if self.cfg.ctrl.control_gains and target_pose.shape[-1] >= 13:
            pose_action = torch.cat((pose_action, target_pose[:, 7:13]), dim=-1)
        return self._fit_action_dim(pose_action)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        actions, action_rep, compliance = self._parse_action_packet(actions)
        if action_rep not in ("rel_ee_pose", "abs_ee_pose", "delta_ee_pose"):
            raise ValueError(f"Unsupported FrankaRobustTrack action_rep={action_rep!r}")
        if action_rep == "rel_ee_pose":
            raise ValueError("FrankaRobustTrackEnv does not support action_rep='rel_ee_pose'.")
        self.current_action_rep = action_rep
        self.packet_stiffness_override = compliance.get("stiffness", None)
        self.packet_deriv_override = compliance.get("damping", None)

        actions = actions.clone().to(self.device)
        if action_rep == "abs_ee_pose":
            actions = self._delta_action_from_abs_ee_pose(actions)
        else:
            actions = self._fit_action_dim(actions)
        self.actions = self.cfg.ctrl.ema_factor * actions.clamp(-1.0, 1.0) + (
            1.0 - self.cfg.ctrl.ema_factor
        ) * self.actions
        if action_rep == "delta_ee_pose" and self.delta_target_mode == "per_control_step":
            # Snapshot the state at the policy/control-step boundary.  _apply_action
            # runs once per decimation substep, so caching the absolute target here
            # prevents the same delta from being repeatedly re-anchored as the EE
            # moves during those substeps.
            target_pos, target_quat = self._get_action_target_pose()
            self._action_target_pos.copy_(target_pos)
            self._action_target_quat.copy_(target_quat)

    def _apply_action(self):
        self._compute_intermediate_values()
        self._update_gains_from_action()
        if self.packet_stiffness_override is not None:
            self.task_prop_gains = self.packet_stiffness_override
            if self.packet_deriv_override is not None:
                self.task_deriv_gains = self.packet_deriv_override
            else:
                self.task_deriv_gains = factory_utils.get_deriv_gains(self.task_prop_gains)
        if self.enable_force:
            # Current step's target wrench (pre-increment episode index), applied
            # as the external disturbance for this physics step.
            self.target_wrench[:] = self._traj_wrench_at_index(self.episode_length_buf)
        self._apply_external_wrenches()
        if self.current_action_rep == "delta_ee_pose" and self.delta_target_mode == "per_control_step":
            target_pos, target_quat = self._action_target_pos, self._action_target_quat
        else:
            target_pos, target_quat = self._get_action_target_pose()

        if self.cfg.ctrl.backend == "factory_control":
            self._apply_factory_control(target_pos, target_quat)
        elif self.cfg.ctrl.backend == "dls_ik":
            self._apply_dls_ik(target_pos, target_quat)
        else:
            raise ValueError(f"Unsupported Franka robust track controller backend: {self.cfg.ctrl.backend}")

    def _apply_external_wrenches(self):
        """Apply the payload weight (and optional target tracking wrench) as external forces.

        Robot gravity is disabled (nominal gravity compensation is assumed perfect), so the payload
        shows up exactly as its uncompensated weight `m * g`, acting at the payload CoM. When force
        tracking is enabled, the sampled target wrench (world-frame 3D force) is additionally applied
        at the wrist `force_sensor` body's origin. Both forces are rotated into their respective link
        frames since the wrench buffers are consumed in that frame, and are set in a single call so
        the two bodies' external wrenches coexist within the step.
        """
        payload_quat_w = self._robot.data.body_quat_w[:, self.payload_body_idx]
        payload_force_b = math_utils.quat_apply_inverse(payload_quat_w, self.payload_mass * self.gravity_vec)

        if not self.enable_force:
            self._robot.set_external_force_and_torque(
                forces=payload_force_b.unsqueeze(1),
                torques=torch.zeros((self.num_envs, 1, 3), device=self.device),
                positions=self.payload_com.unsqueeze(1),
                body_ids=[self.payload_body_idx],
            )
            return

        fs_quat_w = self._robot.data.body_quat_w[:, self.force_sensor_body_idx]
        fs_force_b = math_utils.quat_apply_inverse(fs_quat_w, self.target_wrench)

        forces = torch.stack([payload_force_b, fs_force_b], dim=1)
        torques = torch.zeros((self.num_envs, 2, 3), device=self.device)
        positions = torch.stack(
            [self.payload_com, torch.zeros((self.num_envs, 3), device=self.device)], dim=1
        )
        self._robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            positions=positions,
            body_ids=[self.payload_body_idx, self.force_sensor_body_idx],
        )

    def _update_gains_from_action(self):
        """Map the gain action dims (6:12) to task PD gains when gain control is on.

        The 6 gain actions are clamped to [-1, 1] and affinely mapped onto
        [gain_min, gain_max]; the derivative gains are recomputed for critical
        damping. No-op when `ctrl.control_gains` is False.
        """
        if not self.cfg.ctrl.control_gains:
            return
        gain_actions = self.actions[:, 6:12].clamp(-1.0, 1.0)
        normalized = 0.5 * (gain_actions + 1.0)
        self.task_prop_gains = self.gain_min + (self.gain_max - self.gain_min) * normalized
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.task_prop_gains)

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

    def _apply_factory_control(self, target_pos: torch.Tensor, target_quat: torch.Tensor):
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
            nullspace_joint_target=self.nominal_start_joint_pos,
            apply_task_inertia=self.cfg.ctrl.use_task_space_inertia,
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

    def _compute_obs_parts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the current-frame observation parts.

        Returns (proprio, future_errors, controller_context, privileged): `proprio`
        is stacked over history; `future_errors` and `controller_context` are fed
        once to both policy and critic; and `privileged` is critic-only information.
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
        controller_context = self.task_prop_gains
        privileged = torch.cat(
            [
                self.payload_mass,
                self.payload_com,
                self.joint_friction,
                self.joint_armature,
                self.pos_threshold,
                self.rot_threshold,
            ],
            dim=-1,
        )
        return proprio, future_errors, controller_context, privileged

    def _get_observations(self) -> dict:
        proprio, future_errors, controller_context, privileged = self._compute_obs_parts()
        if self.obs_history_length == 1:
            proprio_stacked = proprio
        else:
            # Shift the proprio history left by one frame and append the newest
            # frame, then flatten oldest->newest into a single vector per env.
            self.proprio_history = torch.roll(self.proprio_history, shifts=-1, dims=1)
            self.proprio_history[:, -1] = proprio
            proprio_stacked = self.proprio_history.reshape(self.num_envs, -1)

        if self.enable_force:
            future_wrench = self._future_wrench()
            policy_obs = torch.cat(
                [proprio_stacked, future_errors, future_wrench, controller_context], dim=-1
            )
            critic_obs = torch.cat(
                [proprio_stacked, future_errors, future_wrench, controller_context, privileged], dim=-1
            )
        else:
            future_wrench = None
            policy_obs = torch.cat([proprio_stacked, future_errors, controller_context], dim=-1)
            critic_obs = torch.cat([proprio_stacked, future_errors, controller_context, privileged], dim=-1)

        # Fail loud if a NaN/Inf reaches the policy/critic input. If this fires, the
        # crash is env-side (controller / sim state) rather than the policy std
        # diverging on its own; the named part tells you where the NaN entered.
        parts = {
            "fingertip_midpoint_pos": self.fingertip_midpoint_pos,
            "fingertip_midpoint_quat": self.fingertip_midpoint_quat,
            "joint_pos": self.joint_pos[:, 0:7],
            "actions": self.actions,
            "future_errors": future_errors,
            "controller_context": controller_context,
            "privileged": privileged,
        }
        if future_wrench is not None:
            parts["future_wrench"] = future_wrench
        bad = {k: int((~torch.isfinite(v)).any(dim=-1).sum()) for k, v in parts.items()}
        if any(bad.values()):
            raise RuntimeError(
                f"Non-finite values in observation before feeding the policy: "
                f"{ {k: n for k, n in bad.items() if n} } (per-part env counts). "
                f"NaN originates in the env/controller, not the policy std."
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
        ee_accel_norm = torch.linalg.norm(
            self.fingertip_midpoint_linvel - self.prev_fingertip_midpoint_linvel, dim=-1
        ) + 0.1 * torch.linalg.norm(self.fingertip_midpoint_angvel - self.prev_fingertip_midpoint_angvel, dim=-1)
        self.prev_fingertip_midpoint_linvel = self.fingertip_midpoint_linvel.clone()
        self.prev_fingertip_midpoint_angvel = self.fingertip_midpoint_angvel.clone()
        # Cartesian delta-pose action rate uses only the first 6 dims so it is
        # unaffected by the optional gain dims, which get their own penalty.
        action_rate = torch.linalg.norm(self.actions[:, 0:6] - self.prev_actions[:, 0:6], dim=-1)
        if self.cfg.ctrl.control_gains:
            gain_rate = torch.linalg.norm(self.actions[:, 6:12] - self.prev_actions[:, 6:12], dim=-1)
        else:
            gain_rate = torch.zeros(self.num_envs, device=self.device)
        joint_vel_norm = torch.linalg.norm(self.joint_vel[:, 0:7], dim=-1)
        joint_limit_penalty = self._joint_limit_penalty()

        rewards = {
            "pos_track": torch.exp(-pos_error_norm / self.cfg.reward.pos_error_temp) * self.cfg.reward.pos_error_scale,
            "rot_track": torch.exp(-rot_error_norm / self.cfg.reward.rot_error_temp) * self.cfg.reward.rot_error_scale,
            "ee_vel": ee_vel_norm * self.cfg.reward.ee_vel_scale,
            "ee_accel": ee_accel_norm * self.cfg.reward.ee_accel_scale,
            "action_rate": action_rate * self.cfg.reward.action_rate_scale,
            "gain_rate": gain_rate * self.cfg.reward.gain_rate_scale,
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
        # Start a fresh window once the previous one filled a full episode length.
        # Reset happens at the top so the window mean is *complete* on the final
        # episode step (the phase the rl_games observer actually logs).
        if self._track_err_count >= self._track_err_window:
            self._track_err_pos_sum.zero_()
            self._track_err_rot_sum.zero_()
            self._track_err_count = 0
            for buf in (
                self._track_free_pos_sum, self._track_free_rot_sum, self._track_free_succ_sum,
                self._track_free_count, self._track_contact_pos_sum, self._track_contact_rot_sum,
                self._track_contact_succ_sum, self._track_contact_count,
            ):
                buf.zero_()
        self._track_err_pos_sum += pos_error_norm.mean()
        self._track_err_rot_sum += rot_error_norm.mean()
        self._track_err_count += 1
        self.extras["tracking_pos_error"] = self._track_err_pos_sum / self._track_err_count
        self.extras["tracking_rot_error"] = self._track_err_rot_sum / self._track_err_count
        if self.enable_force:
            self._accumulate_contact_split_metrics(pos_error_norm, rot_error_norm, successes)
        self._accumulate_perstep_error(pos_error_norm.detach(), rot_error_norm.detach())
        self._accumulate_gain_stats()
        self._vis_pos_error_norm = pos_error_norm.detach()
        self._vis_rot_error_norm = rot_error_norm.detach()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        import time

        reset_start = time.perf_counter()
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
        # Seed prev velocity with the current (post-reset) velocity so the first
        # step after reset sees zero velocity change instead of a spurious spike.
        self.prev_fingertip_midpoint_linvel[env_ids] = self.fingertip_midpoint_linvel[env_ids]
        self.prev_fingertip_midpoint_angvel[env_ids] = self.fingertip_midpoint_angvel[env_ids]
        self._randomize_dynamics(env_ids)
        self._randomize_controller(env_ids)
        sample_start = time.perf_counter()
        sample_stats = self._sample_reachable_start_and_trajectory(env_ids)
        sample_time = time.perf_counter() - sample_start
        if self.enable_force:
            self._sample_force_trajectory(env_ids)
        self._update_command()

        # Seed the proprio history for the reset envs with their current frame so
        # the stacked observation does not mix in stale frames from the prior
        # episode. The subsequent `_get_observations` rolls in the same frame again.
        if self.obs_history_length > 1:
            proprio, _, _, _ = self._compute_obs_parts()
            self.proprio_history[env_ids] = proprio[env_ids].unsqueeze(1)

        # Wall-clock cost of the reset, dominated by the cuRobo reachability search.
        reset_time = time.perf_counter() - reset_start
        self.extras["log"]["Timing/reset_time_s"] = torch.tensor(reset_time, device=self.device)
        self.extras["log"]["Timing/reachability_sample_time_s"] = torch.tensor(
            sample_time, device=self.device
        )
        avg_jump_str = "[" + ", ".join(f"{v:.3f}" for v in sample_stats["avg_joint_jump"]) + "]"
        if sample_stats["filled"] < self.num_envs:
            print(
                f"[FrankaRobustTrack] reset {len(env_ids)} envs in {reset_time:.3f}s "
                f"(reachability sample {sample_time:.3f}s) | "
                f"candidates sampled {sample_stats['sampled']}, good {sample_stats['good']} "
                f"(unreachable {sample_stats['unreachable']}, discontinuous {sample_stats['discontinuous']}, "
                f"singular {sample_stats['singular']}), "
                f"envs filled {sample_stats['filled']}/{sample_stats['total']} "
                f"in {sample_stats['attempts']} attempt(s) | avg joint jump {avg_jump_str}"
            )

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
        """Cache the constant selected-tool<->hand and world->base transforms once.

        cuRobo IK targets `panda_hand` in the `panda_link0` base frame, while the
        env's reference poses are defined at `tool_frame`. Both directions of the
        tool/hand offset and the base pose are fixed, so they are captured a single
        time after the first reset-pose step. tool->hand converts IK targets; the
        inverse offset shifts cuRobo's hand Jacobian to the controller's tool origin.
        """
        if self._kin_frames_cached:
            return
        tool_pos_w = self.fingertip_midpoint_pos + self.scene.env_origins
        tool_quat_w = self.fingertip_midpoint_quat
        hand_pos_w = self._robot.data.body_pos_w[:, self.hand_body_idx]
        hand_quat_w = self._robot.data.body_quat_w[:, self.hand_body_idx]
        self.hand_in_tool_pos[:], self.hand_in_tool_quat[:] = math_utils.subtract_frame_transforms(
            tool_pos_w, tool_quat_w, hand_pos_w, hand_quat_w
        )
        self.tool_in_hand_pos[:], self.tool_in_hand_quat[:] = math_utils.subtract_frame_transforms(
            hand_pos_w, hand_quat_w, tool_pos_w, tool_quat_w
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
        if not self._kin_frames_cached:
            raise RuntimeError("Tool/hand transforms must be cached before launching the cuRobo IK worker.")

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
        # The worker shares the GPU with the training/sim process, whose footprint
        # grows over training. expandable_segments lets the worker's allocator
        # return freed memory and fragment less, reducing collisions that OOM'd
        # mid-solve; the worker also adaptively shrinks its batch on OOM.
        alloc_conf = worker_env.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in alloc_conf:
            worker_env["PYTORCH_CUDA_ALLOC_CONF"] = (
                f"{alloc_conf},expandable_segments:True".lstrip(",")
            )
        worker_cfg = {
            "robot_cfg": cfg.ik_robot_cfg,
            "num_seeds": cfg.ik_num_seeds,
            "max_batch_size": self.num_envs * max(cfg.reach_check_waypoints, cfg.reach_oversample),
            "ik_batch_size": cfg.ik_batch_size,
            "position_tolerance": cfg.reach_pos_tol,
            "orientation_tolerance": cfg.reach_rot_tol,
            "use_cuda_graph": cfg.ik_use_cuda_graph,
            "device": "cuda:0",
            # When present, the worker inserts this fixed child into cuRobo's
            # kinematic tree and makes it the actual IK optimization frame.
            "extra_tool_frame": (
                {
                    "name": self.tool_frame_name,
                    "parent_link_name": self.tool_body_name,
                    "fixed_transform": (
                        self.tool_pos_in_body[0].detach().cpu().tolist()
                        + self.tool_quat_in_body[0].detach().cpu().tolist()
                    ),
                }
                if self.curobo_uses_selected_tool_frame
                else None
            ),
            # For legacy body frames cuRobo still solves at panda_hand, so shift
            # its metric Jacobian to the selected tool origin. A directly modeled
            # extra tool already has the correct Jacobian and needs zero shift.
            "metric_tool_offset_pos": (
                [0.0, 0.0, 0.0]
                if self.curobo_uses_selected_tool_frame
                else self.tool_in_hand_pos[0].detach().cpu().tolist()
            ),
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
        tool_pos: torch.Tensor,
        tool_quat: torch.Tensor,
        env_ids: torch.Tensor,
        seed_arm_q: torch.Tensor | None = None,
    ):
        """Solve cuRobo IK (in the worker) for selected-tool targets.

        `tool_pos`/`tool_quat` are (n, W, ...) env-local targets for the envs in
        `env_ids`. A directly modeled virtual tool is sent to cuRobo unchanged in
        the base frame; legacy body-frame targets are converted tool->hand first.
        The targets are sent to the worker as a flat batch of n*W problems, and
        the returned per-target success and arm joints are reshaped back.
        `seed_arm_q`, if given, is a (n, W, 7) tensor of initial joint seeds
        (panda_joint1..7 order)
        passed to the solver so the returned solution stays near it. Returns
        (success (n, W) bool, arm_q (n, W, 7), manip (n, W), cond (n, W)) where
        `manip`/`cond` are the manipulability and Jacobian condition number of each
        solved config (from the geometric Jacobian at the selected tool), used to
        filter near-singular samples.
        """
        self._ensure_ik_worker()
        n_envs, num_wp = tool_pos.shape[0], tool_pos.shape[1]

        env_origins = self.scene.env_origins[env_ids].unsqueeze(1)  # (n, 1, 3)
        base_pos = self.base_pos_w[env_ids].unsqueeze(1).expand(-1, num_wp, -1)
        base_quat = self.base_quat_w[env_ids].unsqueeze(1).expand(-1, num_wp, -1)
        off_pos = self.hand_in_tool_pos[env_ids].unsqueeze(1).expand(-1, num_wp, -1)
        off_quat = self.hand_in_tool_quat[env_ids].unsqueeze(1).expand(-1, num_wp, -1)

        # Tool target: env-local -> world (env frame is a pure translation).
        tool_pos_w = (tool_pos + env_origins).reshape(-1, 3)
        tool_quat_w = tool_quat.reshape(-1, 4)
        if self.curobo_uses_selected_tool_frame:
            goal_pos_w, goal_quat_w = tool_pos_w, tool_quat_w
        else:
            # Selected tool -> panda_hand for legacy cuRobo robot configs.
            goal_pos_w, goal_quat_w = math_utils.combine_frame_transforms(
                tool_pos_w, tool_quat_w, off_pos.reshape(-1, 3), off_quat.reshape(-1, 4)
            )
        goal_pos_b, goal_quat_b = math_utils.subtract_frame_transforms(
            base_pos.reshape(-1, 3), base_quat.reshape(-1, 4), goal_pos_w, goal_quat_w
        )

        seed_np = None if seed_arm_q is None else seed_arm_q.reshape(-1, 7).detach().cpu().numpy()
        self._ik_worker_mod.send_msg(
            self._ik_proc.stdin,
            (goal_pos_b.detach().cpu().numpy(), goal_quat_b.detach().cpu().numpy(), seed_np),
        )
        response = self._ik_worker_mod.recv_msg(self._ik_proc.stdout)
        if response is None:
            raise RuntimeError(
                f"cuRobo IK worker exited unexpectedly (return code {self._ik_proc.poll()}); see its stderr above."
            )
        success_np, arm_q_np, manip_np, cond_np = response

        success = torch.as_tensor(success_np, device=self.device).view(n_envs, num_wp)
        arm_q = torch.as_tensor(arm_q_np, device=self.device, dtype=torch.float32).view(n_envs, num_wp, 7)
        manip = torch.as_tensor(manip_np, device=self.device, dtype=torch.float32).view(n_envs, num_wp)
        cond = torch.as_tensor(cond_np, device=self.device, dtype=torch.float32).view(n_envs, num_wp)
        return success, arm_q, manip, cond

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
        # Singularity filter (getattr keeps old restored env.pkl configs working:
        # absent -> disabled). A waypoint config is rejected when its geometric-
        # Jacobian manipulability is too low or its condition number too high.
        sing_check = bool(getattr(cfg, "singularity_check", False))
        min_manip = float(getattr(cfg, "min_manipulability", 0.0))
        max_cond = float(getattr(cfg, "max_jac_cond", float("inf")))
        # cuRobo arm-joint solution for each env's start pose; filled during the check.
        start_arm_q = self.joint_pos[:, 0:7].clone()
        waypoint_times = torch.linspace(0.0, self.max_episode_length_s, num_wp, device=self.device)

        bad_envs = env_ids.clone()
        attempt = 0
        total_sampled = 0
        total_good = 0
        total_reach_fail = 0
        total_cont_fail = 0
        total_sing_fail = 0
        # Per-joint |Δq| accumulated over waypoint transitions where both endpoints
        # were reachable, to report the average continuity of the sampled chains.
        joint_jump_sum = torch.zeros(7, device=self.device)
        joint_jump_count = 0
        # Accepted (reachable + continuous) candidates accumulated across attempts.
        # Any envs still pending after the last attempt are backfilled by resampling
        # (with replacement) from these known-good trajectories rather than arbitrary
        # ones; per-env domain randomization still differentiates repeated trajectories.
        accepted_store = {key: [] for key in self._traj_param_buffers()}
        accepted_start_q = []
        # Optionally seed the first waypoint's IK with reset_joints so the start
        # config lands on that arm branch (later waypoints seed from the previous
        # solution and follow it); mirrors factory/forge's teleport-to-reset_joints.
        reset_seed = None
        if getattr(cfg, "seed_ik_with_reset_joints", False):
            reset_seed = torch.tensor(self.cfg.ctrl.reset_joints, device=self.device)
        while True:
            n_bad = len(bad_envs)
            total_sampled += n_bad * k
            # Draw a pool of n_bad*k candidate trajectories. `cand_env_ids` only
            # feeds the IK frame conversion, which is identical across envs, so the
            # candidates are not actually bound to specific envs.
            params = self._sample_candidate_params(bad_envs, k)
            cand_env_ids = bad_envs.repeat_interleave(k)

            # Walk the waypoints in order, seeding each solve with the previous
            # solution. Track reachability and continuity separately so we can report
            # which one is rejecting candidates.
            reach_ok = torch.ones(n_bad * k, dtype=torch.bool, device=self.device)
            cont_ok = torch.ones(n_bad * k, dtype=torch.bool, device=self.device)
            sing_ok = torch.ones(n_bad * k, dtype=torch.bool, device=self.device)
            cand_start_q = torch.zeros((n_bad * k, 7), device=self.device)
            prev_q = None
            prev_success = None
            for i in range(num_wp):
                elapsed_time = torch.full((n_bad * k, 1), waypoint_times[i].item(), device=self.device)
                pos_i, quat_i = self._eval_pose_from_params(params, elapsed_time)
                if prev_q is not None:
                    seed = prev_q.unsqueeze(1)
                elif reset_seed is not None:
                    # First waypoint: seed from reset_joints (does not set prev_q, so
                    # the i==0 continuity check stays skipped as before).
                    seed = reset_seed.unsqueeze(0).expand(n_bad * k, 7).unsqueeze(1)
                else:
                    seed = None
                success, arm_q, manip, cond = self._solve_curobo_ik(
                    pos_i.unsqueeze(1), quat_i.unsqueeze(1), cand_env_ids, seed_arm_q=seed
                )
                success = success[:, 0]
                arm_q = arm_q[:, 0]
                reach_ok &= success
                if sing_check:
                    # Only gate reachable waypoints (unreachable ones already reject
                    # the candidate and carry meaningless IK solutions).
                    wp_sing_ok = (manip[:, 0] >= min_manip) & (cond[:, 0] <= max_cond)
                    sing_ok &= wp_sing_ok | ~success
                if prev_q is not None:
                    joint_diff = (arm_q - prev_q).abs()  # (n*k, 7)
                    cont_ok &= joint_diff.amax(dim=-1) <= cfg.reach_joint_diff_threshold
                    # Only count transitions where both endpoints were reachable.
                    both_ok = prev_success & success
                    joint_jump_sum += joint_diff[both_ok].sum(dim=0)
                    joint_jump_count += int(both_ok.sum().item())
                if i == 0:
                    cand_start_q = arm_q.clone()
                prev_q = arm_q
                prev_success = success

            accept = reach_ok & cont_ok & sing_ok
            total_good += int(accept.sum().item())
            total_reach_fail += int((~reach_ok).sum().item())
            total_cont_fail += int((reach_ok & ~cont_ok).sum().item())
            total_sing_fail += int((reach_ok & cont_ok & ~sing_ok).sum().item())

            # The whole pool of n_bad*k candidates is fungible: the start pose and
            # trajectory are env-independent and the IK is solved in the (shared)
            # base frame, so any accepted candidate is valid for any env. Randomly
            # draw accepted candidates from the pool and assign them to the pending
            # envs rather than tying each env to its own k candidates.
            accepted_idx = accept.nonzero(as_tuple=False).squeeze(-1)
            if accepted_idx.numel() > 0:
                for key in self._traj_param_buffers():
                    accepted_store[key].append(params[key][accepted_idx])
                accepted_start_q.append(cand_start_q[accepted_idx])
                perm = accepted_idx[torch.randperm(accepted_idx.numel(), device=self.device)]
                take = perm[: len(bad_envs)]
                fill_envs = bad_envs[: take.numel()]
                self._assign_params_to_envs(fill_envs, params, take)
                start_arm_q[fill_envs] = cand_start_q[take]
                bad_envs = bad_envs[take.numel() :]

            attempt += 1
            if bad_envs.shape[0] == 0 or attempt >= cfg.max_reach_attempts:
                # Backfill: envs that never got their own accepted candidate reuse a
                # randomly drawn accepted (reachable + continuous) trajectory. Repeats
                # are fine since domain randomization still differs per env. If nothing
                # was ever accepted, fall back to an arbitrary candidate from the pool.
                if bad_envs.shape[0] > 0:
                    if len(accepted_start_q) > 0:
                        store = {key: torch.cat(vals, dim=0) for key, vals in accepted_store.items()}
                        store_q = torch.cat(accepted_start_q, dim=0)
                        pick = torch.randint(store_q.shape[0], (bad_envs.shape[0],), device=self.device)
                        self._assign_params_to_envs(bad_envs, store, pick)
                        start_arm_q[bad_envs] = store_q[pick]
                    else:
                        fb = torch.arange(bad_envs.shape[0], device=self.device)
                        self._assign_params_to_envs(bad_envs, params, fb)
                        start_arm_q[bad_envs] = cand_start_q[fb]
                break

        # Discretize the (now-final) analytic trajectory onto the per-step grid.
        self._build_trajectory_buffer(env_ids)

        # Place every reset env at the cuRobo joint solution that reaches its start pose.
        self.joint_pos[env_ids, 0:7] = start_arm_q[env_ids]
        # Anchor the OSC nullspace posture. "start" uses each env's own cuRobo start
        # config (per-env); "reset_joints"/"default_dof_pos" use a fixed posture so the
        # redundancy resolution biases toward a consistent config (factory/forge use a
        # fixed anchor -> "default_dof_pos" matches them exactly).
        posture = getattr(self.cfg.ctrl, "nullspace_posture", "start")
        if posture == "reset_joints":
            self.nominal_start_joint_pos[env_ids] = torch.tensor(self.cfg.ctrl.reset_joints, device=self.device)
        elif posture == "default_dof_pos":
            self.nominal_start_joint_pos[env_ids] = torch.tensor(
                self.cfg.ctrl.default_dof_pos_tensor, device=self.device
            )
        else:
            self.nominal_start_joint_pos[env_ids] = start_arm_q[env_ids]
        if self.joint_pos.shape[1] > 7:
            self.joint_pos[env_ids, 7:] = 0.04
        self.joint_vel[env_ids] = 0.0
        self.ctrl_target_joint_pos[env_ids] = self.joint_pos[env_ids]
        self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(torch.zeros_like(self.joint_pos))
        self.step_sim_no_action()
        num_filled = int(len(env_ids) - bad_envs.shape[0])
        self.extras["log"]["Init/unreachable_envs"] = torch.tensor(float(bad_envs.shape[0]), device=self.device)
        self.extras["log"]["Init/candidates_sampled"] = torch.tensor(float(total_sampled), device=self.device)
        self.extras["log"]["Init/candidates_good"] = torch.tensor(float(total_good), device=self.device)
        self.extras["log"]["Init/candidates_unreachable"] = torch.tensor(float(total_reach_fail), device=self.device)
        self.extras["log"]["Init/candidates_discontinuous"] = torch.tensor(float(total_cont_fail), device=self.device)
        self.extras["log"]["Init/candidates_singular"] = torch.tensor(float(total_sing_fail), device=self.device)
        avg_joint_jump = joint_jump_sum / max(joint_jump_count, 1)
        for j in range(7):
            self.extras["log"][f"Init/avg_joint_jump_{j}"] = avg_joint_jump[j]
        return {
            "sampled": total_sampled,
            "good": total_good,
            "unreachable": total_reach_fail,
            "discontinuous": total_cont_fail,
            "singular": total_sing_fail,
            "filled": num_filled,
            "total": int(len(env_ids)),
            "attempts": attempt,
            "avg_joint_jump": avg_joint_jump.tolist(),
        }

    @staticmethod
    def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Batched shortest-path spherical interpolation of (w, x, y, z) quats.

        `q0`, `q1` are (m, 4) unit quats; `t` is (m,) in [0, 1]. Falls back to a
        normalized lerp when the two quats are nearly parallel (sin theta ~ 0).
        """
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = dot.abs().clamp(max=1.0)
        theta = torch.acos(dot)  # (m, 1)
        sin_theta = torch.sin(theta)
        small = sin_theta < 1.0e-6
        t = t.unsqueeze(-1)
        w0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta.clamp(min=1.0e-6))
        w1 = torch.where(small, t, torch.sin(t * theta) / sin_theta.clamp(min=1.0e-6))
        q = w0 * q0 + w1 * q1
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)

    def _resample_pose_sequence(self, pos: torch.Tensor, quat: torch.Tensor, length: int):
        """Resample a (n, 3)/(n, 4) pose sequence onto `length` evenly-spaced samples."""
        n = pos.shape[0]
        tgt = torch.linspace(0.0, n - 1, length, device=pos.device)
        i0 = tgt.floor().long().clamp(0, n - 1)
        i1 = (i0 + 1).clamp(max=n - 1)
        frac = (tgt - i0.float()).unsqueeze(-1)
        p = pos[i0] * (1.0 - frac) + pos[i1] * frac
        q = self._quat_slerp(quat[i0], quat[i1], frac.squeeze(-1))
        return p, q

    @staticmethod
    def _resample_vec_sequence(vec: torch.Tensor, length: int) -> torch.Tensor:
        """Linearly resample a (n, d) sequence onto `length` evenly-spaced samples."""
        n = vec.shape[0]
        tgt = torch.linspace(0.0, n - 1, length, device=vec.device)
        i0 = tgt.floor().long().clamp(0, n - 1)
        i1 = (i0 + 1).clamp(max=n - 1)
        frac = (tgt - i0.float()).unsqueeze(-1)
        return vec[i0] * (1.0 - frac) + vec[i1] * frac

    def _load_dataset_trajectories(self):
        """Load + resample offline demonstration EE trajectories for `mode='dataset'`.

        Reads the `dataset_pose_key` (N, 7) pose field (selected-tool pos + quat,
        wxyz, env-local frame) from the HDF5 dataset, splits it into episodes via
        `episode_starts`/`episode_ends`, and resamples every episode onto the
        `_traj_len`-step control grid so the runtime command/lookahead can index it
        directly. When force tracking is enabled and `dataset_force_key` is set, the
        matching per-step force (first 3 components) is resampled the same way.
        """
        import h5py

        tcfg = self.cfg.tracking
        path = tcfg.dataset_path
        if not path:
            raise ValueError("tracking.mode='dataset' requires tracking.dataset_path")
        warp_prob = float(getattr(tcfg, "dataset_warp_prob", 0.0))
        warp_pos_max = float(getattr(tcfg, "dataset_warp_pos_max", 0.0))
        warp_rot_max = float(getattr(tcfg, "dataset_warp_rot_max", 0.0))
        warp_strategy = str(getattr(tcfg, "dataset_warp_strategy", "smooth_bump"))
        shape_fraction = float(getattr(tcfg, "dataset_warp_shape_fraction", 0.0))
        cycles_range = getattr(tcfg, "dataset_warp_cycles_range", [0.5, 1.5])
        valid_warp_strategies = {"smooth_bump", "constant_scale_shape"}
        if warp_strategy not in valid_warp_strategies:
            raise ValueError(
                f"tracking.dataset_warp_strategy must be one of {sorted(valid_warp_strategies)}, "
                f"got {warp_strategy!r}"
            )
        if not 0.0 <= warp_prob <= 1.0:
            raise ValueError(f"tracking.dataset_warp_prob must be in [0, 1], got {warp_prob}")
        if warp_pos_max < 0.0 or warp_rot_max < 0.0:
            raise ValueError("tracking dataset warp magnitudes must be non-negative")
        if not 0.0 <= shape_fraction <= 1.0:
            raise ValueError("tracking.dataset_warp_shape_fraction must be in [0, 1]")
        if len(cycles_range) != 2 or cycles_range[0] < 0.0 or cycles_range[1] < cycles_range[0]:
            raise ValueError("tracking.dataset_warp_cycles_range must be [min, max] with 0 <= min <= max")

        length = self._traj_len
        want_force = self.enable_force and bool(tcfg.dataset_force_key)
        with h5py.File(path, "r") as f:
            pose = torch.as_tensor(f[tcfg.dataset_pose_key][:], dtype=torch.float32)
            starts = f["episode_starts"][:]
            ends = f["episode_ends"][:]
            success = f["trial_success"][:] if "trial_success" in f else None
            force = torch.as_tensor(f[tcfg.dataset_force_key][:], dtype=torch.float32) if want_force else None

        pos_list, quat_list, wr_list = [], [], []
        for e in range(len(starts)):
            if tcfg.dataset_only_success and success is not None and float(success[e]) < 0.5:
                continue
            s, en = int(starts[e]), int(ends[e])
            if en - s < 2:
                continue
            seg_pos = pose[s:en, :3].to(self.device)
            seg_quat = pose[s:en, 3:7].to(self.device)
            seg_quat = seg_quat / seg_quat.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
            p, q = self._resample_pose_sequence(seg_pos, seg_quat, length)
            pos_list.append(p)
            quat_list.append(q)
            if force is not None:
                wr_list.append(self._resample_vec_sequence(force[s:en, :3].to(self.device), length))
            if tcfg.dataset_max_trajs and len(pos_list) >= int(tcfg.dataset_max_trajs):
                break

        if not pos_list:
            raise RuntimeError(f"No usable episodes loaded from {path}")

        self._ds_pos = torch.stack(pos_list)  # (T, L, 3)
        self._ds_quat = torch.stack(quat_list)  # (T, L, 4)
        self._ds_wrench = torch.stack(wr_list) if wr_list else None  # (T, L, 3) or None
        self._ds_num = self._ds_pos.shape[0]

        pos_flat = self._ds_pos.reshape(-1, 3)
        box_min = [round(v, 3) for v in pos_flat.min(dim=0).values.tolist()]
        box_max = [round(v, 3) for v in pos_flat.max(dim=0).values.tolist()]
        force_str = ""
        if self._ds_wrench is not None:
            force_str = f" | force |.| max {self._ds_wrench.norm(dim=-1).max().item():.2f} N"
        warp_str = ""
        if warp_prob > 0.0:
            warp_str = (
                f" | warp={warp_strategy} p={warp_prob:.2f}, "
                f"pos<={warp_pos_max * 1.0e3:.1f} mm, rot<={math.degrees(warp_rot_max):.1f} deg"
            )
            if warp_strategy == "constant_scale_shape":
                warp_str += f", shape={shape_fraction:.2f}, cycles={list(cycles_range)}"
        print(
            f"[FrankaRobustTrack] dataset mode: loaded {self._ds_num} trajectories from {path} "
            f"(resampled to L={length}) | pos box min {box_min} max {box_max}{force_str}{warp_str}"
        )

    def _sample_dataset_candidate_params(self, env_ids: torch.Tensor, num_candidates: int) -> dict:
        """Draw dataset trajectories and optional smooth spatial warps."""
        m = len(env_ids) * num_candidates
        tcfg = self.cfg.tracking
        warp_prob = float(getattr(tcfg, "dataset_warp_prob", 0.0))
        pos_max = float(getattr(tcfg, "dataset_warp_pos_max", 0.0))
        rot_max = float(getattr(tcfg, "dataset_warp_rot_max", 0.0))
        cycles_lo, cycles_hi = getattr(tcfg, "dataset_warp_cycles_range", [0.5, 1.5])
        warp_active = (torch.rand((m, 1), device=self.device) < warp_prob).float()

        # Sample the cone axes plus one perpendicular basis vector. The second
        # perpendicular vector is recovered with a cross product when evaluated.
        def sample_axis_and_perp():
            axis = torch.randn((m, 3), device=self.device)
            axis /= axis.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
            perp = torch.randn((m, 3), device=self.device)
            perp -= (perp * axis).sum(dim=-1, keepdim=True) * axis
            perp /= perp.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
            return axis, perp

        pos_dir, pos_u = sample_axis_and_perp()
        rot_dir, rot_u = sample_axis_and_perp()
        warp_pos = pos_dir * (torch.rand((m, 1), device=self.device) * pos_max) * warp_active
        warp_rot = rot_dir * (torch.rand((m, 1), device=self.device) * rot_max) * warp_active
        return {
            "ds_idx": torch.randint(self._ds_num, (m,), device=self.device),
            "ds_warp_pos": warp_pos,
            "ds_warp_rot": warp_rot,
            "ds_warp_pos_u": pos_u,
            "ds_warp_rot_u": rot_u,
            "ds_warp_phase": 2.0 * math.pi * torch.rand((m, 1), device=self.device),
            "ds_warp_cycles": cycles_lo + (cycles_hi - cycles_lo) * torch.rand((m, 1), device=self.device),
        }

    def _apply_dataset_warp(
        self, pos: torch.Tensor, quat: torch.Tensor, params: dict, traj_phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the configured smooth dataset trajectory warp.

        ``smooth_bump`` applies the original endpoint-preserving quartic envelope.
        ``constant_scale_shape`` rotates a fixed-norm offset along a cone, changing
        path shape without fading or compounding the perturbation.
        """
        warp_strategy = str(getattr(self.cfg.tracking, "dataset_warp_strategy", "smooth_bump"))
        if warp_strategy == "smooth_bump":
            phase = traj_phase.clamp(0.0, 1.0).unsqueeze(-1)
            blend = 16.0 * phase.square() * (1.0 - phase).square()
            pos_offset = blend * params["ds_warp_pos"]
            rot_vec = blend * params["ds_warp_rot"]
        else:
            pos_offset, rot_vec = self._constant_scale_shape_offsets(params, traj_phase)

        pos = pos + pos_offset
        angle = rot_vec.norm(dim=-1)
        axis = rot_vec / angle.unsqueeze(-1).clamp(min=1.0e-6)
        delta_quat = torch_utils.quat_from_angle_axis(angle, axis)
        quat = torch_utils.quat_mul(delta_quat, quat)
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
        return pos, quat

    def _constant_scale_shape_offsets(
        self, params: dict, traj_phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build fixed-norm translation/rotation offsets with varying direction."""
        shape_fraction = float(getattr(self.cfg.tracking, "dataset_warp_shape_fraction", 0.0))
        axial_fraction = math.sqrt(max(0.0, 1.0 - shape_fraction**2))
        theta = (
            2.0 * math.pi * params["ds_warp_cycles"] * traj_phase.unsqueeze(-1)
            + params["ds_warp_phase"]
        )

        def cone_offset(base_vec: torch.Tensor, basis_u: torch.Tensor) -> torch.Tensor:
            magnitude = base_vec.norm(dim=-1, keepdim=True)
            axis = base_vec / magnitude.clamp(min=1.0e-6)
            basis_v = torch.linalg.cross(axis, basis_u, dim=-1)
            direction = axial_fraction * axis + shape_fraction * (
                torch.cos(theta) * basis_u + torch.sin(theta) * basis_v
            )
            return magnitude * direction

        pos_offset = cone_offset(params["ds_warp_pos"], params["ds_warp_pos_u"])
        rot_vec = cone_offset(params["ds_warp_rot"], params["ds_warp_rot_u"])
        return pos_offset, rot_vec

    def _eval_dataset_pose(self, params: dict, elapsed_time: torch.Tensor):
        """Interpolate the selected demonstration trajectories at `elapsed_time`.

        `params["ds_idx"]` is (m,) trajectory indices; `elapsed_time` is (m, 1).
        The stored sequences already live on the per-step grid, so time maps to a
        fractional step index and is linearly (pos) / slerp (quat) interpolated.
        """
        idx = params["ds_idx"]
        length = self._ds_pos.shape[1]
        fidx = (elapsed_time.squeeze(-1) / self.step_dt).clamp(0.0, length - 1)
        i0 = fidx.floor().long()
        i1 = (i0 + 1).clamp(max=length - 1)
        frac = (fidx - i0.float()).unsqueeze(-1)
        pos = self._ds_pos[idx, i0] * (1.0 - frac) + self._ds_pos[idx, i1] * frac
        quat = self._quat_slerp(self._ds_quat[idx, i0], self._ds_quat[idx, i1], frac.squeeze(-1))
        traj_phase = fidx / max(length - 1, 1)
        return self._apply_dataset_warp(pos, quat, params, traj_phase)

    def _sample_dataset_force(self, env_ids: torch.Tensor):
        """Fill the per-step target wrench for `env_ids` from their dataset trajectory."""
        wr = self._ds_wrench[self.traj_ds_idx[env_ids]]  # (n, L, 3)
        self.traj_wrench_buf[env_ids] = wr
        threshold = float(self.cfg.tracking.dataset_contact_force_threshold)
        self.traj_contact_buf[env_ids] = wr.norm(dim=-1) > threshold

    def _traj_param_buffers(self) -> dict:
        """Active param-name -> per-env buffer-name map for the current tracking mode."""
        if self.cfg.tracking.mode == "dataset":
            return self._DS_TRAJ_PARAM_BUFFERS
        return self._TRAJ_PARAM_BUFFERS

    def _sample_candidate_params(self, env_ids: torch.Tensor, num_candidates: int) -> dict:
        """Sample `num_candidates` start-pose + trajectory param sets per env.

        Returns a dict of tensors, each with leading dim ``len(env_ids) *
        num_candidates`` flattened env-major (candidate c = b * k + j belongs to
        env ``env_ids[b]``). Env-specific centers (`nominal_start_pos/quat`) are
        broadcast across the `num_candidates` candidates of each env. Trajectory
        params for the inactive mode are filled with harmless defaults so
        `_assign_params_to_envs` can copy every key uniformly.
        """
        if self.cfg.tracking.mode == "dataset":
            return self._sample_dataset_candidate_params(env_ids, num_candidates)
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

        if tcfg.mode in ("line", "line_fixed"):
            if tcfg.mode == "line_fixed":
                # Fixed direction: same unit vector for every env / candidate.
                direction = torch.tensor(tcfg.line_dir, device=device, dtype=torch.float32)
                direction = direction / torch.clamp(torch.linalg.norm(direction), min=1.0e-6)
                direction = direction.unsqueeze(0).expand(m, 3).contiguous()
            else:
                direction = torch.randn((m, 3), device=device)
                direction = direction / torch.clamp(torch.linalg.norm(direction, dim=-1, keepdim=True), min=1.0e-6)
            params["dir"] = direction
            length_lo, length_hi = tcfg.line_length_range
            length = length_lo + (length_hi - length_lo) * torch.rand((m, 1), device=device)
            params["length"] = length
            # Speed is not sampled independently: the line is traversed exactly
            # once over the full episode, so speed = length / episode_length. This
            # keeps the reference moving for the whole episode (no early saturation)
            # regardless of the sampled length.
            params["speed"] = length / self.max_episode_length_s
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

    # Dataset poses are gathered from the loaded tensors on demand, then bent by
    # the per-env warp vectors (zero vectors mean no augmentation).
    _DS_TRAJ_PARAM_BUFFERS = {
        "ds_idx": "traj_ds_idx",
        "ds_warp_pos": "traj_ds_warp_pos",
        "ds_warp_rot": "traj_ds_warp_rot",
        "ds_warp_pos_u": "traj_ds_warp_pos_u",
        "ds_warp_rot_u": "traj_ds_warp_rot_u",
        "ds_warp_phase": "traj_ds_warp_phase",
        "ds_warp_cycles": "traj_ds_warp_cycles",
    }

    def _assign_params_to_envs(self, env_ids: torch.Tensor, params: dict, cand_idx: torch.Tensor):
        """Copy the selected candidate params (`cand_idx`) into the per-env traj buffers."""
        for key, buf_name in self._traj_param_buffers().items():
            getattr(self, buf_name)[env_ids] = params[key][cand_idx]

    def _eval_pose_from_params(self, params: dict, elapsed_time: torch.Tensor):
        """Evaluate the reference pose at `elapsed_time` for the given trajectory params.

        `params` maps the trajectory keys to tensors with a shared leading dim `M`;
        `elapsed_time` is (M, 1). The trajectory is anchored at `start_pos`/
        `start_quat`, so all modes pass through it at ``elapsed_time == 0``.
        """
        mode = self.cfg.tracking.mode
        if mode == "dataset":
            return self._eval_dataset_pose(params, elapsed_time)
        start_pos = params["start_pos"]
        start_quat = params["start_quat"]
        if mode in ("line", "line_fixed"):
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
        params = {key: getattr(self, buf_name) for key, buf_name in self._traj_param_buffers().items()}
        return self._eval_pose_from_params(params, elapsed_time)

    def _build_trajectory_buffer(self, env_ids: torch.Tensor):
        """Sample the analytic trajectory onto the fixed per-step grid for `env_ids`.

        The grid times are `arange(self._traj_len) * step_dt`, i.e. one waypoint
        per control step, extended past the episode so the furthest lookahead is
        always in-range. After this call the runtime command and lookahead read
        straight out of `traj_pos_buf`/`traj_quat_buf` by integer step index.
        """
        if self.cfg.tracking.mode == "dataset":
            # Sequences are already resampled onto the per-step grid. Apply the
            # accepted candidate's warp in one batch so runtime lookup stays cheap.
            params = {
                key: getattr(self, buf_name)[env_ids] for key, buf_name in self._DS_TRAJ_PARAM_BUFFERS.items()
            }
            pos = self._ds_pos[params["ds_idx"]]
            quat = self._ds_quat[params["ds_idx"]]
            traj_phase = torch.linspace(0.0, 1.0, self._traj_len, device=self.device)
            traj_phase = traj_phase.unsqueeze(0).expand(len(env_ids), -1)
            flat_params = {
                "ds_warp_pos": params["ds_warp_pos"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 3),
                "ds_warp_rot": params["ds_warp_rot"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 3),
                "ds_warp_pos_u": params["ds_warp_pos_u"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 3),
                "ds_warp_rot_u": params["ds_warp_rot_u"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 3),
                "ds_warp_phase": params["ds_warp_phase"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 1),
                "ds_warp_cycles": params["ds_warp_cycles"]
                .unsqueeze(1)
                .expand(-1, self._traj_len, -1)
                .reshape(-1, 1),
            }
            pos, quat = self._apply_dataset_warp(
                pos.reshape(-1, 3), quat.reshape(-1, 4), flat_params, traj_phase.reshape(-1)
            )
            self.traj_pos_buf[env_ids] = pos.view(len(env_ids), self._traj_len, 3)
            self.traj_quat_buf[env_ids] = quat.view(len(env_ids), self._traj_len, 4)
            return
        for i in range(self._traj_len):
            elapsed_time = torch.full((self.num_envs, 1), i * self.step_dt, device=self.device)
            pos_i, quat_i = self._command_pose_at(elapsed_time)
            self.traj_pos_buf[env_ids, i] = pos_i[env_ids]
            self.traj_quat_buf[env_ids, i] = quat_i[env_ids]

    def _traj_pose_at_index(self, idx: torch.Tensor):
        """Gather the discretized reference pose at per-env step index `idx` (num_envs,)."""
        idx = idx.clamp(0, self._traj_len - 1)
        return self.traj_pos_buf[self._env_arange, idx], self.traj_quat_buf[self._env_arange, idx]

    def _traj_wrench_at_index(self, idx: torch.Tensor):
        """Gather the discretized target wrench (world-frame 3D force) at step index `idx`."""
        idx = idx.clamp(0, self._traj_len - 1)
        return self.traj_wrench_buf[self._env_arange, idx]

    def _smooth_osc(self, tau: torch.Tensor, num_components: int = 3) -> torch.Tensor:
        """A smooth per-step signal in [-1, 1] built from a few random sinusoids.

        `tau` is (n, L) time-since-onset in seconds. Frequencies are drawn in
        `tracking.force_osc_freq_range`, phases uniformly, and the components are
        combined with weights that sum to 1 so the weighted sum of unit sines stays
        within [-1, 1]. The result varies continuously in time (no jumps).
        """
        n = tau.shape[0]
        device = tau.device
        freq_lo, freq_hi = self.cfg.tracking.force_osc_freq_range
        freq = freq_lo + (freq_hi - freq_lo) * torch.rand((n, num_components), device=device)
        phase = 2.0 * math.pi * torch.rand((n, num_components), device=device)
        weights = torch.rand((n, num_components), device=device)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
        ang = 2.0 * math.pi * freq.unsqueeze(-1) * tau.unsqueeze(1) + phase.unsqueeze(-1)  # (n, K, L)
        return (weights.unsqueeze(-1) * torch.sin(ang)).sum(dim=1)  # (n, L)

    def _sample_force_trajectory(self, env_ids: torch.Tensor):
        """Sample a per-step target-wrench sequence for `env_ids` into `traj_wrench_buf`.

        Models a handful of contact events: the force is zero for most of the
        episode, punctuated by `force_num_bands_range` bands. The episode is split
        into that many equal segments (per env), and each segment holds one band
        placed at a random onset with a sampled duration, so bands never overlap and
        stay spread out. Each band switches on suddenly (step, no ramp) and, while
        active, varies smoothly: its magnitude oscillates continuously in
        ``[0.5, 1.0] * peak`` (peak sampled per band in `force_mag_range`) and its
        direction drifts smoothly around a random base direction by
        ``force_dir_drift``. With probability ``1 - force_prob`` the env sees no
        contact at all. Independent of the pose-reachability sampling since a wrist
        force does not change IK reachability.
        """
        tcfg = self.cfg.tracking
        if tcfg.mode == "dataset" and tcfg.dataset_force_key and self._ds_wrench is not None:
            # Dataset mode: replay the demonstration's own per-step contact force.
            self._sample_dataset_force(env_ids)
            return
        device = self.device
        n = len(env_ids)

        band_lo, band_hi = tcfg.force_num_bands_range
        max_bands = int(band_hi)
        num_bands = torch.randint(int(band_lo), max_bands + 1, (n, 1), device=device)  # (n, 1)
        seg_len = float(self.max_episode_length) / num_bands.float()  # (n, 1) steps per segment

        steps = torch.arange(self._traj_len, device=device).unsqueeze(0)  # (1, L)
        wrench = torch.zeros((n, self._traj_len, 3), device=device)
        contact = torch.zeros((n, self._traj_len), dtype=torch.bool, device=device)

        dur_lo, dur_hi = tcfg.force_duration_range
        for b in range(max_bands):
            use_b = b < num_bands  # (n, 1) bool: this env actually has a b-th band

            # Duration (steps), capped so the band fits within its own segment.
            duration_s = dur_lo + (dur_hi - dur_lo) * torch.rand((n, 1), device=device)
            duration_steps = (duration_s / self.step_dt).round().clamp(min=1.0)
            duration_steps = torch.minimum(duration_steps, seg_len.clamp(min=1.0))  # (n, 1)

            # Random onset inside segment b: [b * seg_len, (b + 1) * seg_len - duration].
            seg_start = b * seg_len
            onset = seg_start + torch.rand((n, 1), device=device) * (seg_len - duration_steps).clamp(min=0.0)

            active = (steps >= onset) & (steps < onset + duration_steps) & use_b  # (n, L)

            # Smooth time-varying profile within this band (time measured from onset).
            tau = (steps - onset).float() * self.step_dt  # (n, L)
            mag_lo, mag_hi = tcfg.force_mag_range
            peak = (mag_lo + (mag_hi - mag_lo) * torch.rand((n, 1), device=device)).clamp(max=20.0)
            mag = peak * (0.75 + 0.25 * self._smooth_osc(tau))  # (n, L)

            base_dir = torch.randn((n, 3), device=device)
            base_dir = base_dir / torch.clamp(torch.linalg.norm(base_dir, dim=-1, keepdim=True), min=1.0e-6)
            perp1, perp2 = self._orthonormal_basis(base_dir)
            s1 = self._smooth_osc(tau).unsqueeze(-1)
            s2 = self._smooth_osc(tau).unsqueeze(-1)
            dir_raw = base_dir.unsqueeze(1) + tcfg.force_dir_drift * (
                perp1.unsqueeze(1) * s1 + perp2.unsqueeze(1) * s2
            )  # (n, L, 3)
            direction = dir_raw / torch.clamp(torch.linalg.norm(dir_raw, dim=-1, keepdim=True), min=1.0e-6)

            active_f = active.float()
            # Segments are disjoint, so bands never overlap: accumulate is safe.
            wrench = wrench + direction * (mag * active_f).unsqueeze(-1)
            contact = contact | active

        if tcfg.force_prob < 1.0:
            keep = torch.rand((n, 1), device=device) < tcfg.force_prob  # (n, 1)
            wrench = wrench * keep.unsqueeze(-1)
            contact = contact & keep

        self.traj_wrench_buf[env_ids] = wrench
        self.traj_contact_buf[env_ids] = contact

    @staticmethod
    def _orthonormal_basis(axis: torch.Tensor):
        """Return two unit vectors spanning the plane orthogonal to `axis` (n, 3)."""
        helper = torch.zeros_like(axis)
        helper[:, 0] = 1.0
        # Swap the helper axis where `axis` is nearly parallel to x to avoid a zero cross product.
        near_x = axis[:, 0].abs() > 0.9
        helper[near_x, 0] = 0.0
        helper[near_x, 1] = 1.0
        perp1 = torch.linalg.cross(axis, helper)
        perp1 = perp1 / torch.clamp(torch.linalg.norm(perp1, dim=-1, keepdim=True), min=1.0e-6)
        perp2 = torch.linalg.cross(axis, perp1)
        return perp1, perp2

    def _future_wrench(self) -> torch.Tensor:
        """Current + `num_future_steps` lookahead target wrenches, normalized by peak.

        Returns a (num_envs, 3 * num_future_steps) tensor of world-frame forces
        scaled by 1 / force_mag_range[1] so the policy sees a roughly unit-range
        signal (index 0 = current target, one per trajectory step).
        """
        num_steps = self.cfg.tracking.num_future_steps
        base_idx = self.episode_length_buf
        scale = 1.0 / self._force_mag_max
        return torch.cat(
            [self._traj_wrench_at_index(base_idx + i) * scale for i in range(num_steps)], dim=-1
        )

    def _update_command(self):
        """Set the current reference pose (and target wrench) to the current step's waypoint."""
        self.command_pos[:], self.command_quat[:] = self._traj_pose_at_index(self.episode_length_buf)
        if self.enable_force:
            idx = self.episode_length_buf.clamp(0, self._traj_len - 1)
            self.target_wrench[:] = self.traj_wrench_buf[self._env_arange, idx]
            self.current_contact[:] = self.traj_contact_buf[self._env_arange, idx]

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
        scale_range = self.cfg.ctrl.task_prop_gains_randomization_scale_range
        mode = self.cfg.ctrl.task_prop_gains_randomization_mode
        if mode not in {"scalar", "per_axis"}:
            raise ValueError(
                "ctrl.task_prop_gains_randomization_mode must be 'scalar' or "
                f"'per_axis', got {mode!r}"
            )
        if mode == "scalar":
            if scale_range is None and not torch.allclose(noise, noise[:, :1].expand_as(noise)):
                raise ValueError(
                    "Scalar controller-gain randomization requires all entries in "
                    "ctrl.task_prop_gains_noise_level to be equal."
                )
        if scale_range is not None:
            if torch.any(noise > 0):
                raise ValueError(
                    "Set either ctrl.task_prop_gains_randomization_scale_range or "
                    "ctrl.task_prop_gains_noise_level, not both."
                )
            if len(scale_range) != 2:
                raise ValueError("ctrl.task_prop_gains_randomization_scale_range must be [low, high].")
            low, high = (float(value) for value in scale_range)
            if not 0.0 < low <= 1.0 <= high:
                raise ValueError(
                    "ctrl.task_prop_gains_randomization_scale_range must satisfy 0 < low <= 1 <= high."
                )
            sample_shape = (env_ids.numel(), 1) if mode == "scalar" else gains.shape
            multiplier = low + (high - low) * torch.rand(sample_shape, device=self.device)
            gains = gains * multiplier
        elif torch.any(noise > 0):
            if mode == "scalar":
                multiplier = 1.0 + (
                    2.0 * torch.rand((env_ids.numel(), 1), device=self.device) - 1.0
                ) * noise[:, :1]
                gains = gains * multiplier
            else:
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
        # disabled, see `_apply_external_wrenches`). The controller stays blind to all of this because it
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

        if rand_cfg.enable_joint_armature:
            lower, upper = rand_cfg.joint_armature_range
            joint_armature = lower + (upper - lower) * torch.rand((len(env_ids), 7), device=self.device)
        else:
            joint_armature = torch.zeros((len(env_ids), 7), device=self.device)
        self.joint_armature[env_ids] = joint_armature
        self._robot.write_joint_armature_to_sim(
            joint_armature,
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
        # This callback is driven by a timeline event and can fire during
        # shutdown, after the scene has already been torn down.
        if not hasattr(self, "scene") or not hasattr(self, "command_pose_visualizer"):
            return
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

    def _accumulate_contact_split_metrics(
        self, pos_error_norm: torch.Tensor, rot_error_norm: torch.Tensor, successes: torch.Tensor
    ):
        """Accumulate tracking error/success split by free-space vs contact steps.

        `self.current_contact` (num_envs,) flags which envs have a contact band
        active this step. Per-env errors/successes are summed into free and contact
        accumulators with their own counts, so the logged scalar is the mean over
        all (env, step) samples of that category in the running window.
        """
        contact = self.current_contact
        free = ~contact
        succ_f = successes.float()

        self._track_free_pos_sum += pos_error_norm[free].sum()
        self._track_free_rot_sum += rot_error_norm[free].sum()
        self._track_free_succ_sum += succ_f[free].sum()
        self._track_free_count += free.sum()
        self._track_contact_pos_sum += pos_error_norm[contact].sum()
        self._track_contact_rot_sum += rot_error_norm[contact].sum()
        self._track_contact_succ_sum += succ_f[contact].sum()
        self._track_contact_count += contact.sum()

        free_n = self._track_free_count.clamp(min=1.0)
        contact_n = self._track_contact_count.clamp(min=1.0)
        self.extras["tracking_pos_error_free"] = self._track_free_pos_sum / free_n
        self.extras["tracking_rot_error_free"] = self._track_free_rot_sum / free_n
        self.extras["success_free"] = self._track_free_succ_sum / free_n
        self.extras["tracking_pos_error_contact"] = self._track_contact_pos_sum / contact_n
        self.extras["tracking_rot_error_contact"] = self._track_contact_rot_sum / contact_n
        self.extras["success_contact"] = self._track_contact_succ_sum / contact_n
        self.extras["contact_fraction"] = contact.float().mean()

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

    def _accumulate_gain_stats(self):
        """Accumulate per-step policy-scheduled PD gains and log stats periodically."""
        if not self.cfg.ctrl.control_gains or self._perstep_log_every <= 0:
            return
        gains = self.task_prop_gains.detach()
        self._gain_sum += gains.sum(dim=0)
        self._gain_sqsum += gains.square().sum(dim=0)
        self._gain_count += gains.shape[0]

        if self.enable_force:
            contact = self.current_contact
            free = ~contact
            self._gain_free_sum += gains[free].sum(dim=0)
            self._gain_free_sqsum += gains[free].square().sum(dim=0)
            self._gain_free_count += free.sum()
            self._gain_contact_sum += gains[contact].sum(dim=0)
            self._gain_contact_sqsum += gains[contact].square().sum(dim=0)
            self._gain_contact_count += contact.sum()

        gain_min = self.gain_min[0]
        gain_max = self.gain_max[0]
        span = (gain_max - gain_min).clamp(min=1.0e-6)
        normalized = (gains - gain_min) / span
        bin_idx = (normalized * self._gain_hist_num_bins).long().clamp(0, self._gain_hist_num_bins - 1)
        for d in range(self._gain_stats_num_dims):
            self._gain_hist[d].index_add_(0, bin_idx[:, d], torch.ones_like(bin_idx[:, d], dtype=torch.float))

        self._gain_step_counter += 1
        if self._gain_step_counter >= self._perstep_log_every:
            self._log_gain_stats()
            self._reset_gain_stats()

    def _reset_gain_stats(self):
        self._gain_sum.zero_()
        self._gain_sqsum.zero_()
        self._gain_count.zero_()
        self._gain_hist.zero_()
        self._gain_step_counter = 0
        for buf in (
            self._gain_free_sum, self._gain_free_sqsum, self._gain_free_count,
            self._gain_contact_sum, self._gain_contact_sqsum, self._gain_contact_count,
        ):
            buf.zero_()

    def _log_gain_stats(self):
        """Log mean/std scalars and a per-dim histogram of the scheduled PD gains to wandb."""
        import wandb

        if wandb.run is None or float(self._gain_count) <= 0:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mean = self._gain_sum / self._gain_count
        std = (self._gain_sqsum / self._gain_count - mean.square()).clamp(min=0.0).sqrt()

        # Axis labels: first 3 dims are translational stiffness, last 3 rotational.
        axis_names = ["x", "y", "z", "rx", "ry", "rz"]
        log_dict = {}
        for d, name in enumerate(axis_names):
            log_dict[f"gains/mean_{name}"] = float(mean[d])
            log_dict[f"gains/std_{name}"] = float(std[d])

        # Free-space vs contact split of the scheduled-gain mean/std.
        if self.enable_force:
            for phase, gsum, gsqsum, gcount in (
                ("free", self._gain_free_sum, self._gain_free_sqsum, self._gain_free_count),
                ("contact", self._gain_contact_sum, self._gain_contact_sqsum, self._gain_contact_count),
            ):
                if float(gcount) <= 0:
                    continue
                phase_mean = gsum / gcount
                phase_std = (gsqsum / gcount - phase_mean.square()).clamp(min=0.0).sqrt()
                for d, name in enumerate(axis_names):
                    log_dict[f"gains/mean_{name}_{phase}"] = float(phase_mean[d])
                    log_dict[f"gains/std_{name}_{phase}"] = float(phase_std[d])

        gain_min = self.gain_min[0].cpu().numpy()
        gain_max = self.gain_max[0].cpu().numpy()
        hist = self._gain_hist.cpu().numpy()

        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        for d, name in enumerate(axis_names):
            edges = torch.linspace(0.0, 1.0, self._gain_hist_num_bins + 1).cpu().numpy()
            centers = 0.5 * (edges[:-1] + edges[1:])
            centers = gain_min[d] + centers * (gain_max[d] - gain_min[d])
            width = (gain_max[d] - gain_min[d]) / self._gain_hist_num_bins
            axes[d].bar(centers, hist[d], width=width, color="C0", alpha=0.8)
            axes[d].axvline(float(mean[d]), color="C3", linestyle="--", label=f"mean={float(mean[d]):.1f}")
            axes[d].set_xlim(gain_min[d], gain_max[d])
            axes[d].set_xlabel(f"prop gain [{name}]")
            axes[d].set_ylabel("count")
            axes[d].set_title(f"gain[{name}] (std={float(std[d]):.1f})")
            axes[d].legend()
            axes[d].grid(True, alpha=0.3)

        fig.tight_layout()
        log_dict["gains/histogram"] = wandb.Image(fig)
        wandb.log(log_dict)
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
        self.extras["log"]["Dynamics/joint_armature"] = self.joint_armature[env_ids].mean()
