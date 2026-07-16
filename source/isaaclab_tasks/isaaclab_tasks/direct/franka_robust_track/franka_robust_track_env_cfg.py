# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from omegaconf import OmegaConf

from isaaclab_tasks.direct.factory.factory_tasks_cfg import ASSET_DIR


def _to_plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


@configclass
class CtrlCfg:
    """Low-level controller configuration."""

    backend: str = "factory_control"  # factory_control, dls_ik
    ema_factor: float = 1.0
    action_rep: str = "delta_ee_pose"
    # Controls when a delta action is converted to an absolute controller target:
    #   "per_physics_step" -> re-anchor it to the current EE pose on every
    #                         decimation substep (legacy behavior)
    #   "per_control_step" -> anchor it once in _pre_physics_step and hold the
    #                         resulting pose target throughout decimation
    # In both modes the policy still outputs the same normalized 6D delta action.
    delta_target_mode: str = "per_physics_step"

    # By default the OSC controller runs on the *nominal* mass matrix, rebuilt from
    # the default inertial parameters, so it stays blind to the payload and link-mass
    # randomization (that mismatch is the point of the robustness task). Set this to
    # True to instead feed the controller PhysX's ground-truth generalized mass matrix
    # (payload merge + link-mass scaling included), i.e. a perfectly-modeled controller.
    use_gt_mass_matrix: bool = False

    # Use the full operational-space control law: premultiply the task-space PD wrench
    # by the task inertia Λ = (J M⁻¹ Jᵀ)⁻¹ so the closed-loop task dynamics are unit
    # mass and the critical-damping gains (Kd = 2·√Kp) actually hold. Off gives the
    # legacy Jacobian-transpose PD, which is under-damped and oscillates in free space.
    use_task_space_inertia: bool = True

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.15, 0.15, 0.15]

    reset_joints = [0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0]
    default_task_prop_gains = [300.0, 300.0, 300.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Optional asymmetric reset-time multiplier range [low, high]. For example,
    # [0.5, 2.0] around a nominal gain of 80 samples gains uniformly in [40, 160].
    # When set, this replaces the symmetric `task_prop_gains_noise_level` scheme.
    task_prop_gains_randomization_scale_range = None
    # Reset-time gain randomization:
    #   "scalar"   -> draw one multiplier per env and scale all 6 nominal gains,
    #                 preserving their translation-to-rotation proportions
    #   "per_axis" -> draw an independent gain multiplier for each of the 6 axes
    # With no scale range, zero noise keeps the configured default gains.
    task_prop_gains_randomization_mode: str = "per_axis"

    # If True, the policy also outputs the 6 task-space proportional gains
    # (3 translational, 3 rotational) as extra action dims, mapped from the
    # normalized [-1, 1] range onto [task_prop_gains_min, task_prop_gains_max].
    # The derivative gains are recomputed from the commanded proportional gains
    # each step. If False, gains stay at `default_task_prop_gains` (plus optional
    # reset-time noise). This grows the action space by 6.
    control_gains: bool = False
    task_prop_gains_min = [100.0, 100.0, 100.0, 10.0, 10.0, 10.0]
    task_prop_gains_max = [600.0, 600.0, 600.0, 60.0, 60.0, 60.0]
    joint_pos_kp: float = 80.0
    joint_pos_kd: float = 8.0

    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null: float = 10.0
    kd_null: float = 6.3246
    use_full_rotation: bool = True

    # OSC nullspace posture target (the config the redundant DoF is biased toward):
    #   "start"          -> each env's own cuRobo start-pose IK solution (per-env; default)
    #   "reset_joints"   -> the fixed `reset_joints` config (collection-branch posture)
    #   "default_dof_pos"-> the fixed `default_dof_pos_tensor` (matches factory/forge)
    # Fixed anchors give a consistent arm posture instead of whatever IK branch each
    # sampled trajectory happens to land on.
    nullspace_posture: str = "start"


@configclass
class TrackingCfg:
    """Discretized end-effector reference trajectory configuration.

    The trajectory is generated analytically per env at reset and then sampled
    onto a fixed grid of control steps (one waypoint per env step), so the
    reference the policy tracks is a discrete sequence of poses rather than a
    continuously-evaluated function of time.
    """

    mode: str = "line"  # line, line_fixed, circle, dataset

    # Straight-line trajectory: EE moves from its reset pose along a randomized
    # 3D direction, covering a randomized path length. Speed is not sampled
    # independently; the line is traversed exactly once over the full episode
    # (speed = length / episode_length), so the reference keeps moving the whole
    # episode regardless of the sampled length.
    line_length_range = [0.10, 0.30]  # m, total displacement along the line

    # "line_fixed" mode: identical to "line" but the direction is this fixed
    # vector (base frame, normalized internally) instead of a random one. Combined
    # with a fixed start pose (init.start_pos_box collapsed to a point,
    # start_rot_noise=0) and a collapsed line_length_range this yields the exact
    # same straight-line reference every reset.
    line_dir = [1.0, 0.0, 0.0]

    # Circle trajectory: EE traces a circle in a randomized plane, anchored so
    # that t=0 coincides with the reset pose. Direction (cw/ccw) is randomized.
    circle_radius_range = [0.05, 0.15]  # m
    circle_speed_range = [0.20, 0.80]  # rad/s (angular speed)

    # Optional orientation sweep: constant angular velocity about a randomized
    # axis, capped at a randomized total angle. Set ranges to 0 to hold fixed.
    rot_speed_range = [0.0, 0.20]  # rad/s
    rot_angle_range = [0.0, 0.30]  # rad, max sweep

    # Lookahead: the policy observes the next `num_future_steps` reference
    # waypoints (index 0 = current target, one per trajectory step), so it can
    # infer the reference velocity from the future pose sequence.
    num_future_steps: int = 4

    # "dataset" mode: instead of an analytic line/circle, each env tracks a real
    # end-effector trajectory sampled from an offline demonstration dataset (e.g.
    # the peg-insertion `tool_pose` sequences). No frame/representation conversion
    # is performed: the dataset pose must describe the frame selected by the env's
    # top-level `tool_frame`, as pos + quat (w, x, y, z) in the robot-base/env-local
    # frame, i.e. the exact format stored in `traj_pos_buf`/`traj_quat_buf`. At reset
    # a random episode is drawn,
    # its poses are resampled onto the per-step control grid, and the usual
    # cuRobo reachability + singularity filter still applies (episodes that fall
    # outside the reachable/well-conditioned workspace are resampled/backfilled).
    dataset_path: str = ""  # HDF5 path; required when mode == "dataset"
    dataset_pose_key: str = "tool_pose"  # (N, 7) pos(3) + quat(4, wxyz) field
    dataset_only_success: bool = True  # keep only episodes with trial_success set
    dataset_max_trajs: int = 0  # cap loaded episodes (0 = all available)
    # Per-reset spatial augmentation for dataset trajectories. Available strategies:
    #   "smooth_bump": original quartic warp that preserves both endpoints and
    #                  reaches the sampled offset at the trajectory midpoint.
    #   "constant_scale_shape": every pose receives an offset of the same magnitude
    #                  while its direction rotates smoothly along a random cone.
    # For the shape strategy, `dataset_warp_shape_fraction` controls the cone's
    # transverse component (0 = rigid offset, 1 = pure loop), and
    # `dataset_warp_cycles_range` controls its number of turns. Zero probability
    # leaves the demonstration unchanged. Candidates still pass the cuRobo checks.
    dataset_warp_strategy: str = "smooth_bump"
    dataset_warp_prob: float = 0.0
    dataset_warp_pos_max: float = 0.0  # m
    dataset_warp_rot_max: float = 0.0  # rad
    dataset_warp_shape_fraction: float = 0.0
    dataset_warp_cycles_range = [0.5, 1.5]
    # When force tracking is on, source the per-step target wrench from this
    # dataset field so the applied disturbance matches the demonstration's real
    # contact force. The field may be (N, 3) or (N, 6) (only the first 3 force
    # components are used). Empty -> fall back to the synthetic contact-band
    # sampler. NOTE: in this peg dataset `contact_force` is all zeros; the real
    # force lives in `wrench[:, :3]` (peaks ~19 N, matching force_mag_range).
    dataset_force_key: str = "wrench"
    # A dataset step counts as "contact" (for the free/contact metric split) when
    # its target-wrench magnitude exceeds this threshold (N).
    dataset_contact_force_threshold: float = 0.5

    # Force-tracking add-on (independent of `mode`). When enabled, a per-step
    # target-wrench sequence is sampled at reset alongside the pose trajectory,
    # applied as an external force at the wrist `force_sensor` body during the
    # episode, and fed to the policy (current + lookahead) so it can anticipate
    # and compensate the disturbance. The force models *contact events*: it is zero
    # for most of the episode, punctuated by a few contact bands. Each band switches
    # on suddenly at a random onset and, while active, varies smoothly (continuous
    # magnitude oscillation plus a gentle direction drift) rather than staying
    # constant, then switches back off at the end of a sampled duration window.
    enable_force: bool = False
    force_mag_range = [5.0, 20.0]  # N, sampled peak contact magnitude (spec-capped at 20 N)
    # Number of contact bands per episode, sampled per env (inclusive range). The
    # episode is split into that many equal segments, each holding one band, so the
    # bands never overlap and stay spread across the episode.
    force_num_bands_range = [2, 3]
    # Duration (s) each contact band stays active, sampled per band (capped so the
    # band fits inside its segment).
    force_duration_range = [0.3, 1.5]
    # Probability that a contact event occurs at all in an episode; with the
    # complementary probability the force stays zero for the whole episode.
    force_prob: float = 1.0
    # Frequency band (Hz) of the smooth oscillation of the force during contact.
    # The magnitude and the direction drift are built from a few sinusoids drawn
    # in this band, so the active force wobbles continuously instead of being flat.
    force_osc_freq_range = [0.5, 3.0]
    # Direction wobble amplitude (0 = fixed direction). The force direction drifts
    # smoothly around its sampled base direction by roughly this fraction.
    force_dir_drift: float = 0.3


@configclass
class InitCfg:
    """Randomized start pose and trajectory-reachability check at reset.

    At each reset a start EE pose is sampled (uniform position box around the
    nominal reset EE position, plus a random orientation perturbation). The
    trajectory is anchored at that start pose, and the start pose together with
    `reach_check_waypoints` evenly-spaced trajectory waypoints (spanning the
    episode) are solved with cuRobo IK; an env is kept only if every waypoint is
    reachable within the tolerances below, otherwise it is resampled. The robot
    is then placed at the cuRobo joint solution for the start pose.
    """

    # Start-position sampling (base/env-local frame, meters). If `start_pos_box`
    # is set to [[x_min, x_max], [y_min, y_max], [z_min, z_max]], the start EE
    # position is drawn uniformly from that absolute box. Otherwise it is drawn
    # from a box of half-extent `start_pos_noise` centered on the nominal reset EE.
    start_pos_box = None
    start_pos_noise = [0.10, 0.10, 0.10]  # m, per-axis half-extent of the start-pos box
    start_rot_noise = 0.5  # rad, max random rotation angle about a random axis

    reach_check_waypoints: int = 8  # trajectory waypoints checked for reachability (incl. start)
    reach_pos_tol: float = 0.002  # m, cuRobo IK position tolerance for success
    reach_rot_tol: float = 0.01  # rad, cuRobo IK orientation tolerance for success
    # Max allowed per-joint jump (rad) between consecutive waypoint IK solutions.
    # Each waypoint is solved seeded with the previous solution; an env is rejected
    # if any joint moves more than this between neighboring waypoints, so the whole
    # trajectory stays on a single continuous IK branch (no elbow flips / wrist
    # wraps) that the policy can actually track.
    reach_joint_diff_threshold: float = 0.5
    # Candidate trajectories drawn per env each resample attempt. The reachability
    # + continuity check rejects many samples, so oversampling evaluates this many
    # independent candidates per env in one batched IK pass and keeps the first
    # accepted one, drastically cutting the number of sequential resample attempts.
    reach_oversample: int = 4
    ik_num_seeds: int = 4  # cuRobo IK seeds per waypoint
    # Max IK queries solved per cuRobo call. A reset sends num_envs *
    # reach_check_waypoints queries; the worker splits larger requests into
    # chunks of this size to bound GPU memory (avoids OOM at high num_envs).
    ik_batch_size: int = 2048
    ik_robot_cfg: str = "franka.yml"  # cuRobo robot config (base=panda_link0, tool=panda_hand)
    # CUDA-graph replay in the cuRobo worker triggers an illegal memory access at
    # large batch sizes; keep it off (a few tenths of a second slower per solve).
    ik_use_cuda_graph: bool = False
    max_reach_attempts: int = 5  # resampling attempts before accepting the current sample
    # Seed the *first* trajectory-waypoint cuRobo IK solve with `ctrl.reset_joints`
    # (subsequent waypoints seed from the previous solution). This lands the whole
    # start config on the reset_joints arm branch instead of an arbitrary cuRobo seed,
    # matching how factory/forge teleport to reset_joints before their reset IK -- so a
    # tracked demonstration is followed on the same elbow/wrist branch it was collected on.
    seed_ik_with_reset_joints: bool = False

    # Singularity filter. On top of reachability + continuity, reject any candidate
    # trajectory whose per-waypoint arm configuration is too close to a kinematic
    # singularity. Near-singular configs are where the OSC task-space inertia
    # (J M⁻¹ Jᵀ)⁻¹ blows up, so small tracking errors explode into the rare
    # fully-diverged episodes that dominate the tracking-error tail. The metric is
    # computed from cuRobo's geometric Jacobian at the selected `tool_frame` (a
    # virtual frame is inserted into cuRobo's tree directly) from each IK solution.
    # A waypoint fails if
    # manipulability sqrt(det(J Jᵀ)) < `min_manipulability`
    # or condition number sigma_max/sigma_min > `max_jac_cond`.
    singularity_check: bool = True
    min_manipulability: float = 0.02
    max_jac_cond: float = 50.0


@configclass
class RandomizationCfg:
    """Dynamics randomization applied at reset."""

    enable_link_mass: bool = True
    link_mass_scale_range = [0.75, 1.25]

    enable_payload: bool = True
    payload_mass_range = [0.0, 1.5]
    payload_com_range = [[-0.04, 0.04], [-0.04, 0.04], [-0.02, 0.12]]
    payload_body_name: str = "panda_hand"

    enable_joint_friction: bool = True
    joint_friction_range = [0.0, 5.0]

    # Per-joint reflected rotor inertia (kg·m²) added to the diagonal of the arm's
    # joint-space mass matrix in PhysX. Franka harmonic-drive joints have reflected
    # inertias on the order of ~0.1 kg·m² (the real OSC controller lumps ~0.15 onto
    # the wrist joints), so this range brackets the physical value with margin. The
    # OSC controller stays blind to it — its nominal mass matrix keeps the
    # config-default armature — so the sampled armature is a modeled-vs-real inertia
    # mismatch the policy must be robust to.
    enable_joint_armature: bool = True
    joint_armature_range = [0.0, 0.3]


@configclass
class RewardCfg:
    """Reward scales for robust Cartesian tracking."""

    pos_error_scale: float = 8.0
    rot_error_scale: float = 2.0
    # Exponential kernel temperatures: reward = exp(-error / temp). These must be
    # matched to the error magnitude the policy actually operates at, otherwise the
    # kernel saturates to ~0 and provides no gradient (rot error runs ~1 rad, so a
    # tight temp like 0.1 leaves it in a flat dead zone).
    pos_error_temp: float = 0.01  # m
    rot_error_temp: float = 0.1  # rad
    ee_vel_scale: float = -0.01
    # Penalize step-to-step change in EE velocity (acceleration) for smoother motion.
    ee_accel_scale: float = -0.05
    action_rate_scale: float = -0.02
    # Penalize step-to-step change in the commanded controller gains (only active
    # when ctrl.control_gains is True) to encourage smooth gain scheduling instead
    # of chattering stiffness. Computed on the normalized [-1, 1] gain actions.
    gain_rate_scale: float = -0.02
    joint_vel_scale: float = -0.002
    joint_limit_scale: float = -0.05
    success_pos_threshold: float = 0.003
    success_rot_threshold: float = 0.1


@configclass
class FrankaRobustTrackEnvCfg(DirectRLEnvCfg):
    """Robot-only Franka environment for robust commanded EE tracking."""

    episode_length_s = 10.0
    decimation = 4
    action_space = 6

    # Optional observation history: if > 1, the policy and critic observations are
    # the last `obs_history_length` single-step frames concatenated oldest->newest
    # into one flat vector (frame history replaces the recurrent memory). H = 1
    # reproduces the original single-step observation.
    obs_history_length: int = 1

    # Debug visualization of the reference trajectory: current command frame,
    # lookahead targets, and the full episode path (sampled in time).
    debug_vis: bool = False
    debug_vis_path_samples: int = 40

    # Per-episode-step tracking-error profile logged to wandb as a mean±std line
    # plot. The pos/rot tracking error is binned by the step index within the
    # episode and accumulated over this many episodes, after which the profile is
    # logged and the accumulators are reset. Set <= 0 to disable the logging.
    log_perstep_error_episodes: int = 5
    # observation_space / state_space are recomputed in __post_init__. Only the
    # proprioceptive part (20 dims) is stacked over obs_history_length frames so
    # the network can infer velocities/dynamics from the past. The lookahead future
    # errors (6 * num_future_steps), controller gains (6), and the critic's
    # privileged dims are episode-constant/current context, so they are appended
    # once rather than duplicated across the history.
    observation_space = 50
    state_space = 74

    ctrl: CtrlCfg = CtrlCfg()
    tracking: TrackingCfg = TrackingCfg()
    init: InitCfg = InitCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    reward: RewardCfg = RewardCfg()

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 60,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=64,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            # gpu_max_rigid_contact_count=2**23,
            # gpu_max_rigid_patch_count=2**23,
            # gpu_collision_stack_size=2**29,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )

    viewer: ViewerCfg = ViewerCfg(
        # Offsets are relative to the robot base (origin_type="asset_root").
        # 3/4 elevated view pulled back to frame the full arm and the tracking
        # workspace (base-frame center ~ (0.45, 0.0, 0.3)).
        eye=(1.6, 1.1, 0.8),
        lookat=(0.45, 0.45, 0.45),
        origin_type="asset_root",
        asset_name="robot",
        resolution=(720, 720),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, clone_in_fabric=False)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
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
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=-0.001),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    robot_usd_path: str = "franka_mimic.usd"
    # Frame whose pose/Jacobian defines the controlled and observed EE. "auto"
    # preserves the legacy rigid-body priority: panda_fingertip_centered,
    # force_sensor, then panda_hand. "fr3_wsg_tcp" is a virtual frame computed from
    # panda_hand using the fixed transform in panda_arm_wsg_horizontal.urdf; it does
    # not need to exist in the USD.
    tool_frame: str = "auto"
    params = None

    def update_env_params(self):
        if self.params is None:
            self.params = OmegaConf.create({})
        env = self.params.get("env", OmegaConf.create({}))

        if env.get("robot_usd_path", None) is not None:
            self.robot_usd_path = env.robot_usd_path
        if env.get("tool_frame", None) is not None:
            self.tool_frame = str(env.tool_frame)
        if env.get("obs_history_length", None) is not None:
            self.obs_history_length = int(env.obs_history_length)
        if env.get("debug_vis", None) is not None:
            self.debug_vis = env.debug_vis
        if env.get("debug_vis_path_samples", None) is not None:
            self.debug_vis_path_samples = env.debug_vis_path_samples
        if env.get("log_perstep_error_episodes", None) is not None:
            self.log_perstep_error_episodes = int(env.log_perstep_error_episodes)
        # Physics-step control: `decimation` sim substeps per control step and the
        # physics `sim.dt`. step_dt = decimation * sim.dt sets the control rate, so to
        # match the factory/forge peg-collection env exactly use dt=1/120 + decimation=8
        # (120 Hz physics, 15 Hz control) instead of the default 1/60 + 4.
        if env.get("decimation", None) is not None:
            self.decimation = int(env.decimation)
        sim = env.get("sim", OmegaConf.create({}))
        if sim.get("dt", None) is not None:
            self.sim.dt = float(sim.dt)

        ctrl = env.get("ctrl", OmegaConf.create({}))
        for key in [
            "backend",
            "ema_factor",
            "action_rep",
            "delta_target_mode",
            "use_gt_mass_matrix",
            "use_task_space_inertia",
            "pos_action_threshold",
            "rot_action_threshold",
            "default_task_prop_gains",
            "task_prop_gains_noise_level",
            "task_prop_gains_randomization_scale_range",
            "task_prop_gains_randomization_mode",
            "control_gains",
            "task_prop_gains_min",
            "task_prop_gains_max",
            "reset_joints",
            "joint_pos_kp",
            "joint_pos_kd",
            "nullspace_posture",
        ]:
            if ctrl.get(key, None) is not None:
                setattr(self.ctrl, key, _to_plain(ctrl[key]))

        tracking = env.get("tracking", OmegaConf.create({}))
        for key in [
            "mode",
            "line_length_range",
            "line_dir",
            "circle_radius_range",
            "circle_speed_range",
            "rot_speed_range",
            "rot_angle_range",
            "num_future_steps",
            "dataset_path",
            "dataset_pose_key",
            "dataset_only_success",
            "dataset_max_trajs",
            "dataset_warp_strategy",
            "dataset_warp_prob",
            "dataset_warp_pos_max",
            "dataset_warp_rot_max",
            "dataset_warp_shape_fraction",
            "dataset_warp_cycles_range",
            "dataset_force_key",
            "dataset_contact_force_threshold",
            "enable_force",
            "force_mag_range",
            "force_num_bands_range",
            "force_duration_range",
            "force_prob",
            "force_osc_freq_range",
            "force_dir_drift",
        ]:
            if tracking.get(key, None) is not None:
                setattr(self.tracking, key, _to_plain(tracking[key]))

        init = env.get("init", OmegaConf.create({}))
        for key in [
            "start_pos_box",
            "start_pos_noise",
            "start_rot_noise",
            "reach_check_waypoints",
            "reach_pos_tol",
            "reach_rot_tol",
            "reach_joint_diff_threshold",
            "reach_oversample",
            "ik_num_seeds",
            "ik_batch_size",
            "ik_robot_cfg",
            "ik_use_cuda_graph",
            "max_reach_attempts",
            "seed_ik_with_reset_joints",
            "singularity_check",
            "min_manipulability",
            "max_jac_cond",
        ]:
            if init.get(key, None) is not None:
                setattr(self.init, key, _to_plain(init[key]))

        randomization = env.get("randomization", OmegaConf.create({}))
        for key in [
            "enable_link_mass",
            "link_mass_scale_range",
            "enable_payload",
            "payload_mass_range",
            "payload_com_range",
            "payload_body_name",
            "enable_joint_friction",
            "joint_friction_range",
            "enable_joint_armature",
            "joint_armature_range",
        ]:
            if randomization.get(key, None) is not None:
                value = randomization[key]
                setattr(self.randomization, key, _to_plain(value))

        reward = env.get("reward", OmegaConf.create({}))
        for key in [
            "pos_error_scale",
            "rot_error_scale",
            "pos_error_temp",
            "rot_error_temp",
            "ee_vel_scale",
            "ee_accel_scale",
            "action_rate_scale",
            "joint_vel_scale",
            "joint_limit_scale",
            "gain_rate_scale",
            "success_pos_threshold",
            "success_rot_threshold",
        ]:
            if reward.get(key, None) is not None:
                setattr(self.reward, key, _to_plain(reward[key]))

    def __post_init__(self):
        self.update_env_params()
        self.tool_frame = str(self.tool_frame).strip() or "auto"
        self.robot.spawn.usd_path = f"{ASSET_DIR}/{self.robot_usd_path}"
        self.sim.render_interval = self.decimation

        # Action space: 6 Cartesian delta-pose dims (pos3 + rot3), plus 6 gain dims
        # when ctrl.control_gains is enabled (policy schedules its own PD gains).
        action_dim = 6 + (6 if self.ctrl.control_gains else 0)
        self.action_space = action_dim

        # Proprio obs: ee_pos(3)+ee_quat(4)+joint_pos(7)+actions(action_dim) (velocity
        # terms are omitted; the LSTM infers them from history). Each lookahead
        # pose contributes a (pos_error, axis_angle_error) pair = 6 dims. The 6
        # current task gains are fed once to both policy and critic. The critic adds
        # 24 privileged dims: payload_mass(1)+payload_com(3)+joint_friction(7)
        # +joint_armature(7)+pos_threshold(3)+rot_threshold(3).
        proprio_dim = 14 + action_dim
        future_dim = 6 * self.tracking.num_future_steps
        controller_context_dim = 6
        privileged_dim = 24
        # Force-tracking add-on contributes a (current + lookahead) target-wrench
        # block of 3 dims per step to both the policy and the critic observation.
        force_dim = 3 * self.tracking.num_future_steps if self.tracking.enable_force else 0
        history = max(1, int(self.obs_history_length))
        # Only proprio is stacked over history; future errors + target wrench +
        # controller gains (policy+critic) and privileged dims (critic) are appended
        # once from the current frame.
        self.observation_space = proprio_dim * history + future_dim + force_dim + controller_context_dim
        self.state_space = self.observation_space + privileged_dim
