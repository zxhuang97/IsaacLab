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

    backend: str = "factory_osc"  # factory_osc, dls_ik
    ema_factor: float = 0.2

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.15, 0.15, 0.15]

    reset_joints = [0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0]
    default_task_prop_gains = [300.0, 300.0, 300.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    joint_pos_kp: float = 80.0
    joint_pos_kd: float = 8.0

    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null: float = 10.0
    kd_null: float = 6.3246
    use_full_rotation: bool = True


@configclass
class TrackingCfg:
    """Discretized end-effector reference trajectory configuration.

    The trajectory is generated analytically per env at reset and then sampled
    onto a fixed grid of control steps (one waypoint per env step), so the
    reference the policy tracks is a discrete sequence of poses rather than a
    continuously-evaluated function of time.
    """

    mode: str = "line"  # line, circle

    # Straight-line trajectory: EE moves from its reset pose along a randomized
    # 3D direction at constant speed, capped at a randomized path length.
    line_speed_range = [0.03, 0.10]  # m/s
    line_length_range = [0.10, 0.30]  # m, max displacement along the line

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
    reach_pos_tol: float = 0.01  # m, cuRobo IK position tolerance for success
    reach_rot_tol: float = 0.05  # rad, cuRobo IK orientation tolerance for success
    ik_num_seeds: int = 16  # cuRobo IK seeds per waypoint
    # Max IK queries solved per cuRobo call. A reset sends num_envs *
    # reach_check_waypoints queries; the worker splits larger requests into
    # chunks of this size to bound GPU memory (avoids OOM at high num_envs).
    ik_batch_size: int = 1024
    ik_robot_cfg: str = "franka.yml"  # cuRobo robot config (base=panda_link0, tool=panda_hand)
    # CUDA-graph replay in the cuRobo worker triggers an illegal memory access at
    # large batch sizes; keep it off (a few tenths of a second slower per solve).
    ik_use_cuda_graph: bool = False
    max_reach_attempts: int = 20  # resampling attempts before accepting the current sample


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


@configclass
class RewardCfg:
    """Reward scales for robust Cartesian tracking."""

    pos_error_scale: float = 8.0
    rot_error_scale: float = 2.0
    ee_vel_scale: float = -0.01
    action_rate_scale: float = -0.02
    joint_vel_scale: float = -0.002
    joint_limit_scale: float = -0.05
    success_pos_threshold: float = 0.01
    success_rot_threshold: float = 0.10


@configclass
class FrankaRobustTrackEnvCfg(DirectRLEnvCfg):
    """Robot-only Franka environment for robust commanded EE tracking."""

    episode_length_s = 10.0
    decimation = 4
    action_space = 6

    # Debug visualization of the reference trajectory: current command frame,
    # lookahead targets, and the full episode path (sampled in time).
    debug_vis: bool = False
    debug_vis_path_samples: int = 40
    # observation_space / state_space are recomputed in __post_init__ from the
    # number of lookahead poses (tracking.num_future_steps). Values below are for
    # the default num_future_steps = 4 (20 proprio + 6*4 future errors = 44;
    # critic adds 23 privileged dims = 67).
    observation_space = 44
    state_space = 67

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
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**29,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.6, 0.35, 0.35),
        lookat=(0.0, 0.0, 0.25),
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
    params = None

    def update_env_params(self):
        if self.params is None:
            self.params = OmegaConf.create({})
        env = self.params.get("env", OmegaConf.create({}))

        if env.get("robot_usd_path", None) is not None:
            self.robot_usd_path = env.robot_usd_path
        if env.get("debug_vis", None) is not None:
            self.debug_vis = env.debug_vis
        if env.get("debug_vis_path_samples", None) is not None:
            self.debug_vis_path_samples = env.debug_vis_path_samples

        ctrl = env.get("ctrl", OmegaConf.create({}))
        for key in [
            "backend",
            "ema_factor",
            "pos_action_threshold",
            "rot_action_threshold",
            "default_task_prop_gains",
            "task_prop_gains_noise_level",
            "reset_joints",
            "joint_pos_kp",
            "joint_pos_kd",
        ]:
            if ctrl.get(key, None) is not None:
                setattr(self.ctrl, key, _to_plain(ctrl[key]))

        tracking = env.get("tracking", OmegaConf.create({}))
        for key in [
            "mode",
            "line_speed_range",
            "line_length_range",
            "circle_radius_range",
            "circle_speed_range",
            "rot_speed_range",
            "rot_angle_range",
            "num_future_steps",
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
            "ik_num_seeds",
            "ik_batch_size",
            "ik_robot_cfg",
            "ik_use_cuda_graph",
            "max_reach_attempts",
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
        ]:
            if randomization.get(key, None) is not None:
                value = randomization[key]
                setattr(self.randomization, key, _to_plain(value))

    def __post_init__(self):
        self.update_env_params()
        self.robot.spawn.usd_path = f"{ASSET_DIR}/{self.robot_usd_path}"
        self.sim.render_interval = self.decimation

        # Proprio obs: ee_pos(3)+ee_quat(4)+joint_pos(7)+actions(6) = 20 (velocity
        # terms are omitted; the LSTM infers them from history). Each lookahead
        # pose contributes a (pos_error, axis_angle_error) pair = 6 dims. Critic
        # adds 23 privileged dims: payload_mass(1)+payload_com(3)+joint_friction(7)
        # +task_gains(6)+pos_threshold(3)+rot_threshold(3).
        proprio_dim = 20
        future_dim = 6 * self.tracking.num_future_steps
        privileged_dim = 23
        self.observation_space = proprio_dim + future_dim
        self.state_space = self.observation_space + privileged_dim
