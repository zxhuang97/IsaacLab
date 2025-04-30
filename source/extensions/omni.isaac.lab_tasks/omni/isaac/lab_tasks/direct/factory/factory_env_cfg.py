# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.utils import configclass


from omni.isaac.lab.sensors import TiledCameraCfg, ContactSensorCfg, CameraCfg
from omni.isaac.lab.sim.spawners.sensors import PinholeCameraCfg


from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

from .factory_tasks_cfg import ASSET_DIR, FactoryTask, GearMesh, NutThread, PegInsert

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "scales": 1,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
    "scales": 1,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]
    # fixed_asset_pos = [0.0, 0.0, 0.0]


@configclass
class CtrlCfg:
    ema_factor = 1

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]

    reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null = 10.0
    kd_null = 6.3246


@configclass
class FactoryEnvCfg(DirectRLEnvCfg):
    decimation = 8
    action_space = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 72
    obs_order: list = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task_class: FactoryTask = FactoryTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    episode_length_s = 10.0  # Probably need to override.
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
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
            gpu_collision_stack_size=2048*(2**20),# For 1024 parallel scenes, too many collisions
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)

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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
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
                effort_limit=87,
                velocity_limit=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=12,
                velocity_limit=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit=40.0,
                velocity_limit=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    contact_filter_path = [
        "/World/envs/env_.*/FixedAsset/forge_hole_8mm/forge_hole_8mm",
        "/World/envs/env_.*/FixedAsset/factory_bolt_loose",
        "/World/envs/env_.*/FixedAsset/factory_gear_base_loose",
        "/World/envs/env_.*/LargeGearAsset/factory_gear_large",
        "/World/envs/env_.*/SmallGearAsset/factory_gear_small",
    ]

    # Wrench config
    wrench_joint_cfg = SceneEntityCfg("robot", body_names=["panda_link7"])

    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/.*",
        filter_prim_paths_expr=contact_filter_path,
        # debug_vis=True,
        history_length=6,
        update_period=1/60.0,
    )

    cam_offset_cfg = CameraCfg.OffsetCfg(
        pos=(1, 0.0, 0.15),
        rot=[0.4497752, 0.4401843, 0.5533875, 0.545621],
        convention='opengl'
    )
    ph_cfg = PinholeCameraCfg(visible=True,
        semantic_tags=None,
        copy_from_source=True,
        projection_type='pinhole',
        clipping_range=(0.0001, 20.0),
        focal_length=24.0,
        focus_distance=400.0,
        f_stop=0.0,
        horizontal_aperture=20.955,
        vertical_aperture=None,
        horizontal_aperture_offset=0.0,
        vertical_aperture_offset=0.0,
        lock_camera=True
    )
    tiled_camera_cfg = TiledCameraCfg(
        width=720, height=720,
        offset=cam_offset_cfg,
        prim_path='/World/envs/env_.*/Camera',
        spawn=ph_cfg
    )
    # V1
    # obs_camera_cfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/DepthCamera",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1.0, 0.1, 0.12),
    #         rot=[0.4402, -0.4498, -0.5456, 0.5534],
    #         convention="ros",
    #     ),
    #     data_types=["distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.0001, 0.5)),
    #     width=200,
    #     height=200,
    # )
    # V2
    # obs_camera_cfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/DepthCamera",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1, 0.2, 0.15),
    #         rot=[0.34853, -0.3822, -0.62177, 0.5881],
    #         convention="ros",
    #     ),
    #     data_types=["distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.0001, 0.5)),
    #     width=224,
    #     height=224,
    # )
    # V3
    obs_camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/DepthCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8, 0.2, 0.15),
            rot=[0.18913, -0.25231, -0.70188, 0.6387],
            convention="ros",
        ),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.0001, 0.5)),
        width=224,
        height=224,
    )


    # Add in scale randomization of assets.
    randomize_scale_method= "none"    # gaussian/uniform/none, implemented in factory_env
    # Scale range for held asset
    # For uniform, [min, max]
    # For gaussian, [mean, std]
    randomize_scale_range = (0.5, 2.0)

    # To enable experiments with cfg dicts
    params = OmegaConf.create()

    def update_env_params(self):
        """Set default environment parameters."""
        # Initialize params structure
        params = self.params

        # def update_terminals(container, config, keys):
        #     """
        #     Update each attribute in 'container' with the corresponding value
        #     from 'config'. If a key is missing, leave the current value in 'container'.
        #     """
        #     for key in keys:
        #         var = getattr(container, key)
        #         var = config.get(key, var)

        # params.scene = params.get("scene", OmegaConf.create())

        # Hard Coded
        # params.scene.nut = params.scene.get("nut", OmegaConf.create())
        # params.scene.screw_type = params.scene.get("screw_type", "m16_loose")  # m8_tight m16_tight
        # update_terminals(self, params, ["decimation"])

        # Sim
        params_sim = params.get("sim", OmegaConf.create())
        self.sim.dt = params_sim.get("dt", self.sim.dt)

        # # --- Observation Randomization Config ---
        # # Here we keep the whole config group in params.
        params_obs = params.get("observations", OmegaConf.create())
        fixed_asset_pos_noise = params_obs.get("fixed_asset_pos_noise", None)
        if fixed_asset_pos_noise is not None:
            ObsRandCfg.fixed_asset_pos = fixed_asset_pos_noise
            self.obs_rand = ObsRandCfg(fixed_asset_pos=tuple(fixed_asset_pos_noise))
        include_scale = params_obs.get("include_scale", False)
        if include_scale:
            self.obs_order.append("scales")
            self.state_order.append("scales")
            self.observation_space += 1
            self.state_space += 1

        # Update observation camera
        self.use_tiled_camera = params_obs.get("use_tiled_camera", False)
        self.use_obs_camera = params_obs.get("use_obs_camera", False)
        obs_camera_type = params_obs.get("obs_camera_type", ["distance_to_image_plane"])
        self.obs_camera_cfg.data_types = obs_camera_type

        # # NutThread Task related properties
        params_taskcfg = params.get("taskcfg", {})
        asset_scale_randomization = params_taskcfg.get("randomize_scale_method", "none")
        if asset_scale_randomization not in ["gaussian", "uniform", "none"]:
            print(f"Warning: asset_scale_randomization '{asset_scale_randomization}' is not recognized, using 'none'.")
            asset_scale_randomization = "none"
        self.randomize_scale_method = asset_scale_randomization
        scale_range = params_taskcfg.get("randomize_scale_range", None)
        # Update scale
        if self.randomize_scale_method in ["gaussian", "uniform"] \
            and scale_range is not None \
            and len(scale_range) == 2:
            if isinstance(scale_range, ListConfig):
                scale_range = OmegaConf.to_container(scale_range, resolve=True)
            self.randomize_scale_range = tuple(scale_range)
            self.scene = InteractiveSceneCfg(
                num_envs=128,
                env_spacing=2.0,
                replicate_physics=False
            )
        else:
            print(f"Warning: 'randomize_scale_range' should be a list/tuple of length 2, using default {self.randomize_scale_range}.")

    def __post_init__(self):
        """Post initialization."""
        self.update_env_params()
        self.sim.render_interval = self.decimation

        # self.episode_length_s = 24   # 24, 10 for sim quality test
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "fixed_asset"
        # self.viewer.eye = (0.1, 0.1, 0.06)
        # self.viewer.lookat = (0, 0.0, 0.04)
        self.viewer.eye = (0.37, 0.1, 0.12)
        self.viewer.lookat = (0.0, 0.0, 0.03)
        self.viewer.resolution = (720, 720)



@configclass
class FactoryTaskPegInsertCfg(FactoryEnvCfg):
    name = "peg_insert"
    task_class = PegInsert()
    episode_length_s = 10.0


@configclass
class FactoryTaskGearMeshCfg(FactoryEnvCfg):
    name = "gear_mesh"
    task_class = GearMesh()
    episode_length_s = 20.0


from omegaconf import OmegaConf

@configclass
class FactoryTaskNutThreadCfg(FactoryEnvCfg):
    name = "nut_thread"
    task_class = NutThread()
    episode_length_s = 30.0