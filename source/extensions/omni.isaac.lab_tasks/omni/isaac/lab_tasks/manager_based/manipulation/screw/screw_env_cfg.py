# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import functools
import torch
from dataclasses import MISSING
from typing import Literal

from omegaconf import OmegaConf

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, ContactSensorCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from omni.isaac.lab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp

from omni.isaac.lab.markers.config import (  # isort: skip
    DEFORMABLE_TARGET_MARKER_CFG,
    FRAME_MARKER_CFG,
    RED_ARROW_X_MARKER_CFG,
)


# Scene definition
FRAME_MARKER_SMALL_CFG = copy.deepcopy(FRAME_MARKER_CFG)
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.008, 0.008, 0.008)
RED_PLATE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "height": sim_utils.CylinderCfg(
            radius=0.02,
            height=0.001,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0, 0), opacity=0.5),
        )
    }
)
BLUE_PLATE_MARKER_CFG = RED_PLATE_MARKER_CFG.copy()
BLUE_PLATE_MARKER_CFG.markers["height"].visual_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0, 0, 1.0), opacity=0.5
)
PLATE_ARROW_CFG = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.008, 0.008, 0.008),
        ),
    }
)

asset_factory = {
    "m8_loose": {
        "nut_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_nut_m8_loose/factory_nut_m8_loose.usd",
        "bolt_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_bolt_m8_loose/factory_bolt_m8_loose.usd",
        "nut_init_state_tighten": RigidObjectCfg.InitialStateCfg(
            pos=(6.3000e-01, 2.0661e-06, 3.0895e-03), rot=(-2.1609e-01, 6.6671e-05, -6.6467e-05, 9.7637e-01)
        ),
        "nut_init_state_thread": RigidObjectCfg.InitialStateCfg(
            pos=(6.3000e-01, 4.0586e-06, 0.02), rot=(9.9833e-01, 1.2417e-04, -1.2629e-05, 5.7803e-02)
        ),
        "bolt_init_state": RigidObjectCfg.InitialStateCfg(pos=(0.63, 0.0, 0.0)),
        "nut_frame_offset": OffsetCfg(pos=(0.0, 0.0, 0.011)),
        "bolt_bottom_offset": OffsetCfg(pos=(0.0, 0.0, 0.012)),
        "bolt_tip_offset": OffsetCfg(pos=(0.0, 0.0, 0.0261)),
        "float_gain": 10.0,
        "float_damp": 0.01,
    },
    "m8_tight": {
        "nut_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_nut_m8_tight/factory_nut_m8_tight.usd",
        "bolt_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_bolt_m8_tight/factory_bolt_m8_tight.usd",
        "nut_init_state_tighten": RigidObjectCfg.InitialStateCfg(
            pos=(6.3000e-01, 2.0661e-06, 3.0895e-03), rot=(-2.1609e-01, 6.6671e-05, -6.6467e-05, 9.7637e-01)
        ),
        "nut_init_state_thread": RigidObjectCfg.InitialStateCfg(
            pos=(6.3000e-01, 4.0586e-06, 0.02), rot=(9.9833e-01, 1.2417e-04, -1.2629e-05, 5.7803e-02)
        ),
        "bolt_init_state": RigidObjectCfg.InitialStateCfg(pos=(0.63, 0.0, 0.0)),
        "nut_frame_offset": OffsetCfg(pos=(0.0, 0.0, 0.011)),
        "bolt_bottom_offset": OffsetCfg(pos=(0.0, 0.0, 0.0)),
        "bolt_tip_offset": OffsetCfg(pos=(0.0, 0.0, 0.0261)),
        "float_gain": 10.0,
        "float_damp": 0.01,
    },
    "m16_tight": {
        "nut_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_nut_m16_tight/factory_nut_m16_tight.usd",
        "bolt_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_bolt_m16_tight/factory_bolt_m16_tight.usd",
        "nut_init_state_tighten": RigidObjectCfg.InitialStateCfg(
            pos=(6.3000e-01, 2.0661e-06, 3.0895e-03), rot=(-2.1609e-01, 6.6671e-05, -6.6467e-05, 9.7637e-01)
        ),
        "nut_init_state_thread": RigidObjectCfg.InitialStateCfg(
            pos=(6.3000e-01, 4.0586e-06, 0.03), rot=(9.9833e-01, 1.2417e-04, -1.2629e-05, 5.7803e-02)
        ),
        "bolt_init_state": RigidObjectCfg.InitialStateCfg(pos=(0.63, 0.0, 0.0)),
        "bolt_init_state": RigidObjectCfg.InitialStateCfg(pos=(0.63, 0.0, 0.0)),
        "nut_frame_offset": OffsetCfg(pos=(0.0, 0.0, 0.0225)),
        "bolt_bottom_offset": OffsetCfg(pos=(0.0, 0.0, 0.0)),
        "bolt_tip_offset": OffsetCfg(pos=(0.0, 0.0, 0.041)),
        "float_gain": 10.0,
        "float_damp": 0.01,
    },
    "m16_loose": {
        "nut_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_nut_m16_loose/factory_nut_m16_loose.usd",
        "bolt_path": f"{ISAAC_NUCLEUS_DIR}/Props/Factory/factory_bolt_m16_loose/factory_bolt_m16_loose.usd",
        "nut_init_state_tighten": RigidObjectCfg.InitialStateCfg(
            pos=(6.3000e-01, 2.0661e-06, 3.0895e-03), rot=(-2.1609e-01, 6.6671e-05, -6.6467e-05, 9.7637e-01)

        ),
        "nut_init_state_thread": RigidObjectCfg.InitialStateCfg(
            pos=(6.3000e-01, 4.0586e-06, 0.03), rot=(9.9833e-01, 1.2417e-04, -1.2629e-05, 5.7803e-02)
        ),
        "bolt_init_state": RigidObjectCfg.InitialStateCfg(pos=(0.63, 0.0, 0.0)),
        "nut_frame_offset": OffsetCfg(pos=(0.0, 0.0, 0.0225)),
        "bolt_bottom_offset": OffsetCfg(pos=(0.0, 0.0, 0.0)),
        "bolt_tip_offset": OffsetCfg(pos=(0.0, 0.0, 0.041)),
        "nut_geom_name": "factory_nut",
        "bolt_geom_name": "factory_bolt",
        "float_gain": 10.0,
        "float_damp": 0.01,
    },   
    "m16": {
        "nut_path": f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Factory/factory_nut_m16.usd",
        "bolt_path": f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Factory/factory_bolt_m16.usd",
        "nut_init_state_tighten": ArticulationCfg.InitialStateCfg(
            pos=(6.3000e-01, 2.0661e-06, 3.0895e-03), rot=(-2.1609e-01, 6.6671e-05, -6.6467e-05, 9.7637e-01),
            joint_pos={}, joint_vel={}
        ),
        "nut_init_state_thread": RigidObjectCfg.InitialStateCfg(
            pos=(6.3000e-01, 4.0586e-06, 0.03), rot=(9.9833e-01, 1.2417e-04, -1.2629e-05, 5.7803e-02)
        ),
        "bolt_init_state": RigidObjectCfg.InitialStateCfg(pos=(0.63, 0.0, 0.0)),
        "nut_frame_offset": OffsetCfg(pos=(0.0, 0.0, 0.0225)),
        "bolt_bottom_offset": OffsetCfg(pos=(0.0, 0.0, 0.0)),
        "bolt_tip_offset": OffsetCfg(pos=(0.0, 0.0, 0.041)),
        "nut_geom_name": "factory_nut_loose",
        "bolt_geom_name": "factory_bolt_loose",
        "float_gain": 10.0,
        "float_damp": 0.01,
    },
}


@configclass
class ScrewSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    screw_type: Literal["m8_loose", "m8_tight", "m16_loose", "m16_tight"] = "m8_tight"

    def __post_init__(self):
        self.screw_dict = asset_factory[self.screw_type]
        # world
        self.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        )
        self.origin = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Origin",
            spawn=sim_utils.SphereCfg(
                radius=1e-3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
        )

        # robots: will be populated by agent env cfg
        self.robot: ArticulationCfg = MISSING

        # m16 is the usd in Forge paper, for some reasons, it's created as articulation.
        # The old factory usd that we used is also problematic after an recent update. 
        #  - It cannot be created as articulation.
        #  - It contains articulation root properties.Has to set to false when created as rigid object
        if self.screw_type == "m16":
            obj_cfg = functools.partial(ArticulationCfg, actuators={})
        else:
            obj_cfg = functools.partial(RigidObjectCfg)
        # objects
        self.nut: RigidObjectCfg = obj_cfg(
            prim_path="{ENV_REGEX_NS}/Nut",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.screw_dict["nut_path"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                # articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False)
            ),
        )

        self.bolt: RigidObjectCfg = obj_cfg(
            prim_path="{ENV_REGEX_NS}/Bolt",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.screw_dict["bolt_path"],
                # articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False)
                ),
            init_state=self.screw_dict["bolt_init_state"],
        )

        # lights
        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
        )

        self.nut_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Origin",
            debug_vis=True,
            visualizer_cfg=PLATE_ARROW_CFG.replace(prim_path="/Visuals/Nut"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    # prim_path="{ENV_REGEX_NS}/Nut/factory_nut",
                    prim_path="{ENV_REGEX_NS}/Nut/" + f"{self.screw_dict['nut_geom_name']}",

                    name="nut",
                    offset=self.screw_dict["nut_frame_offset"],
                )
            ],
        )

        self.bolt_frame: FrameTransformerCfg = MISSING


##
# MDP settings
##
@configclass
class BaseActionsCfg:
    """Action specifications for the MDP."""

    nut_action: ActionTerm | None = None
    arm_action: ActionTerm | None = None
    gripper_action: ActionTerm | None = None


@configclass
class BaseObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # bolt_pose = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("bolt")})
        nut_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("nut")})
        nut_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("nut")})
        nut_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("nut")})
        nut_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("nut")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.hist_len = 1

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


##
# Environment configuration
##


@configclass
class BaseScrewEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the screw end-effector pose tracking environment."""

    scene: ScrewSceneCfg = MISSING
    # Basic settings
    observations: BaseObservationsCfg = BaseObservationsCfg()
    actions: BaseActionsCfg = BaseActionsCfg()
    # MDP settings
    rewards = MISSING
    terminations = MISSING
    events: EventCfg = EventCfg()
    curriculum = CurriculumCfg()

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_collision_stack_size=2**30,
            gpu_heap_capacity=2**31,
            gpu_temp_buffer_capacity=2**30,
            gpu_max_rigid_patch_count=2**24,
            enable_enhanced_determinism=False,
        ),
    )

    def get_default_env_params(self):
        """Set default environment parameters."""
        # Initialize params structure
        if self.params is None:
            self.params = OmegaConf.create()
        params = self.params
        params.scene = params.get("scene", OmegaConf.create())
        params.sim = params.get("sim", OmegaConf.create())
        params.actions = params.get("actions", OmegaConf.create())
        params.observations = params.get("observations", OmegaConf.create())
        params.rewards = params.get("rewards", OmegaConf.create())
        params.terminations = params.get("terminations", OmegaConf.create())
        params.events = params.get("events", OmegaConf.create())
        params.curriculum = params.get("curriculum", OmegaConf.create())
        params.sim.physx = params.sim.get("physx", OmegaConf.create())
        params.scene.nut = params.scene.get("nut", OmegaConf.create())

        params.scene.screw_type = params.scene.get("screw_type", "m16_loose")  # m8_tight m16_tight
        params.sim.dt = params.sim.get("dt", 1.0 / 60.0)
        params.sim.physx.friction_offset_threshold = params.sim.physx.get("friction_offset_threshold", 0.04)
        params.sim.physx.enable_ccd = params.sim.physx.get("enable_ccd", False)
        params.decimation = params.get("decimation", 2)

        # By default use the default params in USD
        nut_params = params.scene.nut
        nut_params.max_depenetration_velocity = nut_params.get("max_depenetration_velocity", None)
        nut_params.sleep_threshold = nut_params.get("sleep_threshold", None)
        nut_params.stabilization_threshold = nut_params.get("stabilization_threshold", None)
        nut_params.linear_damping = nut_params.get("linear_damping", None)
        nut_params.angular_damping = nut_params.get("angular_damping", None)

    def __post_init__(self):
        """Post initialization."""
        self.get_default_env_params()
        self.screw_type = self.params.scene.screw_type
        self.scene = ScrewSceneCfg(num_envs=4096, env_spacing=2.5, screw_type=self.screw_type)
        # general settings
        self.decimation = self.params.decimation
        self.sim.render_interval = self.decimation
        # self.rerender_on_reset = True
        self.sim.dt = self.params.sim.dt
        self.sim.physx.friction_offset_threshold = self.params.sim.physx.friction_offset_threshold
        self.sim.physx.enable_ccd = self.params.sim.physx.enable_ccd

        nut = self.scene.nut
        nut_params = self.params.scene.nut
        nut.spawn.rigid_props.max_depenetration_velocity = nut_params.max_depenetration_velocity
        nut.spawn.rigid_props.sleep_threshold = nut_params.sleep_threshold
        nut.spawn.rigid_props.stabilization_threshold = nut_params.stabilization_threshold
        nut.spawn.rigid_props.linear_damping = nut_params.linear_damping
        nut.spawn.rigid_props.angular_damping = nut_params.angular_damping

        self.episode_length_s = 24
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "bolt"
        self.viewer.eye = (0.1, 0, 0.04)
        self.viewer.lookat = (0, 0, 0.04)
        self.viewer.resolution = (720, 720)


###################################
#           Nut Tighten           #
def nut_tighten_reward_forge(env: ManagerBasedRLEnv, a: float = 100, b: float = 0, tol: float = 0):
    diff = mdp.rel_nut_bolt_bottom_distance(env)
    rewards = mdp.forge_kernel(diff, a, b, tol)
    return rewards


@configclass
class NutTightenRewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    coarse_nut = RewTerm(func=nut_tighten_reward_forge, params={"a": 100, "b": 2}, weight=1.0)
    fine_nut = RewTerm(
        func=nut_tighten_reward_forge,
        params={
            "a": 500,
            "b": 0,
        },
        weight=1.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.00001)


@configclass
class NutTightenTerminationsCfg:
    """Termination terms for screw tightening."""

    nut_screwed = DoneTerm(func=mdp.nut_fully_screwed, params={"threshold": 1e-4})
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BaseNutTightenEnvCfg(BaseScrewEnvCfg):
    rewards: NutTightenRewardsCfg = NutTightenRewardsCfg()
    terminations: NutTightenTerminationsCfg = NutTightenTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        screw_dict = asset_factory[self.params.scene.screw_type]
        self.scene.nut.init_state = screw_dict["nut_init_state_tighten"]
        self.scene.bolt_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Origin",
            debug_vis=True,
            visualizer_cfg=RED_PLATE_MARKER_CFG.replace(prim_path="/Visuals/Bolt"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Bolt/" + f"{screw_dict['bolt_geom_name']}",
                    name="bolt_bottom",
                    offset=screw_dict["bolt_bottom_offset"],
                ),
            ],
        )


###################################
#           Nut Thread           #
def nut_thread_reward_forge(env: ManagerBasedRLEnv, a: float = 100, b: float = 0, tol: float = 0):
    diff = mdp.rel_nut_bolt_tip_distance(env)
    rewards = mdp.forge_kernel(diff, a, b, tol)
    return rewards


def nut_thread_xy_l2(env: ManagerBasedRLEnv):
    diff = mdp.rel_nut_bolt_tip_distance(env)[..., :2]
    rewards = mdp.l2_norm(diff)
    return 0.01 - rewards


def nut_upright_reward_forge(env: ManagerBasedRLEnv, a: float = 300, b: float = 0, tol: float = 0):
    # penalize if nut is not upright
    # compute the cosine distance between the nut normal and the global up vector
    nut_quat = env.scene["nut_frame"].data.target_quat_w[:, 0]
    up_vec = torch.tensor([[0, 0, 1.0]], device=nut_quat.device)
    up_vecs = up_vec.expand(nut_quat.shape[0], 3)
    nut_up_vec = math_utils.quat_apply(nut_quat, up_vecs)
    cos_sim = torch.sum(nut_up_vec * up_vecs, dim=1, keepdim=True) / torch.norm(nut_up_vec, dim=1, keepdim=True)
    rewards = mdp.forge_kernel(1 - cos_sim, a, b, tol)
    return rewards


@configclass
class NutThreadRewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    xy_nut = RewTerm(func=nut_thread_xy_l2, weight=10)
    coarse_nut = RewTerm(func=nut_thread_reward_forge, params={"a": 50, "b": 1}, weight=0.5)
    fine_nut = RewTerm(
        func=nut_thread_reward_forge,
        params={
            "a": 500,
            "b": 0,
        },
        weight=2.0,
    )
    upright_reward = RewTerm(func=nut_upright_reward_forge, params={"a": 700, "b": 0, "tol": 1e-3}, weight=2)
    success = RewTerm(func=mdp.nut_successfully_threaded, params={"threshold": 3e-4}, weight=1)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.000001)


@configclass
class NutThreadTerminationsCfg:
    """Termination terms for screw tightening."""

    # nut_screwed = DoneTerm(func=mdp.nut_successfully_threaded, params={"threshold": 1e-4})
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BaseNutThreadEnvCfg(BaseScrewEnvCfg):
    rewards: NutThreadRewardsCfg = NutThreadRewardsCfg()
    terminations: NutThreadTerminationsCfg = NutThreadTerminationsCfg()

    def __init__(self):
        super().__init__()  # Call the parent class's __init__ method

    def __post_init__(self):
        super().__post_init__()
        screw_dict = asset_factory[self.params.scene.screw_type]
        self.scene.nut.init_state = screw_dict["nut_init_state_thread"]

        self.scene.nut_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Origin",
            debug_vis=True,
            visualizer_cfg=BLUE_PLATE_MARKER_CFG.replace(prim_path="/Visuals/Nut"),
            # visualizer_cfg=PLATE_ARROW_CFG.replace(prim_path="/Visuals/Nut"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Nut/" + f"{screw_dict['nut_geom_name']}",
                    name="nut",
                    offset=screw_dict["nut_frame_offset"],
                )
            ],
        )
        self.scene.bolt_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Origin",
            debug_vis=True,
            visualizer_cfg=RED_PLATE_MARKER_CFG.replace(prim_path="/Visuals/Bolt"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Bolt/" + f"{screw_dict['bolt_geom_name']}",
                    name="bolt_tip",
                    offset=screw_dict["bolt_tip_offset"],
                )
            ],
        )
        
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Nut/" + f"{screw_dict['nut_geom_name']}",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Bolt/" + f"{screw_dict['bolt_geom_name']}"],
            update_period=0.0,
        )
