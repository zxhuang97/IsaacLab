# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import (
    FactoryTask, GearMesh, NutThread, PegInsert,
    PegInsertTac, GearMeshTac, NutThreadTac
)


@configclass
class ForgeTask(FactoryTask):
    # Applied to the fixed socket USD in peg-insertion tasks only.  Using the
    # same factor for x and y preserves a circular, centered hole; z remains
    # unscaled so the insertion depth is unchanged.
    hole_xy_scale: float = 1.0
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    # Penalize measured end-effector speed above this threshold. Disabled by
    # default through a zero scale.
    ee_speed_penalty_scale: float = 0.0
    ee_speed_penalty_threshold: float = 0.01
    ee_speed_exp_penalty_scale: float = 0.0
    ee_speed_exp_penalty_k: float = 0.25
    ee_speed_exp_penalty_cap: float = 10.0
    # Penalize end-effector angular motion separately from linear motion.  This
    # is disabled by default and enabled by launchers that allow full rotation.
    ee_ang_speed_penalty_scale: float = 0.0
    ee_ang_speed_penalty_threshold: float = 0.1
    # Keep the fingertip top-down even when a compliant gripper lets the held
    # asset remain aligned independently of the robot wrist.
    ee_upright_penalty_scale: float = 0.0
    contact_penalty_scale: float = 0.05
    # Upper bound on the per-step contact penalty (relu(force - threshold)) before
    # scaling. <= 0 disables the cap. Caps prevent rare hard-contact force spikes
    # from producing huge reward outliers that destabilize PPO.
    contact_penalty_cap: float = -1.0
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]


@configclass
class ForgePegInsert(PegInsert, ForgeTask):
    # PegInsert precedes ForgeTask in the MRO, so repeat overrides that also
    # exist on FactoryTask instead of silently inheriting Factory's zeros.
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.2


@configclass
class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = 0.05


@configclass
class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = 0.05


# =============================================================================
# Tactile Finger Forge Task Configurations
# =============================================================================

@configclass
class ForgePegInsertTac(PegInsertTac, ForgeTask):
    """Forge peg insertion task configured for tactile fingers."""
    contact_penalty_scale: float = 0.2


@configclass
class ForgeGearMeshTac(GearMeshTac, ForgeTask):
    """Forge gear mesh task configured for tactile fingers."""
    contact_penalty_scale: float = 0.05


@configclass
class ForgeNutThreadTac(NutThreadTac, ForgeTask):
    """Forge nut threading task configured for tactile fingers."""
    contact_penalty_scale: float = 0.05
