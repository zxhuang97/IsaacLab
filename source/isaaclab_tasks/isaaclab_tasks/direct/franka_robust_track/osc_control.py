# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Operational-space controller for the Franka robust-track env.

This is a self-contained copy of the OSC torque law so changes here never affect
the shared factory/forge controller in
``isaaclab_tasks.direct.factory.factory_control`` (``ForgeEnv`` subclasses
``FactoryEnv``, which uses that module).

Difference from the legacy factory controller: this implements the full
operational-space control law (Khatib). The task-space PD wrench is premultiplied
by the task-space inertia matrix ``Λ = (J M⁻¹ Jᵀ)⁻¹`` so the closed-loop task
dynamics reduce to a unit mass (``ẍ = Kp·e − Kd·ẋ``). The critical-damping gains
``task_deriv_gains = 2·√task_prop_gains`` are then actually critical. The legacy
controller maps a raw ``Jᵀ(Kp·e − Kd·ẋ)`` wrench, whose closed loop still carries
the real (several-kg) task-space inertia and is therefore heavily under-damped —
that is what produces sustained end-effector oscillation for free-space tracking.
"""

import math

import torch

# Pure task-space pose-error math shared with factory_control (not the buggy torque
# law); reusing it does not couple the controller behaviors.
from isaaclab_tasks.direct.factory.factory_control import get_pose_error


def compute_dof_torque(
    cfg,
    dof_pos,
    dof_vel,
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    fingertip_midpoint_linvel,
    fingertip_midpoint_angvel,
    jacobian,
    arm_mass_matrix,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    task_prop_gains,
    task_deriv_gains,
    device,
    nullspace_joint_target=None,
    apply_task_inertia=True,
):
    """Compute Franka arm DOF torque to drive the fingertip toward a target pose.

    When ``apply_task_inertia`` is True (default) the task-space PD wrench is
    premultiplied by ``Λ = (J M⁻¹ Jᵀ)⁻¹`` (full OSC); set it False to fall back to
    the legacy Jacobian-transpose PD.
    """
    num_envs = cfg.scene.num_envs
    dof_torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)

    pos_error, axis_angle_error = get_pose_error(
        fingertip_midpoint_pos=fingertip_midpoint_pos,
        fingertip_midpoint_quat=fingertip_midpoint_quat,
        ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
        jacobian_type="geometric",
        rot_error_type="axis_angle",
    )
    delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

    # Task-space PD wrench: Kp * pose_error - Kd * ee_velocity (per-axis gains).
    ee_vel = torch.cat((fingertip_midpoint_linvel, fingertip_midpoint_angvel), dim=-1)
    task_wrench = task_prop_gains * delta_fingertip_pose - task_deriv_gains * ee_vel

    # Operational-space inertia Λ = (J M⁻¹ Jᵀ)⁻¹ (ETH eq. 3.86; geometric Jacobian).
    arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
    jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
    task_inertia = jacobian @ arm_mass_matrix_inv @ jacobian_T
    arm_mass_matrix_task = torch.inverse(task_inertia)
    j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv

    # Fail loud if the OSC inverses produced non-finite values: at (near-)singular
    # arm configurations `J M⁻¹ Jᵀ` is ill-conditioned and `torch.inverse` returns
    # inf/NaN, which would silently poison the torques -> sim state -> observations
    # -> the policy std (the downstream `normal expects std >= 0` crash).
    if not torch.isfinite(arm_mass_matrix_task).all():
        bad = ~torch.isfinite(arm_mass_matrix_task).reshape(num_envs, -1).all(dim=-1)
        det = torch.linalg.det(task_inertia)
        raise RuntimeError(
            f"OSC task-inertia inverse (J M^-1 J^T)^-1 is non-finite in {int(bad.sum())}/{num_envs} envs "
            f"(near-singular arm configuration). min|det(J M^-1 J^T)|={det.abs().min().item():.3e}. "
            f"This is the source of the downstream NaN, not the policy std."
        )

    # Full OSC: decouple + normalize the task dynamics to unit mass so the
    # critical-damping gains hold (see module docstring).
    if apply_task_inertia:
        task_wrench = (arm_mass_matrix_task @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # Map the task wrench into joint torques: tau = J^T * wrench.
    dof_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

    # Nullspace posture control: bias the redundant DoF toward the posture target
    # without disturbing the task-space motion.
    if nullspace_joint_target is not None:
        default_dof_pos_tensor = nullspace_joint_target.to(device)
    else:
        default_dof_pos_tensor = torch.tensor(cfg.ctrl.default_dof_pos_tensor, device=device).repeat((num_envs, 1))
    distance_to_default_dof_pos = default_dof_pos_tensor - dof_pos[:, :7]
    distance_to_default_dof_pos = (distance_to_default_dof_pos + math.pi) % (2 * math.pi) - math.pi
    u_null = cfg.ctrl.kd_null * -dof_vel[:, :7] + cfg.ctrl.kp_null * distance_to_default_dof_pos
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    torque_null = (torch.eye(7, device=device).unsqueeze(0) - jacobian_T @ j_eef_inv) @ u_null
    dof_torque[:, 0:7] += torque_null.squeeze(-1)

    if not torch.isfinite(dof_torque).all():
        bad = ~torch.isfinite(dof_torque).all(dim=-1)
        raise RuntimeError(
            f"OSC produced non-finite joint torques in {int(bad.sum())}/{num_envs} envs "
            f"before clamping. This is a controller-side NaN, not the policy std."
        )

    dof_torque = torch.clamp(dof_torque, min=-100.0, max=100.0)
    return dof_torque, task_wrench
