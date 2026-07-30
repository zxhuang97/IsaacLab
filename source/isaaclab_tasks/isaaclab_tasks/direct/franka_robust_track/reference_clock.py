"""Pure dual-clock indexing helpers for robust trajectory tracking."""

from __future__ import annotations

import torch


def policy_step_to_reference_index(
    policy_step: torch.Tensor, policy_steps_per_reference: int
) -> torch.Tensor:
    """Map policy counters to native reference indices with zero-order hold."""
    if policy_steps_per_reference < 1:
        raise ValueError("policy_steps_per_reference must be positive")
    return torch.div(
        policy_step, policy_steps_per_reference, rounding_mode="floor"
    )


def reference_coordinates(
    policy_step: torch.Tensor,
    *,
    policy_decimation: int,
    reference_decimation: int,
    completed_physics_substeps: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return native endpoints and phase at a policy/physics clock instant."""
    if policy_decimation < 1:
        raise ValueError("policy_decimation must be positive")
    if reference_decimation < policy_decimation:
        raise ValueError("reference_decimation must be at least policy_decimation")
    if reference_decimation % policy_decimation:
        raise ValueError(
            "reference_decimation must be divisible by policy_decimation"
        )
    if not 0 <= completed_physics_substeps <= policy_decimation:
        raise ValueError(
            "completed_physics_substeps must be in [0, policy_decimation]"
        )
    total_physics_steps = (
        policy_step * policy_decimation + completed_physics_substeps
    )
    i0 = torch.div(
        total_physics_steps, reference_decimation, rounding_mode="floor"
    )
    remainder = torch.remainder(total_physics_steps, reference_decimation)
    phase = remainder.float() / float(reference_decimation)
    return i0, i0 + 1, phase
