"""Small tensor utilities for RobustTrack's geometry-free contact model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_force_directions(
    force: torch.Tensor,
    window: int,
    valid_force_threshold: float,
) -> torch.Tensor:
    """Return temporally smooth unit force directions for complete trajectories.

    Args:
        force: ``(batch, time, 3)`` world-frame force trajectories.
        window: Odd centered moving-average window. ``1`` disables smoothing.
        valid_force_threshold: A smoothed force below this magnitude does not
            provide a trustworthy direction. Such samples hold the most recent
            valid direction; samples before first contact use the first valid
            direction in the trajectory.

    The complete reference trajectory is known at reset, so a centered filter
    avoids the phase lag of a causal EMA. Only direction is returned: callers can
    retain the original demonstrated force magnitude.
    """
    if force.ndim != 3 or force.shape[-1] != 3:
        raise ValueError(f"force must have shape (batch, time, 3), got {tuple(force.shape)}")
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    if valid_force_threshold < 0.0:
        raise ValueError("valid_force_threshold must be non-negative")

    if window == 1:
        force_smooth = force
    else:
        # Replication padding preserves sequence length without introducing zero
        # vectors at episode boundaries. avg_pool1d operates on (N, C, T).
        force_channels = force.transpose(1, 2)
        pad = window // 2
        force_smooth = F.avg_pool1d(
            F.pad(force_channels, (pad, pad), mode="replicate"),
            kernel_size=window,
            stride=1,
        ).transpose(1, 2)

    magnitude = torch.linalg.norm(force_smooth, dim=-1)
    candidate = force_smooth / magnitude.unsqueeze(-1).clamp(min=1.0e-6)
    valid = magnitude > float(valid_force_threshold)

    batch_size, sequence_length, _ = force.shape
    default = torch.zeros((batch_size, 3), dtype=force.dtype, device=force.device)
    default[:, 2] = 1.0
    has_valid = valid.any(dim=1)
    first_valid_idx = valid.long().argmax(dim=1)
    batch_idx = torch.arange(batch_size, device=force.device)
    direction = torch.where(has_valid.unsqueeze(-1), candidate[batch_idx, first_valid_idx], default)

    directions = torch.empty_like(force)
    for step in range(sequence_length):
        direction = torch.where(valid[:, step].unsqueeze(-1), candidate[:, step], direction)
        directions[:, step] = direction
    return directions


def construct_reference_plane_trajectory(
    reference_position: torch.Tensor,
    target_force: torch.Tensor,
    plane_stiffness: float | torch.Tensor,
    direction_smoothing_window: int,
    direction_force_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct one virtual plane per reference timestep.

    The plane normal is the smoothed target-force direction. For outward normal
    ``n`` and plane point ``p``, penetration at reference position ``x_ref`` is
    ``relu(n dot (p - x_ref))``. Therefore choosing

    ``p = x_ref + (|F_goal| / k_p) * n``

    makes the spring-only contact law generate exactly ``|F_goal| * n`` when the
    end effector perfectly follows ``x_ref``. ``k_p`` is the fixed environment
    stiffness and is deliberately independent of the robot/controller stiffness.

    Returns ``(surface_point, surface_normal, aligned_target_force)``.
    """
    if reference_position.shape != target_force.shape or reference_position.shape[-1] != 3:
        raise ValueError(
            "reference_position and target_force must have matching (batch, time, 3) shapes; "
            f"got {tuple(reference_position.shape)} and {tuple(target_force.shape)}"
        )
    stiffness = torch.as_tensor(
        plane_stiffness, dtype=reference_position.dtype, device=reference_position.device
    )
    if torch.any(stiffness <= 0.0):
        raise ValueError("plane_stiffness must be positive")
    if stiffness.ndim == 0:
        stiffness = stiffness.view(1, 1, 1)
    elif stiffness.ndim == 1:
        stiffness = stiffness.view(-1, 1, 1)
    elif stiffness.ndim == 2:
        stiffness = stiffness.unsqueeze(1)

    normal = smooth_force_directions(
        target_force,
        window=direction_smoothing_window,
        valid_force_threshold=direction_force_threshold,
    )
    force_magnitude = torch.linalg.norm(target_force, dim=-1, keepdim=True)
    penetration = force_magnitude / stiffness
    surface_point = reference_position + penetration * normal
    aligned_target_force = force_magnitude * normal
    return surface_point, normal, aligned_target_force


def finite_difference_surface_velocity(
    surface_point: torch.Tensor,
    timestep: float,
) -> torch.Tensor:
    """Estimate per-timestep virtual-plane velocity from its point trajectory.

    Interior samples use centered differences; the first and last samples use
    one-sided differences. The resulting control-grid velocity can be held across
    all physics substeps belonging to that control step.
    """
    if surface_point.ndim != 3 or surface_point.shape[-1] != 3:
        raise ValueError(
            f"surface_point must have shape (batch, time, 3), got {tuple(surface_point.shape)}"
        )
    timestep = float(timestep)
    if timestep <= 0.0:
        raise ValueError("timestep must be positive")

    velocity = torch.zeros_like(surface_point)
    sequence_length = surface_point.shape[1]
    if sequence_length <= 1:
        return velocity
    velocity[:, 0] = (surface_point[:, 1] - surface_point[:, 0]) / timestep
    velocity[:, -1] = (surface_point[:, -1] - surface_point[:, -2]) / timestep
    if sequence_length > 2:
        velocity[:, 1:-1] = (surface_point[:, 2:] - surface_point[:, :-2]) / (2.0 * timestep)
    return velocity


def compute_virtual_plane_contact(
    position: torch.Tensor,
    linear_velocity: torch.Tensor,
    surface_point: torch.Tensor,
    surface_normal: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    transition_width: float,
    force_cap: float,
    surface_velocity: torch.Tensor | None = None,
    contact_model: str = "linear",
    hunt_crossley_dissipation: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed plane distance and a selectable unilateral contact force.

    ``surface_normal`` points out of the solid half-space. Positive signed
    distance is free space; negative distance is penetration. Damping increases
    the reaction while moving into the surface and decreases it while separating,
    but is gated by penetration so it cannot create a pre-contact force.

    ``linear`` uses ``f_n = max(0, k*delta + c*delta_dot)``. The Drake-style
    ``drake_hunt_crossley`` model uses
    ``f_n = k*delta*max(0, 1 + d*delta_dot)``. Here
    ``delta_dot = n dot (v_surface - v_contact)`` is positive during compression.
    """
    if contact_model not in ("linear", "drake_hunt_crossley"):
        raise ValueError(
            "contact_model must be 'linear' or 'drake_hunt_crossley', "
            f"got {contact_model!r}"
        )
    if hunt_crossley_dissipation < 0.0:
        raise ValueError("hunt_crossley_dissipation must be non-negative")
    if surface_velocity is None:
        surface_velocity = torch.zeros_like(linear_velocity)

    distance = ((position - surface_point) * surface_normal).sum(dim=-1, keepdim=True)
    penetration_raw = -distance
    penetration = torch.relu(penetration_raw)
    penetration_rate = ((surface_velocity - linear_velocity) * surface_normal).sum(
        dim=-1, keepdim=True
    )
    spring_force = stiffness * penetration
    if contact_model == "linear":
        force_magnitude = torch.relu(spring_force + damping * penetration_rate)
    else:
        dissipation_factor = torch.relu(
            1.0 + float(hunt_crossley_dissipation) * penetration_rate
        )
        force_magnitude = spring_force * dissipation_factor
    if transition_width > 0.0:
        transition = (penetration / float(transition_width)).clamp(0.0, 1.0)
        # C1-continuous smoothstep: zero value/slope at first contact and unit
        # value/zero slope once penetration reaches the transition width.
        activation = transition.square() * (3.0 - 2.0 * transition)
        force_magnitude = force_magnitude * activation
    force_magnitude = torch.where(
        penetration_raw > 0.0,
        force_magnitude,
        torch.zeros_like(force_magnitude),
    )
    force_magnitude = force_magnitude.clamp(max=float(force_cap))
    return distance, force_magnitude * surface_normal
