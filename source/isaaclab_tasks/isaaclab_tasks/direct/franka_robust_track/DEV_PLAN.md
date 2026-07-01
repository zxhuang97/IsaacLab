# Franka Robust Track — Development Plan

Task: `Isaac-Franka-Robust-Track-v0`

- Registration: `franka_robust_track/__init__.py`
- Config: `franka_robust_track_env_cfg.py`
- Env logic: `franka_robust_track_env.py`
- Controller: `../factory/factory_control.py`
- Launcher: `launchers/launch_rlgames_train_franka_robust_track.py`

This document tracks planned feature changes. Each item lists the current
behavior, the target behavior, and the concrete implementation touch points.

---

## Feature 1: Track a trajectory instead of a single pose

**Motivation:** Track a moving, time-parameterized reference instead of a single
static pose.

**Done**
- `TrackingCfg` (`franka_robust_track_env_cfg.py`): `mode` (`line`/`circle`),
  `line_speed_range`, `line_length_range`, `circle_radius_range`,
  `circle_speed_range`, `rot_speed_range`, `rot_angle_range`, plus lookahead
  params `num_future_steps`, `future_step_dt`.
- Per-env trajectory params sampled at reset in `_sample_trajectory`, anchored at
  the reset fingertip pose so `t=0` = start pose:
  - **line:** randomized unit direction, speed, capped displacement length.
  - **circle:** randomized orthonormal plane basis (u, v via Gram-Schmidt),
    radius, signed angular speed (cw/ccw); `pos(t) = start + r((cosθ−1)u + sinθ v)`.
  - shared orientation sweep: random axis, sweep speed, capped angle.
- `_command_pose_at(t)` evaluates the reference at any time; `_update_command`
  uses it for the current target (reward), and `_future_command_errors` uses it
  for the lookahead observation.
- **Lookahead observation:** policy now sees `num_future_steps` reference poses
  spaced `future_step_dt` apart (index 0 = current target), each as a
  (pos_error, axis_angle_error) pair relative to the current EE — so it can infer
  the reference velocity. `observation_space`/`state_space` are recomputed in
  `__post_init__` (default k=4 → obs 57, state 80).

**Status:** [x] done (line + circle + future-pose lookahead)

---

## Feature 2: Robustness to imprecise link mass

**Motivation:** Real link masses are uncertain. Train the policy to be robust to
mass/inertia mismatch. Two candidate implementations:

**Option A — constant random force per link**
- Sample a fixed random 3D force (and/or torque) per link at reset, apply it as
  an external wrench every physics step for the episode.
- Cheap, keeps `disable_gravity=True`, decoupled from mass scaling.

**Option B — turn on gravity + randomize mass**
- Set `disable_gravity=False` on the robot rigid bodies
  (`franka_robust_track_env_cfg.py:137`).
- Randomize link masses (already partially present via
  `randomization.enable_link_mass`) so gravitational load varies per env.
- More physically faithful; the OSC controller must now compensate real gravity
  torques, which stresses the controller/policy more realistically.

**Current behavior**
- Gravity disabled on robot (`disable_gravity=True`).
- Link mass scaling and payload randomization exist but only affect inertial
  dynamics, not static gravity load (`_randomize_dynamics`).

**Decision needed:** pick A, B, or support both behind a flag.

**Implementation notes**
- Option A: add per-link external force buffers, apply via the articulation's
  external-wrench API each step; add ranges to `RandomizationCfg`.
- Option B: flip `disable_gravity`, verify OSC gravity handling / nullspace
  behavior, ensure critic obs still exposes mass-related privileged info.

**Status:** [ ] not started — needs decision on A vs B

---

## Suggested ordering

1. Feature 2 (mass robustness) — after tracking works; decide A vs B.

## Completed

- Feature 3: removed `pos_action_bounds` (clamp + `CtrlCfg` + `update_env_params`).
- Feature 4: `sim.dt = 1/60`, `decimation = 4` (policy stays 15 Hz).
- Feature 5: removed the resample mechanism. `_sample_commands` /
  `command_step_counter` and the `_get_rewards` resample block are gone; the
  trajectory is now sampled once per reset and followed for the whole episode
  (folded into Feature 1).
