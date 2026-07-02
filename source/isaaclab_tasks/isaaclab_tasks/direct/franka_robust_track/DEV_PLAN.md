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

## Feature 6: Richer joint-friction model (sysid-aligned)

**Motivation:** The current friction randomization uses a single Coulomb
coefficient per joint, but real arm joints exhibit static (breakaway), velocity-
independent kinetic, and velocity-proportional (viscous) losses plus reflected
rotor inertia. Match the physical model UWLab identifies in
`UWLab/scripts_v2/tools/sim2real/sysid_ur5e_osc.py` so the friction disturbance
the policy trains against is representative rather than a coarse Coulomb-only
proxy.

**Reference model (UWLab sysid, 25 params for a 6-DoF arm)**
- Per joint: `armature`, `static_friction`, `dynamic_ratio`, `viscous_friction`.
- `dynamic_friction = dynamic_ratio * static_friction` (ratio in `[0, 1]`).
- One shared `motor_delay` (physics steps), applied via a `DelayedPD` actuator's
  position/velocity/effort delay buffers.
- Applied in sim via `write_joint_armature_to_sim(...)` and
  `write_joint_friction_coefficient_to_sim(static, joint_dynamic_friction_coeff=...,
  joint_viscous_friction_coeff=...)`.

**Current behavior (`franka_robust_track_env.py`)**
- `_randomize_dynamics` (line ~970): samples one coefficient per joint,
  `joint_friction ~ U(joint_friction_range)` with `joint_friction_range = [0.0, 5.0]`
  (`franka_robust_track_env_cfg.py:134-135`), 7 joints.
- Writes only the positional arg of `write_joint_friction_coefficient_to_sim`
  (line ~976) → dynamic & viscous coefficients stay at PhysX defaults.
- Cached in `self.joint_friction` (`(num_envs, 7)`), fed to the critic obs
  (line ~416) and logged as `Dynamics/joint_friction` (line ~1183).
- Critic privileged block is 23 dims: `payload_mass(1)+payload_com(3)+
  joint_friction(7)+task_gains(6)+pos_threshold(3)+rot_threshold(3)`
  (`franka_robust_track_env_cfg.py:296`).

**Target behavior**
- Replace the single-coefficient friction with the 3-component model
  (static + dynamic_ratio + viscous) and add per-joint `armature`.
- Keep it as absolute-range domain randomization for now (no FR3 sysid data yet):
  sample each component independently per joint per reset. If/when a real FR3
  sysid is run, switch to `nominal * U(0.8, 1.2)` centering (UWLab style) and
  optionally add an ADR curriculum — tracked as a follow-up, not this feature.
- Motor delay is **optional/deferred**: it requires the arm to use a
  `DelayedPDActuatorCfg` with a nonzero `max_delay` so the actuator exposes
  `positions/velocities/efforts_delay_buffer`. Confirm the current actuator type
  first; if the env uses an implicit/OSC-effort actuator, adding delay is a
  separate change and should not block the friction refactor.

**Implementation touch points**
- `RandomizationCfg` (`franka_robust_track_env_cfg.py:134-135`): replace
  `joint_friction_range` with
  - `enable_joint_friction: bool`
  - `static_friction_range = [0.0, 5.0]`
  - `dynamic_ratio_range = [0.0, 1.0]`
  - `viscous_friction_range = [0.0, 5.0]`
  - `enable_joint_armature: bool`, `joint_armature_range = [0.0, ...]`
  and update the `update_env_params` allow-list (lines ~373-374) accordingly.
- `_randomize_dynamics` (`franka_robust_track_env.py` ~970-980): sample the 3
  friction components + armature `(len(env_ids), 7)` each; call
  `write_joint_friction_coefficient_to_sim(static, joint_dynamic_friction_coeff=
  dynamic_ratio*static, joint_viscous_friction_coeff=viscous, joint_ids=
  self.arm_joint_ids, env_ids=env_ids)` and `write_joint_armature_to_sim(...)`.
- Buffers: replace `self.joint_friction (N,7)` with
  `self.joint_static_friction`, `self.joint_dynamic_ratio`,
  `self.joint_viscous_friction`, `self.joint_armature` (each `(N,7)`), all
  initialized in `__init__` next to the existing dynamics buffers (line ~125).
- Critic obs (`_get_observations`, concat at line ~413-419): swap the single
  `joint_friction` term for the new per-joint terms. Privileged block grows from
  23 → e.g. 23 - 7 + 28 = 44 dims (static+ratio+viscous+armature). Recompute
  `state_space` in `__post_init__` and update the dim comment at
  `franka_robust_track_env_cfg.py:293-297`.
- Logging (`_reset_idx` extras, line ~1183): log means of each new component.

**Status:** [ ] not started — friction model refactor; motor-delay + sysid
centering are follow-ups gated on actuator type and real FR3 data.

---

## Suggested ordering

1. Feature 2 (mass robustness) — after tracking works; decide A vs B.
2. Feature 6 (richer joint-friction model) — independent of Feature 2; can land
   first since it only touches `_randomize_dynamics` + critic obs.

## Completed

- Feature 3: removed `pos_action_bounds` (clamp + `CtrlCfg` + `update_env_params`).
- Feature 4: `sim.dt = 1/60`, `decimation = 4` (policy stays 15 Hz).
- Feature 5: removed the resample mechanism. `_sample_commands` /
  `command_step_counter` and the `_get_rewards` resample block are gone; the
  trajectory is now sampled once per reset and followed for the whole episode
  (folded into Feature 1).
