# Franka Robust Track — Development Plan

Task: `Isaac-Franka-Robust-Track-v0`

- Registration: `franka_robust_track/__init__.py`
- Config: `franka_robust_track_env_cfg.py`
- Env logic: `franka_robust_track_env.py`
- Controller: `../factory/factory_control.py` shared OSC/Jacobian-transpose torque path
- Launcher: `launchers/launch_rlgames_train_franka_robust_track.py`

This document tracks planned feature changes. Each item lists the current
behavior, the target behavior, and the concrete implementation touch points.

---

## Current implementation handoff: action packets and action representations

**Public step input**
- Tensor actions still work and keep the normal Gym/RL-wrapper path:
  `env.step(action_tensor)`.
- Direct eval/replay code can now pass an action packet:
  `env.step({"action": tensor, "action_rep": rep, "compliance": {...}})`.
- `action_rep` is one of:
  - `rel_ee_pose`: Forge fixed-object-relative normalized action.
  - `abs_ee_pose`: absolute fingertip pose `[x, y, z, qw, qx, qy, qz]`.
  - `delta_ee_pose`: normalized current-EE delta; rotation is axis-angle.
- `compliance` is optional and may contain `stiffness`, `damping`, and
  `clip_pose_target`.

**RobustTrack env behavior**
- `CtrlCfg.action_rep` defaults to `delta_ee_pose`.
- Tensor actions are interpreted using `cfg.ctrl.action_rep`.
- Packet `action_rep="delta_ee_pose"` uses the existing RobustTrack action path:
  position delta is scaled by `pos_action_threshold`; rotation delta is an
  axis-angle vector scaled by `rot_action_threshold`; target quat is
  `delta_quat * current_fingertip_quat`.
- Packet `action_rep="abs_ee_pose"` converts the absolute pose into the native
  normalized delta action inside `_pre_physics_step` using
  `factory_control.get_pose_error(..., rot_error_type="axis_angle")`, then
  reuses the same target-pose path.
- Packet `action_rep="rel_ee_pose"` is invalid for RobustTrack and raises.
- Packet compliance stiffness/damping can override `task_prop_gains` /
  `task_deriv_gains` for that step.

**Forge env behavior**
- `ForgeCtrlCfg.action_rep` defaults to `rel_ee_pose`.
- `use_delta_pose` has been removed from the Forge dispatch path; callers should
  configure `env.ctrl.action_rep`.
- `rel_ee_pose` is the old Forge default: normalized target pose relative to the
  fixed object/action frame.
- `abs_ee_pose` replaces the old external `compliance_pose_target` hook. The
  action tensor is the absolute fingertip pose; Forge calls
  `_apply_abs_ee_pose_action(...)` internally and can clip to
  `pos_threshold`/`rot_threshold`.
- `delta_ee_pose` is implemented natively in Forge to match RobustTrack
  semantics: current EE position plus scaled delta, and axis-angle delta
  quaternion multiplied by current EE quaternion.
- The old `compliance_pose_target`, `compliance_clip_pose_target`,
  `compliance_stiffness_override`, and `compliance_deriv_override` hook path was
  removed from Forge. Compliance now rides in the action packet.

**Eval/control integration**
- `ComplianceController.apply(...)` now returns an action packet descriptor
  instead of mutating env hooks.
- `HierarchicalTrackerController.apply(...)` returns an `abs_ee_pose` packet with
  compliance gains.
- `experiments/exp_planning.py` always calls `env.step(packet)` when compliance
  or a non-default action representation is needed.
- `scripts/track_dataset_tool_pose.py` sends dataset tool poses to RobustTrack as
  `abs_ee_pose` packets; the env converts them to `delta_ee_pose`.
- `scripts/track_dataset_tool_pose_forge.py` sends dataset tool poses to Forge
  as `abs_ee_pose` packets for direct replay, or `rel_ee_pose` packets for the
  Forge action path.
- `launchers/launch_eval_mdf.py` now emits
  `+experiment.interact.env.ctrl.action_rep="rel_ee_pose"` for normal Forge
  evals and uses representation-based compliance labels such as
  `rel_ee_pose_default`, `tool_abs_ee_pose_adaptive`, and
  `vt_abs_ee_pose_pred_dir`.

**Known cleanup left**
- `ComplianceController` still uses the internal name `exec_mode`; it now means
  "which policy output/source to package" rather than env execution. A future
  cleanup should replace it with explicit `source` + `action_rep` fields.
- Some older launcher names/comments may still be semantically tied to historical
  experiments, but local `use_delta_pose` usage was removed from the Forge eval
  and collection launchers touched in this refactor.

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

## Feature 7: Fix the OSC controller (missing task-space inertia)

**Motivation:** A feed-forward diagnostic (`launchers/launch_franka_robust_track_feedforward.py`,
which commands the next reference waypoint directly and plots target vs actual EE
pose) showed the end-effector oscillating heavily — ~100 mm even holding a
near-static target, and diverging to ~350 mm while tracking a moving reference.

**Root cause**
- The controller torque was a plain Jacobian-transpose PD:
  `τ = Jᵀ (Kp·e − Kd·ẋ)`. The task-space inertia matrix `Λ = (J M⁻¹ Jᵀ)⁻¹` was
  computed but only used for the nullspace term, never applied to the motion wrench.
- With `Λ` omitted, the closed-loop task dynamics are `Λ·ẍ = Kp·e − Kd·ẋ`, so the
  damping ratio is `ζ = Kd / (2√(Kp·Λ)) = 1/√Λ` (with `Kd = 2√Kp`). Since the
  Franka's task inertia `Λ` is several kg (>1), the loop is always under-damped;
  the critical-damping gains only hold if `Λ = 1`.
- Factory/forge use the same law but hide the problem (lower Kp=100, 120 Hz
  control, `ema_factor=0.2`, target clipped to a ±5 cm box, quasi-static contact,
  and a trained policy as an outer loop). Robust-track's high Kp=300, 60 Hz, no
  smoothing, fast free-space reference, evaluated open-loop, exposes it.

**Fix**
- `factory_control.compute_dof_torque` supports the full operational-space control
  law: premultiply the PD wrench by `Λ` (`τ = Jᵀ · Λ · (Kp·e − Kd·ẋ)`) so the task
  dynamics reduce to a unit mass (`ẍ = Kp·e − Kd·ẋ`) and `Kd = 2√Kp` is actually
  critical.
- `CtrlCfg.use_task_space_inertia: bool = True` (`franka_robust_track_env_cfg.py`),
  added to the `update_env_params` allow-list; passed through as `apply_task_inertia`.
  `False` reproduces the legacy Jacobian-transpose PD.
- `franka_robust_track_env._apply_factory_control` and `ForgeEnv.generate_ctrl_signals`
  both call `factory_control.compute_dof_torque`.

**Result:** open-loop feed-forward tracking went from ~100–350 mm oscillation to a
steady ~1–2 mm lag (the expected small lag of a critically-damped follower), with
orientation error decaying to ~0.

**Status:** [x] done

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
