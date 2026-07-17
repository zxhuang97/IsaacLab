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
- `HierarchicalTrackerController.apply(...)` returns a `delta_ee_pose` packet with
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

## Dataset tool-pose replay parity: RobustTrack vs. Forge

**Scope:** Make `scripts/track_dataset_tool_pose.py` and
`scripts/track_dataset_tool_pose_forge.py` execute the saved
`0716_peg_track_his8_fut4_task_gain40_120_0708_fut_ref` tracker with the same
reference indexing, horizon, controller inputs, plant model, initialization,
and diagnostics. Evaluate with the held and fixed objects moved away so contact
cannot explain controller-only differences.

**Dataset / evaluation fixture**
- Dataset: `raw_datasets/0708_peg_speed_exp10_zero_noise_flat`
- Trajectories: 0--49
- Saved run:
  `data/rlgames_train/0716_peg_track_his8_fut4_task_gain40_120_0708_fut_ref`
- Raw demonstration length: 141 poses. Saved tracker reference length: 153
  poses. Executable transitions: 149 because the controller consumes a
  four-step future reference.
- Contact-free Forge fixture: held offset `[+0.5, 0.0, +0.2]` m and fixed
  offset `[+0.5, 0.0, +0.2]` m, while preserving the original action frame.
  The earlier negative fixed-x offset placed the hole near the robot base and
  could contact trajectory 49's elbow branch; it is not a valid away fixture.

**Measured baseline before the remaining fixes**
- RobustTrack: `0.182 +/- 0.335 mm`, `0.484 +/- 1.221 deg`, `98.0%`.
- Forge, both objects away: `0.527 +/- 3.683 mm`,
  `0.911 +/- 2.786 deg`, `97.61%`.
- The apparent Forge gap is dominated by sample 148: all 50 Forge envs reset
  automatically before that sample is measured, producing 24--62 mm position
  errors. Excluding the terminal sample gives:
  - RobustTrack: `0.1832 +/- 0.3362 mm`, `0.4870 +/- 1.2248 deg`, `98.027%`.
  - Forge, both away: `0.2323 +/- 0.3015 mm`,
    `0.7198 +/- 1.0917 deg`, `98.270%`.
- Therefore the contact-free residual is approximately `0.049 mm` mean
  position error. Forge has slightly higher threshold success after excluding
  the invalid reset sample.
- Moving only the held object away and moving both objects away are nearly
  identical (`0.23210` vs. `0.23232 mm` without the terminal sample). This
  confirms that the fixed object is not the source once the held object is
  removed.
- Wrist incoming-joint-force diagnostic with both objects away: raw force
  median `0.027 N`, mean `0.103 N`, p95 `0.512 N`, max `1.215 N`; smoothed
  force mean `0.130 N`, p95 `0.659 N`, max `2.071 N`. These are small startup /
  inertial reactions rather than sustained collision loads.

### P0. Future-reference and metric indexing

**Required invariant**
- With current pose `P_t`, the policy observation starts at
  `P_(t + reference_start_offset)` and the post-step tracking metric compares
  the reached pose against `P_(t + 1)`.
- Both scripts must resample the raw 141-pose demonstration onto the saved
  153-pose reference grid and execute the same 149 transitions.

**Status:** [x] fixed and unit-tested.
- The tracker now retains `reference_start_offset` and
  `max_episode_length` from the saved run.
- Full-reference inputs are marked as beginning at the current pose, avoiding
  a second implicit offset.
- Forge now uses the saved reference grid instead of truncating the raw data.
- Native traces now store the post-step command target rather than the
  pre-step policy input.

### P1. Prevent terminal auto-reset from entering tracking metrics

**July 17 revision:** The timeout-extension implementation documented below
has been reverted. Evaluation now retains the saved 10 s / 150-step episode and
executes the raw dataset length of 141 control steps. The 141 raw poses keep
their original indices; future lookahead is supplied only by copies of the
final pose. Because 141 is below DirectRLEnv's internal reset boundary,
evaluation does not encounter a timeout and does not mutate episode length.
The current expected metadata is `steps=141`, `max_episode_length=150`,
`done_count=0`, and `terminal_reset_observed=false`.

Fresh 50-trajectory validation satisfies that contract. Native Mimic reports
`0.189862 mm`, `0.485032 deg`, `98.1844%`; paired Forge Mimic reports
`0.196184 mm`, `0.484952 deg`, `98.1844%`. Both report `steps=141`,
`max_episode_length=150`, `done_count=0`, and no terminal reset. Artifacts:
`data/dataset_tool_pose_track/0717_raw141_0716_policy_native_mimic_gain80_traj000000_n50`
and
`data/dataset_tool_pose_track_forge/0717_raw141_0716_policy_forge_mimic_both_away_gain80_traj000000_n50`.

The remainder of this subsection records the superseded 149-step investigation
because it explains why the old final sample was invalid.

**Root cause:** Replay attempted to override `env.episode_length_s` in the
saved variant, but `FactoryEnvCfg.update_env_params()` did not consume that
field. DirectRLEnv resets timed-out environments inside `step()`, before either
script reads the post-step state.

**Fixes**
- [x] Allow `FactoryEnvCfg.update_env_params()` to apply an explicit
  `env.episode_length_s` override.
- [x] Extend both replay environments by at least one control period before
  construction, so all 149 post-step states are observed before reset.
- [x] Preserve the saved 153-pose dataset resampling grid while padding the
  runtime-only extra lookahead slot with the last pose.
- [x] Record the configured timeout and expected reference horizon in the
  result metadata.
- [x] Add a regression test proving the final measured state is not a reset
  state.

**Acceptance:** 149 valid transitions, no timeout/reset at sample 148, and no
terminal error spike in either trace.

**Status:** [x] complete. One-trajectory runtime validation:
- RobustTrack artifact:
  `data/dataset_tool_pose_track/0717_p1_timeout_grid153_traj000000_n1`.
- Forge both-away artifact:
  `data/dataset_tool_pose_track_forge/0717_p1_timeout_both_away_traj000000_n1`.
- Both report `steps=149`, `max_episode_length=151`, `done_count=0`, and
  `terminal_reset_observed=false`. Final position errors are `0.0195 mm`
  (RobustTrack) and `0.0762 mm` (Forge), with no spike.
- Their 149 metric targets still match: maximum position delta
  `9.31e-10 m`, maximum quaternion `1-|dot| = 5.96e-8`.

### P2. Make force diagnostics explicit and correctly named

**Current pitfall:** `applied_wrench` is the OSC commanded task wrench. It is
not measured contact force and must not be used as collision evidence.

**Fixes**
- [x] Save Forge `force_sensor_world` and `force_sensor_world_smooth` in the
  trace as wrist incoming-joint wrench measurements.
- [x] Add force/torque mean, p95, and max summaries to stats.
- [x] Label these as wrist joint reaction measurements; collision-specific
  attribution requires a link contact sensor and is a separate diagnostic.
- [x] Add tests for trace shape, units, summary reduction, and protection from
  PhysX's in-place tensor updates.

**Acceptance:** A replay result distinguishes commanded wrench, wrist reaction
wrench, and contact state; both-away force remains at the small inertial
baseline above without sustained spikes.

**Status:** [x] complete. Validated artifact:
`data/dataset_tool_pose_track_forge/0717_p2_force_v2_both_away_traj000000_n1`.
Its trace contains separate `applied_wrench`, `force_sensor_world`, and
`force_sensor_world_smooth` arrays with shape `(149, 1, 6)`. Raw wrist reaction
force is mean `0.101 N`, p95 `0.542 N`, max `1.211 N`; raw torque is mean
`0.0378 Nm`, p95 `0.1128 Nm`, max `0.3112 Nm`. The stats exactly reproduce the
trace reduction and remain consistent with contact-free startup/inertial load.

### P3. Align task-space gain selection

**Current mismatch:** The saved RobustTrack configuration samples a scalar
task proportional gain in `[40, 120]` at reset. Forge currently uses a fixed
gain of `80`. The policy observes the gain, so this changes both its input and
plant response. The observed correlation is weak, making this a secondary but
real parity issue.

**Fixes**
- [x] Add an explicit deterministic fixed-gain replay option to RobustTrack:
  `--fixed-tracker-gain`.
- [x] Keep Forge's `--task-inertia-gain` value in result metadata and validate
  that it reaches the action packet unchanged.
- [x] Use `80` in both scripts for paired simulator-parity runs; retain saved
  randomization as a separate training-distribution evaluation mode.
- [x] Assert from traces and stats that both controllers use `[80] * 6`.

**Acceptance:** Identical per-step proportional and derivative gains in paired
runs, with the same gain vector in policy observation and controller command.

**Status:** [x] complete. Native artifact
`data/dataset_tool_pose_track/0717_p3_gain80_traj000000_n1` reports actual
gain min=max=`[80] * 6` over all 149 steps. Forge smoke artifact
`data/dataset_tool_pose_track_forge/0717_p3_gain80_smoke_traj000000_n1` reports
the same actual min/max, and the full P2 trace also contains `[80] * 6` at every
step. Fixed mode disables reset-time gain scale/noise in native replay; omitting
the new flag retains the saved `[40, 120]` distribution.

### P4. Align the robot, gripper, and nominal dynamics model

**Current mismatch**
- RobustTrack uses `franka_mimic.usd`; Forge defaults to
  `franka_gelsight_mini_assembled_z13_x10.usd`.
- Forge includes compliant Gelsight fingers / force-sensor links and commands
  gripper target `0.0`; RobustTrack commands finger target `0.04`.
- With `use_gt_mass_matrix=false`, each environment reconstructs a nominal
  mass matrix from its own loaded USD. Equal gains therefore do not imply equal
  torque or motion when masses and inertias differ.

**Fixes**
- [x] Add a replay robot profile that selects the same USD, finger type,
  compliant-gripper setting, and gripper target in both environments.
- [x] Remove hard-coded gripper targets from the replay path; carry the target
  through replay configuration/action compliance.
- [x] Log the resolved USD, tool body, joint/body IDs, gripper target, and
  nominal-vs-ground-truth mass-matrix mode.
- [x] Smoke-test both candidate parity directions: Forge with the mimic asset,
  and RobustTrack with the Gelsight asset. Select the direction compatible with
  all required bodies/sensors and the trained policy assumptions.

**Acceptance:** Same articulated asset and gripper command; initial one-step
EE displacement difference is below `0.1 mm` under an identical action packet.

**Status:** [x] complete. Selected parity direction: Forge
`--robot-profile native-mimic`.
It resolves to `franka_mimic.usd`, `finger_type=franka`, compliant gripper off,
and gripper target `0.04`. Forge accepts this asset and falls back to the
selected tool body if a dedicated force body is absent (this USD does contain
`force_sensor`). Full artifact
`data/dataset_tool_pose_track_forge/0717_p4_native_mimic_both_away_traj000000_n1`
improves the one-trajectory Forge result from `0.172/0.429` to
`0.128 mm/0.133 deg`; paired native fixed-gain result is
`0.095 mm/0.144 deg`. Reverse-direction Gelsight RobustTrack smoke also runs,
but it changes the plant seen during training and produces larger early error,
so it was not selected.

The remaining first-step mismatch was the legacy Forge kinematic controller
model: it averaged the left/right finger Jacobians and used the unshifted body
velocity, while RobustTrack controls the actual `panda_fingertip_centered`
origin with a COM-to-tool velocity/Jacobian shift. The native-mimic profile now
selects `tool_kinematics_mode=tool_body`. With P5's shared joint state and an
identical action, first-step EE divergence is `0.00050 mm` (previously
`0.371 mm`), satisfying the `0.1 mm` acceptance threshold.
The profile also disables the Forge table, which does not exist in
RobustTrack and is outside the paired plant model.

### P5. Align IK solution, initial state, history, and startup settling

**Current mismatch:** RobustTrack initializes via cuRobo IK seeded from reset
joints; Forge uses `set_pos_inverse_kinematics`. Redundant 7-DoF IK can yield
different joint configurations for the same fingertip pose, and joint position
is a policy input. There is also no controlled, identical settling interval
before the first tracker action.

**Fixes**
- [x] Save initial arm joints and fingertip pose in both traces and stats.
- [x] Provide a shared/replayed initial-joint path so paired runs begin from
  the same joint configuration, not merely the same tool pose.
- [x] Apply the same number of no-action physics settling steps before tracker
  history is reset.
- [x] Seed history only after all object moves, joint writes, and settling are
  complete.
- [x] Save and compare every policy observation/action; report initial and
  rollout deltas.

**Acceptance:** Initial joints and tracker histories match numerically; first
normalized action delta is below `1e-4` and first-step tool displacement agrees
within the P4 tolerance.

**Status:** [x] complete. Native source artifact:
`data/dataset_tool_pose_track/0717_p5_shared_init_source_traj000000_n1`;
paired Forge artifact:
`data/dataset_tool_pose_track_forge/0717_p5_full_parity_both_away_traj000000_n1`.
The pre-settle cuRobo joint command is handed to Forge through
`--initial-joints-npz`; all 9 settled joints match exactly, initial tool
position differs by `0.00055 mm`, first raw policy observation max delta is
`4.77e-7`, and first normalized action L2 delta is `1.89e-5`. Over 149 steps,
EE-position simulator delta is mean `0.0285 mm`, p95 `0.0864 mm`, max
`0.1194 mm`. Native/Forge tracking metrics are respectively
`0.0952/0.1090 mm` and `0.14411/0.14410 deg`; the residual mean tracking gap is
only `0.0138 mm`.

In the final 50-env batch, all initial joints match exactly and the initial EE
position differs by at most `0.00151 mm`. Initial observation max-absolute
delta is `1.43e-6`. The single-env first-action L2 delta meets the original
`1e-4` target (`1.89e-5`); the vectorized batch has mean `1.76e-4`, max
`5.20e-4` from policy sensitivity to micron-scale cloned-environment float
differences. This remains below `1e-3` and produces only micrometer-scale
first-state differences.

### P6. Re-run and close the parity investigation

- [x] Run unit tests for tracker indexing, replay-grid construction, timeout,
  force summaries, gain override, and initialization metadata.
- [x] Run 50-trajectory RobustTrack and Forge both-away evaluations with all
  parity options enabled.
- [x] Verify references match (final max position delta `5.96e-8 m` and
  quaternion `1-|dot| <= 1.19e-7`).
- [x] Compare per-step observations, actions, commanded wrench, wrist reaction
  wrench, and EE state; document any remaining irreducible engine difference.
- [x] Record final commands, artifact paths, metrics, and conclusion here.

**Final acceptance:** No reset-contaminated samples, no collision-like force in
the both-away fixture, matching controller inputs/configuration, and a residual
tool-position gap explained and bounded by a named simulator/model difference.

**Status:** [x] complete. Full test suite: `36 passed`.

**Final commands**

```bash
/home/zixuanh/miniforge3/envs/isaac_tac/bin/python -u \
  scripts/track_dataset_tool_pose.py \
  --dataset raw_datasets/0708_peg_speed_exp10_zero_noise_flat \
  --traj-index 0 --num-trajs 50 --policy rl \
  --run-dir data/rlgames_train/0716_peg_track_his8_fut4_task_gain40_120_0708_fut_ref \
  --device cuda:0 --fixed-tracker-gain 80 \
  --out-dir data/dataset_tool_pose_track/0717_final_parity_gain80_traj000000_n50

/home/zixuanh/miniforge3/envs/isaac_tac/bin/python -u \
  scripts/track_dataset_tool_pose_forge.py \
  --dataset raw_datasets/0708_peg_speed_exp10_zero_noise_flat \
  --traj-index 0 --num-trajs 50 --steps 149 --policy tracker \
  --run-dir data/rlgames_train/0716_peg_track_his8_fut4_task_gain40_120_0708_fut_ref \
  --device cuda:0 --task-inertia-gain 80 --robot-profile native-mimic \
  --initial-joints-npz \
    data/dataset_tool_pose_track/0717_final_parity_gain80_traj000000_n50/trace.npz \
  --initial-settling-steps 1 --move-held-away --move-fixed-away \
  --fixed-away-offset=0.5,0.0,0.2 --no-compare-rgb-video \
  --out-dir \
    data/dataset_tool_pose_track_forge/0717_final_parity_v4_true_both_away_traj000000_n50
```

**Final 50-trajectory results**
- RobustTrack artifact:
  `data/dataset_tool_pose_track/0717_final_parity_gain80_traj000000_n50`.
  Position `0.17170 +/- 0.31263 mm`; rotation
  `0.47050 +/- 1.18878 deg`; tracking success `98.1745%`.
- Forge artifact:
  `data/dataset_tool_pose_track_forge/0717_final_parity_v4_true_both_away_traj000000_n50`.
  Position `0.17872 +/- 0.31016 mm`; rotation
  `0.47058 +/- 1.18873 deg`; tracking success `98.1745%`.
- Both execute 149 transitions with `done_count=0`; gain min=max=`[80] * 6`;
  references match to `5.96e-8 m` / quaternion `1-|dot|=1.19e-7`.
- Cross-simulator EE-position delta over all 7,450 samples: mean
  `0.02421 mm`, p95 `0.06435 mm`, max `0.13391 mm`. Aggregate tracking mean
  differs by only `0.00702 mm`; rotation differs by `0.00008 deg`; success is
  identical.
- Contact-free wrist reaction: raw force mean `0.0397 N`, p95 `0.1370 N`, max
  `0.5580 N`; raw torque mean `0.0234 Nm`, p95 `0.0604 Nm`, max `0.2206 Nm`.
  No collision-like sustained load remains.

**Conclusion:** The large earlier Forge error was not policy future-reference
indexing or collision after the held object was removed. It was primarily a
terminal auto-reset artifact, followed by mismatched gain sampling, robot /
gripper model, legacy finger-averaged Jacobian/velocity convention, redundant
IK branch, and an invalid fixed-away direction that moved the hole toward the
base. With those fixed, the remaining `~0.007 mm` aggregate tracking difference
is bounded numerical divergence between two DirectRLEnv implementations that
now use the same USD, joint state, tool-frame/Jacobian convention, gains,
reference, timeout, policy, and action semantics.

### P1--P6 leave-one-fix-out ablation (50 trajectories)

**Method.** Each Forge ablation starts from the final contact-free parity
configuration and removes one fix while retaining the others. The comparison
uses the same 50 trajectories and 149 requested transitions. The reference
Native result is `0.171704 mm`, `0.470504 deg`, and `98.1745%`; final Forge is
`0.178721 mm`, `0.470583 deg`, and `98.1745%`, giving a baseline signed gap of
`+0.007017 mm`, `+0.000079 deg`, and `0.0000` percentage points.
These are leave-one-out effects in a nonlinear closed loop, so they identify
causal importance but must not be summed as an additive error decomposition.

| Ablation | Forge / paired Native position (mm) | Position gap (mm) | Rotation gap (deg) | Success gap (pp) | Main causal observation |
|---|---:|---:|---:|---:|---|
| Final parity | `0.178721 / 0.171704` | `+0.007017` | `+0.000079` | `0.0000` | Residual numerical divergence. |
| P1: no timeout extension | `0.479505 / 0.171704` | `+0.307801` | `+0.205572` | `-0.6577` | All 50 envs reset at sample 148. |
| P2: no wrist diagnostics or trace | `0.178721 / 0.171704` | `+0.007017` | `+0.000079` | `0.0000` | All six scalar rollout metrics are bit-for-bit unchanged. |
| P3: Native random `[40,120]`, Forge fixed `80` | `0.178721 / 0.180268` | `-0.001547` | `-0.023245` | `+0.2013` | Aggregate gap improves by accidental cancellation despite strongly mismatched policy inputs/actions. |
| P4: Forge Gelsight plant/profile | `0.341465 / 0.171704` | `+0.169761` | `+0.274769` | `-1.2617` | Largest genuine plant/controller-frame contribution. |
| P5a: independent Forge IK | `0.180210 / 0.171704` | `+0.008506` | `+0.001198` | `-0.0268` | Small score change, but joint/observation/action parity is lost. |
| P5b: wrong fixed offset `[-0.5,0,+0.2]` | `0.179057 / 0.171704` | `+0.007353` | `-0.000609` | `-0.0134` | Local trajectory-49 initialization collision/outlier is hidden by the aggregate mean. |
| P6: tests/reporting removed | unchanged by construction | `0` physical | `0` physical | `0` physical | P6 detects and documents errors; it is not in the control loop. |

**P1 detail.** Samples 0--147 are numerically the same as final Forge:
`0.179417 mm` and `0.472495 deg`. At sample 148 all 50 environments report
done/reset; the mean final-sample error is `44.8926 mm` and `30.8061 deg`, its
position range is `25.9237--63.8444 mm`, and final-sample tracking success is
zero. Thus P1 alone added `0.300784 mm` to the historical 149-step absolute
mean position gap. The timeout-extension option has since been removed; the
current 141-step replay never reaches that boundary.

**P2 detail.** The control disables both trace saving and all post-step wrist
tensor reads/copies using `--no-record-wrist-wrench --no-save-trace`. Position
mean/std, rotation mean/std, tracking success, and return are exactly identical
to final Forge. This confirms the diagnostic is observational. The normal
diagnostic remains useful: final raw wrist reaction is mean `0.0397 N`, p95
`0.1370 N`, max `0.5580 N`.

**P3 detail.** Native sampled scalar gains from `40.3703` to `119.6903`, with
mean `78.7791`; mean distance from Forge's gain 80 was `18.7504`. Relative to
Native fixed-80 replay, randomization increased Native position error by
`0.008564 mm`, rotation error by `0.023325 deg`, and reduced success by
`0.2013` pp. The smaller aggregate Native--Forge error gap is not improved
parity: cross-simulator EE delta rises from `0.02421` to `0.06764 mm`, p95 from
`0.06435` to `0.20245 mm`, and max from `0.13391` to `2.82172 mm`. First-action
L2 delta rises from mean `0.000176` to `0.12213`; its correlation with absolute
gain mismatch is `0.876`. Fixed gain therefore matters for a controlled paired
comparison even though the scalar mean error happens to cancel.

**P4 detail.** Reverting the complete Forge profile restores the Gelsight USD,
compliant/closed fingers, table, finger-averaged kinematics, and the associated
nominal dynamics model. Although the nine commanded joints still match, the
different controlled tool definition starts `17.9293 mm` from Native. First
action L2 delta is mean `2.2662`, max `2.7461`; cross-simulator EE delta is mean
`0.25992 mm`, p95 `0.39167 mm`, max `13.7537 mm`. The final profile/tool-body
fix reduces the genuine position-gap contribution by `0.162743 mm` and the
rotation gap by `0.274690 deg`. Objects were still moved away; raw wrist
reaction remained modest (mean `0.119 N`, p95 `0.437 N`, max `2.792 N`).

**P5a detail.** Independent Forge DLS IK differs from Native cuRobo IK by arm
joint L2 mean `0.2983 rad`, max `0.9128 rad`, while initial EE position remains
close (mean `0.0218 mm`, max `0.2029 mm`). Because joints are policy inputs,
initial observation max difference is `0.6890`, and first-action L2 difference
is mean `0.19068`, max `1.35672`. The policy/controller recovers well, so mean
position error increases by only `0.001489 mm` relative to the final absolute
gap. Nevertheless, cross-simulator EE delta nearly doubles from `0.02421` to
`0.04671 mm`, and its max rises to `2.4184 mm`; shared joints are required for
a causal simulator comparison.

**P5b detail.** Moving the fixed object by negative x puts the hole near the
robot base rather than away. Trajectory 49 is disturbed during shared-joint
settling: initial EE delta is `5.7146 mm`, initial observation max difference
is `0.07990`, and first-action L2 delta reaches `1.49333`. Across all 50 envs
the initial EE mean is only `0.1150 mm`, so aggregate tracking changes by just
`0.000336 mm`; this illustrates why the per-environment trace was necessary.
Rollout wrist force is unchanged because the short collision/constraint event
occurs during initialization before rollout force accumulation. The Forge
return change is not a tracking metric: moving the fixed task asset to the
opposite side changes Forge's task reward geometry.

**P6 detail.** P6 adds no runtime policy, controller, or physics operation.
Its numerical ablation is therefore zero. Its practical contribution is
verification: the relevant tracker/replay suite passes `32` tests after adding
the P1 ablation switch, and P6 is what exposed the reset sample, mutable PhysX
force buffer, resampling-grid regression, and initialization outlier.

**Ablation artifacts**

- P1: `data/dataset_tool_pose_track_forge/0717_ablation_p1_no_timeout_traj000000_n50`
- P2: `data/dataset_tool_pose_track_forge/0717_ablation_p2_no_wrist_diagnostics_traj000000_n50`
- P3 Native: `data/dataset_tool_pose_track/0717_ablation_p3_random_gain_traj000000_n50`
- P3 Forge: `data/dataset_tool_pose_track_forge/0717_ablation_p3_gain_mismatch_traj000000_n50`
- P4: `data/dataset_tool_pose_track_forge/0717_ablation_p4_forge_gelsight_traj000000_n50`
- P5a: `data/dataset_tool_pose_track_forge/0717_ablation_p5a_independent_ik_traj000000_n50`
- P5b: `data/dataset_tool_pose_track_forge/0717_ablation_p5b_wrong_fixed_away_traj000000_n50`

All Forge ablations used the historical final Forge command above. The
intentional command deltas were respectively: P1 disabling the then-current
timeout extension; P2
`--no-record-wrist-wrench --no-save-trace`; P3 uses the random-gain Native
trace for `--initial-joints-npz`; P4 `--robot-profile forge-gelsight`; P5a
omits `--initial-joints-npz`; and P5b uses
`--fixed-away-offset=-0.5,0.0,0.2`.

### Reverse parity direction: Gelsight Franka in RobustTrack

The opposite P4 direction was also evaluated: load the Gelsight Franka in
RobustTrack, then run Forge with the same Gelsight USD, shared Native joints,
tool-body pose/Jacobian convention, gain 80, gripper target `0.04`, no table,
and the same positive both-away fixture. Forge profile
`gelsight-native-track` retains the Gelsight body layout and compliant-gripper
setup while matching the controller frame and gripper command used by
RobustTrack.

The first attempt exposed a replay configuration bug: assigning
`env_cfg.robot_usd_path` after loading the saved config changed result metadata
but did not update the already-built `env_cfg.robot.spawn.usd_path`. The alleged
Native-Gelsight run still had mimic body names. Replay now updates both fields,
records `robot_spawn_usd_path`, and has a regression test. The corrected Native
body list contains `gelsight_finger`, `gelsight_finger_0`, elastomer tips, and
Gelsight sensor bodies.

**Corrected 50-trajectory results**

| Metric | Native Gelsight | Forge Gelsight | Signed Forge--Native gap |
|---|---:|---:|---:|
| Position error | `0.207721 +/- 0.284283 mm` | `0.211670 +/- 0.283249 mm` | `+0.003949 mm` |
| Rotation error | `0.721355 +/- 1.087901 deg` | `0.721428 +/- 1.088390 deg` | `+0.000073 deg` |
| Tracking success | `98.2685%` | `98.2685%` | `0.0000 pp` |

Both execute 149 transitions with no reset. Initial joints match exactly;
initial EE position differs by mean `0.000746 mm`, max `0.002054 mm`; initial
observation max difference is `1.91e-6`; first-action L2 delta is mean
`0.000180`, max `0.000594`. Cross-simulator EE delta is mean `0.02512 mm`, p95
`0.06734 mm`, max `0.17411 mm`. References match to `5.96e-8 m`.

The reverse position gap (`0.00395 mm`) is smaller than the mimic pair's
`0.00702 mm`; cross-simulator state divergence is essentially unchanged
(`0.02512` versus `0.02421 mm`). Swapping the asset changes the existing
mimic-trained policy's absolute result similarly in both simulators: Native
position/rotation change by `+0.03602 mm/+0.25085 deg`, and Forge changes by
`+0.03295 mm/+0.25084 deg`. The success change is the same (`+0.0940 pp`). This
shows the Gelsight direction is valid, while the absolute metric change is a
real asset/domain shift rather than a simulator-parity failure.

Contact-free Forge-Gelsight wrist reaction remains small: raw force mean
`0.1066 N`, p95 `0.3948 N`, max `1.8541 N`; raw torque mean `0.0501 Nm`, p95
`0.1281 Nm`, max `0.3746 Nm`.

**Artifacts**

- Native: `data/dataset_tool_pose_track/0717_reverse_gelsight_v2_actual_asset_gain80_traj000000_n50`
- Forge: `data/dataset_tool_pose_track_forge/0717_reverse_gelsight_v2_true_both_away_traj000000_n50`

**Configuration decision:** New `FrankaRobustTrackEnvCfg` and the peg-track
training launcher default to the Gelsight USD. The launcher exposes
`--robot-profile gelsight|mimic` (default `gelsight`) and adds `_gelsight` to
new run names. Existing saved mimic checkpoints retain their serialized asset,
and `--robot-profile mimic` explicitly reproduces old training runs. A new
Gelsight-trained checkpoint is needed before interpreting the absolute
Gelsight tracking score as in-distribution performance.

### Evaluation: 0717 Gelsight-trained tracker on both replay scripts

Policy:
`data/rlgames_train/0717_peg_gs_track_his8_fut4_task_gain40_120_0708_fut_ref_gelsight`.
The paired evaluation uses trajectories 0--49, fixed gain 80, 149 transitions,
the saved future-only reference contract, the actual Gelsight USD in both
simulators, shared Native initialization joints, and the positive both-away
Forge fixture.

| Metric | Native RobustTrack | Forge Gelsight | Signed Forge--Native gap |
|---|---:|---:|---:|
| Position error | `0.624688 +/- 3.755037 mm` | `0.661211 +/- 4.172114 mm` | `+0.036523 mm` |
| Rotation error | `3.736156 +/- 20.290155 deg` | `3.716128 +/- 20.217338 deg` | `-0.020028 deg` |
| Tracking success | `96.0671%` | `96.0671%` | `0.0000 pp` |

Both runs execute all 149 transitions with `done_count=0`, use gains exactly
`[80] * 6`, and have matching references (`5.96e-8 m`, quaternion
`1-|dot|=1.19e-7`). Initial joints match exactly; initial EE position differs
by mean `0.000746 mm`, max `0.002054 mm`; initial observation max difference is
`1.91e-6`; first-action L2 delta is mean `0.000160`, max `0.000503`.

The aggregate is dominated by the same three policy failures in both
simulators:

| Trajectory | First failed step Native/Forge | Native mean pos/rot | Forge mean pos/rot | Action behavior |
|---:|---:|---:|---:|---|
| 12 | `45 / 45` | `8.616 mm / 62.522 deg` | `11.967 mm / 62.196 deg` | Saturation begins at step 39; source demo is marked failed. |
| 39 | `42 / 42` | `13.531 mm / 74.159 deg` | `11.762 mm / 74.473 deg` | Saturated from step 0; source demo is marked successful. |
| 45 | `67 / 67` | `4.136 mm / 41.008 deg` | `3.993 mm / 40.018 deg` | Persistent saturation begins before failure; source demo is marked successful. |

Removing only those three environments, the remaining 47 trajectories give:

- Native: `0.105339 +/- 0.187127 mm`,
  `0.194005 +/- 0.287089 deg`, `100%` success.
- Forge: `0.113586 +/- 0.184759 mm`,
  `0.194030 +/- 0.287006 deg`, `100%` success.
- Cross-simulator EE delta: mean `0.024187 mm`, p95 `0.065659 mm`, max
  `0.142711 mm`.

Forge wrist reaction increases only after each policy has already diverged:
force first exceeds `2 N` one control step after the tracking failure for all
three trajectories. Overall raw force is mean `0.196 N`, p95 `0.744 N`, max
`33.346 N`; trajectory 12 supplies the maximum. Because the held/fixed objects
are away and Native fails at the same time without them, this load is a
consequence of the saturated unstable robot motion, not the cause of the
cross-simulator error.

**Commands**

```bash
/home/zixuanh/miniforge3/envs/isaac_tac/bin/python -u \
  scripts/track_dataset_tool_pose.py \
  --dataset raw_datasets/0708_peg_speed_exp10_zero_noise_flat \
  --traj-index 0 --num-trajs 50 --policy rl \
  --run-dir data/rlgames_train/0717_peg_gs_track_his8_fut4_task_gain40_120_0708_fut_ref_gelsight \
  --device cuda:0 --fixed-tracker-gain 80 \
  --out-dir data/dataset_tool_pose_track/0717_eval_0717_gs_policy_native_gain80_traj000000_n50

/home/zixuanh/miniforge3/envs/isaac_tac/bin/python -u \
  scripts/track_dataset_tool_pose_forge.py \
  --dataset raw_datasets/0708_peg_speed_exp10_zero_noise_flat \
  --traj-index 0 --num-trajs 50 --steps 149 --policy tracker \
  --run-dir data/rlgames_train/0717_peg_gs_track_his8_fut4_task_gain40_120_0708_fut_ref_gelsight \
  --device cuda:0 --task-inertia-gain 80 \
  --robot-profile gelsight-native-track \
  --initial-joints-npz \
    data/dataset_tool_pose_track/0717_eval_0717_gs_policy_native_gain80_traj000000_n50/trace.npz \
  --initial-settling-steps 1 --move-held-away --move-fixed-away \
  --fixed-away-offset=0.5,0.0,0.2 --no-compare-rgb-video \
  --out-dir \
    data/dataset_tool_pose_track_forge/0717_eval_0717_gs_policy_forge_gelsight_both_away_gain80_traj000000_n50
```

### P7. Preserve the dataset timebase: replace resampling with tail padding

**Status:** Evaluation and training paths implemented. Both evaluation scripts
default to 141 control steps, preserve raw samples 0--140, and copy the final
pose for future lookahead. New training runs expose `--chunk-length` (default
150), preserve every raw pose, and copy the final pose through the chunk and
lookahead tail. Sampling a shorter chunk from within a longer trajectory remains
unsupported and raises a clear error. The existing `0716` and `0717` policies
were trained with stretched references and remain legacy until retrained.

**Problem:** The environment currently sets the stored reference length to
`max_episode_length + num_future_steps - 1`. With the saved configuration this
is `150 + 4 - 1 = 153`. A raw episode with 141 poses is expanded to 153 poses
using interpolation over the full episode. This does not merely provide future
lookahead: it changes the demonstrated timebase. The original 140 pose
intervals become 152 intervals at the same controller frequency, increasing
duration by `152 / 140 = 1.085714` and reducing reference velocity to
`140 / 152 = 0.921053` of the recorded velocity (about 7.9% slower).

**Required invariant:** Dataset samples must retain their original indices and
timing. For raw poses `P0 ... P140`, the 153-entry reference buffer must be:

```text
P0, P1, ... P139, P140, P140, ... P140
|------ 141 original samples ------|  |-- 12 padded copies --|
```

Do not interpolate between or redistribute dataset samples. Copy the final
state into every unused reference slot. Apply the same index-preserving tail
padding to every time-aligned dataset signal used by the task (pose,
quaternion, force/wrench, contact, and any future per-step fields) so that
modalities cannot drift out of alignment. Quaternion padding must copy the
stored final quaternion rather than recomputing it.

**Evaluation indexing after the change:** The 141-step rollout uses target
`P[t + 1]`. It executes the 140 recorded transitions through `P139 -> P140`,
then one stationary hold target copied from `P140`. Forge appends four final
pose copies, producing the 145 entries needed by the last four-pose future
window. Native's internal 153-entry buffer contains the same raw prefix and a
longer unused padded tail. Padding adds a terminal hold; it never stretches the
moving portion of the demonstration.

**Implementation scope:**

1. [x] In `franka_robust_track_env.py`, replace the dataset pose resampling path in
   `_load_dataset_trajectories` with an index-preserving pad-to-capacity helper.
   Rename or document `dataset_reference_length` as buffer capacity, not a
   resampling target. Remove resampling from the default training path.
2. [x] In `scripts/track_dataset_tool_pose.py`, set the replay reference length
   to the raw length (141), leaving the internal unused tail as copies of the
   final state. Do not extend the evaluator timeout.
3. [x] In `scripts/track_dataset_tool_pose_forge.py`, use a dedicated padded
   pose-batch builder for tracker replay; preserve raw indices and append four
   final-pose copies for lookahead.
4. If an episode is longer than the supported reference capacity, fail with a
   clear error by default. Do not silently compress or truncate it. Supporting
   longer episodes requires increasing the configured episode/buffer length;
   any explicit truncation mode must be opt-in and reported.
5. If legacy checkpoint reproduction is retained, expose it only as a clearly
   named legacy compatibility option. It must not be the training or evaluation
   default, and results produced with it must state that their trajectory
   timebase was stretched.

**Acceptance tests:**

- A 141-pose input produces a 153-pose Native internal reference and a
  145-pose Forge replay reference; their used 0--144 prefixes are identical.
- Output indices `0:141` equal the raw samples without interpolation; output
  indices `141:153` all equal raw sample 140.
- Finite-difference position and orientation increments for the first 140
  intervals equal those of the raw trajectory; there is no `140 / 152` speed
  factor.
- All aligned force/contact arrays preserve samples `0:141` and repeat their
  final sample in the padded tail.
- Native training, Native evaluation, and Forge evaluation produce identical
  padded pose references for the same raw trajectory.
- Current action/reference indexing remains correct: action at step `t` tracks
  `P[t + 1]`, and the observation still contains four future poses.
- The full 50-trajectory evaluation completes 141 control steps with the saved
  `max_episode_length=150` and no reset entering the metrics. Then retrain and
  re-evaluate the Gelsight policy after the training loader is converted.

### Re-evaluation: 0716 mimic-trained policy versus 0717 Gelsight-trained policy

**Configuration distinction:** The previously reported `~0.17 mm` result is
the 0716 policy on its serialized Mimic plant. A fresh exact rerun reproduces
Native `0.171704 +/- 0.312627 mm`, `0.470504 +/- 1.188777 deg`, `98.1745%`, and
paired Forge `native-mimic` `0.178721 +/- 0.310162 mm`,
`0.470583 +/- 1.188727 deg`, `98.1745%`. The `0.207721 mm` number below is a
different, intentional cross-asset experiment: the same 0716 checkpoint was
forced to run on the Gelsight plant so it could be compared plant-for-plant
with the 0717 Gelsight policy. It does not supersede the `0.171704 mm` native
Mimic result.

The `0716_peg_track_his8_fut4_task_gain40_120_0708_fut_ref` checkpoint was
re-evaluated from scratch on July 17 with exactly the same replay contract used
for the `0717` policy: trajectories 0--49, the Gelsight robot in both
simulators, deterministic policy actions, fixed task gain 80, 149 transitions,
shared Native initialization joints, and both Forge objects moved away. P7 is
not implemented yet, so both policies in this comparison use the same legacy
141-to-153 stretched reference and the relative comparison remains controlled.

| Policy | Simulator | Position error | Rotation error | Tracking success |
|---|---|---:|---:|---:|
| 0716 mimic-trained | Native Gelsight | `0.207721 +/- 0.284283 mm` | `0.721355 +/- 1.087901 deg` | `98.2685%` |
| 0716 mimic-trained | Forge Gelsight | `0.211670 +/- 0.283249 mm` | `0.721428 +/- 1.088390 deg` | `98.2685%` |
| 0717 Gelsight-trained | Native Gelsight | `0.624688 +/- 3.755037 mm` | `3.736156 +/- 20.290155 deg` | `96.0671%` |
| 0717 Gelsight-trained | Forge Gelsight | `0.661211 +/- 4.172114 mm` | `3.716128 +/- 20.217338 deg` | `96.0671%` |

On Native, the 0717 aggregate is `+0.416967 mm`, `+3.014801 deg`, and
`-2.2013 pp` relative to 0716. The fresh 0716 rerun exactly reproduces the
previous reverse-Gelsight result. Native and Forge also agree closely for each
policy, so the difference is a policy behavior rather than a Forge collision or
simulator-parity effect.

The aggregate difference is a tail-risk difference, not worse nominal tracking.
After removing trajectories 12, 39, and 45, both policies have 100% tracking
success and the 0717 policy is better:

| Policy, Native | Position error | Rotation error | Tracking success |
|---|---:|---:|---:|
| 0716, remaining 47 | `0.168010 +/- 0.181221 mm` | `0.509010 +/- 0.288831 deg` | `100%` |
| 0717, remaining 47 | `0.105339 +/- 0.187127 mm` | `0.194005 +/- 0.287089 deg` | `100%` |

The two policies saturate on the same difficult trajectories, but the 0717
policy diverges much farther:

| Trajectory | 0716 mean pos/rot, success | 0717 mean pos/rot, success |
|---:|---:|---:|
| 12 | `1.259 mm / 6.194 deg`, `30.20%` | `8.616 mm / 62.522 deg`, `30.20%` |
| 39 | `0.505 mm / 2.717 deg`, `98.66%` | `13.531 mm / 74.159 deg`, `28.19%` |
| 45 | `0.725 mm / 3.233 deg`, `84.56%` | `4.136 mm / 41.008 deg`, `44.97%` |

Training-time controller-gain randomization does not explain the result. With
the saved random gain sampler instead of fixed gain 80, the 0716/0717 Native
results are respectively `0.223088/0.489120 mm`, `0.738300/3.272479 deg`, and
`97.9597%/96.0268%`. Restricting that diagnostic to the 48 demonstrations
marked successful still gives `0.201983/0.361126 mm`,
`0.629376/2.227142 deg`, and `99.4407%/97.3015%`.

**Why the training records look similar:** The selected checkpoints do have
similar nearby periodic deterministic-eval records. The 0716/0717 entries are
`97.0271/97.2656` reward, `0.335321/0.323448 mm`,
`0.5277/0.5075 deg`, and `100%/99.9023%` final-step success. Those records are
not the fixed 50-trajectory benchmark:

- Training samples only demonstrations marked successful, samples trajectories
  with replacement through the reachability/continuity filter, randomly warps
  20% of resets, and randomizes task gain. Replay deterministically evaluates
  every requested raw demonstration with no warp.
- The tracking reward is bounded and exponential (`8 exp(-pos/0.01)` plus
  `2 exp(-rot/0.1)`). Once a rare trajectory has diverged, making its error much
  larger changes reward only weakly. This is why replay return differs by only
  `0.531` while geometric error differs by several multiples.
- `FrankaRobustTrackEnv` publishes `tracking_pos_error` and
  `tracking_rot_error` as running prefix means. `DeterministicEvalA2CAgent` then
  sums those already-averaged values at every step and divides by the horizon.
  The TensorBoard `eval/tracking_*` value is therefore a mean of prefix means,
  not the direct mean over all environment-step samples used by these replay
  summaries. It also does not expose per-trajectory tails.

**Fresh artifacts**

- 0716 Native Mimic fixed gain: `data/dataset_tool_pose_track/0717_reeval_0716_policy_native_mimic_gain80_traj000000_n50`
- 0716 Forge Mimic fixed gain: `data/dataset_tool_pose_track_forge/0717_reeval_0716_policy_forge_mimic_both_away_gain80_traj000000_n50`
- 0716 Native fixed gain: `data/dataset_tool_pose_track/0717_reeval_0716_policy_native_gelsight_gain80_traj000000_n50`
- 0716 Forge fixed gain: `data/dataset_tool_pose_track_forge/0717_reeval_0716_policy_forge_gelsight_both_away_gain80_traj000000_n50`
- 0716 Native random gain: `data/dataset_tool_pose_track/0717_diagnose_0716_policy_native_gelsight_random_gain_traj000000_n50`
- 0717 Native random gain: `data/dataset_tool_pose_track/0717_diagnose_0717_policy_native_gelsight_random_gain_traj000000_n50`

### P8. Make checkpoint evaluation measure the fixed replay benchmark and tails

**Status:** Planned after P7. Training reward remains useful for optimization,
but it must not be used as the sole checkpoint-quality or policy-comparison
metric.

1. Log direct sums and counts for position error, rotation error, and threshold
   success during deterministic evaluation. Do not average a running mean a
   second time.
2. Add a fixed evaluation manifest with stable trajectory IDs and report the
   raw, unwarped suite separately from randomized training evaluation. Include
   both successful demonstrations and explicitly identified failed/holdout
   demonstrations.
3. Report per-trajectory mean, p95, maximum, first failed step, and action
   saturation. Aggregate mean alone must not hide a small number of unstable
   rollouts.
4. Evaluate at a fixed gain of 80 and at a declared gain grid or random-gain
   seed. Do not compare a fixed-gain replay number directly with an unspecified
   random-gain training number.
5. Store the fixed-suite result with every checkpoint and select/retain a robust
   checkpoint using both nominal mean tracking and a tail-stability constraint.
6. After P7 changes the reference construction, retrain both robot variants and
   run the same fixed suite; do not compare padded-reference policies against
   these legacy stretched-reference numbers as an accuracy ablation.

### P9. Verify the controlled tool Jacobian and add cuRobo reset to Forge

**Status:** Implemented and validated on July 17, 2026.

**Question resolved:** Forge always *reports the pose* of
`panda_fingertip_centered`, but its Jacobian depends on
`ctrl.tool_kinematics_mode`:

- `finger_average` (the legacy `forge-gelsight` profile) uses
  `0.5 * (J_left_finger + J_right_finger)`. This is not the exact geometric
  Jacobian of the reported `panda_fingertip_centered` body origin.
- `tool_body` (the `gelsight-native-track` and `native-mimic` parity profiles)
  takes the PhysX Jacobian for `panda_fingertip_centered` and shifts its linear
  rows from the rigid-body COM to the body origin with `omega x r`. This is the
  same definition used by RobustTrack.
- cuRobo normally returns `panda_hand` kinematics. The diagnostic measures the
  active USD's fixed `T_hand_tool`, applies it to the cuRobo pose, and shifts the
  linear Jacobian rows to the same tool origin. All three results are expressed
  in `panda_link0`, wxyz, with `[linear; angular]` Jacobian rows.

**Diagnostic implementation:**

- `scripts/compare_franka_tool_kinematics.py` launches Forge and RobustTrack in
  separate Isaac processes, writes identical requested joint angles into each,
  queries cuRobo out of process, prints every pose and 6x7 Jacobian, and reports
  pairwise pose/Jacobian deltas.
- `force_tool/utils/curobo_ik_worker.py` now accepts a backward-compatible tagged
  kinematics request in addition to the existing IK tuple protocol.
- `force_tool/utils/curobo_ik_client.py` centralizes GPU pinning, process launch,
  protocol I/O, kinematics queries, IK queries, and shutdown.

Run the three built-in examples with:

```bash
IsaacLab/isaaclab.sh -p scripts/compare_franka_tool_kinematics.py \
  --robot-profile forge-gelsight \
  --output-json data/franka_tool_kinematics/forge_gelsight_three_examples.json
```

Pass any number of explicit configurations by repeating:

```bash
--joint-angles=q1,q2,q3,q4,q5,q6,q7
```

**Measured result:** On three configurations, `forge-gelsight` pose disagreement
versus RobustTrack was only `0.000300--0.000534 mm` and
`0.000026--0.000035 deg`, but its legacy averaged-finger Jacobian had
`max|delta| = 0.03531--0.03690` and RMS `0.01113--0.01152`. cuRobo versus
RobustTrack had `max|delta| = 0.75e-6--1.03e-6` and RMS
`0.24e-6--0.31e-6`. Repeating the reset configuration with
`gelsight-native-track` (`tool_body`) reduced Forge versus RobustTrack to
`max|delta| = 5.78e-7`, RMS `1.84e-7`. Therefore the discrepancy is specifically
the legacy Forge Jacobian selection, not a tool-pose frame mismatch or a
different Jacobian convention in cuRobo.

**cuRobo-based Forge initialization:**

- `scripts/track_dataset_tool_pose_forge.py` now accepts
  `--reset-ik-solver {forge-dls,curobo}`. The default stays `forge-dls` so old
  commands remain reproducible; `--initial-joints-npz` still takes precedence.
- The cuRobo mode measures `tool<->panda_hand` and `world<->panda_link0` from the
  live Forge USD, converts the first dataset tool pose to cuRobo's hand/base goal,
  seeds the solve with `RESET_JOINTS`, writes the returned arm joints, and uses
  the same `--initial-settling-steps` path as shared-joint replay.
- Tunables are `--curobo-num-seeds` (default 8),
  `--curobo-position-tolerance` (default 0.002 m), and
  `--curobo-orientation-tolerance` (default 0.01 rad).
- The selected path is recorded as `initialization_source: curobo_ik` in
  `stats.json`.

Smoke-test command:

```bash
IsaacLab/isaaclab.sh -p scripts/track_dataset_tool_pose_forge.py \
  --dataset raw_datasets/0708_peg_speed_exp10_zero_noise_flat \
  --traj-index 0 --num-trajs 1 --policy direct --steps 1 \
  --reset-ik-solver curobo --no-save-trace \
  --out-dir data/dataset_tool_pose_track_forge/0717_curobo_reset_smoke
```

The smoke test completed without a reset/failure and produced first-step error
`0.007105 mm / 0.001488 deg` with `initialization_source=curobo_ik`.

---

## Feature 6: Domain-randomize richer joint-friction parameters

**Motivation:** The current friction randomization uses a single Coulomb
coefficient per joint, but real arm joints exhibit static (breakaway), velocity-
independent kinetic, and velocity-proportional (viscous) losses plus reflected
rotor inertia. Match the physical model UWLab identifies in
`UWLab/scripts_v2/tools/sim2real/sysid_ur5e_osc.py` so the friction disturbance
the policy trains against is representative, while still treating the parameters
as domain-randomized uncertainty rather than fixed sysid constants.

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
- Replace the single-coefficient friction randomization with domain
  randomization over the richer per-joint parameter set:
  `static_friction`, `dynamic_ratio`, `viscous_friction`, and `armature`.
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
- `RandomizationCfg` (`franka_robust_track_env_cfg.py:134-135`): expose the new
  domain-randomization ranges by replacing `joint_friction_range` with
  - `enable_joint_friction: bool`
  - `static_friction_range = [0.0, 5.0]`
  - `dynamic_ratio_range = [0.0, 1.0]`
  - `viscous_friction_range = [0.0, 5.0]`
  - `enable_joint_armature: bool`, `joint_armature_range = [0.0, ...]`
  and update the `update_env_params` allow-list (lines ~373-374) accordingly.
- `_randomize_dynamics` (`franka_robust_track_env.py` ~970-980): sample the 3
  friction components + armature `(len(env_ids), 7)` each as reset-time domain
  randomization; call
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

**Status:** [ ] not started — domain randomization over richer friction/armature
parameters; motor-delay + sysid centering are follow-ups gated on actuator type
and real FR3 data.

---

## Suggested ordering

1. Dataset replay parity P1--P6, in order; each later comparison depends on the
   preceding invariant.
2. Feature 6 (domain-randomized richer joint-friction parameters) — touches
   `_randomize_dynamics` + critic obs.
