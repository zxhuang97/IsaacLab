# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .franka_robust_track_env import FrankaRobustTrackEnv
from .franka_robust_track_env_cfg import FrankaRobustTrackEnvCfg


gym.register(
    id="Isaac-Franka-Robust-Track-v0",
    entry_point="isaaclab_tasks.direct.franka_robust_track:FrankaRobustTrackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaRobustTrackEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

