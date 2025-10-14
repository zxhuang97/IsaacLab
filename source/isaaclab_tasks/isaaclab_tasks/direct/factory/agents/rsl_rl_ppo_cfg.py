# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

# Import the A2C-specific configuration classes.
# (These names are assumed based on the translation from PPO to A2C.)
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class FactoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Runner / Experiment settings
    # Here we set the number of steps per environment (the rollout horizon),
    # the maximum number of training epochs (iterations), and the save interval.
    num_steps_per_env = 128           # YAML: horizon_length
    max_iterations = 200              # YAML: max_epochs
    save_interval = 100               # YAML: save_frequency
    experiment_name = "FactoryNutThread"          # YAML: full_experiment_name
    run_name = ""
    resume = False
    logger = "wandb"
    wandb_project = "FactoryNutThread"            # Updated to match experiment name
    empirical_normalization = True

    # Policy / Network configuration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 256, 128],
        critic_hidden_dims=[128, 256, 128],
        # actor_hidden_dims=[256, 128, 64],
        # critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # -----------------------------------------------------------------------------
    # PPO Algorithm / Optimization Settings
    # -----------------------------------------------------------------------------
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=10,
        num_mini_batches=8,  # default
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )