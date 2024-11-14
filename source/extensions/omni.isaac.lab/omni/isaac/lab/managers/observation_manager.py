# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation manager for computing observation signals for a given world."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from einops import repeat

from omni.isaac.lab.utils import modifiers

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ObservationGroupCfg, ObservationTermCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class ObservationManager(ManagerBase):
    """Manager for computing observation signals for a given world.

    Observations are organized into groups based on their intended usage. This allows having different observation
    groups for different types of learning such as asymmetric actor-critic and student-teacher training. Each
    group contains observation terms which contain information about the observation function to call, the noise
    corruption model to use, and the sensor to retrieve data from.

    Each observation group should inherit from the :class:`ObservationGroupCfg` class. Within each group, each
    observation term should instantiate the :class:`ObservationTermCfg` class. Based on the configuration, the
    observations in a group can be concatenated into a single tensor or returned as a dictionary with keys
    corresponding to the term's name.

    If the observations in a group are concatenated, the shape of the concatenated tensor is computed based on the
    shapes of the individual observation terms. This information is stored in the :attr:`group_obs_dim` dictionary
    with keys as the group names and values as the shape of the observation tensor. When the terms in a group are not
    concatenated, the attribute stores a list of shapes for each term in the group.

    .. note::
        When the observation terms in a group do not have the same shape, the observation terms cannot be
        concatenated. In this case, please set the :attr:`ObservationGroupCfg.concatenate_terms` attribute in the
        group configuration to False.

    The observation manager can be used to compute observations for all the groups or for a specific group. The
    observations are computed by calling the registered functions for each term in the group. The functions are
    called in the order of the terms in the group. The functions are expected to return a tensor with shape
    (num_envs, ...).

    If a noise model or custom modifier is registered for a term, the function is called to corrupt
    the observation. The corruption function is expected to return a tensor with the same shape as the observation.
    The observations are clipped and scaled as per the configuration settings.
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize observation manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ObservationGroupCfg]``).
            env: The environment instance.

        Raises:
            RuntimeError: If the shapes of the observation terms in a group are not compatible for concatenation
                and the :attr:`~ObservationGroupCfg.concatenate_terms` attribute is set to True.
        """
        super().__init__(cfg, env)

        # compute combined vector for obs group
        self._group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()
        for group_name, group_term_dims in self._group_obs_term_dim.items():
            # if terms are concatenated, compute the combined shape into a single tuple
            # otherwise, keep the list of shapes as is
            if self._group_obs_concatenate[group_name]:
                try:
                    term_dims = [torch.tensor(dims, device="cpu") for dims in group_term_dims]
                    self._group_obs_dim[group_name] = tuple(torch.sum(torch.stack(term_dims, dim=0), dim=0).tolist())
                except RuntimeError:
                    raise RuntimeError(
                        f"Unable to concatenate observation terms in group '{group_name}'."
                        f" The shapes of the terms are: {group_term_dims}."
                        " Please ensure that the shapes are compatible for concatenation."
                        " Otherwise, set 'concatenate_terms' to False in the group configuration."
                    )
            else:
                self._group_obs_dim[group_name] = group_term_dims

    def __str__(self) -> str:
        """Returns: A string representation for the observation manager."""
        msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"

        # add info for each group
        for group_name, group_dim in self._group_obs_dim.items():
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Observation Terms in Group: '{group_name}'"
            if self._group_obs_concatenate[group_name]:
                table.title += f" (shape: {group_dim})"
            table.field_names = ["Index", "Name", "Shape"]
            # set alignment of table columns
            table.align["Name"] = "l"
            # add info for each term
            obs_terms = zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
            )
            for index, (name, dims) in enumerate(obs_terms):
                # resolve inputs to simplify prints
                tab_dims = tuple(dims)
                # add row
                table.add_row([index, name, tab_dims])
            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active observation terms in each group.

        The keys are the group names and the values are the list of observation term names in the group.
        """
        return self._group_obs_term_names

    @property
    def group_obs_dim(self) -> dict[str, tuple[int, ...] | list[tuple[int, ...]]]:
        """Shape of computed observations in each group.

        The key is the group name and the value is the shape of the observation tensor.
        If the terms in the group are concatenated, the value is a single tuple representing the
        shape of the concatenated observation tensor. Otherwise, the value is a list of tuples,
        where each tuple represents the shape of the observation tensor for a term in the group.
        """
        return self._group_obs_dim

    @property
    def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
        """Shape of individual observation terms in each group.

        The key is the group name and the value is a list of tuples representing the shape of the observation terms
        in the group. The order of the tuples corresponds to the order of the terms in the group.
        This matches the order of the terms in the :attr:`active_terms`.
        """
        return self._group_obs_term_dim

    @property
    def group_obs_concatenate(self) -> dict[str, bool]:
        """Whether the observation terms are concatenated in each group or not.

        The key is the group name and the value is a boolean specifying whether the observation terms in the group
        are concatenated into a single tensor. If True, the observations are concatenated along the last dimension.

        The values are set based on the :attr:`~ObservationGroupCfg.concatenate_terms` attribute in the group
        configuration.
        """
        return self._group_obs_concatenate

    def get_term_cfg(self, group_name, term_name: str) -> RewardTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the reward term.

        Returns:
            The configuration of the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        group_term_names = self._group_obs_term_names[group_name]
        if term_name not in group_term_names:
            raise ValueError(f"Obs term '{term_name}' not found.")
        # return the configuration
        return self._group_obs_term_cfgs[group_name][group_term_names.index(term_name)]
    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # call all terms that are classes
        for group_cfg in self._group_obs_class_term_cfgs.values():
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_ids=env_ids)
        # call all modifiers that are classes
        for mod in self._group_obs_class_modifiers:
            mod.reset(env_ids=env_ids)
        # reset history
        for group_name, group_term_hist in self._group_obs_term_hist.items():
            group_term_names = self._group_obs_term_names[group_name]
            obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])
            for name, term_cfg in obs_terms:
                term_hist = group_term_hist[name]
                obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params)[env_ids]
                repeat_obs = repeat(obs, "b ... -> b t ...", t=term_cfg.hist_len).clone()
                term_hist[env_ids] = repeat_obs

        # nothing to log here

        return {}

    def compute(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute the observations per group for all groups.

        The method computes the observations for all the groups handled by the observation manager.
        Please check the :meth:`compute_group` on the processing of observations per group.

        Returns:
            A dictionary with keys as the group names and values as the computed observations.
            The observations are either concatenated into a single tensor or returned as a dictionary
            with keys corresponding to the term's name.
        """
        # create a buffer for storing obs from all the groups
        obs_buffer = dict()
        # iterate over all the terms in each group
        for group_name in self._group_obs_term_names:
            obs_buffer[group_name] = self.compute_group(group_name)
        # otherwise return a dict with observations of all groups
        return obs_buffer

    def compute_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
        """Computes the observations for a given group.

        The observations for a given group are computed by calling the registered functions for each
        term in the group. The functions are called in the order of the terms in the group. The functions
        are expected to return a tensor with shape (num_envs, ...).

        The following steps are performed for each observation term:

        1. Compute observation term by calling the function
        2. Apply custom modifiers in the order specified in :attr:`ObservationTermCfg.modifiers`
        3. Apply corruption/noise model based on :attr:`ObservationTermCfg.noise`
        4. Apply clipping based on :attr:`ObservationTermCfg.clip`
        5. Apply scaling based on :attr:`ObservationTermCfg.scale`

        We apply noise to the computed term first to maintain the integrity of how noise affects the data
        as it truly exists in the real world. If the noise is applied after clipping or scaling, the noise
        could be artificially constrained or amplified, which might misrepresent how noise naturally occurs
        in the data.

        Args:
            group_name: The name of the group for which to compute the observations. Defaults to None,
                in which case observations for all the groups are computed and returned.

        Returns:
            Depending on the group's configuration, the tensors for individual observation terms are
            concatenated along the last dimension into a single tensor. Otherwise, they are returned as
            a dictionary with keys corresponding to the term's name.

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """
        # check ig group name is valid
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_obs_term_names[group_name]
        group_term_hist = self._group_obs_term_hist[group_name]
        # buffer to store obs per group
        group_obs = dict.fromkeys(group_term_names, None)
        # read attributes for each term
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])

        # evaluate terms: compute, add noise, clip, scale, custom modifiers
        for name, term_cfg in obs_terms:
            # compute term's value
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
            # apply post-processing
            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    obs = modifier.func(obs, **modifier.params)
            if term_cfg.noise:
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale:
                obs = obs.mul_(term_cfg.scale)
            cur_hist = group_term_hist[name]
            # cur_hist[:, :-1] = cur_hist[:, 1:]
            cur_hist = torch.roll(cur_hist, shifts=-1, dims=1)
            cur_hist[:, -1] = obs[:]

            flat_obs_hist = cur_hist.flatten(start_dim=1)
            # add value to list
            group_obs[name] = flat_obs_hist
            group_term_hist[name] = cur_hist

        # concatenate all observations in the group together
        if self._group_obs_concatenate[group_name]:
            return torch.cat(list(group_obs.values()), dim=-1)
        else:
            return group_obs

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of observation terms functions."""
        # create buffers to store information for each observation group
        # TODO: Make this more convenient by using data structures.
        self._group_obs_term_names: dict[str, list[str]] = dict()
        self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
        self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_concatenate: dict[str, bool] = dict()

        self._group_obs_term_hist = dict()

        # create a list to store modifiers that are classes
        # we store it as a separate list to only call reset on them and prevent unnecessary calls
        self._group_obs_class_modifiers: list[modifiers.ModifierBase] = list()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            group_cfg_items = self.cfg.items()
        else:
            group_cfg_items = self.cfg.__dict__.items()
        # iterate over all the groups
        for group_name, group_cfg in group_cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # check if the term is a curriculum term
            if not isinstance(group_cfg, ObservationGroupCfg):
                raise TypeError(
                    f"Observation group '{group_name}' is not of type 'ObservationGroupCfg'."
                    f" Received: '{type(group_cfg)}'."
                )
            # initialize list for the group settings
            self._group_obs_term_names[group_name] = list()
            self._group_obs_term_dim[group_name] = list()
            self._group_obs_term_cfgs[group_name] = list()
            self._group_obs_class_term_cfgs[group_name] = list()
            # read common config for the group
            self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms
            self._group_obs_term_hist[group_name] = dict()

            # check if config is dict already
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items()
            # iterate over all the terms in each group
            for term_name, term_cfg in group_cfg.__dict__.items():
                # skip non-obs settings
                if term_name in ["enable_corruption", "concatenate_terms", "hist_len"]:
                    continue
                # check for non config
                if term_cfg is None:
                    continue
                if not isinstance(term_cfg, ObservationTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type ObservationTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # resolve common terms in the config
                self._resolve_common_term_cfg(f"{group_name}/{term_name}", term_cfg, min_argc=1)

                # check noise settings
                if not group_cfg.enable_corruption:
                    term_cfg.noise = None
                # add term config to list to list
                self._group_obs_term_names[group_name].append(term_name)
                self._group_obs_term_cfgs[group_name].append(term_cfg)

                # call function the first time to fill up dimensions
                obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)
                single_obs_dims = list(obs_dims[1:])
                single_obs_dims[0] *= term_cfg.hist_len
                self._group_obs_term_dim[group_name].append(tuple(single_obs_dims))
                self._group_obs_term_hist[group_name][term_name] = torch.zeros(
                    (obs_dims[0], term_cfg.hist_len, *obs_dims[1:]), device=self._env.device
                )

                # prepare modifiers for each observation
                if term_cfg.modifiers is not None:
                    # initialize list of modifiers for term
                    for mod_cfg in term_cfg.modifiers:
                        # check if class modifier and initialize with observation size when adding
                        if isinstance(mod_cfg, modifiers.ModifierCfg):
                            # to list of modifiers
                            if inspect.isclass(mod_cfg.func):
                                if not issubclass(mod_cfg.func, modifiers.ModifierBase):
                                    raise TypeError(
                                        f"Modifier function '{mod_cfg.func}' for observation term '{term_name}'"
                                        f" is not a subclass of 'ModifierBase'. Received: '{type(mod_cfg.func)}'."
                                    )
                                mod_cfg.func = mod_cfg.func(cfg=mod_cfg, data_dim=obs_dims, device=self._env.device)

                                # add to list of class modifiers
                                self._group_obs_class_modifiers.append(mod_cfg.func)
                        else:
                            raise TypeError(
                                f"Modifier configuration '{mod_cfg}' of observation term '{term_name}' is not of"
                                f" required type ModifierCfg, Received: '{type(mod_cfg)}'"
                            )

                        # check if function is callable
                        if not callable(mod_cfg.func):
                            raise AttributeError(
                                f"Modifier '{mod_cfg}' of observation term '{term_name}' is not callable."
                                f" Received: {mod_cfg.func}"
                            )

                        # check if term's arguments are matched by params
                        term_params = list(mod_cfg.params.keys())
                        args = inspect.signature(mod_cfg.func).parameters
                        args_with_defaults = [arg for arg in args if args[arg].default is not inspect.Parameter.empty]
                        args_without_defaults = [arg for arg in args if args[arg].default is inspect.Parameter.empty]
                        args = args_without_defaults + args_with_defaults
                        # ignore first two arguments for env and env_ids
                        # Think: Check for cases when kwargs are set inside the function?
                        if len(args) > 1:
                            if set(args[1:]) != set(term_params + args_with_defaults):
                                raise ValueError(
                                    f"Modifier '{mod_cfg}' of observation term '{term_name}' expects"
                                    f" mandatory parameters: {args_without_defaults[1:]}"
                                    f" and optional parameters: {args_with_defaults}, but received: {term_params}."
                                )

                # add term in a separate list if term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_obs_class_term_cfgs[group_name].append(term_cfg)
                    # call reset (in-case above call to get obs dims changed the state)
                    term_cfg.func.reset()
