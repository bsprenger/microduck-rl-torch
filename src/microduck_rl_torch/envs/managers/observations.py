"""Observation-group manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..task_config import ObservationGroupCfg, ObservationGroupsCfg, ObservationTermCfg
from .base import call_term, reset_term, resolve_term


@dataclass
class ObservationManager:
    config: ObservationGroupsCfg

    def __post_init__(self) -> None:
        self._resolved_terms: dict[tuple[str, str], Any] = {}
        self._resolved_noise: dict[tuple[str, str], Any] = {}
        self._temporal: dict[tuple[str, str], dict[str, Any]] = {}
        self._cache: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}

    @staticmethod
    def _ids(env: Any, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(env.num_envs, device=env.bundle.device)
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.bundle.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=env.bundle.device).reshape(-1)

    def _apply_temporal(
        self,
        env: Any,
        key: tuple[str, str],
        value: torch.Tensor,
        term: ObservationTermCfg,
        group_cfg: ObservationGroupCfg,
        *,
        update_history: bool,
    ) -> torch.Tensor:
        """Apply the delay-then-history observation pipeline."""

        history_length = (
            group_cfg.history_length
            if group_cfg.history_length is not None
            else term.history_length
        )
        if term.delay_max_lag <= 0 and history_length <= 0:
            return value
        if not update_history:
            state = self._temporal.get(key)
            if state is None:
                return value
            if term.delay_max_lag > 0 and state["buffer"]:
                lag = state["lag"]
                if isinstance(lag, torch.Tensor):
                    lag = torch.minimum(
                        lag,
                        torch.full_like(lag, len(state["buffer"]) - 1),
                    )
                    value = torch.stack(
                        [
                            state["buffer"][-1 - int(item)][index]
                            for index, item in enumerate(lag.tolist())
                        ],
                        dim=0,
                    )
                else:
                    value = state["buffer"][-1 - min(int(lag), len(state["buffer"]) - 1)]
            if history_length > 0 and state["history"]:
                if group_cfg.flatten_history_dim and term.flatten_history_dim:
                    value = torch.cat(state["history"], dim=-1)
                else:
                    value = torch.stack(state["history"], dim=-2)
            return value
        state = self._temporal.setdefault(
            key,
            {
                "buffer": [],
                "history": [],
                "lag": 0,
                "step": 0,
                "phase": 0,
                "delay_update_period": term.delay_update_period,
                "delay_per_env_phase": term.delay_per_env_phase,
            },
        )
        is_batch = env.num_envs > 1
        if term.delay_max_lag > 0:
            if is_batch and not isinstance(state["lag"], torch.Tensor):
                state["lag"] = torch.zeros(env.num_envs, dtype=torch.long, device=env.bundle.device)
                state["step"] = torch.zeros(
                    env.num_envs, dtype=torch.long, device=env.bundle.device
                )
                state["phase"] = (
                    torch.randint(
                        0,
                        term.delay_update_period,
                        (env.num_envs,),
                        generator=getattr(env.physics, "generators", [None])[0]
                        if getattr(env.physics, "generators", None)
                        else None,
                        device=env.bundle.device,
                    )
                    if term.delay_update_period > 0 and term.delay_per_env_phase
                    else torch.zeros(env.num_envs, dtype=torch.long, device=env.bundle.device)
                )
            if is_batch:
                if term.delay_update_period > 0:
                    due = ((state["step"] + state["phase"]) % term.delay_update_period) == 0
                else:
                    due = torch.ones(env.num_envs, dtype=torch.bool, device=env.bundle.device)
                sampled = torch.as_tensor(
                    env._sample_delay(
                        term.delay_min_lag,
                        term.delay_max_lag,
                        per_env=term.delay_per_env,
                    ),
                    dtype=torch.long,
                    device=env.bundle.device,
                ).reshape(-1)
                if not term.delay_per_env:
                    sampled = sampled[:1].expand(env.num_envs)
                if term.delay_per_env:
                    hold = (
                        env._random_tensor((env.num_envs,), dtype=env.bundle.dtype)
                        < term.delay_hold_prob
                    )
                else:
                    hold = (
                        env._random_tensor((), dtype=env.bundle.dtype) < term.delay_hold_prob
                    ).expand(env.num_envs)
                state["lag"] = torch.where(due & ~hold, sampled, state["lag"])
                state["step"] += 1
            else:
                due = (
                    (state["step"] + state["phase"]) % term.delay_update_period == 0
                    if term.delay_update_period > 0
                    else True
                )
                hold = bool(
                    env._random_tensor((), dtype=env.bundle.dtype).item() < term.delay_hold_prob
                )
                if due and not hold:
                    state["lag"] = env._sample_delay(term.delay_min_lag, term.delay_max_lag)
                state["step"] += 1
            state["buffer"].append(value.clone())
            if len(state["buffer"]) > term.delay_max_lag + 1:
                state["buffer"].pop(0)
            if is_batch:
                lag = torch.minimum(
                    state["lag"],
                    torch.full_like(state["lag"], len(state["buffer"]) - 1),
                )
                value = torch.stack(
                    [
                        state["buffer"][-1 - int(item)][index]
                        for index, item in enumerate(lag.tolist())
                    ],
                    dim=0,
                )
            else:
                lag = min(int(state["lag"]), len(state["buffer"]) - 1)
                value = state["buffer"][-1 - lag]
        if history_length > 0:
            state["history"].append(value.clone())
            if len(state["history"]) > history_length:
                state["history"].pop(0)
            if group_cfg.flatten_history_dim and term.flatten_history_dim:
                value = torch.cat(state["history"], dim=-1)
            else:
                value = torch.stack(state["history"], dim=-2)
        return value

    def _term_function(self, group: str, name: str, term: Any, env: Any) -> Any:
        key = (group, name)
        if key not in self._resolved_terms:
            self._resolved_terms[key] = resolve_term(term, env)
        return self._resolved_terms[key]

    def reset(self, env: Any, env_ids: torch.Tensor | None = None) -> None:
        for group_name, group_cfg in self.config.groups.items():
            for name, term in group_cfg.terms.items():
                if term.enabled:
                    reset_term(self._term_function(group_name, name, term, env), env_ids)
                    noise = self._resolved_noise.get((group_name, name))
                    if noise is not None:
                        reset = getattr(noise, "reset", None)
                        if callable(reset):
                            reset(env_ids)
        if env_ids is None:
            self._temporal.clear()
            self._cache.clear()
            return
        ids = self._ids(env, env_ids)
        for state in self._temporal.values():
            for name in ("buffer", "history"):
                for value in state.get(name, []):
                    if env.num_envs > 1 and value.ndim >= 2:
                        value[ids] = 0
            if isinstance(state.get("lag"), torch.Tensor):
                state["lag"][ids] = 0
                state["step"][ids] = 0
                period = int(state.get("delay_update_period", 0))
                if period > 0 and state.get("delay_per_env_phase", True):
                    generators = getattr(env.physics, "generators", None)
                    generator = generators[0] if generators else None
                    state["phase"][ids] = torch.randint(
                        0,
                        period,
                        (ids.numel(),),
                        dtype=torch.long,
                        device=state["phase"].device,
                        generator=generator,
                    )
                else:
                    state["phase"][ids] = 0
        self._cache.clear()

    def _noise_value(
        self, env: Any, key: tuple[str, str], term: ObservationTermCfg, value: torch.Tensor
    ) -> torch.Tensor:
        noise = term.noise
        if noise is None:
            return value
        resolved = self._resolved_noise.get(key)
        if resolved is None:
            resolved = noise
            self._resolved_noise[key] = resolved
        apply = getattr(resolved, "apply", None)
        if callable(apply):
            return torch.as_tensor(apply(value), dtype=value.dtype, device=value.device)
        if callable(resolved):
            noise_value = torch.as_tensor(
                resolved(env, value, **term.noise_params),
                dtype=value.dtype,
                device=value.device,
            )
            # Callable noise terms return an additive delta; object noise
            # terms use ``apply`` above and return the complete value.
            return value + noise_value
        raise TypeError(f"Unsupported observation noise object {type(noise).__name__}")

    def compute(
        self, env: Any, group: str = "actor", *, update_history: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute one named observation group."""

        return self.compute_group(env, group, update_history=update_history)

    def compute_group(
        self, env: Any, group: str = "actor", *, update_history: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute a group, optionally without advancing delay/history state."""

        if not update_history and group in self._cache:
            return self._cache[group]
        if group not in self.config.groups:
            raise KeyError(f"Observation group {group!r} is not configured")
        group_cfg = self.config.groups[group]
        if not isinstance(group_cfg, ObservationGroupCfg):
            raise TypeError(f"Observation group {group!r} is not an ObservationGroupCfg")
        if not group_cfg.enabled:
            raise RuntimeError(f"Observation group {group!r} is disabled")
        values: list[torch.Tensor] = []
        named_values: dict[str, torch.Tensor] = {}
        for name, term in group_cfg.terms.items():
            if not isinstance(term, ObservationTermCfg):
                raise TypeError(f"Observation term {name!r} is not an ObservationTermCfg")
            if not term.enabled:
                continue
            function = self._term_function(group, name, term, env)
            if function is None:
                raise RuntimeError(f"Observation term {name!r} has no function")
            value = torch.as_tensor(
                call_term(function, env, term.params),
                dtype=env.bundle.dtype,
                device=env.bundle.device,
            )
            if getattr(env, "num_envs", 1) > 1:
                # Every batched observation term has the canonical shape
                # ``(num_envs, features)``.  This accepts scalar constants,
                # one shared feature vector, and already-batched terms from
                # custom task code without making the manager inspect the
                # term's implementation.
                num_envs = env.num_envs
                if value.ndim == 0:
                    value = value.expand(num_envs).unsqueeze(-1)
                elif value.ndim == 1:
                    if value.shape[0] == num_envs:
                        value = value.unsqueeze(-1)
                    else:
                        value = value.unsqueeze(0).expand(num_envs, -1)
                elif value.ndim != 2 or value.shape[0] != num_envs:
                    raise ValueError(
                        f"Observation term {name!r} returned {tuple(value.shape)}; "
                        f"expected a vector or ({num_envs}, features)"
                    )
            elif value.ndim == 0:
                value = value.reshape(1)
            if term.noise is not None and group_cfg.enable_corruption:
                value = self._noise_value(env, (group, name), term, value)
            if term.clip is not None:
                value = torch.clamp(value, min=term.clip[0], max=term.clip[1])
            if term.scale is not None:
                scale = torch.as_tensor(term.scale, dtype=value.dtype, device=value.device)
                value = value * scale
            value = self._apply_temporal(
                env, (group, name), value, term, group_cfg, update_history=update_history
            )
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Observation term {name!r} returned non-finite values")
            values.append(value)
            named_values[name] = value
        if not values:
            raise RuntimeError(f"Observation group {group!r} has no enabled terms")
        if not group_cfg.concatenate_terms:
            self._cache[group] = named_values
            return named_values
        dim = group_cfg.concatenate_dim
        if dim >= 0 and getattr(env, "num_envs", 1) > 1:
            dim += 1
        observation = torch.cat(values, dim=dim).to(dtype=env.bundle.dtype)
        if group_cfg.expected_size is not None and observation.shape[-1] != group_cfg.expected_size:
            raise RuntimeError(
                f"Observation group {group!r} returned {observation.shape[-1]} values; "
                f"expected {group_cfg.expected_size}"
            )
        self._cache[group] = observation
        return observation
