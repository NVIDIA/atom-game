# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MultiheadAttention
from gymnasium.spaces import flatten_space, flatten
from qds.qcgame.heuristic import SampledGreedyPolicy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import re


class TrainAdapter(nn.Module):
    """Expose the update-time call as ``forward`` so DDP can hook into it.

    Gradients are only all-reduced when backward runs through a DDP forward, so
    the optimisation step has to go through this wrapper rather than the raw
    policy.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state, action):
        return self.policy.get_action_and_value(state, action)


class RunningMeanStd:
    """Welford-style running statistics used to rescale rewards."""

    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, values):
        batch_count = values.numel()
        batch_mean = values.mean().item()
        batch_var = values.var(unbiased=False).item()
        delta = batch_mean - self.mean
        total = self.count + batch_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.mean += delta * batch_count / total
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total


class BestEpisodeBuffer:
    """Keep the cheapest episodes seen, with the states and actions behind them.

    What the benchmark reports is the best schedule found, not the average one,
    and a policy trained only to raise the expected return stops producing new
    candidates once it becomes deterministic. Imitating its own best rollouts
    keeps the search anchored on the incumbent instead.
    """

    def __init__(self, capacity, num_envs):
        self.capacity = capacity
        # Ranked within a circuit: across circuits a cost says more about which
        # circuit was drawn than about the schedule, so one shared ranking would
        # fill up with the cheapest circuit and never revisit the rest.
        self.groups = {}
        self.trace_obs = []
        self.trace_actions = []
        self.offset = 0
        self.starts = [0] * num_envs
        self._cache = None

    @property
    def entries(self):
        return [entry for group in self.groups.values() for entry in group]

    @property
    def cursor(self):
        return self.offset + len(self.trace_obs)

    def append(self, obs_row, action_row):
        self.trace_obs.append(obs_row.clone())
        self.trace_actions.append(action_row.clone())

    def finish(self, env_index, cost, group=0):
        start, end = self.starts[env_index], self.cursor
        self.starts[env_index] = end
        if self._qualifies(cost, group):
            span = range(start - self.offset, end - self.offset)
            self._insert(
                cost,
                torch.stack([self.trace_obs[k][env_index] for k in span]),
                torch.stack([self.trace_actions[k][env_index] for k in span]),
                group,
            )
        self._trim()

    def add_episode(self, cost, obs, actions, group=0):
        if self._qualifies(cost, group):
            self._insert(cost, obs, actions, group)

    def sample(self, size):
        if not self.entries:
            return None
        if self._cache is None:
            self._cache = (
                torch.cat([entry[1] for entry in self.entries]),
                torch.cat([entry[2] for entry in self.entries]),
            )
        obs, actions = self._cache
        index = torch.randint(0, obs.shape[0], (size,), device=obs.device)
        return obs[index], actions[index]

    def _qualifies(self, cost, group=0):
        if not np.isfinite(cost):
            return False
        entries = self.groups.get(group, [])
        return len(entries) < self.capacity or cost < entries[-1][0]

    def _insert(self, cost, obs, actions, group=0):
        entries = self.groups.setdefault(group, [])
        entries.append((cost, obs, actions))
        entries.sort(key=lambda entry: entry[0])
        del entries[self.capacity :]
        self._cache = None

    def _trim(self):
        drop = min(self.starts) - self.offset
        if drop > 0:
            del self.trace_obs[:drop]
            del self.trace_actions[:drop]
            self.offset += drop


class PPOTask:
    def __init__(self, config, envs, logger):
        self.config = config
        self.envs = envs
        self.logger = logger
        self.max_reward = -np.inf
        self.min_cost = np.inf
        self.circuit_min = {}
        self.reward_rms = RunningMeanStd()
        self.return_acc = None
        self.ent_coef = config.training.ent_coef
        self.target_entropy = None
        self.seed_cost = np.inf
        os.makedirs(self.config.model_path, exist_ok=True)

    def entropy_target(self):
        """Absolute entropy to hold, expressed as a fraction of the uniform one.

        An agent picking uniformly among its legal actions -- the empty sites
        plus staying put -- sits at ``log(empties + 1)``, which is what the
        untrained policy measures, so the fraction transfers across board sizes.
        """
        target = self.config.training.get("target_entropy", None)
        if not target:
            return None
        positions = self.config.board.row * self.config.board.col
        # A pool of circuits has no single atom count, and the widest board is
        # the one every circuit shares, so the largest circuit sets the target.
        atoms = self.config.benchmark.n_atoms or self.config.benchmark.max_atoms
        legal = positions - atoms + 1
        return target * math.log(max(legal, 2))

    def adapt_ent_coef(self, entropy):
        """Push the entropy bonus up while the policy is below its target.

        A saturated softmax has vanishing entropy gradient, so recovery is not
        guaranteed once collapse has happened; the point is to react during the
        descent, hence the update every iteration.
        """
        if self.target_entropy is None or math.isnan(entropy):
            return
        args = self.config.training
        if args.get("ent_control", "linear") == "log":
            # Useful entropies span orders of magnitude, so the error that counts
            # is the ratio. At a target of 0.06 and an entropy of 0.01 the linear
            # error is 0.05, which moves the coefficient by half a percent.
            error = math.log(self.target_entropy / max(entropy, 1e-8))
        else:
            error = self.target_entropy - entropy
        step = args.get("ent_coef_lr", 0.1) * error
        self.ent_coef *= math.exp(max(min(step, 0.5), -0.5))
        self.ent_coef = max(
            min(self.ent_coef, args.get("ent_coef_max", 1.0)),
            args.get("ent_coef_min", 1e-5),
        )

    @staticmethod
    def turn_groups(turn_ends):
        """Slice each environment's rollout into complete reconfigurations.

        ``turn_ends`` marks the sub-step whose atom was the last to act, so a
        group is everything since the previous mark. A rollout usually stops
        mid-turn and that remainder is dropped rather than treated as a decision
        that never got its reward.
        """
        steps, num_envs = turn_ends.shape
        groups = [[] for _ in range(num_envs)]
        for env_index in range(num_envs):
            start = 0
            for step in range(steps):
                if turn_ends[step, env_index]:
                    groups[env_index].append((start, step + 1))
                    start = step + 1
        return groups

    def turn_advantages(self, groups, buffers, bootstrap, next_done, args):
        """Generalised advantage estimation over turns instead of atoms."""
        rewards, dones, values = buffers["rewards"], buffers["dones"], buffers["values"]
        steps = rewards.shape[0]
        advantages, returns, group_values, spans = [], [], [], []
        for env_index, env_groups in enumerate(groups):
            per_env = []
            for start, end in env_groups:
                if end < steps:
                    next_value = values[end, env_index]
                    nonterminal = 1.0 - dones[end, env_index]
                else:
                    next_value = bootstrap[env_index]
                    nonterminal = 1.0 - next_done[env_index]
                per_env.append(
                    {
                        "span": (start, end, env_index),
                        "reward": rewards[start:end, env_index].sum(),
                        "value": values[start, env_index],
                        "next_value": next_value,
                        "nonterminal": nonterminal,
                    }
                )
            lastgaelam = 0.0
            for entry in reversed(per_env):
                delta = (
                    entry["reward"]
                    + args.gamma * entry["next_value"] * entry["nonterminal"]
                    - entry["value"]
                )
                lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * entry["nonterminal"] * lastgaelam
                )
                entry["advantage"] = lastgaelam
            for entry in per_env:
                spans.append(entry["span"])
                advantages.append(entry["advantage"])
                group_values.append(entry["value"])
                returns.append(entry["advantage"] + entry["value"])
        return spans, advantages, returns, group_values

    def single_env(self):
        """A single environment to run heuristics on, and whether we own it."""
        envs = getattr(self.envs, "envs", None)
        if envs:
            return envs[0], False
        factories = getattr(self.envs, "env_fns", None)
        if factories:
            return factories[0](), True
        return None, False

    def seed_best_episodes(self, buffer, device):
        """Fill the imitation buffer with epsilon-greedy rollouts.

        The heuristic is strong and uses only the analytic cost model, so
        starting the imitation term from its schedules means the search begins
        where the baseline ends rather than at noise. ``min cost`` is left alone
        so that the reported number stays what the learned policy itself found.
        """
        args = self.config.training
        episodes = args.get("sil_seed_episodes", 0)
        if buffer is None or episodes <= 0:
            return
        env, borrowed = self.single_env()
        if env is None:
            return
        # The heuristics read estimator and board state, which the gym wrappers
        # only forward with a deprecation warning.
        env = env.unwrapped
        policy = SampledGreedyPolicy(
            self.config, epsilon=args.get("sil_seed_epsilon", 0.02)
        )
        space = self.envs.single_observation_space
        for _ in range(episodes):
            observation, _ = env.reset()
            states, actions = [], []
            while True:
                action = policy.action(env)
                states.append(flatten(space, observation))
                actions.append(action)
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            self.seed_cost = min(self.seed_cost, env.log.cost)
            buffer.add_episode(
                env.log.cost,
                torch.as_tensor(np.array(states), dtype=torch.float32, device=device),
                torch.as_tensor(actions, dtype=torch.long, device=device),
                getattr(env.reader, "current_reader_indx", 0),
            )
        if borrowed:
            env.close()

    def scope_advantages(self, advantages):
        """Optionally centre and scale each environment's own advantages.

        A minibatch spans every environment, and with a pool of circuits those
        environments are scheduling different circuits whose costs differ by more
        than any schedule does. Normalising over the mixture therefore divides by
        a spread that mostly says which circuit was drawn. One environment holds
        one circuit for the length of an episode, so its own column is the scope
        the comparison belongs in.
        """
        if self.config.training.get("adv_scope", "batch") != "env":
            return advantages
        mean = advantages.mean(dim=0, keepdim=True)
        std = advantages.std(dim=0, unbiased=False, keepdim=True)
        return (advantages - mean) / (std + 1e-8)

    def scale_rewards(self, reward, next_done):
        """Optionally rescale rewards by the running std of discounted returns."""
        args = self.config.training
        reward = reward * args.get("reward_scale", 1.0)
        if not args.get("normalize_reward", False):
            return reward
        if self.return_acc is None:
            self.return_acc = torch.zeros_like(reward)
        self.return_acc = self.return_acc * args.gamma * (1.0 - next_done) + reward
        self.reward_rms.update(self.return_acc)
        return reward / (self.reward_rms.var**0.5 + 1e-8)

    def update_turns(self, ddp_model, agent, optimizer, rollout, sil, device):
        """Optimise with a whole reconfiguration as the unit of decision.

        The atoms of one turn are still chosen one at a time against the board
        each of them leaves behind, so nothing about the policy or the
        environment changes. What changes is the bookkeeping: the joint log
        probability of a turn is the sum over its atoms, which makes the
        importance ratio exact, and the reward now lands on the same transition
        that earned it instead of eight steps downstream.
        """
        args = self.config.training
        steps, num_envs = rollout["rewards"].shape
        groups = self.turn_groups(rollout["turn_ends"])
        spans, advantages, returns, values = self.turn_advantages(
            groups, rollout, rollout["bootstrap"], rollout["next_done"], args
        )
        if not spans:
            raise RuntimeError(
                "No reconfiguration finished within the rollout; "
                "raise training.rollout_multiplier"
            )
        flat_obs = rollout["obs"].reshape(steps * num_envs, -1)
        flat_actions = rollout["actions"].reshape(-1).long()
        flat_logprobs = rollout["logprobs"].reshape(-1)
        # Rows of each turn in the flattened [step, env] layout.
        members = [
            torch.arange(start, end, device=device) * num_envs + env_index
            for start, end, env_index in spans
        ]
        old_logprobs = torch.stack([flat_logprobs[rows].sum() for rows in members])
        advantages = torch.stack(advantages)
        returns = torch.stack(returns)
        values = torch.stack(values)

        order = np.arange(len(members))
        minibatch = max(1, len(members) // args.num_minibatches)
        stats = {name: [] for name in self.UPDATE_STATS}
        stop_early = False
        for _ in range(args.update_epochs):
            np.random.shuffle(order)
            for offset in range(0, len(order), minibatch):
                mb = torch.as_tensor(
                    order[offset : offset + minibatch], device=device
                )
                rows = torch.cat([members[i] for i in mb.tolist()])
                lengths = torch.tensor(
                    [len(members[i]) for i in mb.tolist()], device=device
                )
                segment = torch.repeat_interleave(
                    torch.arange(len(mb), device=device), lengths
                )
                first = torch.cumsum(lengths, 0) - lengths
                mb_obs = flat_obs[rows]
                mb_actions = flat_actions[rows]

                sil_batch = (
                    sil["buffer"].sample(sil["size"])
                    if sil["buffer"] is not None
                    else None
                )
                if sil_batch is not None:
                    mb_obs = torch.cat([mb_obs, sil_batch[0]])
                    mb_actions = torch.cat([mb_actions, sil_batch[1].long()])

                _, newlogprob, entropy, newvalue = ddp_model(
                    self.unflatten_batched(mb_obs), mb_actions
                )
                if sil_batch is not None:
                    split = rows.shape[0]
                    sil_loss = -newlogprob[split:].mean()
                    newlogprob = newlogprob[:split]
                    entropy = entropy[:split]
                    newvalue = newvalue[:split]
                else:
                    sil_loss = None

                turn_logprob = torch.zeros(len(mb), device=device).index_add_(
                    0, segment, newlogprob
                )
                logratio = turn_logprob - old_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    stats["approx_kls"].append(
                        ((ratio - 1) - logratio).mean().item()
                    )
                    stats["clipfracs"].append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = advantages[mb]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std(unbiased=False) + 1e-8
                    )
                pg_loss = torch.max(
                    -mb_advantages * ratio,
                    -mb_advantages
                    * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                # A turn's value is the value of the board the first atom saw.
                newvalue = newvalue.view(-1)[first]
                if args.clip_vloss:
                    clipped = values[mb] + torch.clamp(
                        newvalue - values[mb], -args.clip_coef, args.clip_coef
                    )
                    v_loss = 0.5 * torch.max(
                        (newvalue - returns[mb]) ** 2, (clipped - returns[mb]) ** 2
                    ).mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb]) ** 2).mean()

                # Kept per atom so that the entropy target means the same thing
                # here as it does when every atom is its own transition.
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * args.vf_coef
                if sil_loss is not None:
                    loss = loss + sil["coef"] * sil_loss
                    stats["sil_losses"].append(sil_loss.item())
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                optimizer.step()
                stats["losses"].append(loss.item())
                stats["pg_losses"].append(pg_loss.item())
                stats["v_losses"].append(v_loss.item())
                stats["entropies"].append(entropy_loss.item())
                stats["grad_norms"].append(grad_norm.item())
            if args.target_kl is not None and stats["approx_kls"][-1] > args.target_kl:
                stop_early = True
            if stop_early:
                break
        stats["values"] = values.detach()
        stats["returns"] = returns.detach()
        stats["turns"] = len(members)
        return stats

    UPDATE_STATS = (
        "losses",
        "pg_losses",
        "v_losses",
        "entropies",
        "approx_kls",
        "grad_norms",
        "clipfracs",
        "sil_losses",
    )

    def run(self, policy, local_rank):
        policy.cuda(local_rank)
        ddp_model = DDP(
            TrainAdapter(policy),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=self.config.training.get(
                "find_unused_parameters", True
            ),
        )
        checkpoint_path, start_iter = self.get_latest_checkpoint()
        if self.config["load_checkpoint"] and checkpoint_path:
            state_dict = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            policy.load_state_dict(state_dict, strict=True)
            print(f"Loaded model from {checkpoint_path}, starting from iteration {start_iter}")
        else:
            start_iter = 0
            print("No checkpoint found, starting from scratch.")
        device = next(ddp_model.parameters()).device
        start_iter_tensor = torch.tensor(
            [start_iter], dtype=torch.long, device=device
        )
        dist.broadcast(start_iter_tensor, src=0)
        start_iter = start_iter_tensor.item()
        
        args = self.config.training
        envs = self.envs
        args.num_steps = self.config["n_components"] * args.get(
            "rollout_multiplier", 1
        )
        args.batch_size = args.num_steps * args.num_envs
        args.minibatch_size = args.batch_size // args.num_minibatches
        self.ent_coef = args.ent_coef
        self.target_entropy = self.entropy_target()
        sil_coef = args.get("sil_coef", 0.0)
        sil_buffer = (
            BestEpisodeBuffer(args.get("sil_capacity", 8), args.num_envs)
            if sil_coef > 0
            else None
        )
        sil_size = max(1, int(args.minibatch_size * args.get("sil_batch_ratio", 0.25)))
        self.seed_best_episodes(sil_buffer, device)
        agent = policy.to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        # ALGO Logic: Storage setup
        flat_obs_space = flatten_space(envs.single_observation_space)
        obs = torch.zeros((args.num_steps, args.num_envs) + flat_obs_space.shape).to(
            device
        )
        actions = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape
        ).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        turn_actions = args.get("turn_actions", False)
        turn_ends = np.zeros((args.num_steps, args.num_envs), dtype=bool)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        next_obs, infos = envs.reset()
        previous_infos = infos
        next_obs = torch.Tensor(self.flatten_batched(next_obs)).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        num_done = 0
        started_at = time.time()
        for iteration in range(start_iter + 1, args.num_iterations + 1):
            episode_costs = []
            episode_rewards = []
            episode_reconfigs = []
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = max(frac, args.get("lr_floor", 0.0)) * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(
                        self.unflatten_batched(next_obs)
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                if sil_buffer is not None:
                    sil_buffer.append(obs[step], actions[step])

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(
                    action.cpu().numpy()
                )
                next_done = np.logical_or(terminations, truncations)
                if turn_actions:
                    # The key is absent on the step that ends an episode, which
                    # is also a turn boundary.
                    ended = infos.get("turn_end")
                    turn_ends[step] = np.logical_or(
                        next_done,
                        np.zeros_like(next_done) if ended is None else ended,
                    )
                rewards[step] = self.scale_rewards(
                    torch.tensor(reward, dtype=torch.float32).to(device).view(-1),
                    torch.tensor(next_done, dtype=torch.float32).to(device).view(-1),
                )
                circuits = previous_infos.get("circuit")
                for j, d in enumerate(next_done):
                    if d:
                        log = previous_infos['log'][j]
                        circuit = 0 if circuits is None else int(circuits[j])
                        if log.reward > self.max_reward:
                            self.max_reward = log.reward
                            self.max_log = log.copy()
                        self.min_cost = min(self.min_cost, log.cost)
                        # With one circuit these coincide; with a pool the global
                        # minimum only tracks the cheapest circuit, so the best
                        # per circuit is what says whether training progresses.
                        self.circuit_min[circuit] = min(
                            self.circuit_min.get(circuit, np.inf), log.cost
                        )
                        if sil_buffer is not None:
                            sil_buffer.finish(j, log.cost, circuit)
                        episode_costs.append(log.cost)
                        episode_rewards.append(log.reward)
                        episode_reconfigs.append(log.reconfig_count)
                        num_done += 1

                next_obs, next_done = torch.Tensor(self.flatten_batched(next_obs)).to(
                    device
                ), torch.Tensor(next_done).to(device)
                previous_infos = infos
            if self.config.task == "test":
                self.log_dict(
                    {
                        "iteration": iteration,
                        "episodes": num_done,
                        "min cost": self.min_cost,
                        "max reward": self.max_reward,
                        "episode cost": np.mean(episode_costs)
                        if episode_costs
                        else float("nan"),
                        "time": time.time() - started_at,
                    },
                    iteration,
                )
                continue
            if turn_actions:
                with torch.no_grad():
                    bootstrap = agent.get_value(
                        self.unflatten_batched(next_obs)
                    ).view(-1)
                stats = self.update_turns(
                    ddp_model,
                    agent,
                    optimizer,
                    {
                        "obs": obs,
                        "actions": actions,
                        "logprobs": logprobs,
                        "rewards": rewards,
                        "dones": dones,
                        "values": values,
                        "turn_ends": turn_ends,
                        "bootstrap": bootstrap,
                        "next_done": next_done,
                    },
                    {"buffer": sil_buffer, "size": sil_size, "coef": sil_coef},
                    device,
                )
                (
                    losses,
                    pg_losses,
                    v_losses,
                    entropies,
                    approx_kls,
                    grad_norms,
                    clipfracs,
                    sil_losses,
                ) = (stats[name] for name in self.UPDATE_STATS)
                b_values, b_returns = stats["values"], stats["returns"]
                turn_count = stats["turns"]
            else:
                # bootstrap value if not done
                with torch.no_grad():
                    next_value = agent.get_value(self.unflatten_batched(next_obs)).reshape(
                        1, -1
                    )
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + args.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        )
                    returns = advantages + values
                    advantages = self.scope_advantages(advantages)

                # flatten the batch
                b_obs = obs.reshape((-1,) + flat_obs_space.shape)
                b_logprobs = logprobs.reshape(-1)
                b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = values.reshape(-1)

                # Optimizing the policy and value network
                b_inds = np.arange(args.batch_size)
                clipfracs = []
                losses = []
                pg_losses = []
                v_losses = []
                entropies = []
                approx_kls = []
                grad_norms = []
                sil_losses = []
                stop_early = False
                for epoch in range(args.update_epochs):
                    np.random.shuffle(b_inds)
                    for offset in range(0, args.batch_size, args.minibatch_size):
                        mb_inds = b_inds[offset : offset + args.minibatch_size]
                        mb_obs = b_obs[mb_inds]
                        mb_actions = b_actions.long()[mb_inds]

                        # One forward for both losses: DDP reduces gradients on the
                        # last graph it saw, so a second forward would go unsynced.
                        sil_batch = (
                            sil_buffer.sample(sil_size)
                            if sil_buffer is not None
                            else None
                        )
                        if sil_batch is not None:
                            mb_obs = torch.cat([mb_obs, sil_batch[0]])
                            mb_actions = torch.cat([mb_actions, sil_batch[1].long()])

                        _, newlogprob, entropy, newvalue = ddp_model(
                            self.unflatten_batched(mb_obs), mb_actions
                        )
                        if sil_batch is not None:
                            split = mb_inds.shape[0]
                            sil_loss = -newlogprob[split:].mean()
                            newlogprob = newlogprob[:split]
                            entropy = entropy[:split]
                            newvalue = newvalue[:split]
                        else:
                            sil_loss = None
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [
                                ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                            ]

                        mb_advantages = b_advantages[mb_inds]
                        if args.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                                mb_advantages.std(unbiased=False) + 1e-8
                            )

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio, 1 - args.clip_coef, 1 + args.clip_coef
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if args.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = (
                            pg_loss - self.ent_coef * entropy_loss + v_loss * args.vf_coef
                        )
                        if sil_loss is not None:
                            loss = loss + sil_coef * sil_loss
                            sil_losses.append(sil_loss.item())
                        optimizer.zero_grad()
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(
                            agent.parameters(), args.max_grad_norm
                        )
                        optimizer.step()
                        losses.append(loss.item())
                        pg_losses.append(pg_loss.item())
                        v_losses.append(v_loss.item())
                        entropies.append(entropy_loss.item())
                        approx_kls.append(approx_kl.item())
                        grad_norms.append(grad_norm.item())
                    if args.target_kl is not None and approx_kls[-1] > args.target_kl:
                        stop_early = True
                    if stop_early:
                        break
                turn_count = float("nan")
            self.adapt_ent_coef(np.mean(entropies) if entropies else math.nan)
            if (
                dist.get_rank() == 0
                and iteration % self.config.save_frequency == 0
            ):
                save_path = os.path.join(
                    self.config.model_path, f"model_iter_{iteration}.pt"
                )
                torch.save(policy.state_dict(), save_path)
            metrics = {
                "iteration": iteration,
                "loss": np.mean(losses),
                "lr": optimizer.param_groups[0]["lr"],
                "pg_loss": np.mean(pg_losses),
                "v_loss": np.mean(v_losses),
                "entropy": np.mean(entropies),
                "ent_coef": self.ent_coef,
                "sil_loss": np.mean(sil_losses) if sil_losses else float("nan"),
                "seed cost": self.seed_cost,
                "approx_kl": np.mean(approx_kls),
                "clipfrac": np.mean(clipfracs),
                "grad_norm": np.mean(grad_norms),
                "turns": turn_count,
                "explained_variance": self.explained_variance(b_values, b_returns),
                "return_mean": b_returns.mean().item(),
                "return_std": b_returns.std(unbiased=False).item(),
                "episodes": num_done,
                "min cost": self.min_cost,
                "max reward": self.max_reward,
                "time": time.time() - started_at,
            }
            if episode_costs:
                metrics.update(
                    {
                        "episode cost": np.mean(episode_costs),
                        "episode reward": np.mean(episode_rewards),
                        "episode reconfig": np.mean(episode_reconfigs),
                    }
                )
            if len(self.circuit_min) > 1:
                metrics.update(
                    {
                        "circuit min cost": float(
                            np.mean(list(self.circuit_min.values()))
                        ),
                        "circuits seen": len(self.circuit_min),
                    }
                )
            if hasattr(self, "max_log"):
                metrics.update(
                    {
                        "cost": self.max_log.cost,
                        "best duration": self.max_log.duration,
                        "best touches": self.max_log.n_touches,
                        "reconfig count": self.max_log.reconfig_count,
                    }
                )
            self.log_dict(metrics, iteration)
        self.logger.save()
        self.logger.finalize("success")

    @staticmethod
    def explained_variance(values, returns):
        variance = returns.var(unbiased=False)
        if variance.item() == 0:
            return float("nan")
        return (1 - (returns - values).var(unbiased=False) / variance).item()

    def get_latest_checkpoint(self):
        files = os.listdir(self.config.model_path)
        pattern = re.compile(r"model_iter_(\d+)\.pt")
        checkpoints = [(int(m.group(1)), f) for f in files if (m := pattern.match(f))]
        if not checkpoints:
            return None, 0
        checkpoints.sort()
        latest_iter, latest_file = checkpoints[-1]
        return os.path.join(self.config.model_path, latest_file), latest_iter

    def log_dict(self, dict, iteration):
        self.logger.log_metrics(dict, step=iteration)

    def flatten_batched(self, obs):
        obs_batch = []
        for i in range(self.envs.num_envs):
            item = {}
            for key, _ in self.envs.single_observation_space.spaces.items():
                item[key] = obs[key][i]
            obs_batch.append(item)
        flat_obs = np.array(
            [flatten(self.envs.single_observation_space, ob) for ob in obs_batch]
        )
        return flat_obs

    def unflatten_batched(self, flat_tensor):
        space = self.envs.single_observation_space
        obs = {}
        index = 0
        batch_size, _ = flat_tensor.shape
        for key, subspace in space.spaces.items():
            size = int(np.prod(subspace.shape))
            shape = subspace.shape
            if isinstance(subspace, gym.spaces.Discrete):
                size = subspace.n
                shape = (subspace.n,)
            space = (batch_size,) + shape
            obs[key] = flat_tensor[:, index : index + size].reshape(space)
            index += size
        return obs
