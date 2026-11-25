# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cost of a trained policy on circuits it never saw, one rollout per circuit.

Every number reported so far is the best of thousands of sampled episodes on the
circuit the policy trained on, which measures a search. What a policy trained
across circuits is for is the opposite: schedule a new circuit in one shot, with
no search at all. So each held out circuit is scheduled once by the policy's own
argmax and once by each heuristic, on the same board, and the costs are paired.

    PYTHONPATH=. python bin/evaluate.py --config-name=transfer_50 task=test \\
        benchmark.test_count=10 test.training_exp_tag=v1
"""

import json
import random
from collections import Counter

import hydra
import torch
from gymnasium.spaces import flatten
from ml_collections import ConfigDict
from omegaconf import DictConfig, OmegaConf

from bin.imitate import cost_row, needs_cost, shortlist_mask
from qds.qcgame.heuristic import POLICIES, GreedyPolicy
from qds.qcgame.meter import counting, evaluations, steps
from qds.task.container import Container
from qds.task.obs import unflatten_batch
from bin.train import latest_checkpoint


def episode(env, choose):
    """One scheduled circuit, reporting which one it was."""
    observation, _ = env.reset()
    circuit = int(getattr(env.reader, "current_reader_indx", 0))
    atoms = int(env.reader.num_qubits())
    while True:
        observation, _, terminated, truncated, _ = env.step(choose(observation))
        if terminated or truncated:
            return {"circuit": circuit, "atoms": atoms, "cost": float(env.log.cost)}


def learned(policy, space, device, env=None, shortlist=0, expert=None, temperature=0.0):
    """The policy's own choice, restricted to the shortlist it was fitted under.

    Under a shortlist the estimator proposes the cheapest few sites and the policy
    only re-ranks those, which is the same budget greedy pays; evaluating without
    it would measure a different policy from the one that was trained.

    A temperature above zero samples instead of taking the argmax, which is what
    the policy is for when it is given a budget of rollouts rather than one.
    """

    @torch.no_grad()
    def choose(observation):
        flat = torch.as_tensor(
            flatten(space, observation), dtype=torch.float32, device=device
        ).unsqueeze(0)
        row = (
            cost_row(expert, env, env.action_space.n, device)
            if needs_cost(policy, shortlist)
            else None
        )
        logits = policy(unflatten_batch(space, flat), row)
        if shortlist > 0:
            seat = torch.as_tensor(
                [env.board.atom_map[env.turn.current_player()]], device=logits.device
            )
            logits = torch.where(
                shortlist_mask(row, seat, shortlist), logits, -torch.inf
            )
        if temperature > 0:
            draw = torch.distributions.Categorical(logits=logits / temperature)
            return int(draw.sample().item())
        return int(logits.argmax(dim=-1).item())

    return choose


def guides(policy, space, device, expert=None):
    """The model's ranking of candidates and its score for a half built plan.

    A guided beam asks for both at states it reached on a copy of the game, so
    neither may cost the estimator anything the search has not already paid for:
    the ranking is handed the scores the node computed, and the critic reads them
    only if the policy was fitted to take them as an input.
    """

    def observe(env):
        flat = torch.as_tensor(
            flatten(space, env.state), dtype=torch.float32, device=device
        ).unsqueeze(0)
        return unflatten_batch(space, flat)

    def row_of(env, actions, costs):
        row = torch.full(
            (1, env.action_space.n), torch.inf, dtype=torch.float32, device=device
        )
        row[0, list(actions)] = torch.as_tensor(
            list(costs), dtype=torch.float32, device=device
        )
        return row

    @torch.no_grad()
    def rank(env, actions, costs):
        row = row_of(env, actions, costs)
        logits = policy(observe(env), row if needs_cost(policy, 0) else None)[0]
        scores = logits.tolist()
        return [scores[action] for action in actions]

    @torch.no_grad()
    def value(env):
        row = (
            cost_row(expert, env, env.action_space.n, device)
            if needs_cost(policy, 0)
            else None
        )
        return float(policy.get_value(observe(env), row).item())

    return rank, value


def passes(build, container, resolved, count, seed, budget, offset=0):
    """Schedule every held out circuit ``budget`` times over.

    Reseeding at the start of each pass redraws the same circuits on the same
    boards, so a rollout in one pass is comparable with the rollout of any other
    method in any other pass, and the best of a method's passes on a circuit is
    what a budget buys it.
    """
    costs = [[] for _ in range(count)]
    spent = Counter()
    for attempt in range(budget):
        # Building the environment reseeds both streams from the run's seed, which
        # is what repeats the boards; the sampling stream has to be moved on after
        # that or every pass replays the same rollout.
        env = container.get_env(resolved).unwrapped
        torch.manual_seed(seed * 10007 + offset + attempt)
        # What the pass asks of the cost model, which is the budget a method is
        # compared at: a search spends it on the candidates it scores, and a
        # learned policy on the shortlist it is handed.
        with counting() as counts:
            choose = build(env, offset + attempt)
            for index in range(count):
                costs[index].append(episode(env, choose)["cost"])
        spent["calls"] += evaluations(counts)
        spent["steps"] += steps(counts)
        env.close()
    return costs, {
        "calls_per_pass": spent["calls"] / budget,
        "steps_per_pass": spent["steps"] / budget,
        "calls": spent["calls"],
        "steps": spent["steps"],
    }


def best_of_curve(costs, budget):
    """Mean cost per circuit once a method may keep its cheapest of k rollouts."""
    points = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= budget]
    if budget not in points:
        points.append(budget)
    return {
        k: sum(min(row[:k]) for row in costs) / len(costs)
        for k in points
        if all(len(row) >= k for row in costs)
    }


def build_heuristic(name, resolved, attempt, seed, extras=None):
    """A restart has to differ from the last one or the budget buys nothing."""
    policy_class = POLICIES[name][1]
    if getattr(policy_class, "deterministic", True):
        return policy_class(resolved, **(extras or {}))
    return policy_class(resolved, seed=seed * 9973 + attempt, **(extras or {}))


def weights_path(config):
    if config.get("init_model"):
        return config.init_model
    path, _ = latest_checkpoint(config.test.training_model_path)
    if path is None:
        raise ValueError(f"No checkpoint under {config.test.training_model_path}")
    return path


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(config: DictConfig) -> None:
    resolved = ConfigDict(OmegaConf.to_container(config, resolve=True))
    if resolved.mode == "fixed":
        problem, n_atoms = resolved.benchmark.id.rsplit("-", maxsplit=1)
        resolved.benchmark.problem = problem
        resolved.benchmark.n_atoms = int(n_atoms)
    device = resolved.system.device if torch.cuda.is_available() else "cpu"
    container = Container(resolved.mode)

    # The held out circuits are visited in order and their boards are drawn from
    # the global stream, so reseeding before each policy puts every policy on the
    # same circuit and the same board at the same episode.
    count = resolved.benchmark.get("test_count", 1)
    seed = resolved.get("seed", 0)
    runs = {}

    policy = container.get_policy(resolved).to(device)
    path = weights_path(resolved)
    policy.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=True
    )
    policy.eval()
    shortlist = int(resolved.imitation.get("shortlist", 0))
    # One rollout each is the one shot claim; a budget above one is the comparison
    # a surrogate is actually used under, sampling against restarting a heuristic.
    budget = int(resolved.get("budget", 1))
    # A budget splits across processes: each takes its own slice of the sampling
    # stream so their rollouts can be pooled afterwards.
    offset = int(resolved.get("budget_offset", 0))
    temperature = float(resolved.get("sample_temperature", 0.5)) if budget > 1 else 0.0
    space = container.get_env(resolved).unwrapped.observation_space
    curves = {}

    def build_learned(env, _attempt):
        return learned(
            policy,
            space,
            device,
            env,
            shortlist,
            GreedyPolicy(resolved) if needs_cost(policy, shortlist) else None,
            temperature,
        )

    spending = {}
    costs, spending["learned"] = passes(
        build_learned, container, resolved, count, seed, budget, offset
    )
    runs["learned"] = [{"cost": min(row), "rollouts": row} for row in costs]
    curves["learned"] = best_of_curve(costs, budget)

    # The guided beam is the same search with the fitted model in the two places
    # the plain one uses the estimator's ranking, so it is handed the weights that
    # were just loaded rather than built from the config alone.
    rank, value = guides(policy, space, device, GreedyPolicy(resolved))
    extras = {
        "guided_beam": {
            "proposer": rank if resolved.get("guide_branch", True) else None,
            "valuer": value if resolved.get("guide_value", True) else None,
        }
    }

    for name in resolved.get("policies", ["greedy", "no_reconfiguration"]):
        repeats = 1 if getattr(POLICIES[name][1], "deterministic", True) else budget
        def build(env, attempt, name=name):
            heuristic = build_heuristic(
                name, resolved, attempt, seed, extras.get(name)
            )
            return lambda _observation: heuristic.action(env)

        costs, spending[name] = passes(
            build, container, resolved, count, seed, repeats, offset
        )
        runs[name] = [{"cost": min(row), "rollouts": row} for row in costs]
        curves[name] = best_of_curve(costs, repeats)

    reference = runs.get("greedy", runs["learned"])
    print(
        json.dumps(
            {
                "weights": str(path),
                "circuits": count,
                "per_episode": runs,
                "budget": budget,
                "best_of_curve": curves,
                "spent": spending,
                # What a row means depends on these, and a report that recovers
                # them from a file name is a report that mislabels one eventually.
                "settings": {
                    "circuit": str(resolved.get("test_circuit_path", "")),
                    "board": [resolved.board.row, resolved.board.col],
                    "shortlist": shortlist,
                    "temperature": temperature,
                    # Charged once per gate step whatever the schedule, so it sets
                    # the constant every ratio in a report is measured against.
                    "inter_zone_time": resolved.device.inter_zone_time,
                    "beam": dict(resolved.get("beam", {})),
                },
                "mean_cost": {
                    name: sum(e["cost"] for e in episodes) / len(episodes)
                    for name, episodes in runs.items()
                },
                # The share of circuits where one shot scheduling beat the
                # heuristic, which is what a surrogate has to do to be useful.
                "wins_vs_greedy": sum(
                    learned_episode["cost"] < greedy_episode["cost"]
                    for learned_episode, greedy_episode in zip(
                        runs["learned"], reference
                    )
                ),
                "gap_percent_vs_greedy": 100.0
                * (
                    sum(e["cost"] for e in runs["learned"])
                    - sum(e["cost"] for e in reference)
                )
                / sum(e["cost"] for e in reference),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
