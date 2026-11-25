# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the policy PPO starts from, by imitating the heuristic.

Label states with greedy's choice and the actor learns to represent it, which is
what lets training start at the baseline's quality instead of at noise. Two things
decide whether that start survives contact with PPO. The critic is warm started on
the same data, because an untrained one makes the first advantages noise and the
actor's entropy is spent before they mean anything. And the roll-in can be handed
to the policy itself over several rounds, which is where a 90% accurate policy on
a long circuit compounds its mistakes.

    PYTHONPATH=. python bin/imitate.py benchmark.id=qaoa-14 board.row=10 board.col=4 imitation.rounds=3
"""

import copy
import json
import random
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
from gymnasium.spaces import flatten
from ml_collections import ConfigDict
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Categorical
from torch.nn import functional as F

from qds.qcgame.heuristic import POLICIES, GreedyPolicy, SampledGreedyPolicy
from qds.task.container import Container
from qds.task.obs import unflatten_batch


FIELDS = ("states", "labels", "seats", "scores", "rewards", "lengths")


def save_dataset(path, data, costs, edges):
    """Keep a collected set on disk, because a search teacher is expensive.

    Thousands of restarts per episode cost hours, the same set is then wanted by
    every training variant, and the restarts parallelise across processes while
    the fit does not.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **dict(zip(FIELDS, data)),
        costs=np.asarray(costs, dtype=np.float64),
        edges=np.asarray(edges, dtype=np.float64),
    )


def load_datasets(paths):
    """Concatenate sets collected by separate processes."""
    parts = [np.load(path) for path in paths]
    data = [np.concatenate([part[field] for part in parts]) for field in FIELDS]
    costs = np.concatenate([part["costs"] for part in parts]).tolist()
    edges = np.concatenate([part["edges"] for part in parts]).tolist()
    return data, costs, edges


def rollout_cost(env, policy):
    """Play a policy out on a copy of the state the episode starts in."""
    clone = copy.deepcopy(env)
    actions = []
    while True:
        action = policy.action(clone)
        actions.append(action)
        _, _, terminated, truncated, _ = clone.step(action)
        if terminated or truncated:
            return clone.log.cost, actions


def demonstrate(env, restarts, epsilon, seed, config, name="sampled_greedy"):
    """The schedule the teacher produces from this starting state.

    Imitating greedy caps a policy at greedy, and on an unseen circuit the policy
    is measured against greedy, so the teacher has to be something greedy loses to.
    A restarted sampled greedy is the cheapest such teacher but a poor one: the
    rollout that wins is the one whose random departures happened to pay, so its
    individual moves are mostly noise and fitting them better does not schedule
    better. A deterministic search decides every move for a reason and needs one
    rollout. Greedy is played either way, so the edge being taught is known.
    """
    reference, _ = rollout_cost(env, GreedyPolicy(config))
    if name != "sampled_greedy":
        cost, actions = rollout_cost(env, POLICIES[name][1](config))
        return cost, actions, reference
    best = None
    for index in range(restarts):
        cost, actions = rollout_cost(
            env, SampledGreedyPolicy(config, epsilon, seed=seed * restarts + index)
        )
        if best is None or cost < best[0]:
            best = (cost, actions)
    return best[0], best[1], reference


def collect(env, episodes, behaviour, expert, search=None, teacher=None):
    """Roll in with ``behaviour`` and label every state with the expert's choice.

    A ``teacher`` decides at the state it is asked about rather than planning a
    whole schedule, which is what lets the roll-in be somebody else: the states
    worth labelling are the ones the learner reaches, and a plan cannot say
    anything about those.

    The return to go is recorded alongside the label because the critic has to be
    warm started too: handed a trained actor and an untrained critic, PPO spends
    its first twenty iterations on advantages that are noise, and the actor's
    entropy is gone by the time the critic is worth anything.
    """
    space = env.observation_space
    width = env.action_space.n
    states, labels, seats, rewards, lengths, costs = [], [], [], [], [], []
    scores = []
    edges = []
    for episode in range(episodes):
        observation, _ = env.reset()
        # With a search teacher the schedule is planned before the episode is
        # walked, because a plan is what a search produces; the walk then only
        # replays it and records what the state looked like at each decision.
        plan = search(env, episode) if search is not None else None
        if plan is not None:
            edges.append((plan[2] - plan[0]) / plan[2])
        length = 0
        while True:
            states.append(flatten(space, observation))
            # The expert's whole ranking, not just its choice: one label out of
            # two hundred sites is a thin target to fit, and most of the sites
            # it rejects are rejected by a wide and informative margin.
            actions, values = expert.scores(env)
            row = np.full(width, np.inf, dtype=np.float32)
            row[actions] = values
            scores.append(row)
            if teacher is not None:
                labels.append(teacher.action(env))
            elif plan is not None:
                labels.append(plan[1][length])
            else:
                labels.append(actions[int(np.argmin(row[actions]))])
            # Where the acting atom already is, so that staying put can be told
            # apart from a move: on a board this size the expert stays most of
            # the time, and an actor that only learns that scores well while
            # rolling out exactly the no-reconfiguration cost.
            seats.append(env.board.atom_map[env.turn.current_player()])
            observation, reward, terminated, truncated, _ = env.step(
                plan[1][length]
                if plan is not None and teacher is None
                else behaviour(env, observation)
            )
            rewards.append(reward)
            length += 1
            if terminated or truncated:
                break
        lengths.append(length)
        costs.append(env.log.cost)
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(seats, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(lengths, dtype=np.int64),
        costs,
        edges,
    )


def value_targets(rewards, values, lengths, gamma, lam):
    """The target PPO regresses on, so that the warm started critic is on its scale.

    The full return to go is far noisier than what PPO ever fits: its own targets
    come from a truncated advantage, which is why its critic reaches an explained
    variance of 0.6 while a critic fitted on the total return stays below zero.
    Episodes end here, so the last step has nothing to bootstrap from.
    """
    targets = torch.zeros_like(values)
    offset = 0
    for length in lengths.tolist():
        advantage = 0.0
        for index in reversed(range(offset, offset + length)):
            last = index == offset + length - 1
            next_value = 0.0 if last else values[index + 1]
            delta = rewards[index] + gamma * next_value - values[index]
            advantage = delta + gamma * lam * (0.0 if last else advantage)
            targets[index] = advantage + values[index]
        offset += length
    return targets


@torch.no_grad()
def sample_action(policy, space, device, env, observation):
    """The learned policy's own choice, sampled, for rolling itself in."""
    policy.eval()
    flat = torch.as_tensor(
        flatten(space, observation), dtype=torch.float32, device=device
    ).unsqueeze(0)
    logits = policy(unflatten_batch(space, flat))
    return int(Categorical(logits=logits).sample().item())


def cost_row(expert, env, width, device, dtype=torch.float32):
    """The estimator's cost for every legal site, laid out over the action space.

    Occupied sites are infinite, which is what marks them illegal everywhere the
    row is consumed: as a mask, as a target, and as an input to the policy.
    """
    actions, values = expert.scores(env)
    row = torch.full((1, width), torch.inf, dtype=dtype, device=device)
    row[0, actions] = torch.as_tensor(values, dtype=dtype, device=device)
    return row


def needs_cost(policy, shortlist):
    return shortlist > 0 or getattr(policy, "cost_mode", "off") != "off"


def masked_sample(policy, space, device, shortlist, expert, temperature):
    """Roll the policy in the way it is actually used: inside the shortlist.

    Rolling in unmasked visits states the policy will never see, and a shortlisted
    policy has no reason to have learnt anything sensible about the sites the mask
    hides from it, so the walk would wander off immediately.
    """

    @torch.no_grad()
    def choose(env, observation):
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
        if temperature <= 0:
            return int(logits.argmax(dim=-1).item())
        return int(Categorical(logits=logits / temperature).sample().item())

    return choose


def expert_target(scores, labels, temperature):
    """Boltzmann distribution over the expert's costs, on a per state scale.

    Costs differ by orders of magnitude between boards and between steps, so the
    spread of each state's own candidates sets the scale; the temperature then
    says how much of the ranking below the best move to keep.
    """
    if temperature <= 0:
        return F.one_hot(labels, scores.shape[-1]).float()
    legal = torch.isfinite(scores)
    deltas = torch.where(legal, scores, 0.0)
    deltas = deltas - deltas.amin(dim=-1, keepdim=True)
    scale = (deltas.sum(-1, keepdim=True) / legal.sum(-1, keepdim=True)).clamp(min=1e-6)
    logits = torch.where(legal, -deltas / (temperature * scale), -torch.inf)
    return torch.softmax(logits, dim=-1)


def anneal(settings, epoch):
    """Geometric path from the ranking temperature down to the handover one."""
    start = float(settings.get("soft_temperature_start", settings.soft_temperature))
    final = float(settings.get("final_temperature", 0.0))
    if start <= 0 or final <= 0 or settings.epochs < 2:
        return start
    return start * (final / start) ** (epoch / (settings.epochs - 1))


def shortlist_mask(scores, seats, k):
    """Keep only the k sites the estimator likes best, plus staying put.

    The estimator is available to any scheduler at decision time, greedy pays for
    exactly this ranking, and the measurement says a search teacher's move is
    inside the three cheapest 90% of the time. Handing the policy that shortlist
    replaces a fifty five way ranking, which it cannot do, with a three way choice,
    and bounds what a wrong answer costs below what staying costs.
    """
    legal = torch.isfinite(scores)
    filled = torch.where(legal, scores, torch.inf)
    keep = torch.zeros_like(legal)
    keep.scatter_(1, filled.topk(k, dim=1, largest=False).indices, True)
    keep.scatter_(1, seats.unsqueeze(1), True)
    return keep & legal


def cost_targets(scores, legal):
    """Each candidate's cost below the best one, on the state's own scale.

    Fitting the expert's choice supplies one number per state out of a hundred
    sites, and the decision only needs the ranking near the top. Regressing the
    whole row supplies a hundred, and its argmax is the choice, so the quantity
    being minimised is the regret the schedule actually pays.
    """
    deltas = torch.where(legal, scores, 0.0)
    deltas = deltas - deltas.amin(dim=-1, keepdim=True)
    scale = (deltas.sum(-1, keepdim=True) / legal.sum(-1, keepdim=True)).clamp(min=1e-6)
    return -deltas / scale


def cost_loss(logits, scores, legal, focus=0.0):
    """Regress the row, weighted towards its top if ``focus`` is set.

    Spread evenly the fit is dominated by the many sites the expert rejects by a
    wide margin, and the argmax stays poor where it matters. ``focus`` is the
    temperature of a weight that concentrates on the sites near the best one.
    """
    target = cost_targets(scores, legal)
    error = F.smooth_l1_loss(
        torch.where(legal, logits, 0.0),
        torch.where(legal, target, 0.0),
        reduction="none",
    )
    if focus > 0:
        weight = torch.softmax(torch.where(legal, target / focus, -torch.inf), dim=-1)
    else:
        weight = legal.float() / legal.sum(-1, keepdim=True)
    return (error * weight).sum(-1)


def imitation_loss(policy, space, states, labels, seats, scores, targets, settings):
    """Match the expert's choice and its return, at a chosen level of confidence.

    Fitting a one-hot target leaves the policy so certain that PPO's first thirty
    iterations are spent with an entropy of 0.005 and no new episodes to learn
    from. Spreading a little of the target over the legal moves sets the entropy
    the warm start hands over, which is the level exploration needs.

    Staying put is a single class holding four fifths of the labels while the
    moves are spread over a hundred sites, so on a large board the cheapest fit
    is to never move. ``move_weight`` pays for the states that separate the two.
    """
    obs = unflatten_batch(space, states)
    # The estimator's full ranking, before any shortlist trims it: what the policy
    # is allowed to choose from and what it is allowed to know are separate.
    cost = scores
    logits = policy(obs, cost)
    shortlist = int(settings.get("shortlist", 0))
    if shortlist > 0:
        keep = shortlist_mask(scores, seats, shortlist)
        logits = torch.where(keep, logits, -torch.inf)
        # The target has to live on the shortlist too, or it puts mass where the
        # policy now has none and the loss is infinite.
        scores = torch.where(keep, scores, torch.inf)
    log_probs = torch.log_softmax(logits, dim=-1)
    temperature = float(settings.get("soft_temperature", 0.0))
    smoothing = settings.get("smoothing", 0.0)
    if settings.get("objective", "choice") == "cost":
        sample = cost_loss(
            logits,
            scores,
            torch.isfinite(logits),
            float(settings.get("cost_focus", 0.0)),
        )
    elif temperature > 0 or smoothing > 0:
        legal = torch.isfinite(logits)
        target = expert_target(scores, labels, temperature)
        # A search teacher's move is often not the myopically cheapest one, which
        # is the whole reason to use it, so the ranking alone would train it away.
        demo = float(settings.get("demo_weight", 0.0))
        if demo > 0:
            target = target * (1.0 - demo) + demo * F.one_hot(
                labels, target.shape[-1]
            ).float()
        if smoothing > 0:
            target = target * (1.0 - smoothing)
            target = target + smoothing * legal.float() / legal.sum(-1, keepdim=True)
        sample = -(target * torch.where(legal, log_probs, 0.0)).sum(-1)
    else:
        sample = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    weight = torch.where(
        labels != seats,
        torch.as_tensor(float(settings.get("move_weight", 1.0)), device=sample.device),
        torch.ones((), device=sample.device),
    )
    chosen = (weight * sample).sum() / weight.sum()
    # The regression is Huber because a squared error on returns of this size
    # drowns the actor's gradient.
    value = policy.get_value(obs, cost).squeeze(-1)
    return chosen + settings.value_coef * F.smooth_l1_loss(value, targets)


def stay_penalty(labels, seats, scores):
    """What leaving the atom alone costs where the expert moves it.

    A policy that never reconfigures is what both imitation and PPO fall onto, so
    the size of the mistake it is making per decision has to be known before any
    of it is blamed on the model: if staying is nearly free at every single step,
    no per step target can teach the schedule.
    """
    moves = labels != seats
    if not moves.any():
        return {}
    rows = np.arange(labels.shape[0])
    stay = scores[rows, seats]
    # The cheapest site available, not the one the teacher took: a search teacher
    # departs from the myopic choice on purpose, and the bar staying sets does not
    # depend on who is being imitated.
    best = np.where(np.isfinite(scores), scores, np.inf).min(axis=1)
    excess = (stay - best)[moves]
    excess = excess[np.isfinite(excess)]
    typical = np.nanmean(np.where(np.isfinite(scores), scores, np.nan))
    # How much precision moving demands: a pick is worth making only if it lands
    # within the stay penalty of the best site, so the share of sites that
    # qualify says whether the target is a needle or a wide basin, and a uniform
    # pick among the legal ones is the regret to beat.
    rows_moved = np.where(moves)[0]
    deltas = scores[rows_moved] - best[rows_moved, None]
    departures = float(
        (scores[rows, labels][moves] > best[moves] + 1e-9).mean()
    )
    finite = np.isfinite(deltas)
    threshold = float(excess.mean())
    within = ((deltas <= threshold) & finite).sum(1) / finite.sum(1)
    random_excess = np.where(finite, deltas, 0.0).sum(1) / finite.sum(1)
    return {
        "stay_excess_mean": threshold,
        "stay_excess_median": float(np.median(excess)),
        "stay_excess_p90": float(np.quantile(excess, 0.9)),
        # Against the cost of a decision, so that "nearly free" has a scale.
        "stay_excess_over_typical": float(threshold / typical),
        "sites_within_stay": float(within.mean()),
        "random_choice_excess": float(random_excess.mean()),
        # Zero when greedy is the teacher; with a search teacher it is the share of
        # moves that greedy would not have made, which is what there is to gain.
        "teacher_departures": departures,
    }


def keep_moves(data, val_fraction):
    """Drop the states where the expert leaves the atom alone.

    Fitting the moves alone answers whether they are learnable at all, which the
    joint fit cannot: there the majority class is available as an escape. The
    surviving states no longer form episodes, so they are handed over as two
    blocks that the usual split separates into training and validation; the value
    targets built along those blocks are meaningless and the critic is ignored.
    """
    states, labels, seats, scores, rewards, lengths = data
    moves = labels != seats
    total = int(moves.sum())
    held = max(1, int(total * val_fraction))
    return (
        states[moves],
        labels[moves],
        seats[moves],
        scores[moves],
        rewards[moves],
        np.asarray([total - held, held], dtype=np.int64),
    )


def fit_round(
    policy, space, data, config, device, optimizer, settings, env=None, expert=None
):
    """Report the bar the data sets, then fit whichever states are being fitted."""
    penalty = stay_penalty(data[1], data[2], data[3])
    print(
        "  staying instead of moving costs "
        + "  ".join(f"{key}={value:.4f}" for key, value in penalty.items()),
        flush=True,
    )
    fit = (
        keep_moves(data, settings.val_fraction)
        if settings.get("moves_only", False)
        else data
    )
    return train(policy, space, fit, config, device, optimizer, env, expert)


def train(policy, space, data, config, device, optimizer, env=None, expert=None):
    settings = config.imitation
    states, labels, seats, scores, rewards, lengths = data
    # Whole episodes on each side, since a value target is built along an episode.
    episodes = len(lengths)
    keep = max(1, int(episodes * (1.0 - settings.val_fraction)))
    split = int(lengths[:keep].sum())
    parts = (states, labels, seats, scores, rewards)
    tensors = [torch.as_tensor(array, device=device) for array in parts]
    all_states, all_labels, all_seats, all_scores, all_rewards = tensors
    train_states, train_labels = all_states[:split], all_labels[:split]
    train_seats, train_scores = all_seats[:split], all_scores[:split]

    history = []
    # What the schedule pays is the regret of the chosen move, and fitting the
    # expert's exact choice keeps improving the accuracy long after the regret has
    # turned around, so the weights are taken from where the regret was lowest.
    select = settings.get("select", "last")
    best = None
    if "soft_temperature_start" not in settings:
        settings.soft_temperature_start = settings.soft_temperature
    for epoch in range(settings.epochs):
        # Learning the ranking wants a soft target; handing the weights to PPO
        # wants a confident one, since an entropy of four nats over two hundred
        # sites samples nothing the controller can work with. Do both in order.
        settings.soft_temperature = anneal(settings, epoch)
        targets = fitted_targets(
            policy, space, all_states, all_rewards, lengths, config, settings, all_scores
        )
        train_returns = targets[:split]
        validation = (
            all_states[split:],
            all_labels[split:],
            all_seats[split:],
            all_scores[split:],
            targets[split:],
        )
        policy.train()
        order = torch.randperm(train_states.shape[0], device=device)
        losses = []
        for offset in range(0, order.shape[0], settings.batch_size):
            batch = order[offset : offset + settings.batch_size]
            loss = imitation_loss(
                policy,
                space,
                train_states[batch],
                train_labels[batch],
                train_seats[batch],
                train_scores[batch],
                train_returns[batch],
                settings,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                **score(
                    policy,
                    space,
                    *validation,
                    settings.batch_size,
                    int(settings.get("shortlist", 0)),
                ),
            }
        )
        # Under a shortlist the regret against the myopic best is the wrong
        # selector, because departing from it is exactly what a search teacher is
        # being imitated for. Agreement with the teacher is no better: across
        # poolings the epochs that agree most schedule worst, so ``cost`` plays the
        # weights out on a fixed handful of boards and reads the answer off those.
        ranked = {
            "regret": history[-1]["regret"],
            "move": -history[-1]["move_accuracy"],
            "cost": partial(
                rollout_ratio,
                policy,
                env,
                device,
                int(settings.get("shortlist", 0)),
                expert,
                int(settings.get("select_episodes", 3)),
                int(config.seed) + 991,
            ),
        }.get(select)
        if callable(ranked):
            ranked = ranked()
            history[-1]["cost_ratio"] = ranked
        if ranked is not None and (best is None or ranked < best[0]):
            best = (
                ranked,
                {
                    name: tensor.detach().clone()
                    for name, tensor in policy.state_dict().items()
                },
            )
        print(
            f"  epoch {history[-1]['epoch']:>3}  loss {history[-1]['loss']:.4f}"
            f"  val accuracy {history[-1]['val_accuracy']:.4f}"
            f"  move accuracy {history[-1]['move_accuracy']:.4f}"
            f"  regret {history[-1]['regret']:.3f}"
            f"  value ev {history[-1]['value_explained_variance']:.3f}"
            f"  entropy {history[-1]['entropy']:.3f}"
            + (
                f"  cost {history[-1]['cost_ratio']:+.4f}"
                if "cost_ratio" in history[-1]
                else ""
            ),
            flush=True,
        )
    if best is not None:
        policy.load_state_dict(best[1])
        print(f"  kept the weights scoring {best[0]:.4f} on {select}", flush=True)
    return history


@torch.no_grad()
def fitted_targets(
    policy, space, states, rewards, lengths, config, settings, scores=None
):
    """Value targets from the critic as it stands, refreshed once per epoch."""
    policy.eval()
    values = []
    for offset in range(0, states.shape[0], settings.batch_size):
        window = slice(offset, offset + settings.batch_size)
        cost = None if scores is None else scores[window]
        values.append(
            policy.get_value(unflatten_batch(space, states[window]), cost).squeeze(-1)
        )
    return value_targets(
        rewards,
        torch.concat(values),
        lengths,
        config.training.gamma,
        config.training.gae_lambda,
    )


@torch.no_grad()
def score(policy, space, states, labels, seats, scores, returns, batch_size, shortlist=0):
    """Accuracy of the actor and explained variance of the critic, held out.

    Accuracy is also reported on the states where the expert moves the atom,
    because those are the only ones that separate the policy from leaving the
    board alone, and they are a minority of the data. Matching the expert
    exactly matters less than what missing costs, so the regret of the chosen
    move against the best one is reported in the units the benchmark uses.
    """
    policy.eval()
    correct = 0
    moved_correct = 0
    predictions = []
    entropies = []
    regrets = []
    for offset in range(0, states.shape[0], batch_size):
        window = slice(offset, offset + batch_size)
        obs = unflatten_batch(space, states[window])
        logits = policy(obs, scores[window])
        if shortlist > 0:
            logits = torch.where(
                shortlist_mask(scores[window], seats[window], shortlist),
                logits,
                -torch.inf,
            )
        choice = logits.argmax(dim=-1)
        hit = choice == labels[window]
        correct += hit.sum().item()
        moved_correct += (hit & (labels[window] != seats[window])).sum().item()
        picked = scores[window].gather(1, choice.unsqueeze(1)).squeeze(1)
        best = torch.where(
            torch.isfinite(scores[window]), scores[window], torch.inf
        ).amin(dim=-1)
        regrets.append(picked - best)
        predictions.append(policy.get_value(obs, scores[window]).squeeze(-1))
        entropies.append(Categorical(logits=logits).entropy())
    predicted = torch.concat(predictions)
    residual = (returns - predicted).var(unbiased=False)
    moves = int((labels != seats).sum().item())
    return {
        "val_accuracy": correct / states.shape[0],
        "move_accuracy": moved_correct / max(1, moves),
        "move_fraction": moves / states.shape[0],
        # An illegal pick has an infinite score, which would hide everything else.
        "regret": float(torch.concat(regrets).nan_to_num(posinf=0.0).mean()),
        "value_explained_variance": float(
            1.0 - residual / returns.var(unbiased=False).clamp(min=1e-8)
        ),
        # What PPO starts from, and what its entropy controller would otherwise
        # have to claw back at one percent per iteration.
        "entropy": float(torch.concat(entropies).mean()),
    }


@torch.no_grad()
def rollout(policy, env, device, shortlist=0, expert=None):
    """Cost of one deterministic rollout, the cell the benchmark never reported.

    Under a shortlist the estimator ranks the sites and the policy only re-ranks
    the cheapest few, so the rollout has to be played the same way or its cost
    says nothing about the weights being fitted.
    """
    policy.eval()
    space = env.observation_space
    observation, _ = env.reset()
    # Greedy on the very board this episode drew, since in a pool the board and the
    # circuit change with every reset and a cost on its own compares to nothing.
    reference = rollout_cost(env, expert)[0] if expert is not None else None
    while True:
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
        action = logits.argmax(dim=-1).item()
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return env.log.cost, reference


def rollout_ratio(policy, env, device, shortlist, expert, count, seed):
    """What the weights cost against greedy on the same few boards every epoch.

    The pool draws another circuit on every reset and the circuits differ in cost
    by far more than any schedule does, so the seed pins which ones are played and
    the ratio to greedy is what carries across them.
    """
    state = random.getstate()
    random.seed(seed)
    ratios = []
    for _ in range(count):
        cost, reference = rollout(policy, env, device, shortlist, expert)
        ratios.append(cost / reference)
    random.setstate(state)
    return float(np.mean(ratios))


def compare(policy, env, device, shortlist, expert, count):
    """Argmax rollouts against greedy on the boards they were played on.

    One rollout on a pool says nothing: the next reset draws another circuit whose
    cost differs from this one by more than any schedule does.
    """
    ours, theirs = [], []
    for _ in range(count):
        cost, reference = rollout(policy, env, device, shortlist, expert)
        ours.append(cost)
        theirs.append(reference)
    wins = sum(a < b for a, b in zip(ours, theirs))
    print(
        f"argmax rollouts: {np.mean(ours):.2f} against greedy's {np.mean(theirs):.2f} "
        f"on the same boards, {100 * (np.mean(ours) / np.mean(theirs) - 1):+.2f}%, "
        f"winning {wins} of {count}",
        flush=True,
    )
    return float(np.mean(ours))


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(config: DictConfig) -> None:
    resolved = ConfigDict(OmegaConf.to_container(config, resolve=True))
    if resolved.mode == "fixed":
        problem, n_atoms = resolved.benchmark.id.rsplit("-", maxsplit=1)
        resolved.benchmark.problem = problem
        resolved.benchmark.n_atoms = int(n_atoms)
    device = resolved.system.device if torch.cuda.is_available() else "cpu"

    # In transferable mode every reset draws another circuit, so the same
    # collection loop turns into cloning greedy across the training pool.
    container = Container(resolved.mode)
    env = container.get_env(resolved).unwrapped
    policy = container.get_policy(resolved).to(device)
    settings = resolved.imitation

    space = env.observation_space
    noisy = SampledGreedyPolicy(resolved, epsilon=settings.epsilon)
    expert = GreedyPolicy(resolved)
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate)

    if settings.get("init_model"):
        policy.load_state_dict(
            torch.load(settings.init_model, map_location="cpu", weights_only=True),
            strict=True,
        )
        print(f"started from {settings.init_model}", flush=True)

    restarts = int(settings.get("demo_restarts", 0))
    demo_policy = settings.get("demo_policy", "sampled_greedy")
    search = (
        partial(
            lambda env, episode: demonstrate(
                env,
                restarts,
                float(settings.get("demo_epsilon", 0.02)),
                resolved.seed * 1000 + episode,
                resolved,
                demo_policy,
            )
        )
        if (restarts > 0 or demo_policy != "sampled_greedy")
        and not settings.get("dagger", False)
        else None
    )
    # A teacher that answers per state replaces the plan, and then the walk belongs
    # to the learner: label the states its own mistakes take it to. Planning as well
    # would only report what the teacher scores, at the price of a second rollout.
    teacher = (
        POLICIES[demo_policy][1](resolved)
        if settings.get("dagger", False) and demo_policy != "sampled_greedy"
        else None
    )

    rounds = settings.get("rounds", 1)
    episodes = max(1, settings.episodes // rounds)
    data = None
    expert_costs = []
    round_costs = []

    if settings.get("dataset_inputs"):
        paths = sorted(Path().glob(settings.dataset_inputs))
        if not paths:
            raise FileNotFoundError(f"No dataset matched {settings.dataset_inputs!r}")
        # Keeping a prefix of the collected files is how much data is varied, since
        # every file is an independent collector on the same pool.
        limit = int(settings.get("dataset_limit", 0))
        if limit > 0:
            paths = paths[:limit]
        data, expert_costs, edges = load_datasets(paths)
        print(
            f"loaded {len(data[0])} states from {len(paths)} files, "
            f"{len(expert_costs)} episodes, teacher best "
            f"{min(expert_costs):.2f}, mean {np.mean(expert_costs):.2f}",
            flush=True,
        )
        if edges:
            print(
                f"  the teacher beats greedy by {100 * np.mean(edges):.2f}% "
                f"on average, {100 * max(edges):.2f}% at best",
                flush=True,
            )
        rounds = 0

    for index in range(rounds):
        # The first round rolls in with the noisy expert because there is nothing
        # else yet; afterwards the policy visits its own states, which is where
        # its mistakes compound and where labels are therefore worth most.
        if teacher is not None:
            policy.eval()
            roll_in = masked_sample(
                policy,
                space,
                device,
                int(settings.get("shortlist", 0)),
                expert,
                float(settings.get("roll_in_temperature", 0.05)),
            )
        elif index == 0:
            roll_in = lambda env, observation: noisy.action(env)
        else:
            roll_in = partial(sample_action, policy, space, device)
        batch = collect(env, episodes, roll_in, expert, search, teacher)
        expert_costs.extend(batch[6])
        data = batch[:6] if data is None else [
            np.concatenate([old, new]) for old, new in zip(data, batch[:6])
        ]
        print(
            f"round {index + 1}: {len(batch[0])} new states, {len(data[0])} total; "
            f"roll-in best {min(batch[6]):.2f}, mean {np.mean(batch[6]):.2f}",
            flush=True,
        )
        if batch[7]:
            print(
                f"  the teacher beats greedy by {100 * np.mean(batch[7]):.2f}% "
                f"on average, {100 * max(batch[7]):.2f}% at best",
                flush=True,
            )
        if settings.get("dataset_path"):
            save_dataset(settings.dataset_path, data, batch[6], batch[7])
            print(f"saved the collected set to {settings.dataset_path}", flush=True)
        if settings.get("collect_only", False):
            return
        history = fit_round(
            policy, space, data, resolved, device, optimizer, settings, env, expert
        )
        round_costs.append(
            compare(
                policy,
                env,
                device,
                int(settings.get("shortlist", 0)),
                expert,
                int(settings.get("eval_episodes", 5)),
            )
        )

    if data is not None and not round_costs:
        history = fit_round(
            policy, space, data, resolved, device, optimizer, settings, env, expert
        )
        round_costs.append(
            compare(
                policy,
                env,
                device,
                int(settings.get("shortlist", 0)),
                expert,
                int(settings.get("eval_episodes", 5)),
            )
        )

    expert = min(expert_costs)
    costs = expert_costs
    imitated = round_costs[-1]
    if settings.get("save_path"):
        path = Path(settings.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), path)
        print(f"saved imitation weights to {path}", flush=True)
    print(
        json.dumps(
            {
                "benchmark": resolved.benchmark.id,
                "board": [resolved.board.row, resolved.board.col],
                "seed": resolved.seed,
                "geometry": resolved.policy.geometry,
                "cross_task_layers": resolved.policy.cross_task_layers,
                "pool": resolved.policy.pool,
                "states": len(data[0]),
                "rounds": rounds,
                "val_accuracy": history[-1]["val_accuracy"],
                "move_accuracy": history[-1]["move_accuracy"],
                "move_fraction": history[-1]["move_fraction"],
                "regret": history[-1]["regret"],
                "soft_temperature": settings.get("soft_temperature", 0.0),
                "value_explained_variance": history[-1]["value_explained_variance"],
                "entropy": history[-1]["entropy"],
                "expert_best_cost": expert,
                "expert_mean_cost": float(np.mean(costs)),
                "round_costs": round_costs,
                "imitated_cost": imitated,
                "gap_percent": 100.0 * (imitated - expert) / expert,
            },
            indent=2,
        )
    )
    env.close()


if __name__ == "__main__":
    main()
