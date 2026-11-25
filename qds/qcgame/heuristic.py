# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search policies that act on the environment directly.

These use the cost estimator, which is an analytic model of the hardware and is
therefore available to any scheduler at decision time. They serve both as
benchmark baselines and as a source of good trajectories for training.
"""

import copy
import random


class GreedyPolicy:
    """Choose the legal move with the lowest immediate physical cost."""

    deterministic = True

    def __init__(self, config=None):
        self.evaluations = 0

    def action(self, env):
        player = env.turn.current_player()
        current_position = env.board.atom_map[player]
        return self._best(env, player, current_position)

    def scores(self, env):
        """Cost of every legal move, which choosing the best one already computes.

        Imitation throws all but the argument of the minimum away, though the
        ranking of the other candidates is the part that says how a board is
        read. Handing it out costs nothing.
        """
        player = env.turn.current_player()
        current_position = env.board.atom_map[player]
        actions = [current_position, *env.board.empties()]
        return actions, [self._score(env, player, action) for action in actions]

    def _best(self, env, player, current_position):
        scored_actions = [
            (
                self._score(env, player, action),
                action != current_position,
                action,
            )
            for action in [current_position, *env.board.empties()]
        ]
        return min(scored_actions)[2]

    def _score(self, env, player, action):
        self.evaluations += 1
        candidate_board = env.board.transform(player, action)
        candidate_moves = list(env.turn.moves)
        move = env.board.get_atom_move(player, action)
        if move is not None:
            candidate_moves.append(move)

        gate_cost = env.estimator.gate_cost(
            candidate_board, env.next_task.twoq_gates(tuple=True)
        )
        reconfiguration_cost = (
            env.estimator.reconfiguration_cost(candidate_moves)
            if candidate_moves
            else 0
        )
        return gate_cost + reconfiguration_cost


class NoReconfigurationPolicy:
    """Keep every atom at its current position."""

    deterministic = True

    def __init__(self, config=None):
        self.evaluations = 0

    @staticmethod
    def action(env):
        player = env.turn.current_player()
        return env.board.atom_map[player]


class RandomPolicy:
    """Pick uniformly among the legal moves.

    Together with greedy this brackets how much the choice of move matters at
    all, which is what decides whether a benchmark can separate methods.
    """

    deterministic = False

    def __init__(self, config=None, seed=None):
        self.evaluations = 0
        if seed is None:
            seed = config.seed if config else None
        self.rng = random.Random(seed)

    def action(self, env):
        player = env.turn.current_player()
        return self.rng.choice([env.board.atom_map[player], *env.board.empties()])


class SampledGreedyPolicy(GreedyPolicy):
    """Greedy that now and then takes a random legal move instead.

    Deterministic greedy is judged on one rollout while PPO reports the best of
    tens of thousands, so restarting this policy is what puts a search baseline
    on the same budget as the learner.
    """

    deterministic = False

    def __init__(self, config=None, epsilon=None, seed=None):
        super().__init__(config)
        # Restarts have to differ from each other, so the seed is separable from
        # the run's: sharing the run's would make every restart the same rollout.
        if seed is None:
            seed = config.seed if config else None
        self.rng = random.Random(seed)
        if epsilon is None:
            epsilon = config.get("epsilon", 0.1) if config else 0.1
        self.epsilon = epsilon

    def action(self, env):
        player = env.turn.current_player()
        current_position = env.board.atom_map[player]
        if self.rng.random() < self.epsilon:
            return self.rng.choice([current_position, *env.board.empties()])
        return self._best(env, player, current_position)


class ShortlistRandomPolicy(GreedyPolicy):
    """Pick uniformly among the k sites the estimator likes best.

    A learned policy under a shortlist chooses inside exactly this set, so this is
    what the shortlist is worth before anything has been learned, and the learned
    one has to beat it to have contributed at all.
    """

    deterministic = False

    def __init__(self, config=None, k=None, seed=None):
        super().__init__(config)
        if seed is None:
            seed = config.seed if config else None
        self.rng = random.Random(seed)
        if k is None:
            k = config.imitation.get("shortlist", 3) if config else 3
        self.k = max(1, int(k))

    def action(self, env):
        player = env.turn.current_player()
        seat = env.board.atom_map[player]
        actions = [seat, *env.board.empties()]
        ranked = sorted(
            (self._score(env, player, action), action != seat, action)
            for action in actions
        )
        return self.rng.choice([entry[2] for entry in ranked[: self.k]])


class LookaheadPolicy(GreedyPolicy):
    """Judge the shortlist by playing greedy on from each of its sites.

    Greedy cannot see that a cheap site now leaves the atom in the way of the next
    reconfiguration, and a restarted sampled greedy only finds that out by luck, so
    the moves it demonstrates are mostly noise that happened to pay. Rolling each
    candidate forward for a fixed horizon answers the same question systematically,
    at a cost of ``k`` times the horizon in estimator calls per decision.
    """

    deterministic = True
    planning = True

    def __init__(self, config=None, k=None, turns=None):
        super().__init__(config)
        settings = config.get("lookahead", {}) if config is not None else {}
        if k is None:
            k = settings.get("k", 3)
        if turns is None:
            turns = settings.get("turns", 2)
        self.k = max(1, int(k))
        # Whole turns, because a reconfiguration is paid for when its last atom has
        # moved: cutting the rollout inside one compares a candidate that has paid
        # against a candidate that has only deferred.
        self.turns = max(1, int(turns))

    def shortlist(self, env):
        player = env.turn.current_player()
        seat = env.board.atom_map[player]
        ranked = sorted(
            (self._score(env, player, action), action != seat, action)
            for action in [seat, *env.board.empties()]
        )
        return [entry[2] for entry in ranked[: self.k]]

    def action(self, env):
        candidates = self.shortlist(env)
        if len(candidates) == 1:
            return candidates[0]
        return min(candidates, key=lambda action: self._continuation(env, action))

    def _continuation(self, env, action):
        """Cost of taking this site and then behaving greedily for a few turns."""
        twin = copy.deepcopy(env)
        _, _, terminated, truncated, info = twin.step(action)
        turns = int(bool(info.get("turn_end")))
        while not (terminated or truncated) and turns < self.turns:
            player = twin.turn.current_player()
            seat = twin.board.atom_map[player]
            _, _, terminated, truncated, info = twin.step(
                self._best(twin, player, seat)
            )
            turns += int(bool(info.get("turn_end")))
        return twin.log.cost


class BeamSearchPolicy:
    """Plan a whole reconfiguration rather than one atom at a time.

    Greedy takes the cheapest destination for each atom in turn, but what a
    reconfiguration costs is a property of the set of moves it contains: its
    duration is the longest move among them and its conflicts come from atoms
    that end up sharing a row or a column. Choosing the atoms of a turn together
    is what that structure asks for, and it is the standard search baseline a
    per atom rule has to be measured against. Greedy is this policy at width one.
    """

    deterministic = True
    # Ninety times the cost of a greedy rollout, so it is asked for by name
    # rather than swept up by the run that collects the cheap references.
    planning = True

    def __init__(self, config=None, width=None, branch=None, turns=None):
        settings = config.get("beam", {}) if config is not None else {}
        self.width = width if width is not None else settings.get("width", 4)
        self.branch = branch if branch is not None else settings.get("branch", 4)
        # How many reconfigurations to plan ahead. Beyond the first the plan is
        # discarded, but its cost still ranks the moves that led there.
        self.turns = turns if turns is not None else settings.get("turns", 1)
        self.scorer = GreedyPolicy(config)
        self.evaluations = 0
        self.plan = []

    @property
    def _spent(self):
        return self.scorer.evaluations

    def action(self, env):
        player = env.turn.current_player()
        if not self.plan or self.plan[0][0] != player:
            self.plan = self._search(env)
        self.evaluations = self._spent
        return self.plan.pop(0)[1]

    def _search(self, env):
        """Destinations for the atoms of the coming turn, chosen together."""
        beams = [(0.0, [], copy.deepcopy(env), 0, None)]
        done = []
        while beams:
            children = []
            for _, plan, state, turns, cut in beams:
                children.extend(self._expand(plan, state, turns, cut, done))
            beams = sorted(children, key=lambda entry: entry[0])[: self.width]
        # Within a turn nothing is realised yet, so the accumulated cost is what
        # ranks the finished plans; ties fall to the shortest one.
        cost, plan, cut = min(done, key=lambda entry: (entry[0], len(entry[1])))
        return plan[:cut] if cut else plan

    def _expand(self, plan, state, turns, cut, done):
        player = state.turn.current_player()
        actions, costs = self.scorer.scores(state)
        order = sorted(range(len(actions)), key=lambda index: costs[index])
        children = []
        for index in self._branches(state, actions, costs, order):
            twin = copy.deepcopy(state)
            _, _, over, truncated, info = twin.step(actions[index])
            step = plan + [(player, actions[index])]
            ended = bool(info.get("turn_end"))
            # Only the coming turn is played out; further ones just rank it.
            boundary = cut if cut is not None else (len(step) if ended else None)
            if over or truncated or turns + ended >= self.turns:
                done.append((twin.log.cost, step, boundary))
            else:
                children.append(
                    (self._rank(twin, costs[index]), step, twin, turns + ended, boundary)
                )
        return children

    def _branches(self, state, actions, costs, order):
        """Which candidates get a subtree, cheapest by the one step cost."""
        return order[: self.branch]

    def _rank(self, state, cost):
        """What decides which beams survive, lowest kept."""
        return cost


class GuidedBeamPolicy(BeamSearchPolicy):
    """Beam search whose branches a model proposes and whose beams it ranks.

    Beam widens greedy but is ranked by the same quantity greedy is: the one step
    cost of a partial reconfiguration, which within a turn is an estimate of
    something nobody has paid for yet. So the sites it opens a subtree on and the
    plans it keeps are chosen by exactly the myopia the search was built to
    escape, and widening the beam buys less than it should. A fitted actor ranks
    the same candidates having been trained on what the schedule ends up costing,
    and a fitted critic scores a half built plan by what is still to come.

    Neither is asked to propose a site the estimator has not already shortlisted.
    The weights were fitted to choose inside that shortlist and mean nothing
    outside one, and the search pays for the full ranking at every node anyway.
    """

    def __init__(
        self,
        config=None,
        proposer=None,
        valuer=None,
        shortlist=None,
        value_weight=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        settings = config.get("beam", {}) if config is not None else {}
        self.proposer = proposer
        self.valuer = valuer
        # Wider than the branch or the model has nothing left to choose: with a
        # shortlist the size of the branch, its ranking only reorders siblings
        # that all get expanded anyway.
        if shortlist is None:
            shortlist = settings.get("shortlist", 2 * self.branch)
        self.shortlist = max(self.branch, int(shortlist))
        if value_weight is None:
            value_weight = settings.get("value_weight", 1.0)
        self.value_weight = float(value_weight)

    def _branches(self, state, actions, costs, order):
        if self.proposer is None:
            return order[: self.branch]
        pool = order[: self.shortlist]
        preference = self.proposer(state, actions, costs)
        return sorted(pool, key=lambda index: -preference[index])[: self.branch]

    def _rank(self, state, cost):
        if self.valuer is None:
            return cost
        # The critic is fitted on the reward, which is the cost the schedule saves
        # against leaving the board alone, so it is already in the units the beam
        # ranks in. Subtracting it turns the ranking from what a plan has spent
        # into what it is going to have spent.
        return cost - self.value_weight * self.valuer(state)


POLICIES = {
    "greedy": ("greedy_one_step", GreedyPolicy),
    "beam": ("beam_search", BeamSearchPolicy),
    # Needs the fitted model handed to it, so only evaluation builds this one.
    "guided_beam": ("guided_beam", GuidedBeamPolicy),
    "no_reconfiguration": ("no_reconfiguration", NoReconfigurationPolicy),
    "random": ("random", RandomPolicy),
    "sampled_greedy": ("sampled_greedy", SampledGreedyPolicy),
    "shortlist_random": ("shortlist_random", ShortlistRandomPolicy),
    "lookahead": ("lookahead", LookaheadPolicy),
}
