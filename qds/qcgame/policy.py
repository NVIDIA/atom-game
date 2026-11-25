# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.distributions import Categorical

pair = lambda x: x if isinstance(x, tuple) else (x, x)


def _is_transferable(cfg):
    return cfg.get("mode") == "transferable"


def _atom_capacity(cfg):
    if _is_transferable(cfg):
        return cfg.benchmark.max_atoms
    return cfg.benchmark.n_atoms


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout),
    )


class MLPMixer(nn.Module):
    def __init__(
        self,
        image_size,
        channels,
        patch_size,
        dim,
        depth,
        expansion_factor=4,
        expansion_factor_token=0.5,
        dropout=0.0,
    ):
        super().__init__()
        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (
            image_w % patch_size
        ) == 0, "image must be divisible by patch size"
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.net = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear((patch_size**2) * channels, dim),
            *[
                nn.Sequential(
                    PreNormResidual(
                        dim,
                        FeedForward(num_patches, expansion_factor, dropout, chan_first),
                    ),
                    PreNormResidual(
                        dim,
                        FeedForward(dim, expansion_factor_token, dropout, chan_last),
                    ),
                )
                for _ in range(depth)
            ],
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return self.net(x)


class PairGeometry(nn.Module):
    """Coordinate relation of a position pair, projected into the embedding.

    The cost model charges ``sqrt(max(|dx|, |dy|))`` per move and counts
    reconfiguration conflicts from exact row/column matches and order
    inversions. A learned index embedding exposes none of those quantities, so
    ``add`` supplies them alongside it and ``only`` replaces it.
    """

    FEATURES = 10

    def __init__(self, cfg, dim):
        super().__init__()
        mode = cfg.policy.get("geometry", "off")
        # YAML 1.1 turns a bare ``off`` into a boolean; accept both spellings.
        self.mode = "off" if mode in (False, None) else mode
        if self.mode not in ("off", "add", "only"):
            raise ValueError(f"Unsupported geometry mode: {self.mode}")
        self.row = cfg.board.row
        self.size = cfg.board.row * cfg.board.col
        self.scale = float(max(cfg.board.row, cfg.board.col))
        if self.mode != "off":
            self.project = nn.Linear(self.FEATURES, dim)

    def features(self, pairs):
        """``pairs[..., 0]`` is the reference position, ``[..., 1]`` the candidate."""
        inside = ((pairs >= 0) & (pairs < self.size)).all(dim=-1).float()
        index = pairs.clamp(0, self.size - 1)
        column = torch.div(index, self.row, rounding_mode="floor").float()
        row = (index % self.row).float()
        dx = column[..., 0] - column[..., 1]
        dy = row[..., 0] - row[..., 1]
        span = torch.maximum(dx.abs(), dy.abs())
        stacked = torch.stack(
            [
                dx / self.scale,
                dy / self.scale,
                dx.abs() / self.scale,
                dy.abs() / self.scale,
                span / self.scale,
                torch.sign(dx),
                torch.sign(dy),
                (dx == 0).float(),
                (dy == 0).float(),
            ],
            dim=-1,
        )
        # Transferable boards carry a dummy index outside the grid, whose
        # coordinates are meaningless; flag it and zero the rest.
        return torch.cat([stacked * inside.unsqueeze(-1), inside.unsqueeze(-1)], dim=-1)

    def forward(self, diff, pairs):
        if self.mode == "off":
            return diff
        geometry = self.project(self.features(pairs))
        return geometry if self.mode == "only" else diff + geometry


class MoveEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.size = cfg.board.row * cfg.board.col
        self.atom_capacity = _atom_capacity(cfg)
        self.L = self.size + self.atom_capacity
        self.n_heads = cfg.policy.n_heads
        dummy_position = 1 if _is_transferable(cfg) else 0
        self.position_embedding = nn.Embedding(
            self.size + dummy_position, cfg.policy.move_embedding_dim
        )
        self.geometry = PairGeometry(cfg, cfg.policy.move_embedding_dim)

    def build_attention_mask(self, batch_size, mask):
        expanded_row_mask = mask.unsqueeze(1).expand(-1, self.L, -1)
        attn_mask = torch.zeros(batch_size, self.L, self.L, device=mask.device).fill_(
            float("-inf")
        )
        attn_mask[:, :, : self.atom_capacity] = 0
        attn_mask[expanded_row_mask] = -torch.inf
        torch.diagonal(attn_mask, dim1=1, dim2=2).fill_(0)
        return attn_mask.repeat(self.n_heads, 1, 1)

    def forward(self, moves, player_position):
        merged = moves
        pe_encoded = self.position_embedding(merged)
        diff = self.geometry(pe_encoded[:, :, 0, :] - pe_encoded[:, :, 1, :], merged)
        mask = torch.zeros(
            (moves.shape[0], self.L),
            dtype=torch.bool,
            device=moves.device,
        )
        actual_mask = merged[:, :, 0] == merged[:, :, 1]
        mask[:, : self.atom_capacity] = actual_mask
        player_position = player_position.unsqueeze(1).repeat(1, self.size)
        player_next = (
            torch.arange(self.size, device=moves.device)
            .unsqueeze(0)
            .repeat(diff.shape[0], 1)
        )
        player_merged = torch.stack([player_position, player_next], dim=2)
        player_pe_encoded = self.position_embedding(player_merged)
        player_diff = self.geometry(
            player_pe_encoded[:, :, 0, :] - player_pe_encoded[:, :, 1, :],
            player_merged,
        )
        tensor = torch.concat([diff, player_diff], dim=1)
        return tensor, self.build_attention_mask(moves.shape[0], mask)


class TaskContext(nn.Module):
    """How far ahead each lookahead task is, and who still has to move after it.

    ``GateEmbedding`` folds the tasks into the batch dimension, so a task is encoded
    in complete isolation: nothing tells the gate about to run from the one seven
    layers away, and nothing says whether the partner an atom is being placed beside
    will itself be pulled elsewhere immediately afterwards. Both decide whether a
    move pays for itself, which is the judgement the estimator cannot make and the
    only thing left for the policy to contribute.
    """

    FEATURES = 7

    def __init__(self, dim):
        super().__init__()
        self.project = nn.Linear(self.FEATURES, dim)

    def forward(self, tasks, player):
        batch, count, atoms = tasks.shape
        rows = torch.arange(batch, device=tasks.device)
        # An atom with no gate in a task is written as its own partner.
        acts = tasks != torch.arange(atoms, device=tasks.device)
        # What each atom still has to do after each task, which is what makes a
        # partner a moving target rather than a fixed one.
        later = acts.flip(1).cumsum(1).flip(1).float() - acts.float()
        before = acts.cumsum(1).float() - acts.float()
        total = acts.sum(dim=1).float()

        partner = tasks[rows, :, player]
        player_index = player.unsqueeze(1).expand(-1, count)
        slot = (
            torch.arange(count, device=tasks.device, dtype=torch.float32)
            / max(count - 1, 1)
        ).expand(batch, -1)
        features = torch.stack(
            [
                acts[rows, :, player].float(),
                slot,
                before.gather(2, player_index.unsqueeze(-1)).squeeze(-1) / count,
                later.gather(2, player_index.unsqueeze(-1)).squeeze(-1) / count,
                total.gather(1, partner) / count,
                later.gather(2, partner.unsqueeze(-1)).squeeze(-1) / count,
                acts.sum(dim=2).float() / atoms,
            ],
            dim=-1,
        )
        return self.project(features)


class GateEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.size = cfg.board.row * cfg.board.col
        self.feature_count = cfg.env.feature_count
        self.atom_capacity = _atom_capacity(cfg)
        self.L = self.atom_capacity + self.size
        self.n_heads = cfg.policy.n_heads
        dummy_position = 1 if _is_transferable(cfg) else 0
        self.position_embedding = nn.Embedding(
            self.size + dummy_position, cfg.policy.gate_embedding_dim
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.policy.gate_embedding_dim,
            num_heads=cfg.policy.n_heads,
            batch_first=True,
        )
        self.geometry = PairGeometry(cfg, cfg.policy.gate_embedding_dim)
        self.context = (
            TaskContext(cfg.policy.gate_embedding_dim)
            if cfg.policy.get("task_context", False)
            else None
        )

    def build_attention_mask(self, batch_size, mask):
        L = self.size + self.atom_capacity
        expanded_row_mask = mask.unsqueeze(1).expand(-1, L, -1)
        attn_mask = torch.zeros(batch_size, L, L, device=mask.device).fill_(
            float("-inf")
        )
        attn_mask[:, :, : self.atom_capacity] = 0
        attn_mask[expanded_row_mask] = -torch.inf
        torch.diagonal(attn_mask, dim1=1, dim2=2).fill_(0)
        return attn_mask.repeat(self.n_heads, 1, 1)

    def forward(self, board, tasks, player):
        batch_size = board.shape[0]
        board_exp = board.unsqueeze(1).expand(-1, self.feature_count, -1)
        tasks_by_positions = torch.gather(board_exp, 2, tasks)
        board_positions = board_exp[:, : tasks_by_positions.shape[1], :]
        merged = torch.stack([board_positions, tasks_by_positions], dim=2)
        gate_mask = merged[:, :, 0] == merged[:, :, 1]
        player_gate_mask = tasks == player.unsqueeze(1).repeat(
            1, tasks.shape[1]
        ).unsqueeze(-1)
        pe_encoded = self.position_embedding(merged)
        diff = self.geometry(
            pe_encoded[:, :, 0, :] - pe_encoded[:, :, 1, :], merged.movedim(2, -1)
        )
        board_exp = board.unsqueeze(1).expand(-1, self.feature_count, -1)
        pairs = tasks[
            torch.arange(batch_size, device=tasks.device), :, player
        ].unsqueeze(-1)
        pair_gate_mask = tasks == pairs
        gate_mask = torch.logical_or(
            torch.logical_or(gate_mask, player_gate_mask), pair_gate_mask
        )
        pair_positions = torch.gather(board_exp, 2, pairs).repeat(1, 1, self.size)
        player_next = (
            torch.arange(self.size, device=board.device)
            .unsqueeze(0)
            .repeat(batch_size, tasks.shape[1], 1)
        )
        player_merged = torch.stack([pair_positions, player_next], dim=3)
        player_pe_encoded = self.position_embedding(player_merged)
        player_diff = self.geometry(
            player_pe_encoded[:, :, :, 0, :] - player_pe_encoded[:, :, :, 1, :],
            player_merged,
        )
        tensor = torch.concat([diff, player_diff], dim=2)
        if self.context is not None:
            tensor = tensor + self.context(tasks, player).unsqueeze(2)
        # fold into the batch dimension
        tensor = tensor.reshape(batch_size * tasks.shape[1], -1, tensor.shape[-1])
        mask = torch.zeros(
            (batch_size, tasks.shape[1], self.L),
            dtype=torch.bool,
            device=tasks.device,
        )
        mask[:, :, : self.atom_capacity] = gate_mask
        mask = mask.reshape(batch_size * tasks.shape[1], self.L)
        return tensor, self.build_attention_mask(mask.shape[0], mask)


class MoveTransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.policy.move_embedding_dim,
            num_heads=cfg.policy.n_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(cfg.policy.move_embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.policy.move_embedding_dim, cfg.policy.move_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.policy.move_hidden_dim, cfg.policy.move_embedding_dim),
        )

    def forward(self, tensor, attention_mask):
        query = self.layer_norm(tensor)
        out, _ = self.mha(
            query,
            query,
            query,
            attn_mask=attention_mask,
        )
        tensor = tensor + out
        tensor = tensor + self.ffn(self.layer_norm(tensor))
        return tensor


class GateTransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.policy.gate_embedding_dim,
            num_heads=cfg.policy.n_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(cfg.policy.gate_embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.policy.gate_embedding_dim, cfg.policy.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.policy.gate_hidden_dim, cfg.policy.gate_embedding_dim),
        )

    def forward(self, tensor, attention_mask):
        query = self.layer_norm(tensor)
        out, _ = self.mha(
            query,
            query,
            query,
            attn_mask=attention_mask,
        )
        tensor = tensor + out
        tensor = tensor + self.ffn(self.layer_norm(tensor))
        return tensor


class CrossTaskLayer(nn.Module):
    """Attend along the task axis.

    ``GateEmbedding`` folds the lookahead tasks into the batch dimension, so
    without this the tasks are encoded in complete isolation even though the
    cost of a move depends on what the other tasks are asking for.
    """

    def __init__(self, cfg):
        super().__init__()
        dim = cfg.policy.gate_embedding_dim
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=cfg.policy.n_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, cfg.policy.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.policy.gate_hidden_dim, dim),
        )

    def forward(self, tensor):
        batch, tasks, positions, dim = tensor.shape
        flat = tensor.permute(0, 2, 1, 3).reshape(batch * positions, tasks, dim)
        query = self.layer_norm(flat)
        out, _ = self.mha(query, query, query)
        flat = flat + out
        flat = flat + self.ffn(self.layer_norm(flat))
        return flat.reshape(batch, positions, tasks, dim).permute(0, 2, 1, 3)


class MoveTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.atom_capacity = _atom_capacity(cfg)
        self.embedding = MoveEmbedding(cfg)
        self.layers = nn.ModuleList(
            [MoveTransformerLayer(cfg) for _ in range(cfg.policy.move_layers)]
        )

    def forward(self, moves, player_position):
        tensor, attention_mask = self.embedding(moves, player_position)
        for layer in self.layers:
            tensor = layer(tensor, attention_mask)
        return tensor[:, self.atom_capacity :, :]


class GateTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.size = cfg.board.row * cfg.board.col
        self.atom_capacity = _atom_capacity(cfg)
        self.embedding = GateEmbedding(cfg)
        self.layers = nn.ModuleList(
            [GateTransformerLayer(cfg) for _ in range(cfg.policy.gate_layers)]
        )
        self.cross_task_layers = nn.ModuleList(
            [
                CrossTaskLayer(cfg)
                for _ in range(cfg.policy.get("cross_task_layers", 0))
            ]
        )

    def forward(self, board, tasks, player):
        tensor, attention_mask = self.embedding(board, tasks, player)
        for layer in self.layers:
            tensor = layer(tensor, attention_mask)
        tensor = tensor.reshape(
            tasks.shape[0], tasks.shape[1], self.atom_capacity + self.size, -1
        )[:, :, self.atom_capacity :, :]
        for layer in self.cross_task_layers:
            tensor = layer(tensor)
        return tensor


class MergeLayer(nn.Module):
    """Pool the lookahead-task channels into one embedding per board position.

    ``merge_move`` decides whether the move branch participates at all; with it
    disabled the ``MoveTransformer`` output never reaches the logits.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.merge_move = cfg.policy.get("merge_move", False)
        self.pool = cfg.policy.get("pool", "mean")
        if self.merge_move and (
            cfg.policy.move_embedding_dim != cfg.policy.gate_embedding_dim
        ):
            raise ValueError(
                "merge_move requires move_embedding_dim == gate_embedding_dim"
            )
        if self.pool == "attention":
            self.score = nn.Linear(cfg.policy.gate_embedding_dim, 1)
        elif self.pool == "weighted":
            # One learned weight per channel: the tasks arrive in a fixed order,
            # so the mean is throwing away which slot a channel came from.
            channels = cfg.env.feature_count + (1 if self.merge_move else 0)
            self.channel_weight = nn.Parameter(torch.zeros(channels))
        elif self.pool != "mean":
            raise ValueError(f"Unsupported pooling: {self.pool}")

    def forward(self, gate_tensor, move_tensor):
        if self.merge_move:
            tensor = torch.cat([gate_tensor, move_tensor.unsqueeze(1)], dim=1)
        else:
            tensor = gate_tensor
        tensor = tensor.transpose(1, 2)
        if self.pool == "attention":
            weights = torch.softmax(self.score(tensor), dim=2)
            return (tensor * weights).sum(dim=2)
        if self.pool == "weighted":
            weights = torch.softmax(self.channel_weight, dim=0)
            return (tensor * weights.view(1, 1, -1, 1)).sum(dim=2)
        return tensor.mean(dim=2)


class LocalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.move_transformer = MoveTransformer(cfg)
        self.gate_transformer = GateTransformer(cfg)
        self.merge_layer = MergeLayer(cfg)

    def forward(self, board, tasks, moves, player, player_position):
        gate_tensor = self.gate_transformer(board, tasks, player)
        move_tensor = self.move_transformer(moves, player_position)
        return self.merge_layer(gate_tensor, move_tensor)

class GlobalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.row = cfg.board.row
        self.col = cfg.board.col
        self.board_size = self.row * self.col
        self.nqubits = cfg.benchmark.n_atoms
        self.mixer1 = MLPMixer(
            channels=self.board_size,
            depth=cfg.policy.mlp_depth,
            dim=cfg.policy.p_hsize,
            image_size=(self.nqubits, 1),
            patch_size=1,
        )
        self.mixer2 = MLPMixer(
            channels=cfg.policy.p_hsize,
            depth=cfg.policy.mlp_depth,
            dim=cfg.policy.p_hsize,
            image_size=(self.nqubits, 1),
            patch_size=1,
        )
        self.player_embedding = nn.Embedding(
            self.nqubits, cfg.policy.player_embedding_dim
        )
        self.time_embedding = nn.Embedding(
            cfg.n_components + 1, cfg.policy.time_embedding_dim
        )
        embedding_dim = (
            cfg.policy.player_embedding_dim
            + cfg.policy.time_embedding_dim
            + cfg.policy.p_hsize
        )
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, self.board_size),
        )

    def forward(self, grids, time, player):
        player_embedding = self.player_embedding(player)
        time_embedding = self.time_embedding(time)
        batch_size = grids.shape[0]
        features = self.mixer1(grids.unsqueeze(-1))
        _, nqubits, dim = features.shape
        board_embedding = self.mixer2(
            features.reshape(batch_size, 1, nqubits, dim, 1)
            .permute(0, 1, 3, 2, 4)
            .flatten(1, 2)
        )[torch.arange(batch_size, device=grids.device), player, :]
        return self.fc(
            torch.cat([player_embedding, time_embedding, board_embedding], dim=1)
        )


class BasePolicy(nn.Module):
    """The scheduler's actor and critic over board positions.

    ``policy.cost`` decides what the one step estimator contributes. Off, the
    policy is handed the board and has to infer the cost landscape from geometry,
    while the estimator's numbers are used outside it as a mask. ``feature`` adds
    them to each candidate's embedding, and ``prior`` makes the logits the
    estimator's own ranking plus a learned correction, so the policy starts as
    greedy and training only decides where to depart from it.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.row = cfg.board.row
        self.col = cfg.board.col
        self.board_size = self.row * self.col
        self.layer_norm = nn.LayerNorm(normalized_shape=self.board_size)
        dim = cfg.policy.gate_embedding_dim
        self.local_block = LocalBlock(cfg)
        self.neg_inf = torch.tensor(-torch.inf)
        mode = cfg.policy.get("cost", "off")
        # YAML 1.1 turns a bare ``off`` into a boolean; accept both spellings.
        self.cost_mode = "off" if mode in (False, None) else mode
        if self.cost_mode not in ("off", "feature", "prior"):
            raise ValueError(f"Unsupported cost mode: {self.cost_mode}")
        if self.cost_mode == "feature":
            self.cost_project = nn.Linear(1, dim)
        if self.cost_mode == "prior":
            self.cost_weight = nn.Parameter(torch.ones(()))
        self.critic = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        # Kept for strict compatibility with transferable checkpoints.
        self.global_critic = nn.Sequential(
            nn.Linear(self.board_size, self.board_size),
            nn.ReLU(),
            nn.Linear(self.board_size, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        if cfg.policy.get("orthogonal_init", False):
            self._apply_orthogonal_init()
        if self.cost_mode == "prior":
            # Start exactly on the estimator: a correction of zero that still has a
            # gradient, so the first epochs answer where greedy is wrong rather
            # than relearning what it already knows.
            last = [m for m in self.actor if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def normalise_cost(cost):
        """The estimator's costs on a per state scale, cheapest candidate at zero.

        Costs differ by orders of magnitude between boards and between steps, so
        the spread of a state's own candidates is the only scale available.
        """
        legal = torch.isfinite(cost)
        base = torch.where(legal, cost, torch.inf).amin(dim=-1, keepdim=True)
        deltas = torch.where(legal, cost - base, 0.0)
        scale = (
            deltas.sum(-1, keepdim=True) / legal.sum(-1, keepdim=True).clamp(min=1)
        ).clamp(min=1e-6)
        return deltas / scale

    def with_cost(self, out, cost):
        if self.cost_mode != "feature" or cost is None:
            return out
        return out + self.cost_project(self.normalise_cost(cost).unsqueeze(-1))

    def cost_prior(self, logits, cost):
        if self.cost_mode != "prior" or cost is None:
            return logits
        return logits - self.cost_weight * self.normalise_cost(cost)

    def _apply_orthogonal_init(self):
        for head, output_std in (
            (self.actor, 0.01),
            (self.critic, 1.0),
            (self.global_critic, 1.0),
        ):
            linears = [m for m in head if isinstance(m, nn.Linear)]
            for index, layer in enumerate(linears):
                std = output_std if index == len(linears) - 1 else 2**0.5
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, 0.0)

    @staticmethod
    def mean_pooling(out, mask):
        valid_mask = (~mask).unsqueeze(-1).float()
        return (out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

    def act(self, state, deterministic=False):
        logits = self.forward(state)
        distribution = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits)
        else:
            action = distribution.sample()
        return action.item(), distribution.log_prob(action)


class FixedPolicy(BasePolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.nqubits = cfg.benchmark.n_atoms
        self.global_block = GlobalBlock(cfg)

    def build_obs(self, state):
        player = state["player"].argmax(dim=-1)
        atom_map = state["board"].long()
        batch_size, num_players = atom_map.shape
        batch_index = torch.arange(batch_size, device=player.device)
        mask = torch.zeros(
            batch_size, self.board_size, device=player.device, dtype=torch.bool
        )
        atom_index = batch_index.unsqueeze(1).repeat(1, num_players)
        mask[atom_index, atom_map] = True
        player_position = atom_map[batch_index, player]
        mask[batch_index, player_position] = False

        board = torch.full(
            (batch_size, self.board_size),
            -1,
            device=player.device,
            dtype=torch.long,
        )
        atom_ids = torch.arange(self.nqubits, device=player.device).expand(
            batch_size, -1
        )
        board[batch_index.unsqueeze(1), atom_map] = atom_ids
        grid = torch.zeros(
            batch_size,
            self.board_size,
            self.nqubits,
            device=player.device,
        )
        row_index, column_index = (board >= 0).nonzero(as_tuple=True)
        grid[row_index, column_index, board[row_index, column_index]] = 1
        return {
            "mask": mask,
            "grid": grid,
            "player": player,
            "player_position": player_position,
            "board": state["board"].long(),
            "tasks": state["tasks"].long(),
            "moves": state["moves"].long(),
            "time": state["current_index"].argmax(dim=-1),
        }

    def shared_net(self, state, cost=None):
        obs = self.build_obs(state)
        global_out = self.global_block(obs["grid"], obs["time"], obs["player"])
        local_out = self.local_block(
            obs["board"],
            obs["tasks"],
            obs["moves"],
            obs["player"],
            obs["player_position"],
        )
        return self.with_cost(local_out, cost), global_out, obs["mask"]

    def forward(self, state, cost=None):
        local_out, global_out, mask = self.shared_net(state, cost)
        logits = self.cost_prior(
            self.actor(local_out).squeeze(-1) + global_out, cost
        )
        return torch.where(mask, self.neg_inf, logits)

    def get_value(self, state, cost=None):
        local_out, global_out, mask = self.shared_net(state, cost)
        return self.critic(self.mean_pooling(local_out, mask)) + self.global_critic(
            global_out
        )

    def get_action_and_value(self, state, action=None, cost=None):
        local_out, global_out, mask = self.shared_net(state, cost)
        value = self.critic(
            self.mean_pooling(local_out, mask)
        ) + self.global_critic(global_out)
        logits = self.cost_prior(
            self.actor(local_out).squeeze(-1) + global_out, cost
        )
        distribution = Categorical(logits=torch.where(mask, self.neg_inf, logits))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value


class TransferablePolicy(BasePolicy):
    def build_obs(self, state):
        player = state["player"].argmax(dim=-1)
        atom_map = state["board"].long()
        batch_size, num_players = atom_map.shape
        batch_index = torch.arange(batch_size, device=player.device)
        mask = torch.zeros(
            batch_size,
            self.board_size + 1,
            device=player.device,
            dtype=torch.bool,
        )
        atom_index = batch_index.unsqueeze(1).repeat(1, num_players)
        mask[atom_index, atom_map] = True
        player_position = atom_map[batch_index, player]
        mask[batch_index, player_position] = False
        return {
            "mask": mask[:, : self.board_size],
            "player": player,
            "player_position": player_position,
            "board": state["board"].long(),
            "tasks": state["tasks"].long(),
            "moves": state["moves"].long(),
        }

    def shared_net(self, state, cost=None):
        obs = self.build_obs(state)
        out = self.local_block(
            obs["board"],
            obs["tasks"],
            obs["moves"],
            obs["player"],
            obs["player_position"],
        )
        return self.with_cost(out, cost), obs["mask"]

    def forward(self, state, cost=None):
        out, mask = self.shared_net(state, cost)
        logits = self.cost_prior(self.actor(out).squeeze(-1), cost)
        return torch.where(mask, self.neg_inf, logits)

    def get_value(self, state, cost=None):
        out, mask = self.shared_net(state, cost)
        return self.critic(self.mean_pooling(out, mask))

    def get_action_and_value(self, state, action=None, cost=None):
        out, mask = self.shared_net(state, cost)
        value = self.critic(self.mean_pooling(out, mask))
        logits = self.cost_prior(self.actor(out).squeeze(-1), cost)
        distribution = Categorical(logits=torch.where(mask, self.neg_inf, logits))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value
