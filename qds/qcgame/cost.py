# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cost of a reconfiguration or a gate step, as log infidelity.

This implements Section "Reconfiguration cost estimation" of arXiv:2506.05536.
A step is charged

    J(D, M) = alpha * D * N + beta * M,

the idling of all N atoms for the duration D of the move plus a fixed penalty
for each of the M atoms picked up and dropped off. Neither D nor M is obtained
by scheduling the AOD moves; both are bounded from the number of moves a
log-depth reconfiguration heuristic would need, which is read off the conflict
graph of the step.

`reconfigure_layout` charges a layout change and `gate_forward` a set of gates,
the latter being a layout change that brings each pair together followed by a
trip to the gate zone and back. `estimate_cost_batch` is the same arithmetic over
a batch of steps, which is the path the search actually runs on.
"""

from qds.qcgame.cupy import Cupy
import pydantic
if Cupy.use_cupy:
    import cupy as np
else:
    import numpy as np

class RewardModel:
    def __init__(self, N, idle_error=1.0, move_error=0.1, inter_zone_time=10.0):
        self.N = N
        self.idle_error = idle_error
        self.move_error = move_error
        self.inter_zone_time = inter_zone_time


class Qubit(int):
    def __repr__(self):
        return f"Q[{int(self)}]"


def _move_count(n_edges: int) -> float:
    """The number of moves a log-depth reconfiguration needs, n_m.

    Partitioning the conflict graph with a MaxCut heuristic removes an extensive
    share of its edges per round, so the rounds go as the log of the edge count.
    The one is added so that a single conflict still costs a move.
    """
    return np.log(n_edges) + 1


def _slowest_move(active_sites) -> float:
    """The characteristic time of one move, tau, at unit acceleration.

    Under constant acceleration the time goes as the square root of the distance.
    The two AOD axes are driven in parallel, so the slower axis sets the time for
    a move and the furthest-travelling atom sets it for the whole step.
    """
    span = max(
        max(abs(start[0] - end[0]), abs(start[1] - end[1]))
        for start, end in active_sites
    )
    return np.sqrt(span)


# Runtime type checking, which costs time in the inner loop. Safe to remove.
@pydantic.validate_call(
    config=pydantic.ConfigDict(arbitrary_types_allowed=True)
)
def gate_forward(
    model: RewardModel,
    state: dict[int, tuple[int, int]],
    gates: set[tuple[int, int]],
):
    """Charge executing `gates` on the atoms laid out as `state`.

    The gates are paid for as a layout change that brings each pair adjacent,
    plus `inter_zone_time` for the trip out to the gate zone and back, which is
    charged once because the pairs make the trip together. Both the swap space
    for the reconfiguration and the grid of gate sites are taken as unbounded.
    """
    active_sites = _get_active_gate_sites(state, gates)
    if len(active_sites) == 0:
        return {"total_time": 0, "total_touches": 0, "cost": 0}

    n_edges = reconfiguation_conflicts(active_sites)
    moves = _move_count(n_edges)
    total_time = _slowest_move(active_sites) * moves + model.inter_zone_time
    total_touches = len(active_sites) * moves
    return {
        "total_time": total_time,
        "total_touches": total_touches,
        "cost": total_time * model.N * model.idle_error
        + total_touches * model.move_error,
        "edges": n_edges,
    }


# Runtime type checking, which costs time in the inner loop. Safe to remove.
@pydantic.validate_call(
    config=pydantic.ConfigDict(arbitrary_types_allowed=True)
)
def reconfigure_layout(
    model: RewardModel,
    start_state: dict[int, tuple[int, int]],
    end_state: dict[int, tuple[int, int]],
):
    """Charge moving the atoms in the storage region from one layout to another.

    Both states map a qubit to the (row, column) of the trap it occupies. The
    swap space available for the reconfiguration is taken as unbounded.
    """
    return do_reconfigure_layout(
        model, _get_active_reconfiguration_sites(start_state, end_state)
    )


def estimate_cost_batch(
    model, active_sites: np.ndarray, num_active_sites: np.ndarray, is_gate=False
) -> dict:
    """
    active_sites: shape (B, N, 2, 2)
    n_edges: shape (B,)  # バッチごとの衝突数など
    model: コスト計算に使用するパラメータ (idle_error, move_error をもつ)

    戻り値:
      {
        "total_time": shape (B,),
        "total_touches": shape (B,),
        "cost": shape (B,)
      }
    """
    # B=バッチサイズ, N=サイト数
    B, _, _, _ = active_sites.shape
    n_edges = reconfiguation_conflicts_batch_np(active_sites)

    # active_sites[b, i, 0] = (x1, y1)
    # active_sites[b, i, 1] = (x2, y2)
    x1 = active_sites[:, :, 0, 0]  # (B, N)
    y1 = active_sites[:, :, 0, 1]  # (B, N)
    x2 = active_sites[:, :, 1, 0]  # (B, N)
    y2 = active_sites[:, :, 1, 1]  # (B, N)

    # moved_x, moved_y をバッチ一括で計算
    moved_x = np.abs(x1 - x2)  # (B, N)
    moved_y = np.abs(y1 - y2)  # (B, N)

    # それぞれバッチ内で max をとる
    dx = moved_x.max(axis=1)  # (B,)
    dy = moved_y.max(axis=1)  # (B,)

    # Tc = sqrt( max(dx, dy) )
    max_xy = np.maximum(dx, dy)  # (B,)
    Tc = np.sqrt(max_xy)  # (B,)

    # total_moves = log(n_edges) + 1
    total_moves = np.log(n_edges) + 1  # (B,)

    # total_time = Tc * total_moves
    total_time = Tc * total_moves  # (B,)
    if is_gate:
        total_time += model.inter_zone_time

    # total_touches = N * total_moves  (len(active_sites) 相当)
    total_touches = num_active_sites * total_moves  # (B,)

    # cost = total_time * N * model.idle_error + total_touches * model.move_error
    cost = (
        total_time * model.N * model.idle_error + total_touches * model.move_error
    )  # (B,)

    return {
        "total_time": total_time,  # shape (B,)
        "total_touches": total_touches,  # shape (B,)
        "cost": cost,  # shape (B,)
        "n_edges": n_edges,
    }


def do_reconfigure_layout(model, active_sites):
    """Charge a layout change whose active participants are already known."""
    if len(active_sites) == 0:
        return {"total_time": 0.0, "total_touches": 0, "cost": 0.0}
    return estimate_cost(model, active_sites, reconfiguation_conflicts(active_sites))


def estimate_cost(model, active_sites, nedges):
    """J(D, M) for a step of `active_sites` spanning a graph of `nedges` edges.

    Every participant is touched once per move, so the touches are the moves
    times the number of participants; the constant share of them that a single
    move actually picks up is folded into `move_error`.
    """
    moves = _move_count(nedges)
    total_time = _slowest_move(active_sites) * moves
    total_touches = len(active_sites) * moves
    return {
        "total_time": total_time,
        "total_touches": total_touches,
        "cost": total_time * model.N * model.idle_error
        + total_touches * model.move_error,
        "n_edges": nedges,
    }


def _validate_state(state: dict[Qubit, tuple[int, int]]):
    """Reject a layout that puts two atoms in one trap."""
    if len(set(state.values())) != len(state):
        raise ValueError("State has duplicate values", state)
    return True


def _get_active_reconfiguration_sites(
    start_state: dict[Qubit, tuple[int, int]],
    end_state: dict[Qubit, tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """The atoms a layout change picks up, each as (trap left, trap taken).

    The movers are the atoms whose trap differs between the two layouts. The
    rows and columns they occupy at either end are the active ones, and an atom
    that stays where it is still gets caught by the crossed AOD when it sits at
    the intersection of an active row and an active column. A caught atom is
    returned as a move onto its own trap, since it is touched but travels no
    distance.
    """
    _validate_state(start_state)
    _validate_state(end_state)

    movers = {
        qubit: (start_state[qubit], end_state[qubit])
        for qubit in start_state
        if start_state[qubit] != end_state[qubit]
    }
    active_rows = {trap[0] for move in movers.values() for trap in move}
    active_cols = {trap[1] for move in movers.values() for trap in move}

    caught = {
        start_state[qubit]
        for qubit in start_state
        if qubit not in movers
        and start_state[qubit][0] in active_rows
        and start_state[qubit][1] in active_cols
    }
    return list(set(movers.values()) | {(trap, trap) for trap in caught})


def _get_active_gate_sites(
    state: dict[int, tuple[int, int]], gates: set[tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """The traps each gate asks to be brought together, as (one, the other).

    A gate is charged like a move between the traps of its two atoms, which is
    what bringing the pair adjacent costs.
    """
    return [(state[control], state[target]) for control, target in gates]


def reconfiguation_conflicts(
    active_sites: list[tuple[tuple[int, int], tuple[int, int]]],
) -> int:
    """Edges in the conflict graph of a step, plus one.

    Two moves conflict, and so share an edge, when the crossed AODs cannot carry
    them at once: either one row or column would have to split or merge to serve
    both, or the two would have to change order along an axis mid-flight. A pair
    contributes at most one edge however many of the four ways it conflicts. The
    one added at the end keeps the log of this defined for a step that has no
    conflicts at all.
    """
    n_edges = 0
    for i, _ in enumerate(active_sites):
        for j in range(i + 1, len(active_sites)):
            # X one-to-many
            if (active_sites[i][0][0] == active_sites[j][0][0]) ^ (
                active_sites[i][1][0] == active_sites[j][1][0]
            ):
                n_edges += 1
            # Y one-to-many
            elif (active_sites[i][0][1] == active_sites[j][0][1]) ^ (
                active_sites[i][1][1] == active_sites[j][1][1]
            ):
                n_edges += 1

            # X ordering
            elif (active_sites[i][0][0] < active_sites[j][0][0]) ^ (
                active_sites[i][1][0] < active_sites[j][1][0]
            ):
                n_edges += 1
            # Y ordering
            elif (active_sites[i][0][1] < active_sites[j][0][1]) ^ (
                active_sites[i][1][1] < active_sites[j][1][1]
            ):
                n_edges += 1

    return n_edges + 1


def reconfiguation_conflicts_batch_cp(sites: np.ndarray) -> np.ndarray:
    """
    NumPy でバッチ処理する版。
    sites の形状は [B, N, 2, 2] を想定:
      B = バッチサイズ
      N = 1バッチあたりのアクティブサイト数
      sites[b, i, 0] = (x1, y1)  # i 番目のサイトの座標1
      sites[b, i, 1] = (x2, y2)  # i 番目のサイトの座標2

    Returns:
      conflicts: shape (B,) の ndarray
                 バッチ内の各サンプルに対してカウントした衝突数+1 を格納
    """
    # sites[b, i, 0, 0] = x1, sites[b, i, 0, 1] = y1
    # sites[b, i, 1, 0] = x2, sites[b, i, 1, 1] = y2
    print(np)
    x1 = sites[..., 0, 0]  # shape (B, N)
    y1 = sites[..., 0, 1]  # shape (B, N)
    x2 = sites[..., 1, 0]  # shape (B, N)
    y2 = sites[..., 1, 1]  # shape (B, N)

    B, N = x1.shape

    # それぞれ (B, N, 1) と (B, 1, N) の形にしてブロードキャスト比較できるようにする
    x1_i = x1[:, :, np.newaxis]  # (B, N, 1)
    x1_j = x1[:, np.newaxis, :]  # (B, 1, N)
    x2_i = x2[:, :, np.newaxis]  # (B, N, 1)
    x2_j = x2[:, np.newaxis, :]  # (B, 1, N)
    y1_i = y1[:, :, np.newaxis]
    y1_j = y1[:, np.newaxis, :]
    y2_i = y2[:, :, np.newaxis]
    y2_j = y2[:, np.newaxis, :]

    # ---- 4つの条件をブールで表現 ----
    # X one-to-many
    cond_x_1tomany = np.logical_xor((x1_i == x1_j), (x2_i == x2_j))
    # Y one-to-many
    cond_y_1tomany = np.logical_xor((y1_i == y1_j), (y2_i == y2_j))
    # X ordering
    cond_x_ordering = np.logical_xor((x1_i < x1_j), (x2_i < x2_j))
    # Y ordering
    cond_y_ordering = np.logical_xor((y1_i < y1_j), (y2_i < y2_j))

    # 元の if/elif では、いずれかの条件が True ならカウント(=1)なので OR を取る
    cond_any = cond_x_1tomany | cond_y_1tomany | cond_x_ordering | cond_y_ordering

    # i < j のペアのみカウント (i == j や i>j は数えない)
    # 上三角行列部分だけ True のマスクを作成
    upper_mask = np.triu(np.ones((N, N), dtype=bool), k=1)  # (N, N)
    # バッチ次元にブロードキャスト (B, N, N) で要素同士を AND
    cond_any_upper = cond_any & upper_mask

    # (B, N, N) の True の数をバッチごとに集計
    n_edges = cond_any_upper.sum(axis=(1, 2))

    # 元の関数同様、最後に +1
    return n_edges + 1


def reconfiguation_conflicts_batch_np(sites: np.ndarray) -> np.ndarray:
    """
    NumPy でバッチ処理する版。
    sites の形状は [B, N, 2, 2] を想定:
      B = バッチサイズ
      N = 1バッチあたりのアクティブサイト数
      sites[b, i, 0] = (x1, y1)  # i 番目のサイトの座標1
      sites[b, i, 1] = (x2, y2)  # i 番目のサイトの座標2

    Returns:
      conflicts: shape (B,) の ndarray
                 バッチ内の各サンプルに対してカウントした衝突数+1 を格納
    """
    # sites[b, i, 0, 0] = x1, sites[b, i, 0, 1] = y1
    # sites[b, i, 1, 0] = x2, sites[b, i, 1, 1] = y2
    x1 = sites[..., 0, 0]  # shape (B, N)
    y1 = sites[..., 0, 1]  # shape (B, N)
    x2 = sites[..., 1, 0]  # shape (B, N)
    y2 = sites[..., 1, 1]  # shape (B, N)

    B, N = x1.shape

    # それぞれ (B, N, 1) と (B, 1, N) の形にしてブロードキャスト比較できるようにする
    x1_i = x1[:, :, np.newaxis]  # (B, N, 1)
    x1_j = x1[:, np.newaxis, :]  # (B, 1, N)
    x2_i = x2[:, :, np.newaxis]  # (B, N, 1)
    x2_j = x2[:, np.newaxis, :]  # (B, 1, N)
    y1_i = y1[:, :, np.newaxis]
    y1_j = y1[:, np.newaxis, :]
    y2_i = y2[:, :, np.newaxis]
    y2_j = y2[:, np.newaxis, :]

    # ---- 4つの条件をブールで表現 ----
    # X one-to-many
    cond_x_1tomany = np.logical_xor((x1_i == x1_j), (x2_i == x2_j))
    # Y one-to-many
    cond_y_1tomany = np.logical_xor((y1_i == y1_j), (y2_i == y2_j))
    # X ordering
    cond_x_ordering = np.logical_xor((x1_i < x1_j), (x2_i < x2_j))
    # Y ordering
    cond_y_ordering = np.logical_xor((y1_i < y1_j), (y2_i < y2_j))

    # 元の if/elif では、いずれかの条件が True ならカウント(=1)なので OR を取る
    cond_any = cond_x_1tomany | cond_y_1tomany | cond_x_ordering | cond_y_ordering

    # i < j のペアのみカウント (i == j や i>j は数えない)
    # 上三角行列部分だけ True のマスクを作成
    upper_mask = np.triu(np.ones((N, N), dtype=bool), k=1)  # (N, N)
    # バッチ次元にブロードキャスト (B, N, N) で要素同士を AND
    cond_any_upper = cond_any & upper_mask

    # (B, N, N) の True の数をバッチごとに集計
    n_edges = cond_any_upper.sum(axis=(1, 2))

    # 元の関数同様、最後に +1
    return n_edges + 1


# --------------------------------
