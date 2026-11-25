# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Dict, List
import math
import matplotlib.pyplot as plt
import numpy as np


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_horizontal(self):
        return self.x == 0

    def is_vertical(self):
        return self.y == 0

    def is_left_to_right(self):
        return self.x > 0

    def is_bottom_to_top(self):
        return self.y > 0

    def has_same_topology(self, v2):
        v1 = self
        hol1 = v1.is_horizontal()
        left_to_right1 = v1.is_left_to_right()

        hol2 = v2.is_horizontal()
        left_to_right2 = v2.is_left_to_right()

        horizontally_ok = False
        if hol1 and hol2:
            horizontally_ok = True
        elif hol1 or hol2:
            horizontally_ok = False
        elif left_to_right1 == left_to_right2:
            horizontally_ok = True
        if not horizontally_ok:
            return False

        ver1 = v1.is_vertical()
        ver2 = v2.is_vertical()
        if ver1 and ver2:
            return True
        elif ver1 or ver2:
            return False

        bottom_to_top1 = v1.is_bottom_to_top()
        bottom_to_top2 = v2.is_bottom_to_top()
        return bottom_to_top1 == bottom_to_top2


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def copy(self):
        return Point(self.x, self.y)

    def from_array(d):
        return Point(d[0], d[1])

    def from_arrays(ds):
        return [Point.from_array(d) for d in ds]

    def to_tuple(self):
        return (self.x, self.y)

    def __eq__(self, p):
        if not isinstance(p, Point):
            return False
        if self.x == p.x and self.y == p.y:
            return True

    def to_dict(self):
        return {"x": self.x, "y": self.y}

    def __hash__(self):
        return hash((self.x, self.y))

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __add__(self, other: Vector):
        return Point(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"({self.x},{self.y})"


class Move:
    """
    The transition from one point to another point.
    """

    def __init__(self, frm: Point, to: Point):
        self.frm = frm
        self.to = to

    def inverse(self):
        return Move(self.to, self.frm)

    def dist(self):
        return math.sqrt((self.frm.x - self.to.x) ** 2 + (self.frm.y - self.to.y) ** 2)

    def to_dict(self):
        return {"frm": self.frm.to_dict(), "to": self.to.to_dict()}

    def __eq__(self, __value):
        if not isinstance(__value, Move):
            return False
        return self.frm == __value.frm and self.to == __value.to

    def __hash__(self):
        return hash((self.frm, self.to))

    def __repr__(self):
        return f"{self.frm}->{self.to}"

    def copy(self):
        return Move(self.frm, self.to)


class AtomMove(Move):
    def __init__(self, atom_id, frm: Point, to: Point):
        super().__init__(frm, to)
        self.atom_id = atom_id

    def __eq__(self, __value):
        if not isinstance(__value, AtomMove):
            return False
        return (
            self.atom_id == __value.atom_id
            and self.frm == __value.frm
            and self.to == __value.to
        )

    def to_dict(self):
        return {
            "atom_id": self.atom_id,
            "frm": self.frm.to_dict(),
            "to": self.to.to_dict(),
        }

    def __hash__(self):
        return hash((self.atom_id, self.frm, self.to))

    def __repr__(self):
        return f"{self.atom_id}:{self.frm}->{self.to}"

    def copy(self):
        return AtomMove(self.atom_id, self.frm, self.to)


class Board:
    def __init__(
        self, points: list[Point], atom_map: Dict[int, int], point_to_id=None
    ):
        """
        points:
        atom_map
        point_to_id: reverse index, shared with copies because ``points`` is
            fixed for the lifetime of a board lineage.
        """
        self.points = points
        self.atom_map = atom_map
        if point_to_id is None:
            point_to_id = {p: g_id for g_id, p in enumerate(points)}
        self.point_to_id = point_to_id

    def copy(self):
        return Board(self.points, self.atom_map.copy(), self.point_to_id)

    def get_atom_position_map(self):
        result = {}
        for atom_id, point_id in self.atom_map.items():
            result[atom_id] = self.points[point_id]
        return result

    def get_atom_position(self, atom_id) -> Point:
        return self.points[self.atom_map[atom_id]]

    def get_atom_move(self, atom_id, grid_id) -> AtomMove:
        if self.atom_map[atom_id] == grid_id:
            return None
        return AtomMove(atom_id, self.get_atom_position(atom_id), self.points[grid_id])

    def transform(self, a_id, grid_id, check_empty=True):
        if grid_id == self.atom_map[a_id]:
            return self
        if check_empty and (
            not 0 <= grid_id < len(self.points)
            or grid_id in self.atom_map.values()
        ):
            raise AttributeError(f"{grid_id} is not empty.")
        result = self.copy()
        result.atom_map.pop(a_id)
        result.atom_map[a_id] = grid_id
        return result

    def size(self):
        return len(self.points)

    def movable_mask(self, player):
        oks = set(self.empties())
        if player in self.atom_map:
            oks.add(self.atom_map[player])
        results = []
        for j in range(len(self.points)):
            if j in oks:
                results.append(0)
            else:
                results.append(1)
        return results

    def empties(self):
        """
        Return list of empty grids
        """
        results = []
        filled = set(self.atom_map.values())
        for j in range(len(self.points)):
            if j not in filled:
                results.append(j)
        return results

    def width(self):
        min_x = sys.maxsize
        max_x = -sys.maxsize
        for p in self.points:
            if min_x > p.x:
                min_x = p.x
            if max_x < p.x:
                max_x = p.x
        return max_x - min_x

    def height(self):
        min_y = sys.maxsize
        max_y = -sys.maxsize
        for p in self.points:
            if min_y > p.y:
                min_y = p.y
            if max_y < p.y:
                max_y = p.y
        return max_y - min_y

    def shift(self, vector):
        results = []
        for p in self.points:
            results.append(Point(p.x + vector.x, p.y + vector.y))
        return Board(results, self.atom_map.copy())

    def draw(self):
        plt.clf()
        fig, ax = plt.subplots()
        xs = []
        ys = []
        for p in self.points:
            xs.append(p.x)
            ys.append(p.y)
        plt.plot(xs, ys, linewidth=0, marker=".")

        for id, p_id in self.atom_map.items():
            point = self.points[p_id]
            plt.annotate(
                str(id),
                (point.x, point.y),
                bbox={"boxstyle": "circle", "color": "gray"},
            )

        fig_size = self.fig_size()
        plt.xlim(fig_size[0], fig_size[1])
        plt.ylim(fig_size[2], fig_size[3])
        return fig, ax

    def fig_size(self):
        return [
            -self.width() * 0.2,
            self.width() * 1.2,
            -self.height() * 0.2,
            self.height() * 1.2,
        ]

    def vector(self):
        inverse_map = {}
        for a_id, p_id in self.atom_map.items():
            inverse_map[p_id] = a_id

        results = []
        for j, point in enumerate(self.points):
            if j not in inverse_map:
                results.append(-1)
            else:
                results.append(inverse_map[j])
        return results


class ParallelMove:
    """
    The class wraps multiple moves.
    """

    def __init__(self, moves: [Move]):
        self.moves = moves

    def inverse(self):
        results = []
        for move in self.moves:
            results.append(move.inverse())
        return ParallelMove(results)

    def count(self):
        return len(self.moves)

    def to_dict(self):
        results = {}
        array = []
        for move in self.moves:
            array.append(move.to_dict())
        results["moves"] = array
        return results

    def __eq__(self, __value):
        if not isinstance(__value, ParallelMove):
            return False
        return self.moves == __value.moves

    def __repr__(self):
        return f"AOD:{str(self.moves)}"

    def copy(self):
        return ParallelMove([m.copy() for m in self.moves])


class Reconfiguration:
    def __init__(self, atom_ids, pm: ParallelMove) -> None:
        self.atom_ids = atom_ids
        self.pm = pm


class Grid:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.points = {}
        for x in range(width):
            for y in range(height):
                self.points[self._key(x, y)] = Point(x, y)

    def get(self, x, y) -> Point:
        k = self._key(x, y)
        return self.points[k]

    def _key(self, x, y):
        return f"{x}_{y}"


class AtomConfiguration:
    def __init__(self, grid: Grid, point_map: Dict[int, Point]):
        self.grid = grid
        self.point_map = point_map

    def get(self, id) -> Point:
        return self.point_map[id]

    def update(self, id, point: Point):
        self.point_map[id] = point

    def update_all(self, ids, points):
        if type(points) is np.ndarray:
            points = [Point(x, y) for x, y in points]
        for id, point in zip(ids, points):
            self.point_map[id] = point

    def draw(self):
        plt.clf()
        fig, ax = plt.subplots()
        xs = []
        ys = []
        for p in self.grid.points.values():
            xs.append(p.x)
            ys.append(p.y)
        plt.plot(xs, ys, linewidth=0, marker=".")

        for id, point in self.point_map.items():
            plt.annotate(
                str(id),
                (point.x, point.y),
                bbox={"boxstyle": "circle", "color": "gray"},
            )

        plt.xlim(-0.5, self.grid.width - 0.5)
        plt.ylim(-0.5, self.grid.height - 0.5)
        return fig, ax

    def to_matrix(self, format="sparse"):
        """Allowed: sparse, dense"""
        if format == "sparse":
            coords = [(point.x, point.y) for point in self.point_map.values()]
            matrix = np.array(coords)
        elif format == "dense":
            matrix = np.zeros((self.grid.width, self.grid.height))
            for id, point in self.point_map.items():
                matrix[point.x, point.y] = id
        else:
            raise AttributeError("Invalid format")
        return matrix

    def position_counts(self):
        pos_matrix = self.to_matrix()
        position_counts = {}
        for pos in map(tuple, pos_matrix):
            if pos in position_counts:
                position_counts[pos] += 1
            else:
                position_counts[pos] = 1
        return position_counts

    def num_atoms(self):
        return len(self.point_map)

    @staticmethod
    def moves(frm, to):
        if len(frm.point_map) != len(to.point_map):
            raise AttributeError("# of atoms are different")
        results = []
        for q_id, point in frm.point_map.items():
            if q_id not in to.point_map:
                raise AttributeError("Atoms are different")
            point2 = to.point_map[q_id]
            if point != point2:
                results.append(Move(point, point2))
        return results

    def __repr__(self):
        return str(self.point_map)


class DeviceConfig:
    def __init__(
        self,
        num_atoms,
        aod_count=1,
        decoherence_per_atom=0.999,
        fidelity_per_dist=0.999,
        two_q_gate_fidelity=0.999,
    ):
        self.num_atoms = num_atoms
        self.aod_count = aod_count
        self.decoherence_per_atom = decoherence_per_atom
        self.fidelity_per_dist = fidelity_per_dist
        self.two_q_gate_fidelity = two_q_gate_fidelity
        self.position_penalty = -1.0
