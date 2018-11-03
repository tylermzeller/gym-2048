import gym
from gym import error, spaces, utils
import numpy as np

import sys

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

map = {
    UP: (-1,  0),
    RIGHT: ( 0,  1),
    DOWN: ( 1,  0),
    LEFT: ( 0, -1),
}

class Grid:
    def __init__(self, size):
        if len(size) != 2:
            raise ValueError('Argument size must be tuple of length 2')
        self.size = size
        self.grid = [[None for _ in range(size[1])] for __ in range(size[0])]
        self.max_tile = 0

    def __iter__(self):
        for row in self.grid:
            yield row

    def flatten(self):
        return [x.value if x else 0 for row in self.grid for x in row]

    def get_tile(self, r, c):
        if self.within_bounds(r, c):
            return self.grid[r][c]
        return None

    def available_cells(self):
        return [(i, j) for i in range(self.size[0]) for j in range(self.size[1]) if not self.grid[i][j]]

    def cells_available(self):
        return len(self.available_cells()) > 0

    def random_available_cell(self):
        cells = self.available_cells()
        assert len(cells) > 0
        i = np.random.choice(len(cells))
        return cells[i]

    def insert_tile(self, tile):
        self.grid[tile.r][tile.c] = tile
        self.max_tile = max(tile.value, self.max_tile)

    def remove_tile(self, tile):
        self.grid[tile.r][tile.c] = None

    def within_bounds(self, r, c):
        return (r >= 0 and r < self.size[0] and
                c >= 0 and c < self.size[1])

    def cell_available(self, r, c):
        return not self.cell_occupied(r, c)

    def cell_occupied(self, r, c):
        return bool(self.get_tile(r, c))

class Tile:
    def __init__(self, r, c, value):
        self.update_position(r, c)
        self.value = value
        self.previous_position = None
        self.merged_from = None

    def save_position(self):
        self.previous_position = (self.r, self.c)

    def update_position(self, r, c):
        self.r = r
        self.c = c

class Env2048(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=(4,4), n_start_tiles=2, start_tiles=[1, 2], p_start_tiles=[0.9, 0.1]):
        self.size = size
        self._n_start_tiles = n_start_tiles
        self._p_start_tiles = p_start_tiles
        self._start_tiles = start_tiles
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,size[0] * size[1]), dtype=np.uint8)
        self.reset()

    def step(self, action, force_move=False):
        assert not self.over
        assert action >= 0 and action < self.action_space.n
        moved, reward = self._move(action)
        if not moved and force_move:
            for i in range(self.action_space.n - 1):
                action = (action + 1) % self.action_space.n
                moved, reward = self._move(action)
                if moved: break
        return self.grid.flatten(), reward, self.over, {'action': action, 'max_tile': self.grid.max_tile}

    def reset(self):
        self.grid = Grid(self.size)
        self.score = 0
        self.over = False
        self.won = False
        self._add_start_tiles()
        return self.grid.flatten()

    def render(self, mode='human'):
        outfile = sys.stdout
        for row in self.grid:
            row_string = ('%4d ' * self.size[1]) % tuple([2 ** x.value if x else 0 for x in row])
            outfile.write("\t{0}\n".format(row_string))
        if mode != 'human': return outfile

    def _add_start_tiles(self):
        for i in range(self._n_start_tiles):
            self._add_random_tile()

    def _add_random_tile(self):
        if self.grid.cells_available():
            value = np.random.choice(self._start_tiles, p=self._p_start_tiles)
            tile = Tile(*self.grid.random_available_cell(), value)
            self.grid.insert_tile(tile)

    def _prepare_tiles(self):
        for row in self.grid:
            for tile in row:
                if tile:
                    tile.merged_from = None
                    tile.save_position()

    def _move_tile(self, tile, r, c):
        self.grid.grid[tile.r][tile.c] = None
        self.grid.grid[r][c] = tile
        tile.update_position(r, c)

    def _move(self, direction):
        if self.over: return
        vector = self._get_vector(direction)
        rtravs, ctravs = self._build_traversals(direction)
        moved = False
        reward = 0
        self._prepare_tiles()

        for r in rtravs:
            for c in ctravs:
                cell = (r, c)
                tile = self.grid.get_tile(*cell)

                if tile:
                    farthest, next = self._find_farthest_pos(*cell, vector)
                    next_tile = self.grid.get_tile(*next)
                    if next_tile and next_tile.value == tile.value and not next_tile.merged_from:
                        merged_tile = Tile(*next, tile.value + 1)
                        merged_tile.merged_from = (tile, next)
                        self.grid.insert_tile(merged_tile)
                        self.grid.remove_tile(tile)
                        tile.update_position(*next)
                        self.score += merged_tile.value
                        reward += 2**merged_tile.value

                        if merged_tile.value == 2048: self.won = True
                    else:
                        self._move_tile(tile, *farthest)
                    if not (r == tile.r and c == tile.c):
                        moved = True
        if moved:
            self._add_random_tile()
            if not self._moves_available():
                self.over = True
        return moved, reward

    def _get_vector(self, direction):
        return map[direction]

    def _build_traversals(self, direction):
        rtravs = [i for i in range(self.size[0])]
        ctravs = [i for i in range(self.size[1])]
        if direction == RIGHT: ctravs.reverse()
        if direction ==  DOWN: rtravs.reverse()
        return rtravs, ctravs

    def _find_farthest_pos(self, r, c, vector):
        prev = (r, c)
        cell = (r + vector[0], c + vector[1])
        while self.grid.within_bounds(*cell) and self.grid.cell_available(*cell):
            prev = cell
            cell = (prev[0] + vector[0], prev[1] + vector[1])
        return prev, cell

    def _moves_available(self):
        return self.grid.cells_available() or self._tile_matches_available()

    def _tile_matches_available(self):
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                tile = self.grid.get_tile(r, c)
                if tile:
                    for direction in range(self.action_space.n):
                        vector = self._get_vector(direction)
                        cell = (r + vector[0], c + vector[1])
                        other = self.grid.get_tile(*cell)
                        if other and other.value == tile.value:
                            return True
        return False
