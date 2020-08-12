"""Othello environments for reinforcement learning."""

import enum
import gym
from gym import spaces
from gym.envs.classic_control import rendering
import pyglet
from pyglet import gl
import numpy as np
import time

BLACK_DISK = -1
NO_DISK = 0
WHITE_DISK = 1

IMAGE_SIZE = 96
MAX_INT = (1<<31)

WINDOW_H = 480
WINDOW_W = 480
BOARDFIELD = 480


class OthelloEnv(gym.Env):
    """Wrapper of OthelloBaseEnv."""

    metadata = {'render.modes': ['np_array', 'human']}

    def __init__(self,
                 white_policy=None,
                 black_policy=None,
                 protagonist=WHITE_DISK,
                 board_size=8,
                 sudden_death_on_invalid_move=True,
                 render_in_step=False):

        # Create the inner environment.
        self.env = OthelloBaseEnv(board_size, sudden_death_on_invalid_move)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render_in_step = render_in_step

        # Initialize policies.
        self.protagonist = protagonist
        if self.protagonist == BLACK_DISK:
            self.opponent = white_policy
        else:
            self.opponent = black_policy

    def reset(self):
        obs = self.env.reset()

        # This provides the opponent a chance to get env.possible_moves.
        if hasattr(self.opponent, 'reset'):
            self.opponent.reset(self)

        if self.env.player_turn == self.protagonist:
            return obs
        else:
            action = self.opponent.get_action(obs)
            obs, _, done, _ = self.env.step(action)
            if done:
                print('done==True in reset(), do it again.')
                return self.reset()
            else:
                return obs

    def step(self, action):
        assert self.env.player_turn == self.protagonist
        obs, reward, done, _ = self.env.step(action)  # My move.
        if self.render_in_step:
            self.render()
        if done:
            return obs, reward, done, None
        while not done and self.env.player_turn != self.protagonist:
            opponent_move = self.opponent.get_action(obs)
            obs, reward, done, _ = self.env.step(opponent_move)
            if self.render_in_step:
                self.render()
        return obs, -reward, done, None

    def render(self, mode='human', close=False):
        self.env.render(mode=mode, close=close)

    def close(self):
        self.env.close()

    @property
    def player_turn(self):
        return self.env.player_turn

    @property
    def possible_moves(self):
        return self.env.possible_moves


class OthelloBaseEnv(gym.Env):
    """Othello base environment."""

    metadata = {'render.modes': ['np_array', 'human']}

    def __init__(self, board_size=8, sudden_death_on_invalid_move=True):

        # Initialize internal states.
        self.board_size = max(4, board_size)
        self.board_state = self._reset_board()
        self.player_turn = WHITE_DISK
        self.winner = NO_DISK
        self.possible_moves = []
        self.sudden_death_on_invalid_move = sudden_death_on_invalid_move
        self.viewer = None
        self.terminated = False

        # Initialize action space:
        #   One action for each board position, a policy losses immediately
        #   if it tries to place a disk on an invalid position.
        self.action_space = spaces.Discrete(self.board_size ** 2)

        # Initialize observation space.
        self.observation_space = spaces.Box(np.zeros([self.board_size] * 2),
                                            np.ones([self.board_size] * 2))

    def _reset_board(self):
        board_state = np.zeros([self.board_size] * 2, dtype=int)
        center_row_ix = center_col_ix = self.board_size // 2
        board_state[center_row_ix - 1][center_col_ix - 1] = WHITE_DISK
        board_state[center_row_ix][center_col_ix] = WHITE_DISK
        board_state[center_row_ix][center_col_ix - 1] = BLACK_DISK
        board_state[center_row_ix - 1][center_col_ix] = BLACK_DISK
        return board_state

    def reset(self):
        self.board_state = self._reset_board()
        self.player_turn = WHITE_DISK
        self.winner = NO_DISK
        self.terminated = False
        self.possible_moves = self.get_possible_actions()
        return self.get_observation()

    def is_valid_position(self, board, x, y, delta_x, delta_y):
        # We overload WHITE_DISK to be our disk, and BLACK_DISK to be enemies.
        # (x, y) is a valid position if the following pattern exists:
        #    "(x, y), BLACK_DISK, ..., BLACK_DISK, WHITE_DISK"

        next_x = x + delta_x
        next_y = y + delta_y

        # The neibor must be an enemy.
        if (next_x < 0 or
            next_x >= self.board_size or
            next_y < 0 or
            next_y >= self.board_size or
            board[next_x][next_y] != BLACK_DISK
        ):
            return 0

        # Keep scanning in the direction.
        cnt = 0
        while (next_x >= 0 and
               next_x < self.board_size and
               next_y >= 0 and
               next_y < self.board_size and
               board[next_x][next_y] == BLACK_DISK
        ):
            next_x += delta_x
            next_y += delta_y
            cnt += 1

        if (next_x < 0 or
            next_x >= self.board_size or
            next_y < 0 or
            next_y >= self.board_size or
            board[next_x][next_y] != WHITE_DISK
        ):
            return 0
        else:
            return cnt

    def get_possible_actions(self, board=None):
        actions=[]
        if board is None:
            if self.player_turn == WHITE_DISK:
                board = self.board_state
            else:
                board = -self.board_state

        for row_ix in range(self.board_size):
            for col_ix in range(self.board_size):
                if board[row_ix][col_ix] == NO_DISK:
                    if (self.is_valid_position(board, row_ix, col_ix, 1, 1) or
                        self.is_valid_position(board, row_ix, col_ix, 1, 0) or
                        self.is_valid_position(board, row_ix, col_ix, 1, -1) or
                        self.is_valid_position(board, row_ix, col_ix, 0, 1) or
                        self.is_valid_position(board, row_ix, col_ix, 0, -1) or
                        self.is_valid_position(board, row_ix, col_ix, -1, 1) or
                        self.is_valid_position(board, row_ix, col_ix, -1, 0) or
                        self.is_valid_position(board, row_ix, col_ix, -1, -1)
                    ):
                        actions.append(row_ix * self.board_size + col_ix)
        return actions

    def print_board(self, print_valid_moves=True):
        valid_actions = self.get_possible_actions()

        if print_valid_moves:
            board = self.board_state.copy().ravel()
            for p in valid_actions:
                board[p] = 2
            board = board.reshape(*self.board_state.shape)
        else:
            board = self.board_state

        print('Turn: {}'.format(
            'WHITE' if self.player_turn == WHITE_DISK else 'BLACK'))
        print('Valid actions: {}'.format(valid_actions))
        for row in board:
            print(' '.join(map(lambda x: ['B', 'O', 'W', 'V'][x + 1], row)))
        print('-' * 10)

    def get_observation(self):
        if self.player_turn == WHITE_DISK:
            # White turn, we don't negate state since white=1.
            return self.board_state
        else:
            # Black turn, we negate board state such that black=1.
            return -self.board_state

    def set_board_state(self, board_state, perspective=WHITE_DISK):
        """Force setting the board state, necessary in model-based RL."""
        if perspective == WHITE_DISK:
            self.board_state = np.array(board_state)
        else:
            self.board_state = np.array(-board_state)

    def update_board(self, action):
       x = action // self.board_size
       y = action % self.board_size

       if self.player_turn == BLACK_DISK:
           self.board_state = -self.board_state

       for delta_x in [-1, 0, 1]:
           for delta_y in [-1, 0, 1]:
               if not (delta_x == 0 and delta_y == 0):
                   kill_cnt = self.is_valid_position(
                       self.board_state, x, y, delta_x, delta_y)
                   for i in range(kill_cnt):
                       dx = (i + 1) * delta_x
                       dy = (i + 1) * delta_y
                       self.board_state[x + dx][y + dy] = WHITE_DISK
       self.board_state[x][y] = WHITE_DISK

       if self.player_turn == BLACK_DISK:
           self.board_state = -self.board_state

    def step(self, action):
        if action not in self.possible_moves:
            if self.sudden_death_on_invalid_move:
                print('Invalid position')
                if self.player_turn == WHITE_DISK:
                    print('BLACK wins')
                    self.winner = BLACK_DISK
                else:
                    print('WHITE wins')
                    self.winner = WHITE_DISK
                done = True
                self.terminated = True
            else:
                done = False
        else:
            self.update_board(action)
            num_vacant_positions = (self.board_state == NO_DISK).sum()
            done = num_vacant_positions == 0
            if done:
                self.winner = self.determine_winner()

        current_player = self.player_turn
        if not done:
            self.set_player_turn(-self.player_turn)
            if len(self.possible_moves) == 0:
                self.set_player_turn(-self.player_turn)
                if len(self.possible_moves) == 0:
                    print('No more possibe moves.')
                    done = True
                    self.winner = self.determine_winner()

        reward = self.winner * current_player
        return self.get_observation(), reward, done, None

    def set_player_turn(self, turn):
        self.player_turn = turn
        self.possible_moves = self.get_possible_actions()

    def count_disks(self):
        white_cnt = (self.board_state == WHITE_DISK).sum()
        black_cnt = (self.board_state == BLACK_DISK).sum()
        return white_cnt, black_cnt

    def determine_winner(self):
        white_cnt, black_cnt = self.count_disks()
        print('white: {}, black: {}'.format(white_cnt, black_cnt))
        self.terminated = True
        if white_cnt > black_cnt:
            print('WHITE wins')
            return WHITE_DISK
        elif black_cnt > white_cnt:
            print('BLACK wins')
            return BLACK_DISK
        else:
            print('DRAW')
            return NO_DISK

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'np_array':
            self.print_board()
        else:
            self.show_gui_board()

    def show_gui_board(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)

        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        self.draw_board()

        win.flip()
        return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def draw_board(self):
        # Draw the green background.
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-BOARDFIELD, +BOARDFIELD, 0)
        gl.glVertex3f(+BOARDFIELD, +BOARDFIELD, 0)
        gl.glVertex3f(+BOARDFIELD, -BOARDFIELD, 0)
        gl.glVertex3f(-BOARDFIELD, -BOARDFIELD, 0)
        gl.glEnd()

        # Draw the grid.
        gl.glBegin(gl.GL_LINES)
        gl.glColor4f(0, 0, 0, 1.0)
        grid_width = BOARDFIELD / self.board_size
        for i in range(1, self.board_size):
            offset = grid_width * i
            gl.glVertex2f(offset, 0)
            gl.glVertex2f(offset, BOARDFIELD)
            gl.glVertex2f(0, offset)
            gl.glVertex2f(BOARDFIELD, offset)
        gl.glEnd()

        # Draw disks.
        half_grid_width = grid_width / 2
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board_state[i][j] == WHITE_DISK:
                    gl.glColor4f(1, 1, 1, 1.)
                elif self.board_state[i][j] == BLACK_DISK:
                    gl.glColor4f(0, 0, 0, 1.)
                if self.board_state[i][j] != NO_DISK:
                    disk = self._make_disk_at(
                        x=i * grid_width + half_grid_width,
                        y=j * grid_width + half_grid_width,
                        radius=half_grid_width - 3)
                    disk.render1()

        # Draw possible positions.
        if self.player_turn == WHITE_DISK:
            gl.glColor4f(1, 1, 1, 1.)
            font_color = (255, 255, 255, 255)
        else:
            gl.glColor4f(0, 0, 0, 1.)
            font_color = (0, 0, 0, 255)
        for p in self.possible_moves:
            i = p // self.board_size
            j = p % self.board_size
            x=i * grid_width + half_grid_width
            y=j * grid_width + half_grid_width
            disk = self._make_disk_at(
                x=x, y=y, radius=half_grid_width - 3, filled=False)
            disk.render1()
            label = pyglet.text.Label('{}'.format(i * self.board_size + j),
                font_name='Times New Roman', font_size=10, color=font_color,
                x=x, y=y, anchor_x='center', anchor_y='center')
            label.draw()

    def _make_disk_at(self, x, y, radius, res=30, filled=True):
        points = []
        for i in range(res):
            ang = 2 * np.pi * i / res
            points.append((np.cos(ang)*radius + x, np.sin(ang)*radius + y))
        if filled:
            return rendering.FilledPolygon(points)
        else:
            return rendering.PolyLine(points, True)

