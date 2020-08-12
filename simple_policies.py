"""Simple policies for Othello."""

import numpy as np


WHITE_DISK = 1
BLACK_DISK = -1
PROTAGONIST_TURN = 1
OPPONENT_TURN = -1


def copy_env(env, mute_env=True):
    new_env = env.__class__(
        board_size=env.board_size,
        sudden_death_on_invalid_move=env.sudden_death_on_invalid_move,
        mute=mute_env)
    new_env.reset()
    return new_env


class RandomPolicy(object):
    """Random policy for Othello."""

    def __init__(self, seed=0):
        self.rnd = np.random.RandomState(seed=seed)
        self.env = None

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):
        possible_moves = self.env.possible_moves
        ix = self.rnd.randint(0, len(possible_moves))
        action = possible_moves[ix]
        return action


class GreedyPolicy(object):
    """Greed is good."""

    def __init__(self):
        self.env = None

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def get_action(self, obs):
        my_perspective = self.env.player_turn
        new_env = copy_env(self.env)

        # For each move, replicate the current board and make the move.
        possible_moves = self.env.possible_moves
        disk_cnts = []
        for move in possible_moves:
            new_env.reset()
            new_env.set_board_state(
                board_state=obs, perspective=my_perspective)
            new_env.set_player_turn(my_perspective)
            new_env.step(move)
            white_disks, black_disks = new_env.count_disks()
            if my_perspective == WHITE_DISK:
                disk_cnts.append(white_disks)
            else:
                disk_cnts.append(black_disks)

        new_env.close()
        ix = np.argmax(disk_cnts)
        return possible_moves[ix]


class MaxiMinPolicy(object):
    """Maximin algorithm."""

    def __init__(self, max_search_depth=1):
        self.env = None
        self.max_search_depth = max_search_depth

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def search(self, env, depth, perspective, my_perspective):

        # Search at a node stops if
        #   1. Game terminated
        #   2. depth has reached max_search_depth
        #   3. No more possible moves
        if (
                env.terminated or
                depth >= self.max_search_depth or
                len(env.possible_moves) == 0
        ):
            white_disks, black_disks = env.count_disks()
            if my_perspective == WHITE_DISK:
                return white_disks, None
            else:
                return black_disks, None
        else:
            assert env.player_turn == perspective
            new_env = copy_env(env)

            # For each move, replicate the current board and make the move.
            possible_moves = env.possible_moves
            disk_cnts = []
            for move in possible_moves:
                new_env.reset()
                new_env.set_board_state(env.get_observation(), env.player_turn)
                new_env.set_player_turn(perspective)
                new_env.step(move)
                if (
                        not new_env.terminated and
                        new_env.player_turn == perspective
                ):
                    # The other side had no possible moves.
                    new_env.set_player_turn(-perspective)
                disk_cnt, _ = self.search(
                    new_env, depth + 1, -perspective, my_perspective)
                disk_cnts.append(disk_cnt)

            new_env.close()

            # Max-min.
            ix = int(np.argmin(disk_cnts))
            if perspective == my_perspective:
                ix = int(np.argmax(disk_cnts))
            return disk_cnts[ix], possible_moves[ix]

    def get_action(self, obs):
        my_perspective = self.env.player_turn
        disk_cnt, move = self.search(env=self.env,
                                     depth=0,
                                     perspective=my_perspective,
                                     my_perspective=my_perspective)
        return move


class HumanPolicy(object):
    """Human policy."""

    def __init__(self, board_size):
        self.board_size = board_size

    def reset(self, env):
        pass

    def get_action(self, obs):
        return int(input('Enter action index:'))
