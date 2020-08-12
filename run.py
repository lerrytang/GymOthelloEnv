"""Play Othello."""

import argparse
import othello
import simple_policies


def create_policy(policy_type='rand', board_size=8, seed=0, search_depth=1):
    if policy_type == 'rand':
        policy = simple_policies.RandomPolicy(seed=seed)
    elif policy_type == 'greedy':
        policy = simple_policies.GreedyPolicy()
    elif policy_type == 'maximin':
        policy = simple_policies.MaxiMinPolicy(search_depth)
    else:
        policy = simple_policies.HumanPolicy(board_size)
    return policy


def play(protagonist,
         protagonist_agent_type='greedy',
         opponent_agent_type='rand',
         board_size=8,
         num_rounds=100,
         search_depth=1,
         rand_seed=0,
         render=True):

    protagonist_policy = create_policy(
        policy_type=protagonist_agent_type,
        board_size=board_size, seed=rand_seed, search_depth=search_depth)
    opponent_policy = create_policy(
        policy_type=opponent_agent_type,
        board_size=board_size, seed=rand_seed, search_depth=search_depth)

    if protagonist == 1:
        white_policy = protagonist_policy
        black_policy = opponent_policy
    else:
        white_policy = opponent_policy
        black_policy = protagonist_policy

    if protagonist_agent_type == 'human' and opponent_agent_type == 'human':
        render_in_step = True
    else:
        render_in_step = False

    env = othello.OthelloEnv(white_policy=white_policy,
                             black_policy=black_policy,
                             protagonist=protagonist,
                             board_size=board_size,
                             render_in_step=render_in_step and render)

    win_cnts = draw_cnts = lose_cnts = 0
    for _ in range(num_rounds):
        obs = env.reset()
        protagonist_policy.reset(env)
        if render:
            env.render()
        done = False
        while not done:
            action = protagonist_policy.get_action(obs)
            obs, reward, done, _ = env.step(action)
            if render:
                env.render()
            if done:
                if reward == 1:
                    win_cnts += 1
                elif reward == 0:
                    draw_cnts += 1
                else:
                    lose_cnts += 1
    print('#Wins: {}, #Draws: {}, #Loses: {}'.format(
        win_cnts, draw_cnts, lose_cnts))
    env.close()


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--protagonist', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human'])
    parser.add_argument('--opponent', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human'])
    parser.add_argument('--protagonist-plays-black', default=False,
                        action='store_true')
    parser.add_argument('--board-size', default=8, type=int)
    parser.add_argument('--search-depth', default=1, type=int)
    parser.add_argument('--rand-seed', default=0, type=int)
    parser.add_argument('--num-rounds', default=100, type=int)
    parser.add_argument('--no-render', default=False, action='store_true')
    args, _ = parser.parse_known_args()

    # Run test plays.
    protagonist = -1 if args.protagonist_plays_black else 1
    protagonist_agent_type = args.protagonist
    opponent_agent_type = args.opponent
    play(protagonist=protagonist,
         protagonist_agent_type=protagonist_agent_type,
         opponent_agent_type=opponent_agent_type,
         board_size=args.board_size,
         num_rounds=args.num_rounds,
         search_depth=args.search_depth,
         rand_seed=args.rand_seed,
         render=not args.no_render)

