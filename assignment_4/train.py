import argparse
import string
import itertools
from tqdm import tqdm
from state import State
from agent import (
    Agent,
    INIT_L2_PENALTY,
    INIT_LEARNING_RATE,
    INIT_DISCOUNT_FACTOR,
    RewardFunction,
    FeatureFunction,
    GameScheduler,
)

PLAYER_SYMBOLS = string.digits + string.ascii_uppercase + string.ascii_lowercase + string.punctuation.replace('.', '')
def generate_players(n_players):
    players = []
    symbol_length = 1
    while len(players) < n_players:
        for combo in itertools.product(PLAYER_SYMBOLS, repeat=symbol_length):
            players.append(''.join(combo))
            if len(players) == n_players:
                break
        symbol_length += 1
    return players

def play_game(
    n_rows: int,
    n_cols: int,
    n_players: int,
    n_steps: int,
    agent: Agent,
    reward_function: RewardFunction,
    feature_function: FeatureFunction,
    training: bool = True,
    verbose: bool = False,
):
    players = generate_players(n_players)
    state = State(n_rows, n_cols, players)
    if verbose:
        print(state)
    for t in range(n_steps):
        for _ in range(n_players):
            agent.player = state.next_player
            reward_function.player = state.next_player
            feature_function.player = state.next_player
            action = agent.select_action(state, training=training)
            next_state = state.copy()
            next_state.step(action)
            if training:
                agent.update_model(state, action, next_state, t == n_steps - 1)
            state = next_state
        if verbose:
            print(state)
    return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1, help='Number of games (episodes) to play.')
    args = parser.parse_args()
    reward_function = RewardFunction(None)
    feature_function = FeatureFunction(None)
    agent = Agent(
        player=None,
        l2_penalty=INIT_L2_PENALTY,
        learning_rate=INIT_LEARNING_RATE,
        discount_factor=INIT_DISCOUNT_FACTOR,
        reward_function=reward_function,
        feature_function=feature_function,
    )
    scheduler = GameScheduler(n_episodes=args.episodes)
    for episode in tqdm(range(args.episodes)):
        play_game(
            **scheduler(episode),
            agent=agent,
            reward_function=reward_function,
            feature_function=feature_function,
            training=True,
            verbose=False,
        )
    play_game(
        **scheduler(None),
        agent=agent,
        reward_function=reward_function,
        feature_function=feature_function,
        training=False,
        verbose=True,
    )
