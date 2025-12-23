from __future__ import annotations
import copy
import numpy as np
from typing import List

# Define constants
WAIT_ACTION = 0
MOVE_ACTIONS = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0)
}
MOVE_TO_ATTACK = 4

class State:
    def __init__(self, n_rows: int, n_cols: int, players: List[str]) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.players = players
        all_positions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
        random_indexes = np.random.choice(len(all_positions), size=len(players), replace=False)
        self.player_positions = {player: all_positions[i] for player, i in zip(players, random_indexes)}
        self.player_scores = {player: {'it_count': 0, 'tag_count': 0} for player in players}
        self.next_player = players[0]
        self.tagged_player = players[0]
        self.just_tagged = True

    def get_legal_actions(self) -> List[int]:
        player = self.next_player
        tagged = player == self.tagged_player
        r, c = self.player_positions[player]
        opponent_positions = [v for k, v in self.player_positions.items() if k != player]
        legal_actions = []
        for action, (dr, dc) in MOVE_ACTIONS.items():
            new_r, new_c = r + dr, c + dc
            if 0 <= new_r < self.n_rows and 0 <= new_c < self.n_cols:
                if (new_r, new_c) in opponent_positions:
                    if tagged and not self.just_tagged:
                        legal_actions.append(action + MOVE_TO_ATTACK)
                else:
                    legal_actions.append(action)
        if tagged or not legal_actions:
            legal_actions.append(WAIT_ACTION)
        return legal_actions

    def step(self, action: int) -> None:
        players = self.players
        player = self.next_player
        tagged = player == self.tagged_player
        if action not in self.get_legal_actions():
            raise ValueError(f'Action {action} is not a legal action for state {self}.')
        if action == WAIT_ACTION:
            if tagged:
                self.just_tagged = False
                self.player_scores[player]['it_count'] += 1
        elif action in MOVE_ACTIONS:
            r, c = self.player_positions[player]
            dr, dc = MOVE_ACTIONS[action]
            self.player_positions[player] = (r + dr, c + dc)
            if tagged:
                self.just_tagged = False
                self.player_scores[player]['it_count'] += 1
        else:
            r, c = self.player_positions[player]
            dr, dc = MOVE_ACTIONS[action - MOVE_TO_ATTACK]
            self.tagged_player = next(k for k, v in self.player_positions.items() if v == (r + dr, c + dc))
            self.just_tagged = True
            self.player_scores[player]['tag_count'] += 1
        self.next_player = players[(players.index(player) + 1) % len(players)]

    def copy(self) -> State:
        return copy.deepcopy(self)

    def __str__(self) -> str:
        border = '\n' + '---' * self.n_rows + '\n'
        grid = [[' . ' for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for player, (r, c) in self.player_positions.items():
            if self.tagged_player == player:
                grid[r][c] = f'[{player}]'
            else:
                grid[r][c] = f' {player} '
        return border + '\n'.join(''.join(row) for row in grid) + border
    __repr__ = __str__
