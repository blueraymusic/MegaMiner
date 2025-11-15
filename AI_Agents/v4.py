import sys
import json
import random
import os
from typing import List, Tuple

# --- AI ACTION CLASS ---
class AIAction:
    def __init__(
        self,
        action: str,
        x: int,
        y: int,
        tower_type: str = "",
        merc_direction: str = "",
        provoke_demons: bool = False
    ):
        self.action = action.lower().strip()
        self.x = x
        self.y = y
        self.tower_type = tower_type.strip()
        self.merc_direction = merc_direction.upper().strip()
        self.provoke_demons = provoke_demons
    
    def to_dict(self):
        return {
            'action': self.action,
            'x': self.x,
            'y': self.y,
            'tower_type': self.tower_type,
            'merc_direction': self.merc_direction,
            'provoke_demons': self.provoke_demons
        }
    
    def to_json(self):
        return json.dumps(self.to_dict())

# --- HELPER FUNCTIONS ---
def is_out_of_bounds(game_state: dict, x: int, y: int) -> bool:
    return x < 0 or x >= len(game_state['FloorTiles'][0]) or y < 0 or y >= len(game_state['FloorTiles'])

def get_available_queue_directions(game_state: dict, team_color: str) -> List[str]:
    result = []
    offsets = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W"}
    player = game_state['PlayerBaseR'] if team_color == 'r' else game_state['PlayerBaseB']
    for dx, dy in offsets.keys():
        x, y = player['x'] + dx, player['y'] + dy
        if not is_out_of_bounds(game_state, x, y) and game_state['FloorTiles'][y][x] == "O":
            result.append(offsets[(dx, dy)])
    return result

def get_available_build_spaces(game_state: dict, team_color: str) -> List[Tuple[int,int]]:
    result = []
    for y, row in enumerate(game_state['FloorTiles']):
        for x, c in enumerate(row):
            if c == team_color and game_state['EntityGrid'][y][x] == '':
                result.append((x, y))
    return result

def get_my_towers(game_state: dict, team_color: str) -> List[dict]:
    return [t for t in game_state['Towers'] if t["Team"] == team_color]

def get_my_money_amount(game_state: dict, team_color: str) -> int:
    return game_state["RedTeamMoney"] if team_color == 'r' else game_state["BlueTeamMoney"]

def choose_best_build_position(build_spaces: List[Tuple[int,int]], my_towers: List[dict], enemy_units: List[dict], base_x: int, base_y: int) -> Tuple[int,int]:
    """Evaluate each available build space for threat coverage and clustering"""
    best_score = -1
    best_pos = None
    for x, y in build_spaces:
        cluster_penalty = sum(1 for t in my_towers if abs(t["x"] - x) + abs(t["y"] - y) < 2)
        coverage_score = sum(1/(1+abs(ex-x)+abs(ey-y)) for ex, ey in [(e["x"], e["y"]) for e in enemy_units])
        score = coverage_score - cluster_penalty
        if score > best_score:
            best_score = score
            best_pos = (x, y)
    if best_pos is None and build_spaces:
        best_pos = random.choice(build_spaces)
    return best_pos

def choose_best_merc_direction(enemies: List[dict], base_x: int, base_y: int, directions: List[str]) -> str:
    if not enemies:
        return random.choice(directions)
    threat_map = {d:0 for d in directions}
    for e in enemies:
        dx = e["x"] - base_x
        dy = e["y"] - base_y
        if dx > 0 and "E" in threat_map: threat_map["E"] += 1
        if dx < 0 and "W" in threat_map: threat_map["W"] += 1
        if dy > 0 and "S" in threat_map: threat_map["S"] += 1
        if dy < 0 and "N" in threat_map: threat_map["N"] += 1
    best_dir = max(threat_map, key=threat_map.get)
    if threat_map[best_dir] == 0:
        return random.choice(directions)
    return best_dir

# --- STRATEGIC AI v4.2 ---
class StrategicAI_v4_2:
    def initialize_and_set_name(self, initial_game_state: dict, team_color: str) -> str:
        self.team_color = team_color
        self.tower_count = {"house":0, "crossbow":0, "cannon":0, "minigun":0, "church":0}
        return "StrategicAI_v4.2"

    def do_turn(self, game_state: dict) -> AIAction:
        build_spaces = get_available_build_spaces(game_state, self.team_color)
        my_towers = get_my_towers(game_state, self.team_color)
        my_money = get_my_money_amount(game_state, self.team_color)
        q_dirs = get_available_queue_directions(game_state, self.team_color)
        turn = game_state["CurrentTurn"]
        base = game_state['PlayerBaseR'] if self.team_color == 'r' else game_state['PlayerBaseB']
        base_x, base_y = base['x'], base['y']
        enemy_units = game_state.get("EnemyUnits", [])

        # --- Turn-timed demon provocation ---
        provoke_demons = ((turn == 30 and self.team_color == 'r') or 
                          (turn == 31 and self.team_color == 'b') or 
                          (turn == 38))

        # --- Early game: prioritize houses ---
        if turn == 0 and build_spaces:
            x, y = random.choice(build_spaces)
            self.tower_count["house"] += 1
            return AIAction("build", x, y, 'house', provoke_demons=provoke_demons)

        # --- Early to mid game: houses or mercs ---
        if turn < 25 and build_spaces:
            if random.random() < 0.5 and q_dirs:
                return AIAction("nothing", 0, 0, merc_direction=random.choice(q_dirs), provoke_demons=provoke_demons)
            else:
                x, y = choose_best_build_position(build_spaces, my_towers, enemy_units, base_x, base_y)
                self.tower_count["house"] += 1
                return AIAction("build", x, y, 'house', provoke_demons=provoke_demons)

        # --- Mid to late game: smart tower placement with weighted randomness ---
        if build_spaces:
            tower_choices = ['cannon', 'crossbow', 'minigun', 'church']
            weights = [0.35, 0.35, 0.2, 0.1]
            tower = random.choices(tower_choices, weights=weights, k=1)[0]
            x, y = choose_best_build_position(build_spaces, my_towers, enemy_units, base_x, base_y)
            self.tower_count[tower] += 1
            merc_direction = random.choice(q_dirs) if q_dirs else ''
            return AIAction("build", x, y, tower, merc_direction=merc_direction, provoke_demons=provoke_demons)

        # --- Late game: no build space, send mercs or destroy towers if turn > 50 ---
        if q_dirs:
            if turn >= 50 and my_towers:
                to_destroy = random.choice(my_towers)
                return AIAction("destroy", to_destroy["x"], to_destroy["y"])
            best_dir = choose_best_merc_direction(enemy_units, base_x, base_y, q_dirs)
            return AIAction("nothing", 0, 0, merc_direction=best_dir, provoke_demons=provoke_demons)

        return AIAction("nothing", 0, 0)

# --- DRIVER CODE ---
if __name__ == '__main__':
    team_color = 'r' if input() == "--YOU ARE RED--" else 'b'

    input_buffer = [input()]
    while input_buffer[-1] != "--END INITIAL GAME STATE--":
        input_buffer.append(input())
    game_state_init = json.loads(''.join(input_buffer[:-1]))

    agent = StrategicAI_v4_2()
    print(agent.initialize_and_set_name(game_state_init, team_color))
    print(agent.do_turn(game_state_init).to_json())

    while True:
        input_buffer = [input()]
        while input_buffer[-1] != "--END OF TURN--":
            input_buffer.append(input())
        game_state_this_turn = json.loads(''.join(input_buffer[:-1]))
        print(agent.do_turn(game_state_this_turn).to_json())
