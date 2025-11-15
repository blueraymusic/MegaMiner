import sys
import json
import random
import os
from typing import List, Tuple

import torch
from stable_baselines3 import PPO
import numpy as np

# --- AIAction class ---
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

# --- Helper functions ---
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

def choose_best_build_position(build_spaces, my_towers, enemy_units, base_x, base_y):
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

def choose_best_merc_direction(enemies, base_x, base_y, directions):
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


# --- Hybrid AI class ---
class HybridAI_v4_PPO:
    def initialize_and_set_name(self, initial_game_state, team_color):
        self.team_color = team_color
        self.turn = 0

        # Towers count by type for tracking
        self.tower_count = {"house":0, "crossbow":0, "cannon":0, "minigun":0, "church":0}

        # File to save/load persistent memory
        self.memory_file = f"ai_memory_{team_color}.json"
        self.memory = {
            "tower_builds": [],
            "tower_destroys": [],
            "merc_uses": [],
            "enemy_heatmap": [],
            "provoke_history": []
        }
        self.load_memory()

        # Load PPO model if exists
        best_model_path = "training/models/best_model/best_model.zip"
        if os.path.exists(best_model_path):
            self.ppo = PPO.load(best_model_path, device="cpu")
            self.ppo_available = True
        else:
            print("WARNING: PPO model not found. Hybrid will run pure rule-based fallback.")
            self.ppo_available = False

        return "HybridAI_v4_PPO"

    def load_memory(self):
        if os.path.isfile(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    self.memory = json.load(f)
            except Exception:
                pass

    def save_memory(self):
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f, indent=2)
        except Exception:
            pass

    def encode_state(self, game_state):
        vec = []

        vec.append(game_state.get("CurrentTurn", 0))

        me = game_state['PlayerBaseR'] if self.team_color == 'r' else game_state['PlayerBaseB']
        vec.append(me['x'])
        vec.append(me['y'])

        enemy_units = game_state.get("EnemyUnits", [])
        vec.append(len(enemy_units))

        vec.append(get_my_money_amount(game_state, self.team_color))
        vec.append(len(get_my_towers(game_state, self.team_color)))

        # Incorporate simple enemy heatmap sum as a feature (example)
        heatmap = self.memory.get("enemy_heatmap", [])
        if heatmap:
            heat_sum = sum(sum(row) for row in heatmap)
            vec.append(heat_sum)
        else:
            vec.append(0)

        return np.array(vec, dtype=np.float32)

    def ppo_to_action(self, ppo_action, game_state):
        build_spaces = get_available_build_spaces(game_state, self.team_color)
        q_dirs = get_available_queue_directions(game_state, self.team_color)
        my_towers = get_my_towers(game_state, self.team_color)

        a = int(ppo_action)

        if a == 1 and build_spaces:
            x, y = random.choice(build_spaces)
            tow = random.choice(["house","crossbow","cannon","minigun","church"])
            # Track build for memory
            self.memory["tower_builds"].append({"x": x, "y": y, "type": tow})
            return AIAction("build", x, y, tow)

        if a == 2 and q_dirs:
            merc_dir = random.choice(q_dirs)
            self.memory["merc_uses"].append(merc_dir)
            return AIAction("nothing", 0, 0, merc_direction=merc_dir)

        if a == 3 and my_towers:
            t = random.choice(my_towers)
            self.memory["tower_destroys"].append({"x": t["x"], "y": t["y"], "type": t.get("Type", "")})
            return AIAction("destroy", t["x"], t["y"])

        return AIAction("nothing", 0, 0)

    def update_enemy_heatmap(self, game_state):
        enemies = game_state.get("EnemyUnits", [])
        floor = game_state.get("FloorTiles", [])
        if not floor:
            return
        h = len(floor)
        w = len(floor[0])
        if not self.memory.get("enemy_heatmap") or len(self.memory["enemy_heatmap"]) != h or len(self.memory["enemy_heatmap"][0]) != w:
            self.memory["enemy_heatmap"] = [[0]*w for _ in range(h)]
        for e in enemies:
            x, y = e.get("x", 0), e.get("y", 0)
            if 0 <= y < h and 0 <= x < w:
                self.memory["enemy_heatmap"][y][x] += 1

    def do_turn(self, game_state):
        turn = game_state["CurrentTurn"]
        self.turn = turn

        # Update heatmap for enemies
        self.update_enemy_heatmap(game_state)

        build_spaces = get_available_build_spaces(game_state, self.team_color)
        my_towers = get_my_towers(game_state, self.team_color)
        q_dirs = get_available_queue_directions(game_state, self.team_color)
        enemy_units = game_state.get("EnemyUnits", [])

        base = game_state['PlayerBaseR'] if self.team_color == 'r' else game_state['PlayerBaseB']
        base_x, base_y = base['x'], base['y']

        provoke = (turn == 30 and self.team_color=='r') \
                  or (turn == 31 and self.team_color=='b') \
                  or (turn == 38)
        self.memory["provoke_history"].append(provoke)

        # PHASE 1: Rule-based early game
        if turn < 25:
            if build_spaces:
                x, y = choose_best_build_position(build_spaces, my_towers, enemy_units, base_x, base_y)
                self.tower_count["house"] += 1
                self.memory["tower_builds"].append({"x": x, "y": y, "type": "house"})
                self.save_memory()
                return AIAction("build", x, y, "house", provoke_demons=provoke)

            if q_dirs:
                merc_dir = random.choice(q_dirs)
                self.memory["merc_uses"].append(merc_dir)
                self.save_memory()
                return AIAction("nothing", 0, 0, merc_direction=merc_dir, provoke_demons=provoke)

            self.save_memory()
            return AIAction("nothing", 0, 0, provoke_demons=provoke)

        # PHASE 2: Hybrid Rule/PPO
        if 25 <= turn < 60:
            use_ppo = self.ppo_available and random.random() < 0.3
            if use_ppo:
                obs = self.encode_state(game_state)
                act, _ = self.ppo.predict(obs, deterministic=False)
                hybrid_action = self.ppo_to_action(act, game_state)
                hybrid_action.provoke_demons = provoke
                self.save_memory()
                return hybrid_action

            if build_spaces:
                tower_choices = ['cannon','crossbow','minigun','church']
                weights = [0.35,0.35,0.2,0.1]
                tow = random.choices(tower_choices, weights=weights, k=1)[0]
                x, y = choose_best_build_position(build_spaces, my_towers, enemy_units, base_x, base_y)
                self.tower_count[tow] += 1
                self.memory["tower_builds"].append({"x": x, "y": y, "type": tow})
                self.save_memory()
                return AIAction("build", x, y, tow, provoke_demons=provoke)

            if q_dirs:
                best_dir = choose_best_merc_direction(enemy_units, base_x, base_y, q_dirs)
                self.memory["merc_uses"].append(best_dir)
                self.save_memory()
                return AIAction("nothing", 0, 0, merc_direction=best_dir, provoke_demons=provoke)

            self.save_memory()
            return AIAction("nothing", 0, 0, provoke_demons=provoke)

        # PHASE 3: Full PPO control
        if self.ppo_available:
            obs = self.encode_state(game_state)
            act, _ = self.ppo.predict(obs, deterministic=False)
            action = self.ppo_to_action(act, game_state)
            action.provoke_demons = provoke
            self.save_memory()
            return action

        # PPO not available fallback
        self.save_memory()
        return AIAction("nothing",0,0,provoke_demons=provoke)

if __name__ == '__main__':
    team_color = "r" if input() == "--YOU ARE RED--" else "b"
    input_buffer = [input()]
    while input_buffer[-1] != "--END INITIAL GAME STATE--":
        input_buffer.append(input())
    game_state_init = json.loads("".join(input_buffer[:-1]))

    ai = HybridAI_v4_PPO()
    print(ai.initialize_and_set_name(game_state_init, team_color))
    print(ai.do_turn(game_state_init).to_json())

    while True:
        input_buffer = [input()]
        while input_buffer[-1] != "--END OF TURN--":
            input_buffer.append(input())
        game_state = json.loads("".join(input_buffer[:-1]))
        print(ai.do_turn(game_state).to_json())
