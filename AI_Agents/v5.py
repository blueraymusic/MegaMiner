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
        x: int = 0,
        y: int = 0,
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

def get_base(game_state: dict, team_color: str) -> dict:
    return game_state['PlayerBaseR'] if team_color == 'r' else game_state['PlayerBaseB']

def get_available_queue_directions(game_state: dict, team_color: str) -> List[str]:
    directions = []
    offsets = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
    base = get_base(game_state, team_color)
    for d, (dx, dy) in offsets.items():
        nx, ny = base["x"] + dx, base["y"] + dy
        if not is_out_of_bounds(game_state, nx, ny) and game_state["FloorTiles"][ny][nx] == "O":
            directions.append(d)
    return directions

def get_available_build_spaces(game_state: dict, team_color: str) -> List[Tuple[int, int]]:
    return [(x, y) for y, row in enumerate(game_state['FloorTiles'])
            for x, tile in enumerate(row) if tile == team_color and not game_state['EntityGrid'][y][x]]

def get_my_towers(game_state: dict, team_color: str) -> List[dict]:
    return [t for t in game_state['Towers'] if t["Team"] == team_color]

def get_enemies(game_state: dict, team_color: str) -> List[dict]:
    return [e for e in game_state.get("Mercenaries", []) + game_state.get("Demons", []) if e.get("Team") != team_color]

def get_my_money(game_state: dict, team_color: str) -> int:
    return game_state["RedTeamMoney"] if team_color == 'r' else game_state["BlueTeamMoney"]

def cluster_threat(enemies: List[dict], x: int, y: int) -> int:
    close, mid, far = 0, 0, 0
    for e in enemies:
        dist = abs(e["x"] - x) + abs(e["y"] - y)
        if dist <= 3:
            close += 1
        elif dist <= 6:
            mid += 1
        elif dist <= 10:
            far += 1
    return 6*close + 3*mid + far

def learning_penalty(memory: dict, tower_type: str = None, merc_direction: str = None) -> float:
    penalty = 0.0
    if tower_type:
        perf = memory.get("tower_performance", {}).get(tower_type, {"built": 1, "destroyed": 0, "kills": 1})
        destroyed_ratio = perf["destroyed"] / (perf["built"] + 1)
        kill_ratio = perf["kills"] / (perf["built"] + 1)
        penalty += destroyed_ratio * 50
        penalty -= kill_ratio * 20
    if merc_direction:
        perf = memory.get("merc_performance", {}).get(merc_direction, {"used": 1, "success": 1})
        fail_ratio = 1 - (perf["success"] / (perf["used"] + 1))
        penalty += fail_ratio * 30
    return penalty

def score_build_with_map(self, tower_type, x, y, money, towers, enemies, base_x, base_y, turn, house_count, memory):
    repeat_penalty = 0
    if 'build_history' in memory:
        recent_builds = memory['build_history'][-5:]
        if (x, y) in recent_builds:
            repeat_penalty = 50
    enemy_types = set(e.get("Type", "") for e in enemies)
    counter_bonus = 0
    if tower_type == 'cannon' and 'heavy' in enemy_types:
        counter_bonus = 30
    if tower_type == 'crossbow' and 'flying' in enemy_types:
        counter_bonus = 30
    if tower_type == 'church' and 'undead' in enemy_types:
        counter_bonus = 20
    if tower_type == "house" and (turn < 10 or house_count < 8):
        base_score = 1000 - 50 * abs(base_x - x) - 50 * abs(base_y - y)
    else:
        threat = cluster_threat(enemies, x, y)
        cluster = sum(1 for t in towers if abs(t["x"] - x) + abs(t["y"] - y) < 2)
        base_score = {
            "crossbow": threat * 8 + 100 - cluster * 20,
            "cannon": threat * 15 + 200 - cluster * 25,
            "minigun": threat * 12 + 300 - cluster * 25,
            "church": 300 + threat * 4 - cluster * 15
        }.get(tower_type, 100)
    heat = memory.get('enemy_heatmap', [])
    if heat and 0 <= y < len(heat) and 0 <= x < len(heat[0]):
        base_score += heat[y][x] * 5
    return base_score - repeat_penalty + counter_bonus - learning_penalty(memory, tower_type=tower_type)

def score_merc(direction: str, enemies: List[dict], base_x: int, base_y: int, memory: dict) -> float:
    score = 0
    for e in enemies:
        dx, dy = e["x"] - base_x, e["y"] - base_y
        if direction == "E" and dx > 0:
            score += 5 / (1 + abs(dx))
        if direction == "W" and dx < 0:
            score += 5 / (1 + abs(dx))
        if direction == "S" and dy > 0:
            score += 2 / (1 + abs(dy))
        if direction == "N" and dy < 0:
            score += 2 / (1 + abs(dy))
    if direction in memory.get('merc_directions', [])[-3:]:
        score *= 0.8
    score -= learning_penalty(memory, merc_direction=direction)
    return score

def best_build(space: Tuple[int, int], towers: List[dict], enemies: List[dict], base_x: int, base_y: int,
               turn: int, money: int, house_count: int, team: str, state: dict, memory: dict, ai_instance) -> str:
    tower_prices = state['TowerPricesR'] if team == 'r' else state['TowerPricesB']
    affordable = [t for t in ["minigun", "cannon", "crossbow", "church", "house"]
                  if money >= tower_prices.get(t, 10000)]
    best_score, best_type = float("-inf"), "house"
    for t in affordable:
        val = ai_instance.score_build_with_map(t, space[0], space[1], money, towers, enemies, base_x, base_y, turn, house_count, memory)
        if val > best_score:
            best_score, best_type = val, t
    return best_type

def choose_best_merc_direction(enemies: List[dict], base_x: int, base_y: int, directions: List[str], memory: dict) -> str:
    if not enemies:
        return random.choice(directions) if directions else ""
    threat_map = {d: 0 for d in directions}
    for e in enemies:
        dx = e["x"] - base_x
        dy = e["y"] - base_y
        if dx > 0 and "E" in threat_map:
            threat_map["E"] += 1
        if dx < 0 and "W" in threat_map:
            threat_map["W"] += 1
        if dy > 0 and "S" in threat_map:
            threat_map["S"] += 1
        if dy < 0 and "N" in threat_map:
            threat_map["N"] += 1
    best_dir = max(threat_map, key=threat_map.get, default="")
    if best_dir == "" or threat_map[best_dir] == 0:
        best_dir = random.choice(directions) if directions else ""
    recent_dirs = memory.get('merc_directions', [])
    if best_dir in recent_dirs[-3:]:
        candidates = [d for d in directions if d not in recent_dirs[-3:]]
        if candidates:
            best_dir = random.choice(candidates)
    return best_dir

# --- STRATEGIC AI WITH MAP MEMORY AND LEARNING ---
class StrategicAI_MergedWithMemory:
    def initialize_and_set_name(self, state: dict, team_color: str) -> str:
        self.team = team_color
        self.memory_file = f"ai_memory_{team_color}.json"
        self.tower_count = {k: 0 for k in ["house", "crossbow", "cannon", "minigun", "church"]}
        self.memory = {
            "past_enemy_composition": [],
            "build_history": [],
            "merc_directions": [],
            "provoke_history": [],
            "tower_performance": {k: {"built": 0, "destroyed": 0, "kills": 0} for k in self.tower_count},
            "merc_performance": {},
            "last_actions": [],
            "map_layout": [],
            "tower_positions": [],
            "enemy_heatmap": []
        }
        self.load_memory()
        return "v5"

    def load_memory(self):
        if os.path.isfile(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    loaded = json.load(f)
                    for k in self.memory.keys():
                        if k in loaded:
                            self.memory[k] = loaded[k]
            except Exception:
                pass

    def save_memory(self):
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f, indent=2)
        except Exception:
            pass

    def save_map_memory(self, state: dict):
        self.memory['map_layout'] = state['FloorTiles']
        towers = get_my_towers(state, self.team)
        self.memory['tower_positions'] = [(t['x'], t['y'], t.get('Type', '')) for t in towers]

    def update_enemy_heatmap(self, state: dict):
        enemies = get_enemies(state, self.team)
        if not self.memory['enemy_heatmap']:
            height = len(state['FloorTiles'])
            width = len(state['FloorTiles'][0])
            self.memory['enemy_heatmap'] = [[0]*width for _ in range(height)]
        for e in enemies:
            x, y = e['x'], e['y']
            if 0 <= y < len(self.memory['enemy_heatmap']) and 0 <= x < len(self.memory['enemy_heatmap'][0]):
                self.memory['enemy_heatmap'][y][x] += 1

    def update_performance(self, state):
        """
        Track performance for towers and mercs.
        - Tower performance: built, destroyed, kills
        - Merc performance: used, success
        """
        # Update tower performance
        my_towers = get_my_towers(state, self.team)
        current_positions = {(t["x"], t["y"]): t for t in my_towers}

        # Track destroyed towers
        for t_type, perf in self.memory["tower_performance"].items():
            # Check how many of this type currently exist
            existing_count = sum(1 for t in my_towers if t.get("Type") == t_type)
            destroyed = perf["built"] - existing_count
            if destroyed > perf["destroyed"]:
                perf["destroyed"] = destroyed

        # Track tower kills (simplified: any enemy in range of tower this turn counts)
        enemies = get_enemies(state, self.team)
        for t in my_towers:
            t_type = t.get("Type", "")
            if t_type not in self.memory["tower_performance"]:
                continue
            perf = self.memory["tower_performance"][t_type]
            kills = sum(1 for e in enemies if abs(e["x"] - t["x"]) + abs(e["y"] - t["y"]) <= 2)
            perf["kills"] += kills

        # Update merc performance
        merc_history = self.memory.get("merc_performance", {})
        last_dirs = self.memory.get("merc_directions", [])[-3:]  # last 3 mercs
        for d in last_dirs:
            if d not in merc_history:
                merc_history[d] = {"used": 0, "success": 0}
            merc_history[d]["used"] += 1
            # Success if any enemy is in direction within 3 tiles
            bx, by = get_base(state, self.team)["x"], get_base(state, self.team)["y"]
            success = any(
                (d == "E" and e["x"] > bx) or
                (d == "W" and e["x"] < bx) or
                (d == "N" and e["y"] < by) or
                (d == "S" and e["y"] > by)
                for e in enemies
            )
            if success:
                merc_history[d]["success"] += 1

        self.memory["merc_performance"] = merc_history


    def score_build_with_map(self, *args, **kwargs):
        return score_build_with_map(self, *args, **kwargs)

    def do_turn(self, state: dict) -> AIAction:
        self.save_map_memory(state)
        self.update_enemy_heatmap(state)
        self.update_performance(state)

        team = self.team
        turn = state["CurrentTurn"]
        base = get_base(state, team)
        bx, by = base["x"], base["y"]
        spaces = get_available_build_spaces(state, team)
        towers = get_my_towers(state, team)
        enemies = get_enemies(state, team)
        dirs = get_available_queue_directions(state, team)
        money = get_my_money(state, team)

        current_enemy_types = tuple(sorted(e.get("Type", "") for e in enemies))
        self.memory["past_enemy_composition"].append(current_enemy_types)

        recent_builds = self.memory.get("build_history", [])
        spaces = [s for s in spaces if s not in recent_builds[-5:]]

        threat_index = cluster_threat(enemies, bx, by)
        house_count = self.tower_count["house"]

        provoke = (sum(self.tower_count.values()) > 7 and turn >= 15 and threat_index > 9) or (turn in {28, 37, 46, 55})
        self.memory["provoke_history"].append(provoke)

        if turn == 0 and spaces:
            x, y = random.choice(spaces)
            self.tower_count["house"] += 1
            self.memory["build_history"].append((x, y))
            self.save_memory()
            return AIAction("build", x, y, "house", provoke_demons=provoke)

        if turn < 25 and spaces:
            if random.random() < 0.5 and dirs:
                merc_dir = random.choice(dirs)
                self.memory["merc_directions"].append(merc_dir)
                self.save_memory()
                return AIAction("nothing", 0, 0, merc_direction=merc_dir, provoke_demons=provoke)
            else:
                build_options = []
                for pos in spaces:
                    t = best_build(pos, towers, enemies, bx, by, turn, money, house_count, team, state, self.memory, self)
                    val = self.score_build_with_map(t, pos[0], pos[1], money, towers, enemies, bx, by, turn, house_count, self.memory)
                    build_options.append((val, pos, t))
                build_options.sort(reverse=True)
                if build_options:
                    best_val, pos, t = build_options[0]
                    self.tower_count[t] += 1
                    self.memory["build_history"].append(pos)
                    self.save_memory()
                    return AIAction("build", pos[0], pos[1], t, provoke_demons=provoke)

        if spaces and money >= 10:
            build_options = []
            for pos in spaces:
                t = best_build(pos, towers, enemies, bx, by, turn, money, house_count, team, state, self.memory, self)
                val = self.score_build_with_map(t, pos[0], pos[1], money, towers, enemies, bx, by, turn, house_count, self.memory)
                build_options.append((val, pos, t))
            build_options.sort(reverse=True)
            best_val, pos, t = build_options[0]
            if t == "house" and (turn < 10 or house_count < 8):
                self.tower_count["house"] += 1
                self.memory["build_history"].append(pos)
                self.save_memory()
                return AIAction("build", pos[0], pos[1], "house", provoke_demons=provoke)
            elif t != "house" and money >= 8:
                self.tower_count[t] += 1
                self.memory["build_history"].append(pos)
                self.save_memory()
                return AIAction("build", pos[0], pos[1], t, provoke_demons=provoke)

        if dirs:
            best_dir = choose_best_merc_direction(enemies, bx, by, dirs, self.memory)
            if best_dir:
                self.memory["merc_directions"].append(best_dir)
                self.save_memory()
                return AIAction("nothing", 0, 0, merc_direction=best_dir, provoke_demons=provoke)

        if not spaces and len(towers) > 12:
            old_tower = random.choice(towers)
            self.save_memory()
            return AIAction("destroy", old_tower["x"], old_tower["y"], provoke_demons=provoke)

        self.save_memory()
        return AIAction("nothing", 0, 0, provoke_demons=provoke)


if __name__ == '__main__':
    team_color = 'r' if input() == "--YOU ARE RED--" else 'b'
    input_buffer = [input()]
    while input_buffer[-1] != "--END INITIAL GAME STATE--":
        input_buffer.append(input())
    game_state_init = json.loads(''.join(input_buffer[:-1]))

    agent = StrategicAI_MergedWithMemory()
    print(agent.initialize_and_set_name(game_state_init, team_color))
    print(agent.do_turn(game_state_init).to_json())

    while True:
        input_buffer = [input()]
        while input_buffer[-1] != "--END OF TURN--":
            input_buffer.append(input())
        game_state_turn = json.loads(''.join(input_buffer[:-1]))
        print(agent.do_turn(game_state_turn).to_json())
