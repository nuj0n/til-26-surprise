"""
Advanced Heuristic Agent for TIL-26 Surprise.
Features: Persistent map memory, damage pooling (no overkill), cutoff phase-shifting,
terrain-aware pathfinding, and strategic expansion.
"""

from __future__ import annotations

import random
from agent_base import PlayerAgent
from engine.actions import (
    ActionPayload,
    AttackAction,
    ConstructBuildingAction,
    HoldAction,
    MoveAction,
    ProduceUnitAction,
    ProposeTreatyAction,
)
from engine.constants import BUILDING_STATS, UNIT_STATS
from engine.hex_grid import HexCoord, HexGrid

_PRODUCTION_BUILDINGS = ("Barracks", "Factory", "Airbase")

class AlgoAgent(PlayerAgent):
    def __init__(self):
        # State memory to persist through stateless observations
        self.known_enemy_bases = {}  # type: dict[tuple[int, int], str]
        self.known_rich_resources = set() # type: set[tuple[int, int]]

    async def decide(self, observation: dict) -> ActionPayload:
        pid = observation["player_id"]
        turn = observation.get("turn_number", 0)
        max_turns = observation.get("max_turns", 300)
        gold = observation.get("resources", {}).get("gold", 0)
        grid = HexGrid(
            observation.get("map_width", 35), observation.get("map_height", 30)
        )
        
        actions = []
        own_units, own_buildings, enemies, occupied, visible_empty, difficult_terrain = self._parse_and_update_state(observation, pid)
        own_entity_coords = {(u["q"], u["r"]) for u in own_units} | {(b["q"], b["r"]) for b in own_buildings}

        # ── Phase 1: Diplomacy & Phase Shifting ────────────────────────────────
        # Force open war prep 10 turns before the Turn 200 cutoff
        is_endgame = turn >= 190
        
        if not is_endgame:
            known_players = observation.get("known_players", [])
            active_treaties = [t["partner_id"] for t in observation.get("treaties", []) if t["treaty_type"] == "peace"]
            for player in known_players:
                if player not in active_treaties:
                    actions.append(ProposeTreatyAction(target_player_id=player, treaty_type="peace"))

        # ── Phase 2: Economy & Production ──────────────────────────────────────
        complete_buildings = [b for b in own_buildings if b.get("is_complete", True)]
        prod_buildings = [b for b in complete_buildings if b["type"] in _PRODUCTION_BUILDINGS]
        
        bases = sum(1 for b in own_buildings if b["type"] == "Base")
        mines = sum(1 for b in own_buildings if b["type"] == "Mine")
        
        # In the endgame, dump all gold into units. Otherwise, build economy.
        if is_endgame and prod_buildings and gold >= UNIT_STATS["Artillery"].gold_cost:
            b = prod_buildings[-1] # Produce from the safest (oldest) building
            spot = self._free_neighbour(grid, (b["q"], b["r"]), occupied)
            if spot:
                actions.append(ProduceUnitAction(building_id=b["id"], unit_type="Artillery", target=HexCoord(*spot)))
                occupied.add(spot)
                gold -= UNIT_STATS["Artillery"].gold_cost
        elif not is_endgame:
            # Expand bases if we have fewer than 3
            if bases < 3 and gold >= BUILDING_STATS["Base"].gold_cost:
                base_spot = None
                # Prefer rich resources
                for r_spot in self.known_rich_resources:
                    if r_spot in visible_empty:
                        base_spot = r_spot
                        break
                if not base_spot and visible_empty:
                    # Pick a random empty tile
                    base_spot = random.choice(list(visible_empty))
                
                if base_spot:
                    actions.append(ConstructBuildingAction(building_type="Base", coord=HexCoord(*base_spot)))
                    occupied.add(base_spot)
                    visible_empty.discard(base_spot)
                    gold -= BUILDING_STATS["Base"].gold_cost
            
            # Build Mines first, Barracks second
            want = "Mine" if mines < 4 else "Barracks"
            cost = BUILDING_STATS[want].gold_cost
            
            if gold >= cost and complete_buildings:
                spot = None
                for b in complete_buildings:
                    # Check if we can build next to rich resource first
                    for n in grid.neighbors(HexCoord(b["q"], b["r"])):
                        n_tuple = (n.q, n.r)
                        if n_tuple in visible_empty:
                            if want == "Mine" and n_tuple in self.known_rich_resources:
                                spot = n_tuple
                                break
                    if spot: break
                    
                    spot = self._free_neighbour(grid, (b["q"], b["r"]), occupied)
                    if spot and spot in visible_empty:
                        break
                        
                if spot:
                    actions.append(ConstructBuildingAction(building_type=want, coord=HexCoord(*spot)))
                    occupied.add(spot)
                    visible_empty.discard(spot)
                    gold -= cost

            # Produce Scouts if we have few, otherwise Infantry
            scouts = sum(1 for u in own_units if u["type"] == "Scout")
            unit_to_build = "Scout" if scouts < 2 else "Infantry"
            if prod_buildings and gold >= UNIT_STATS[unit_to_build].gold_cost:
                b = prod_buildings[0]
                spot = self._free_neighbour(grid, (b["q"], b["r"]), occupied)
                if spot:
                    actions.append(ProduceUnitAction(building_id=b["id"], unit_type=unit_to_build, target=HexCoord(*spot)))
                    occupied.add(spot)
                    gold -= UNIT_STATS[unit_to_build].gold_cost

        # ── Phase 3: Combat (Damage Pooling) ───────────────────────────────────
        # Track pending damage on enemies to prevent overkill
        pending_damage = {e["id"]: 0 for e in enemies}
        
        for u in own_units:
            ar = u.get("attack_range", 0)
            mr = u.get("movement_range", 0)
            atk_power = u.get("attack_power", 0)
            unit_type = u.get("type", "")
            here = HexCoord(u["q"], u["r"])
            
            # Find a target that isn't already dead from our other units
            target = None
            best_dist = 10**9
            
            for e in enemies:
                if pending_damage.get(e["id"], 0) >= e.get("hp", 100):
                    continue # Target is already mathematically dead
                
                ec = HexCoord(e["q"], e["r"])
                d = grid.distance(here, ec)
                
                # If artillery, avoid friendly fire splash
                if unit_type == "Artillery":
                    splash_hits_friendly = False
                    for n in grid.neighbors(ec):
                        if (n.q, n.r) in own_entity_coords or (n.q, n.r) == (here.q, here.r):
                            splash_hits_friendly = True
                            break
                    if splash_hits_friendly:
                        continue
                
                if d < best_dist:
                    target, best_dist = e, d
            
            # If we remember an enemy base but can't see it, move towards it
            if target is None and self.known_enemy_bases:
                base_coords = list(self.known_enemy_bases.keys())[0]
                target_coord = HexCoord(*base_coords)
                step = self._step_toward(grid, here, target_coord, occupied, difficult_terrain, mr)
                if step is not None and mr >= 1:
                    actions.append(MoveAction(unit_id=u["id"], path=[here, step]))
                    occupied.add((step.q, step.r))
                    continue

            if target is None:
                continue

            tc = HexCoord(target["q"], target["r"])
            dist = grid.distance(here, tc)
            
            # Attack if in range, otherwise push forward
            if ar >= 1 and 0 < dist <= ar:
                actions.append(AttackAction(unit_id=u["id"], target=tc))
                # Accumulate integer damage (ignoring elevation multipliers for safe lower-bound)
                pending_damage[target["id"]] += atk_power 
            elif mr >= 1:
                step = self._step_toward(grid, here, tc, occupied, difficult_terrain, mr)
                if step is not None:
                    actions.append(MoveAction(unit_id=u["id"], path=[here, step]))
                    # Update occupied tracker to prevent movement collisions
                    occupied.discard((here.q, here.r))
                    occupied.add((step.q, step.r))
                else:
                    actions.append(HoldAction(unit_id=u["id"]))

        return ActionPayload(player_id=pid, turn_number=turn, actions=actions)

    # ── Internal State & Math Helpers ─────────────────────────────────────────
    def _parse_and_update_state(self, observation: dict, pid: str):
        own_units, own_buildings, enemies = [], [], []
        occupied: set[tuple[int, int]] = set()
        visible_empty: set[tuple[int, int]] = set()
        difficult_terrain: set[tuple[int, int]] = set()
        
        for tile in observation.get("visible_tiles", []):
            q, r = tile["q"], tile["r"]
            if tile.get("terrain") == "rich_resource":
                self.known_rich_resources.add((q, r))
            elif tile.get("terrain") == "difficult":
                difficult_terrain.add((q, r))
                
            entities = tile.get("entities", [])
            if not entities:
                visible_empty.add((q, r))
                
            for e in entities:
                occupied.add((q, r))
                if e.get("owner_id") == pid:
                    if e.get("type") in BUILDING_STATS:
                        own_buildings.append(e)
                    else:
                        own_units.append(e)
                else:
                    enemies.append(e)
                    # Cache enemy base locations permanently
                    if e.get("type") == "Base":
                        self.known_enemy_bases[(q, r)] = e["owner_id"]
                        
            # Clean up destroyed bases from memory if we have vision of the tile and it's empty
            if (q, r) in self.known_enemy_bases:
                if not any(e.get("type") == "Base" for e in entities):
                    del self.known_enemy_bases[(q, r)]
                    
        return own_units, own_buildings, enemies, occupied, visible_empty, difficult_terrain

    @staticmethod
    def _free_neighbour(grid, coord, occupied):
        for n in grid.neighbors(HexCoord(*coord)):
            if (n.q, n.r) not in occupied:
                return (n.q, n.r)
        return None

    @staticmethod
    def _step_toward(grid, here, target, occupied, difficult_terrain, move_range):
        best, best_d = None, grid.distance(here, target)
        for n in grid.neighbors(here):
            nt = (n.q, n.r)
            if nt in occupied:
                continue
            if nt in difficult_terrain and move_range < 2:
                continue
            d = grid.distance(n, target)
            if d < best_d:
                best, best_d = n, d
        return best