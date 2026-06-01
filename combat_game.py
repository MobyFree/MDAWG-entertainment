import sys, random, os, traceback
from transformers import pipeline, logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

# --- CONFIGURATION ---
os.environ["HF_HUB_DISABLE_SYSLOG_WARNINGS"] = "1"
logging.set_verbosity_error()

# Color codes
CLR_RESET = "\033[0m"
CLR_RED = "\033[31m"
CLR_CYAN = "\033[36m"
CLR_YELLOW = "\033[33m"
CLR_GREEN = "\033[32m"
CLR_MAGENTA = "\033[35m"
CLR_BLUE = "\033[34m"
CLR_WHITE = "\033[37m"

class ItemRarity(Enum):
    COMMON = 1
    UNCOMMON = 2
    RARE = 3
    LEGENDARY = 4

@dataclass
class Item:
    name: str
    item_type: str
    damage_bonus: int = 0
    armor_bonus: int = 0
    to_hit_bonus: int = 0
    description: str = ""
    rarity: ItemRarity = ItemRarity.COMMON

@dataclass
class CombatCharacter:
    name: str
    hp: int
    max_hp: int
    atk: int
    defense: int
    weapon: str
    weapon_bonus: int
    location: str = "Wasteland"
    initiative_roll: int = 0
    is_player: bool = False
    inventory: List[Item] = field(default_factory=list)
    active_armor: Optional[Item] = None
    active_weapon: Optional[Item] = None

    def get_hp_bar(self) -> str:
        """Generate visual HP bar with color"""
        filled = max(0, self.hp * 10 // self.max_hp)
        bar = "■" * filled + "□" * (10 - filled)
        
        if self.hp > self.max_hp * 0.5:
            color = CLR_GREEN
        elif self.hp > self.max_hp * 0.25:
            color = CLR_YELLOW
        else:
            color = CLR_RED
        
        return f"{color}{bar}{CLR_RESET}"

    def display_status(self) -> str:
        """Display character status with color"""
        hp_bar = self.get_hp_bar()
        name_color = CLR_GREEN if self.is_player else CLR_RED
        
        hp_color = CLR_GREEN if self.hp > self.max_hp * 0.5 else (CLR_YELLOW if self.hp > self.max_hp * 0.25 else CLR_RED)
        
        weapon_info = f"{self.weapon}"
        if self.active_weapon:
            weapon_info += f" (+{self.active_weapon.to_hit_bonus} to-hit, +{self.active_weapon.damage_bonus} dmg)"
        
        armor_info = ""
        if self.active_armor:
            armor_info = f" | {CLR_CYAN}Armor: {self.active_armor.name} (+{self.active_armor.armor_bonus}){CLR_RESET}"
        
        return f"{name_color}{self.name:20}{CLR_RESET} [{hp_bar}] {hp_color}{self.hp:3}/{self.max_hp:3}{CLR_RESET} HP | {weapon_info}{armor_info}"

    def display_inventory(self) -> str:
        """Display inventory items"""
        if not self.inventory:
            return f"{CLR_YELLOW}Inventory empty{CLR_RESET}"
        
        items_str = f"{CLR_CYAN}Inventory ({len(self.inventory)} items):{CLR_RESET}\n"
        for i, item in enumerate(self.inventory, 1):
            rarity_color = {
                ItemRarity.COMMON: CLR_WHITE,
                ItemRarity.UNCOMMON: CLR_GREEN,
                ItemRarity.RARE: CLR_BLUE,
                ItemRarity.LEGENDARY: CLR_MAGENTA
            }.get(item.rarity, CLR_WHITE)
            
            bonus_info = ""
            if item.to_hit_bonus > 0:
                bonus_info += f" +{item.to_hit_bonus}to-hit"
            if item.damage_bonus > 0:
                bonus_info += f" +{item.damage_bonus}dmg"
            if item.armor_bonus > 0:
                bonus_info += f" +{item.armor_bonus}armor"
            
            items_str += f"  {i}. {rarity_color}{item.name}{CLR_RESET} ({item.item_type}){bonus_info} - {item.description}\n"
        return items_str

    def get_full_status(self) -> str:
        """Get full character status with all details"""
        status = f"\n{CLR_CYAN}{'='*90}\n"
        status += f"CHARACTER STATUS\n"
        status += f"{'='*90}{CLR_RESET}\n"
        status += f"{CLR_GREEN}Name:{CLR_RESET} {self.name}\n"
        status += f"{CLR_GREEN}Location:{CLR_RESET} {self.location}\n"
        status += f"{CLR_RED}Health:{CLR_RESET} {self.hp}/{self.max_hp} {self.get_hp_bar()}\n"
        status += f"{CLR_YELLOW}Attack:{CLR_RESET} {self.atk}\n"
        status += f"{CLR_BLUE}Defense:{CLR_RESET} {self.defense}\n"
        status += f"{CLR_MAGENTA}Weapon:{CLR_RESET} {self.weapon} (bonus: +{self.weapon_bonus})\n"
        if self.active_armor:
            status += f"{CLR_CYAN}Armor:{CLR_RESET} {self.active_armor.name} (+{self.active_armor.armor_bonus})\n"
        return status

# --- PART 1: Advanced DM AI with Story Narration ---
class GameAI:
    def __init__(self):
        print(f"{CLR_YELLOW}[INIT] Loading AI DM model...{CLR_RESET}", flush=True)
        sys.stdout.flush()
        self.pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
        print(f"{CLR_GREEN}[INIT] AI DM loaded and ready!{CLR_RESET}\n", flush=True)

    def narrate_story(self, player_location: str, last_action: str = "", turn_num: int = 1) -> str:
        """Generate story narration for exploration/downtime"""
        prompt = (
            f"You are a gritty post-apocalyptic DM narrating a wasteland adventure. "
            f"The player is at: {player_location}. Turn {turn_num}/100. "
            f"Write 3-4 vivid sentences describing the wasteland environment, atmosphere, or NPCs. "
            f"Be descriptive but leave room for player interaction. What do they see, hear, smell?"
        )
        
        try:
            output = self.pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.85, top_p=0.9)
            response = output[0]['generated_text']
            # Extract only generated narrative
            if prompt in response:
                narrative = response.split(prompt)[-1].strip()
            else:
                narrative = response
            # Clean up and ensure complete sentences
            sentences = narrative.split('.')
            clean_sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 10]
            return ' '.join(clean_sentences[:4])
        except:
            locations = {
                "Wasteland Settlement": "The settlement around you is a collection of ramshackle buildings and makeshift shelters. The wind carries the scent of dust and decay. You can see merchants haggling in the distance.",
                "Desert Ruins": "Crumbling ruins of the old world stretch across the horizon. Metal frameworks jut from the sand like skeletal remains. The sun beats down mercilessly on this desolate place.",
                "Underground Bunker": "The concrete walls of the bunker are cold and damp. Emergency lighting flickers overhead, casting eerie shadows. The air is stale but breathable.",
                "Trading Post": "A bustling hub of activity where survivors gather to trade. Guards watch from elevated platforms. The smell of food and fuel fills the air.",
                "Abandoned Lab": "High-tech equipment lies dormant in the darkness, half-buried by sand and time. Computer terminals hang silent on the walls. Something feels wrong here."
            }
            return locations.get(player_location, "The wasteland stretches endlessly before you, empty and unforgiving. The wind whispers of ancient secrets lost to time.")

    def narrate_encounter_start(self, enemy_names: str) -> str:
        """Narrate the start of an encounter"""
        return (f"Your journey is interrupted! {enemy_names} emerge from the shadows, "
                f"blocking your path forward. Combat is unavoidable!")

    def narrate_round_start(self, round_num: int, active_combatants: List[str]) -> str:
        """Generate round narration"""
        combatants_str = ", ".join(active_combatants)
        return (f"Round {round_num}: The battle intensifies! {combatants_str} clash once more. "
                f"Weapons clash, dust rises, and survival hangs by a thread.")

    def narrate_attack(self, attacker: str, defender: str, 
                      attack_roll: int, hit: bool, damage: int = 0, 
                      critical: Optional[str] = None) -> str:
        """Generate attack narration"""
        
        if critical == "failure":
            return (f"{attacker} swings wildly, completely missing {defender}. "
                   f"The weapon whistles through empty air. A critical failure!")
        elif critical == "success":
            return (f"{attacker} unleashes a DEVASTATING strike! "
                   f"The blow connects with a sickening crunch, dealing {damage} damage! "
                   f"Blood sprays as {defender} reels from the catastrophic impact!")
        elif hit:
            return (f"{attacker} strikes {defender}, dealing {damage} damage! "
                   f"The weapon connects with brutal force. "
                   f"{defender} staggers backward from the impact.")
        else:
            return (f"{attacker} swings at {defender} but the attack misses! "
                   f"The weapon passes just inches away. "
                   f"{defender} sidesteps with practiced ease.")

    def narrate_death(self, defeated: str, killer: str) -> str:
        """Generate death narration"""
        return (f"{defeated} collapses to the ground, their life extinguished. "
               f"Another body falls in the wasteland. "
               f"The desert claims yet another soul.")

    def narrate_defend(self, character: str) -> str:
        """Generate defend narration"""
        return (f"{character} takes a defensive stance, bracing for incoming attacks. "
               f"Every muscle tenses in preparation. "
               f"They're ready to weather whatever comes next.")

    def narrate_item_use(self, character: str, item: str) -> str:
        """Generate item usage narration"""
        if "health" in item.lower():
            return (f"{character} quickly deploys a {item}! "
                   f"Relief floods through their body as wounds begin to close.")
        else:
            return f"{character} uses {item}!"

    def narrate_victory(self, player: str) -> str:
        """Generate victory narration"""
        return (f"{player} stands victorious over the fallen enemies! "
               f"Blood stains the wasteland, but {player} survives. "
               f"Another day won in this harsh world.")

    def narrate_defeat(self, player: str) -> str:
        """Generate defeat narration"""
        return (f"Darkness closes in as {player}'s vision fades. "
               f"The wasteland claims {player} as yet another victim. "
               f"Another story ends in the sand.")

    def narrate_exploration_action(self, action: str, location: str) -> str:
        """Narrate exploration actions"""
        actions = {
            "search": f"You search the area around {location} carefully. You find remnants of the old world scattered about.",
            "rest": f"You find shelter and rest at {location}. The wasteland is quiet for once.",
            "investigate": f"You investigate the surroundings at {location}. There are signs of past habitation.",
            "listen": f"You listen carefully at {location}. Strange sounds echo in the distance.",
            "gather": f"You gather resources from {location}. A few useful items are collected."
        }
        return actions.get(action.lower(), f"You perform an action at {location}.")

# --- PART 2: Persistent World Combat Game ---
class WastelandAdventure:
    def __init__(self):
        self.ai = GameAI()
        
        # Player character
        self.player = CombatCharacter(
            name="You",
            hp=100,
            max_hp=100,
            atk=8,
            defense=2,
            weapon="Plasma Rifle",
            weapon_bonus=3,
            location="Wasteland Settlement",
            is_player=True,
            inventory=[
                Item("Plasma Rifle", "weapon", damage_bonus=3, to_hit_bonus=0, 
                     description="High-tech energy weapon", rarity=ItemRarity.RARE),
                Item("Laser Rifle", "weapon", damage_bonus=4, to_hit_bonus=1,
                     description="Precise energy weapon", rarity=ItemRarity.RARE),
                Item("Combat Armor", "armor", armor_bonus=2, 
                     description="Kevlar reinforced", rarity=ItemRarity.UNCOMMON),
                Item("Health Pack", "consumable", damage_bonus=25,
                     description="Restores 25 HP", rarity=ItemRarity.COMMON),
                Item("Health Pack", "consumable", damage_bonus=25,
                     description="Restores 25 HP", rarity=ItemRarity.COMMON),
                Item("Energy Cell", "consumable",
                     description="Refills energy weapon", rarity=ItemRarity.COMMON)
            ]
        )
        self.player.active_weapon = self.player.inventory[0]
        self.player.active_armor = self.player.inventory[2]
        
        # Game state
        self.turn_count = 0
        self.max_turns = 100
        self.in_combat = False
        self.current_enemies: List[CombatCharacter] = []
        self.round_count = 0
        self.game_over = False
        
        # Available locations
        self.locations = [
            "Wasteland Settlement",
            "Desert Ruins",
            "Underground Bunker",
            "Trading Post",
            "Abandoned Lab"
        ]

    def roll_initiative(self):
        """Roll initiative for all combatants"""
        print(f"\n{CLR_YELLOW}{'='*90}")
        print(f"INITIATIVE PHASE")
        print(f"{'='*90}{CLR_RESET}\n")
        
        all_combatants = [self.player] + self.current_enemies
        for combatant in all_combatants:
            d20_roll = random.randint(1, 20)
            modifier = combatant.atk // 2
            combatant.initiative_roll = d20_roll + modifier
            
            role = f"{CLR_GREEN}[PLAYER]{CLR_RESET}" if combatant.is_player else f"{CLR_RED}[ENEMY]{CLR_RESET}"
            print(f"{role} {combatant.name:20} d20: {d20_roll:2} + {modifier} = {combatant.initiative_roll:3}")
        
        self.initiative_order = sorted(all_combatants, key=lambda x: x.initiative_roll, reverse=True)

    def display_combat_status(self):
        """Display battle status"""
        print(f"\n{CLR_BLUE}{'='*90}")
        print(f"ROUND {self.round_count} - TURN {self.turn_count}/{self.max_turns}")
        print(f"{'='*90}{CLR_RESET}\n")
        
        print(f"{CLR_GREEN}[PLAYER]{CLR_RESET}")
        print(self.player.display_status())
        print(f"  Location: {self.player.location}\n")
        
        print(f"{CLR_RED}[ENEMIES]{CLR_RESET}")
        for enemy in self.current_enemies:
            if enemy.hp > 0:
                print(enemy.display_status())
                print(f"  Location: {enemy.location}")

    def parse_action(self, command: str):
        """Parse player action"""
        command = command.lower().strip()
        parts = command.split()
        
        if not parts:
            return None, None
        
        action = parts[0]
        target = " ".join(parts[1:]) if len(parts) > 1 else None
        
        return action, target

    def find_enemy(self, name: str) -> Optional[CombatCharacter]:
        """Find enemy by partial name"""
        for enemy in self.current_enemies:
            if enemy.hp > 0 and (name.lower() in enemy.name.lower() or enemy.name.lower() in name.lower()):
                return enemy
        return None

    def find_item(self, character: CombatCharacter, name: str) -> Optional[Item]:
        """Find item in inventory"""
        for item in character.inventory:
            if name.lower() in item.name.lower():
                return item
        return None

    def roll_attack(self, attacker: CombatCharacter, defender: CombatCharacter) -> Dict:
        """Execute attack roll"""
        d20_roll = random.randint(1, 20)
        
        to_hit_bonus = attacker.active_weapon.to_hit_bonus if attacker.active_weapon else 0
        total_roll = d20_roll + attacker.atk + to_hit_bonus
        
        defense_ac = 12 + (defender.defense if defender.active_armor else 0)
        
        if d20_roll == 1:
            return {"hit": False, "d20": d20_roll, "total": total_roll, "damage": 0, "critical": "failure"}
        elif d20_roll == 20:
            base_damage = random.randint(attacker.atk + 5, attacker.atk + 10)
            dmg_bonus = attacker.active_weapon.damage_bonus if attacker.active_weapon else 0
            damage = (base_damage + attacker.weapon_bonus + dmg_bonus) * 2
            return {"hit": True, "d20": d20_roll, "total": total_roll, "damage": damage, "critical": "success"}
        elif total_roll >= defense_ac:
            base_damage = random.randint(attacker.atk, attacker.atk + 6)
            dmg_bonus = attacker.active_weapon.damage_bonus if attacker.active_weapon else 0
            damage = base_damage + attacker.weapon_bonus + dmg_bonus
            return {"hit": True, "d20": d20_roll, "total": total_roll, "damage": damage, "critical": None}
        else:
            return {"hit": False, "d20": d20_roll, "total": total_roll, "damage": 0, "critical": None}

    def execute_player_turn(self):
        """Execute player's combat turn"""
        self.display_combat_status()
        
        while True:
            print(f"\n{CLR_GREEN}[YOUR TURN]{CLR_RESET}")
            print(f"{CLR_YELLOW}Combat Commands: attack <enemy>, defend, use <item>, equip <weapon>, inventory, help{CLR_RESET}")
            command = input(f"{CLR_GREEN}> {CLR_RESET}").strip()
            
            action, target = self.parse_action(command)
            
            if action == "help":
                print(f"{CLR_CYAN}[COMMANDS]\n  attack <enemy> - Attack an enemy\n  defend - Raise defenses\n  use <item> - Use consumable\n  equip <weapon> - Switch weapons\n  inventory - View items\n  help - This message{CLR_RESET}\n")
                continue
            
            if action == "inventory":
                print(f"\n{self.player.display_inventory()}\n")
                continue
            
            if action is None:
                print(f"{CLR_RED}Invalid command.{CLR_RESET}\n")
                continue
            
            if action == "equip":
                if not target:
                    print(f"{CLR_RED}Equip what?{CLR_RESET}\n")
                    continue
                item = self.find_item(self.player, target)
                if not item or item.item_type != "weapon":
                    print(f"{CLR_RED}Weapon not found.{CLR_RESET}\n")
                    continue
                self.player.active_weapon = item
                self.player.weapon = item.name
                print(f"{CLR_GREEN}Equipped {item.name}! (+{item.to_hit_bonus} to-hit, +{item.damage_bonus} damage){CLR_RESET}\n")
                return "equip"
            
            if action == "attack":
                if not target:
                    print(f"{CLR_RED}Attack whom?{CLR_RESET}\n")
                    continue
                enemy = self.find_enemy(target)
                if not enemy:
                    print(f"{CLR_RED}No such enemy.{CLR_RESET}\n")
                    continue
                
                result = self.roll_attack(self.player, enemy)
                print(f"\n{CLR_BLUE}[ATTACK]{CLR_RESET}")
                print(f"{CLR_CYAN}d20: {result['d20']} + {self.player.atk} + {self.player.active_weapon.to_hit_bonus if self.player.active_weapon else 0} = {result['total']}{CLR_RESET}")
                
                if result["hit"]:
                    enemy.hp = max(enemy.hp - result["damage"], 0)
                    status = f"{CLR_GREEN}CRITICAL HIT!{CLR_RESET}" if result["critical"] == "success" else f"{CLR_GREEN}HIT!{CLR_RESET}"
                    print(f"{status} {result['damage']} damage!\n")
                else:
                    status = f"{CLR_RED}CRITICAL FAILURE!{CLR_RESET}" if result["critical"] == "failure" else f"{CLR_RED}MISS!{CLR_RESET}"
                    print(f"{status}\n")
                
                narrative = self.ai.narrate_attack(self.player.name, enemy.name, result['d20'], 
                                                  result["hit"], result["damage"], result["critical"])
                print(f"{CLR_MAGENTA}{narrative}{CLR_RESET}\n")
                
                if enemy.hp <= 0:
                    death_text = self.ai.narrate_death(enemy.name, self.player.name)
                    print(f"{CLR_RED}[DEATH]{CLR_RESET} {death_text}\n")
                    self.current_enemies.remove(enemy)
                    if enemy in self.initiative_order:
                        self.initiative_order.remove(enemy)
                
                return "attack"
            
            elif action == "defend":
                text = self.ai.narrate_defend(self.player.name)
                print(f"\n{CLR_MAGENTA}{text}{CLR_RESET}\n")
                return "defend"
            
            elif action == "use":
                if not target:
                    print(f"{CLR_RED}Use what?{CLR_RESET}\n")
                    continue
                item = self.find_item(self.player, target)
                if not item:
                    print(f"{CLR_RED}Item not found.{CLR_RESET}\n")
                    continue
                
                text = self.ai.narrate_item_use(self.player.name, item.name)
                print(f"\n{CLR_MAGENTA}{text}{CLR_RESET}\n")
                
                if item.item_type == "consumable" and "health" in item.name.lower():
                    heal = item.damage_bonus
                    self.player.hp = min(self.player.hp + heal, self.player.max_hp)
                    print(f"{CLR_GREEN}Restored {heal} HP! {self.player.hp}/{self.player.max_hp}{CLR_RESET}\n")
                    self.player.inventory.remove(item)
                
                return "use"

    def execute_enemy_turn(self, enemy: CombatCharacter):
        """Execute enemy turn"""
        if enemy.hp <= 0:
            return
        
        result = self.roll_attack(enemy, self.player)
        print(f"{CLR_RED}[{enemy.name.upper()}]{CLR_RESET}")
        print(f"{CLR_CYAN}d20: {result['d20']} + {enemy.atk} = {result['total']}{CLR_RESET}")
        
        if result["hit"]:
            self.player.hp = max(self.player.hp - result["damage"], 0)
            status = f"{CLR_RED}CRITICAL HIT!{CLR_RESET}" if result["critical"] == "success" else f"{CLR_RED}HIT!{CLR_RESET}"
            print(f"{status} You take {result['damage']} damage!\n")
        else:
            status = f"{CLR_GREEN}CRITICAL FAILURE!{CLR_RESET}" if result["critical"] == "failure" else f"{CLR_GREEN}MISS!{CLR_RESET}"
            print(f"{status}\n")
        
        text = self.ai.narrate_attack(enemy.name, self.player.name, result['d20'], result["hit"], 
                                     result["damage"], result["critical"])
        print(f"{CLR_MAGENTA}{text}{CLR_RESET}\n")
        
        if self.player.hp <= 0:
            death_text = self.ai.narrate_defeat(self.player.name)
            print(f"{CLR_RED}[DEATH] {death_text}{CLR_RESET}\n")
            self.game_over = True
            self.in_combat = False

    def start_combat_encounter(self, num_enemies: int = 2):
        """Start a random combat encounter"""
        self.in_combat = True
        self.round_count = 0
        self.current_enemies = []
        
        enemy_templates = [
            CombatCharacter(
                name="Super Mutant", hp=60, max_hp=60, atk=7, defense=1,
                weapon="Super Sledge", weapon_bonus=3, location="Combat Zone"
            ),
            CombatCharacter(
                name="Feral Ghoul", hp=30, max_hp=30, atk=5, defense=0,
                weapon="Claws", weapon_bonus=1, location="Combat Zone"
            ),
            CombatCharacter(
                name="Raider", hp=40, max_hp=40, atk=6, defense=1,
                weapon="Hunting Rifle", weapon_bonus=2, location="Combat Zone"
            )
        ]
        
        selected = random.sample(enemy_templates, min(num_enemies, len(enemy_templates)))
        self.current_enemies = selected
        
        enemy_names = ", ".join([e.name for e in self.current_enemies])
        encounter_text = self.ai.narrate_encounter_start(enemy_names)
        print(f"\n{CLR_RED}{encounter_text}{CLR_RESET}\n")
        
        self.roll_initiative()

    def run_combat_round(self):
        """Execute one complete combat round"""
        self.round_count += 1
        
        active = [c.name for c in self.initiative_order if c.hp > 0]
        round_text = self.ai.narrate_round_start(self.round_count, active)
        print(f"\n{CLR_YELLOW}[ROUND {self.round_count}]{CLR_RESET}")
        print(f"{CLR_MAGENTA}{round_text}{CLR_RESET}\n")
        
        for combatant in self.initiative_order:
            if self.player.hp <= 0 or len(self.current_enemies) == 0:
                break
            if combatant.hp <= 0:
                continue
            
            if combatant.is_player:
                self.execute_player_turn()
            else:
                self.execute_enemy_turn(combatant)
        
        # Check combat end
        if len(self.current_enemies) == 0 and self.in_combat:
            print(f"\n{CLR_GREEN}{'='*90}")
            print(f"VICTORY!")
            print(f"{'='*90}{CLR_RESET}\n")
            victory_text = self.ai.narrate_victory(self.player.name)
            print(f"{CLR_GREEN}{victory_text}{CLR_RESET}\n")
            self.in_combat = False

    def run_exploration_action(self):
        """Execute player exploration action"""
        self.turn_count += 1
        
        # Show world state
        print(f"\n{CLR_BLUE}{'='*90}")
        print(f"TURN {self.turn_count}/{self.max_turns}")
        print(f"{'='*90}{CLR_RESET}")
        print(f"\n{CLR_GREEN}[LOCATION]{CLR_RESET} {self.player.location}")
        print(f"{self.player.display_status()}\n")
        
        # Generate environment narration
        env_narration = self.ai.narrate_story(self.player.location, "", self.turn_count)
        print(f"{CLR_MAGENTA}{env_narration}{CLR_RESET}\n")
        
        # Get player action
        print(f"{CLR_YELLOW}Actions: move <location>, search, rest, examine, travel, inventory, status, help{CLR_RESET}")
        command = input(f"{CLR_GREEN}> {CLR_RESET}").strip()
        
        action, target = self.parse_action(command)
        
        if action == "help":
            print(f"\n{CLR_CYAN}[EXPLORATION COMMANDS]")
            print(f"  move <location>  - Travel to: {', '.join(self.locations)}")
            print(f"  search           - Search current location for items/info")
            print(f"  rest             - Rest and recover HP")
            print(f"  examine          - Examine surroundings")
            print(f"  inventory        - View items")
            print(f"  status           - Full character sheet")
            print(f"  travel           - Chance to encounter enemies{CLR_RESET}\n")
            return True
        
        if action == "inventory":
            print(f"\n{self.player.display_inventory()}\n")
            return True
        
        if action == "status":
            print(self.player.get_full_status())
            return True
        
        if action == "move":
            if not target or target not in self.locations:
                print(f"{CLR_RED}Move to where? {self.locations}{CLR_RESET}\n")
                return True
            self.player.location = target
            print(f"{CLR_GREEN}You travel to {target}.{CLR_RESET}\n")
            return True
        
        if action == "rest":
            heal = min(20, self.player.max_hp - self.player.hp)
            self.player.hp += heal
            rest_text = self.ai.narrate_story(self.player.location, "rest", self.turn_count)
            print(f"{CLR_MAGENTA}{rest_text}{CLR_RESET}\n")
            print(f"{CLR_GREEN}Restored {heal} HP! {self.player.hp}/{self.player.max_hp}{CLR_RESET}\n")
            return True
        
        if action == "search":
            search_text = self.ai.narrate_exploration_action("search", self.player.location)
            print(f"{CLR_MAGENTA}{search_text}{CLR_RESET}\n")
            return True
        
        if action == "examine":
            examine_text = self.ai.narrate_exploration_action("investigate", self.player.location)
            print(f"{CLR_MAGENTA}{examine_text}{CLR_RESET}\n")
            return True
        
        if action == "travel":
            print(f"\n{CLR_YELLOW}You venture deeper into the wasteland...{CLR_RESET}\n")
            encounter_roll = random.randint(1, 20)
            if encounter_roll > 12:
                num_enemies = 1 if encounter_roll > 16 else 2
                self.start_combat_encounter(num_enemies)
            else:
                print(f"{CLR_GREEN}You travel safely, encountering nothing.{CLR_RESET}\n")
            return True
        
        print(f"{CLR_RED}Unknown action.{CLR_RESET}\n")
        return True

    def run(self):
        """Main game loop"""
        print(f"\n{CLR_MAGENTA}{'='*90}")
        print(f"WASTELAND ADVENTURE - A PERSISTENT WORLD")
        print(f"{'='*90}{CLR_RESET}\n")
        
        opening = self.ai.narrate_story(self.player.location, "", 0)
        print(f"{CLR_MAGENTA}{opening}{CLR_RESET}\n")
        
        while self.turn_count < self.max_turns and not self.game_over:
            if self.in_combat:
                self.run_combat_round()
            else:
                if not self.run_exploration_action():
                    break
        
        if self.game_over:
            print(f"\n{CLR_RED}{'='*90}")
            print(f"ADVENTURE ENDED")
            print(f"{'='*90}{CLR_RESET}\n")
        elif self.turn_count >= self.max_turns:
            print(f"\n{CLR_YELLOW}{'='*90}")
            print(f"100 TURNS COMPLETED - YOUR ADVENTURE ENDS HERE")
            print(f"{'='*90}{CLR_RESET}\n")

def main():
    """Main program loop"""
    while True:
        game = WastelandAdventure()
        game.run()
        
        print(f"\n{CLR_CYAN}[GAME END]")
        print(f"  new   - Start new adventure")
        print(f"  exit  - Exit program\n")
        
        while True:
            command = input(f"{CLR_GREEN}> {CLR_RESET}").strip().lower()
            if command in ["new", "n"]:
                break
            elif command in ["exit", "e", "quit", "q"]:
                print(f"\n{CLR_YELLOW}Thanks for playing! Window will close in 10 seconds...{CLR_RESET}\n")
                import time
                time.sleep(10)
                return
            else:
                print(f"{CLR_RED}Invalid command.{CLR_RESET}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{CLR_YELLOW}Game interrupted.{CLR_RESET}")
        import time
        time.sleep(5)
    except Exception as e:
        print(f"{CLR_RED}Error: {e}{CLR_RESET}")
        traceback.print_exc()
        import time
        time.sleep(10)
