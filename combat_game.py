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
    item_type: str  # weapon, armor, consumable
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

# --- PART 1: Advanced DM AI with Proper Narration ---
class GameAI:
    def __init__(self):
        print(f"{CLR_YELLOW}[INIT] Loading AI DM model...{CLR_RESET}", flush=True)
        sys.stdout.flush()
        self.pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
        print(f"{CLR_GREEN}[INIT] AI DM loaded and ready!{CLR_RESET}\n", flush=True)

    def _extract_clean_narrative(self, text: str) -> str:
        """Extract only the AI's narration without prompt"""
        # Take only the generated part, not the prompt
        if "Write" in text or "Describe" in text or "post-apocalyptic" in text:
            # Find where actual narration likely starts
            lines = text.split('\n')
            narrative_lines = []
            started = False
            
            for line in lines:
                # Skip obvious prompt lines
                if any(x in line for x in ["Write", "Describe", "post-apocalyptic DM", "d20 roll"]):
                    started = False
                    continue
                if line.strip() and not line.startswith(("You", "Write", "DM")):
                    narrative_lines.append(line.strip())
                    started = True
            
            narrative = ' '.join(narrative_lines)
        else:
            narrative = text
        
        # Ensure complete sentences
        if narrative:
            sentences = []
            parts = narrative.split('.')
            for part in parts:
                part = part.strip()
                if len(part) > 10:
                    sentences.append(part + '.')
                if len(sentences) >= 3:
                    break
            
            if sentences:
                return ' '.join(sentences)
        
        return text.strip()[:200]

    def narrate_combat_start(self, player: 'CombatCharacter', enemies: List['CombatCharacter']) -> str:
        """Generate opening narration for combat"""
        enemy_names = ", ".join([e.name for e in enemies])
        return (f"The wasteland erupts into chaos as {player.name} faces off against {enemy_names}. "
                f"Sand swirls around your feet as your enemies close in, weapons ready. "
                f"Every heartbeat feels like thunder as you prepare for the fight of your life.")

    def narrate_round_start(self, round_num: int, active_combatants: List[str]) -> str:
        """Generate round narration"""
        combatants_str = ", ".join(active_combatants)
        return (f"Round {round_num}: The battle intensifies! {combatants_str} clash once more. "
                f"Weapons clash, dust rises, and survival hangs by a thread.")

    def narrate_attack(self, attacker: 'CombatCharacter', defender: 'CombatCharacter', 
                      attack_roll: int, hit: bool, damage: int = 0, 
                      critical: Optional[str] = None) -> str:
        """Generate attack narration"""
        
        if critical == "failure":
            return (f"{attacker.name} swings {attacker.weapon} wildly, completely missing {defender.name}. "
                   f"The weapon whistles through empty air as {attacker.name} stumbles off-balance. "
                   f"A critical failure that leaves {attacker.name} vulnerable!")
        
        elif critical == "success":
            return (f"{attacker.name} unleashes a DEVASTATING strike with {attacker.weapon}! "
                   f"The blow connects with a sickening crunch, slamming into {defender.name} for {damage} damage! "
                   f"Blood sprays across the wasteland as {defender.name} reels from the catastrophic impact!")
        
        elif hit:
            return (f"{attacker.name} strikes {defender.name} with {attacker.weapon}, dealing {damage} damage! "
                   f"The weapon connects with brutal force, drawing blood. "
                   f"{defender.name} staggers backward from the impact.")
        
        else:
            return (f"{attacker.name} attempts to hit {defender.name} but the attack misses! "
                   f"The weapon passes just inches away from its target. "
                   f"{defender.name} sidesteps the incoming attack with practiced ease.")

    def narrate_death(self, defeated: 'CombatCharacter', killer: 'CombatCharacter') -> str:
        """Generate death narration"""
        return (f"{defeated.name} collapses to the ground, their life extinguished by {killer.name}. "
               f"Another body falls in the wasteland. Another story ends. "
               f"The desert claims yet another soul.")

    def narrate_defend(self, character: 'CombatCharacter') -> str:
        """Generate defend narration"""
        return (f"{character.name} takes a defensive stance, bracing for incoming attacks. "
               f"Every muscle tenses in preparation. "
               f"They're ready to weather whatever comes next.")

    def narrate_item_use(self, character: 'CombatCharacter', item: Item) -> str:
        """Generate item usage narration"""
        if "health" in item.name.lower():
            return (f"{character.name} quickly deploys a {item.name}! "
                   f"Relief floods through their body as wounds begin to close. "
                   f"They're ready to continue fighting.")
        else:
            return (f"{character.name} uses {item.name}! {item.description}")

    def narrate_victory(self, player: 'CombatCharacter') -> str:
        """Generate victory narration"""
        return (f"{player.name} stands victorious over their fallen enemies! "
               f"Blood stains the wasteland, but {player.name} survives. "
               f"The desert has tested them, and they have emerged triumphant.")

    def narrate_defeat(self) -> str:
        """Generate defeat narration"""
        return (f"Darkness closes in as your vision fades. "
               f"The wasteland claims you as yet another victim. "
               f"Your story ends here, in the sand.")

# --- PART 2: Enhanced Combat Game ---
class AdvancedCombatGame:
    def __init__(self):
        self.ai = GameAI()
        
        # Initialize player with detailed equipment
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
                Item("Energy Cell", "consumable",
                     description="Refills energy weapon", rarity=ItemRarity.COMMON)
            ]
        )
        self.player.active_weapon = self.player.inventory[0]
        self.player.active_armor = self.player.inventory[2]
        
        # Initialize enemies
        self.enemies: List[CombatCharacter] = [
            CombatCharacter(
                name="Super Mutant",
                hp=60,
                max_hp=60,
                atk=7,
                defense=1,
                weapon="Super Sledge",
                weapon_bonus=3,
                location="Wasteland - Northeast",
                inventory=[Item("Scrap Metal", "junk", description="Worthless but heavy")]
            ),
            CombatCharacter(
                name="Feral Ghoul",
                hp=30,
                max_hp=30,
                atk=5,
                defense=0,
                weapon="Claws",
                weapon_bonus=1,
                location="Wasteland - Underground",
                inventory=[Item("Spoiled Meat", "junk", description="Smells of death")]
            )
        ]
        
        self.all_combatants: List[CombatCharacter] = [self.player] + self.enemies
        self.initiative_order: List[CombatCharacter] = []
        self.round_count = 0
        self.in_combat = True
        
    def roll_initiative(self):
        """Roll and display initiative"""
        print(f"\n{CLR_YELLOW}{'='*90}")
        print(f"INITIATIVE PHASE - ROLLING FOR COMBAT ORDER")
        print(f"{'='*90}{CLR_RESET}\n")
        
        for combatant in self.all_combatants:
            d20_roll = random.randint(1, 20)
            modifier = combatant.atk // 2
            combatant.initiative_roll = d20_roll + modifier
            
            role = f"{CLR_GREEN}[PLAYER]{CLR_RESET}" if combatant.is_player else f"{CLR_RED}[ENEMY]{CLR_RESET}"
            print(f"{role} {combatant.name:20} d20: {d20_roll:2} + {modifier} = {combatant.initiative_roll:3}")
        
        self.initiative_order = sorted(self.all_combatants, key=lambda x: x.initiative_roll, reverse=True)
        
        print(f"\n{CLR_CYAN}{'='*90}")
        print(f"COMBAT ORDER:")
        print(f"{'='*90}{CLR_RESET}")
        for i, c in enumerate(self.initiative_order, 1):
            role = "PLAYER" if c.is_player else "ENEMY"
            print(f"  {i}. {CLR_WHITE}{c.name:20}{CLR_RESET} (Init: {c.initiative_roll}) - {role}")
        print()

    def display_combat_status(self):
        """Display combat status"""
        print(f"\n{CLR_BLUE}{'='*90}")
        print(f"ROUND {self.round_count} - BATTLEFIELD STATUS")
        print(f"{'='*90}{CLR_RESET}\n")
        
        print(f"{CLR_GREEN}[PLAYER]{CLR_RESET}")
        print(self.player.display_status())
        print(f"  Location: {self.player.location}")
        
        print(f"\n{CLR_RED}[ENEMIES]{CLR_RESET}")
        for enemy in self.enemies:
            if enemy.hp > 0:
                print(enemy.display_status())
                print(f"  Location: {enemy.location}")
        print()

    def parse_player_action(self, command: str):
        """Parse player commands"""
        command = command.lower().strip()
        parts = command.split()
        
        if not parts:
            return None, None
        
        action = parts[0]
        
        if action in ["attack", "a"] and len(parts) > 1:
            return "attack", " ".join(parts[1:])
        elif action in ["defend", "d"]:
            return "defend", None
        elif action in ["use", "u"] and len(parts) > 1:
            return "use", " ".join(parts[1:])
        elif action in ["equip", "eq"] and len(parts) > 1:
            return "equip", " ".join(parts[1:])
        elif action in ["inventory", "inv", "i"]:
            return "inventory", None
        elif action in ["status", "s"]:
            return "status", None
        elif action in ["flee", "f"]:
            return "flee", None
        elif action in ["help", "h", "?"]:
            return "help", None
        
        return None, None

    def find_enemy(self, name: str) -> Optional[CombatCharacter]:
        """Find enemy by partial name"""
        for enemy in self.enemies:
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
        """Execute attack roll with bonuses"""
        d20_roll = random.randint(1, 20)
        
        # Apply to-hit bonuses
        to_hit_bonus = attacker.active_weapon.to_hit_bonus if attacker.active_weapon else 0
        total_roll = d20_roll + attacker.atk + to_hit_bonus
        
        # Defense AC
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
        """Execute player's turn"""
        self.display_combat_status()
        
        while True:
            print(f"{CLR_GREEN}[YOUR TURN]{CLR_RESET}")
            print(f"{CLR_YELLOW}Commands: attack <enemy>, defend, use <item>, equip <weapon>, inventory, status, flee, help{CLR_RESET}")
            command = input(f"{CLR_GREEN}> {CLR_RESET}").strip()
            
            action, target = self.parse_player_action(command)
            
            if action == "help":
                print(f"{CLR_CYAN}\n[COMMANDS]")
                print(f"  attack <enemy>    - Attack by name")
                print(f"  defend            - Raise defenses")
                print(f"  use <item>        - Use consumable")
                print(f"  equip <weapon>    - Equip different weapon")
                print(f"  inventory         - View items")
                print(f"  status            - View full status")
                print(f"  flee              - Escape (DC 12)")
                print(f"  help              - Show this{CLR_RESET}\n")
                continue
            
            if action == "inventory":
                print(f"\n{self.player.display_inventory()}\n")
                continue
            
            if action == "status":
                print(self.player.get_full_status())
                continue
            
            if action is None:
                print(f"{CLR_RED}Invalid command. Type 'help' for options.{CLR_RESET}\n")
                continue
            
            if action == "equip":
                if not target:
                    print(f"{CLR_RED}Equip what? Specify weapon name.{CLR_RESET}\n")
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
                    print(f"{CLR_RED}Attack whom? Use: attack <enemy name>{CLR_RESET}\n")
                    continue
                
                enemy = self.find_enemy(target)
                if not enemy:
                    print(f"{CLR_RED}No such enemy found.{CLR_RESET}\n")
                    continue
                
                result = self.roll_attack(self.player, enemy)
                
                print(f"\n{CLR_BLUE}[ATTACK ROLL]{CLR_RESET}")
                print(f"{CLR_CYAN}d20: {result['d20']} + {self.player.atk} (ATK) + {self.player.active_weapon.to_hit_bonus if self.player.active_weapon else 0} (weapon) = {result['total']}{CLR_RESET}")
                
                if result["hit"]:
                    enemy.hp = max(enemy.hp - result["damage"], 0)
                    status = f"{CLR_GREEN}CRITICAL HIT!{CLR_RESET}" if result["critical"] == "success" else f"{CLR_GREEN}HIT!{CLR_RESET}"
                    print(f"{status} {result['damage']} damage!\n")
                else:
                    status = f"{CLR_RED}CRITICAL FAILURE!{CLR_RESET}" if result["critical"] == "failure" else f"{CLR_RED}MISS!{CLR_RESET}"
                    print(f"{status}\n")
                
                narrative = self.ai.narrate_attack(self.player, enemy, result['d20'], 
                                                  result["hit"], result["damage"], result["critical"])
                print(f"{CLR_MAGENTA}{narrative}{CLR_RESET}\n")
                
                if enemy.hp <= 0:
                    death_narration = self.ai.narrate_death(enemy, self.player)
                    print(f"{CLR_RED}[DEATH]{CLR_RESET} {death_narration}\n")
                    self.enemies.remove(enemy)
                    if enemy in self.initiative_order:
                        self.initiative_order.remove(enemy)
                
                return "attack"
            
            elif action == "defend":
                narration = self.ai.narrate_defend(self.player)
                print(f"\n{CLR_MAGENTA}{narration}{CLR_RESET}\n")
                return "defend"
            
            elif action == "use":
                if not target:
                    print(f"{CLR_RED}Use what? Specify item name.{CLR_RESET}\n")
                    continue
                
                item = self.find_item(self.player, target)
                if not item:
                    print(f"{CLR_RED}Item not found.{CLR_RESET}\n")
                    continue
                
                narration = self.ai.narrate_item_use(self.player, item)
                print(f"\n{CLR_MAGENTA}{narration}{CLR_RESET}\n")
                
                if item.item_type == "consumable":
                    if "health" in item.name.lower():
                        heal_amount = item.damage_bonus
                        self.player.hp = min(self.player.hp + heal_amount, self.player.max_hp)
                        print(f"{CLR_GREEN}Restored {heal_amount} HP! Current: {self.player.hp}/{self.player.max_hp}{CLR_RESET}\n")
                    self.player.inventory.remove(item)
                
                return "use"
            
            elif action == "flee":
                flee_roll = random.randint(1, 20)
                print(f"{CLR_BLUE}[FLEE ATTEMPT] Flee Roll: {flee_roll}{CLR_RESET}\n")
                
                if flee_roll > 12:
                    print(f"{CLR_GREEN}You escape the combat!{CLR_RESET}\n")
                    self.in_combat = False
                    return "flee"
                else:
                    print(f"{CLR_RED}Failed! Enemies pursue!{CLR_RESET}\n")
                    continue

    def execute_enemy_turn(self, enemy: CombatCharacter):
        """Execute enemy's turn"""
        if enemy.hp <= 0 or not self.enemies:
            return
        
        target = self.player
        result = self.roll_attack(enemy, target)
        
        print(f"{CLR_RED}[{enemy.name.upper()} ATTACKS]{CLR_RESET}")
        print(f"{CLR_CYAN}d20: {result['d20']} + {enemy.atk} (ATK) = {result['total']}{CLR_RESET}")
        
        if result["hit"]:
            target.hp = max(target.hp - result["damage"], 0)
            status = f"{CLR_RED}CRITICAL HIT!{CLR_RESET}" if result["critical"] == "success" else f"{CLR_RED}HIT!{CLR_RESET}"
            print(f"{status} You take {result['damage']} damage!\n")
        else:
            status = f"{CLR_GREEN}CRITICAL FAILURE!{CLR_RESET}" if result["critical"] == "failure" else f"{CLR_GREEN}MISS!{CLR_RESET}"
            print(f"{status}\n")
        
        narrative = self.ai.narrate_attack(enemy, target, result['d20'], result["hit"], 
                                          result["damage"], result["critical"])
        print(f"{CLR_MAGENTA}{narrative}{CLR_RESET}\n")
        
        if target.hp <= 0:
            death_narration = self.ai.narrate_defeat()
            print(f"{CLR_RED}[YOUR DEATH] {death_narration}{CLR_RESET}\n")
            self.in_combat = False

    def show_end_menu(self):
        """Show menu after game ends"""
        while True:
            print(f"\n{CLR_YELLOW}{'='*90}")
            print(f"GAME OVER")
            print(f"{'='*90}{CLR_RESET}")
            print(f"\n{CLR_CYAN}Options:{CLR_RESET}")
            print(f"  new   - Start a new game")
            print(f"  exit  - Exit the program\n")
            
            command = input(f"{CLR_GREEN}> {CLR_RESET}").strip().lower()
            
            if command in ["new", "n"]:
                return True
            elif command in ["exit", "e", "quit", "q"]:
                return False
            else:
                print(f"{CLR_RED}Invalid command.{CLR_RESET}")

    def run(self):
        """Main game loop"""
        print(f"\n{CLR_MAGENTA}{'='*90}")
        print(f"WASTELAND COMBAT SYSTEM - INITIALIZED")
        print(f"{'='*90}{CLR_RESET}\n")
        
        opening = self.ai.narrate_combat_start(self.player, self.enemies)
        print(f"{CLR_MAGENTA}{opening}{CLR_RESET}\n")
        
        self.roll_initiative()
        
        while self.in_combat and self.player.hp > 0 and len(self.enemies) > 0:
            self.round_count += 1
            
            active = [c.name for c in self.initiative_order if c.hp > 0]
            round_narration = self.ai.narrate_round_start(self.round_count, active)
            print(f"\n{CLR_YELLOW}[ROUND {self.round_count}]{CLR_RESET}")
            print(f"{CLR_MAGENTA}{round_narration}{CLR_RESET}\n")
            
            for combatant in self.initiative_order:
                if not self.in_combat or self.player.hp <= 0 or len(self.enemies) == 0:
                    break
                if combatant.hp <= 0:
                    continue
                
                if combatant.is_player:
                    self.execute_player_turn()
                else:
                    self.execute_enemy_turn(combatant)
            
            if len(self.enemies) == 0:
                print(f"\n{CLR_GREEN}{'='*90}")
                print(f"VICTORY!")
                print(f"{'='*90}{CLR_RESET}\n")
                victory_text = self.ai.narrate_victory(self.player)
                print(f"{CLR_GREEN}{victory_text}{CLR_RESET}\n")
                self.in_combat = False
            elif self.player.hp <= 0:
                print(f"\n{CLR_RED}{'='*90}")
                print(f"DEFEAT!")
                print(f"{'='*90}{CLR_RESET}\n")
                self.in_combat = False

def main():
    """Main program loop"""
    while True:
        game = AdvancedCombatGame()
        game.run()
        
        if not game.show_end_menu():
            break
    
    print(f"\n{CLR_YELLOW}Thanks for playing! Exiting in 10 seconds...{CLR_RESET}\n")
    
    try:
        import time
        time.sleep(10)
    except:
        pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{CLR_YELLOW}Game interrupted.{CLR_RESET}")
        import time
        time.sleep(5)
    except Exception as e:
        print(f"{CLR_RED}Error occurred:{CLR_RESET}")
        traceback.print_exc()
        import time
        time.sleep(10)
