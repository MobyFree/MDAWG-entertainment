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
    initiative_roll: int = 0
    is_player: bool = False
    inventory: List[Item] = field(default_factory=list)
    active_armor: Optional[Item] = None

    def get_hp_bar(self) -> str:
        """Generate visual HP bar"""
        filled = max(0, self.hp * 10 // self.max_hp)
        bar = "■" * filled + "□" * (10 - filled)
        return bar

    def display_status(self) -> str:
        """Display character status with color"""
        hp_bar = self.get_hp_bar()
        color = CLR_GREEN if self.is_player else CLR_RED
        armor_str = f" | Armor: {self.active_armor.name}" if self.active_armor else ""
        return f"{color}{self.name:20} [{hp_bar}] {self.hp:3}/{self.max_hp:3} HP | {self.weapon}{armor_str}{CLR_RESET}"

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
            items_str += f"  {i}. {rarity_color}{item.name}{CLR_RESET} ({item.item_type}) - {item.description}\n"
        return items_str

# --- PART 1: Advanced DM AI with Combat Narration ---
class GameAI:
    def __init__(self):
        print(f"{CLR_YELLOW}[INIT] Loading AI DM model... (this may take a minute){CLR_RESET}", flush=True)
        sys.stdout.flush()
        self.pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
        print(f"{CLR_GREEN}[INIT] AI DM loaded and ready!{CLR_RESET}", flush=True)

    def narrate_combat_start(self, player: CombatCharacter, enemies: List[CombatCharacter]) -> str:
        """Generate opening narration for combat"""
        enemy_names = ", ".join([e.name for e in enemies])
        prompt = (
            f"You are a gritty post-apocalyptic DM. Write a 3-4 sentence vivid, intense opening to combat. "
            f"The player faces {enemy_names} in the wasteland. Be dramatic and set a dark mood."
        )
        
        try:
            output = self.pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.85)
            response = output[0]['generated_text'].strip()
            return response[:300] if len(response) > 300 else response
        except:
            return f"The wasteland trembles as {enemy_names} emerge from the shadows, ready for brutal combat."

    def narrate_initiative_roll(self, character: CombatCharacter, roll: int) -> str:
        """Generate narration for initiative roll"""
        prompt = (
            f"Write a 1-sentence dramatic description of {character.name} rolling for initiative "
            f"with result {roll}. Gritty, post-apocalyptic tone. Very brief."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=30, do_sample=True, temperature=0.7)
            response = output[0]['generated_text'].strip()
            return response if len(response) > 5 else f"{character.name} readies for action..."
        except:
            return f"{character.name} prepares to act..."

    def narrate_round_start(self, round_num: int) -> str:
        """Generate narration for round start"""
        prompt = (
            f"Write a 1-2 sentence dramatic description of combat round {round_num} beginning. "
            f"Post-apocalyptic setting, tense atmosphere. Be vivid and brief."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=40, do_sample=True, temperature=0.8)
            response = output[0]['generated_text'].strip()
            return response if len(response) > 5 else f"Round {round_num} erupts with violence..."
        except:
            return f"Round {round_num} rages on!"

    def narrate_attack(self, attacker: CombatCharacter, defender: CombatCharacter, 
                      attack_roll: int, hit: bool, damage: int = 0, critical: Optional[str] = None) -> str:
        """Generate vivid attack narration"""
        
        if critical == "failure":
            prompt = (
                f"Describe {attacker.name} CRITICALLY FAILING an attack against {defender.name}. "
                f"Attack: {attacker.weapon}. d20 roll: {attack_roll}. "
                f"Write 1-2 graphic, embarrassing sentences. Include the bungle."
            )
        elif critical == "success":
            prompt = (
                f"Describe {attacker.name} landing a DEVASTATING CRITICAL HIT on {defender.name}! "
                f"Weapon: {attacker.weapon}. Damage: {damage}! d20 roll: {attack_roll}. "
                f"Write 1-2 ultra-violent, graphic sentences with GORE. Make it brutal."
            )
        elif hit:
            prompt = (
                f"Describe {attacker.name} hitting {defender.name} with {attacker.weapon}. "
                f"Damage dealt: {damage}. d20 roll: {attack_roll}. "
                f"Write 1-2 violent sentences with blood and impact."
            )
        else:
            prompt = (
                f"Describe {attacker.name} MISSING their attack against {defender.name}. "
                f"Weapon: {attacker.weapon}. d20 roll: {attack_roll}. "
                f"Write 1-2 dramatic sentences about the miss."
            )
        
        try:
            output = self.pipe(prompt, max_new_tokens=80, do_sample=True, temperature=0.85, repetition_penalty=1.2)
            response = output[0]['generated_text'].strip()
            
            # Clean up
            for phrase in ["Write 1-2", "Describe", "Attack:", "Damage:", "d20"]:
                if phrase in response:
                    response = response.split(phrase)[-1].strip()
            
            if len(response) < 10:
                if critical == "success":
                    response = f"{attacker.name}'s {attacker.weapon} TEARS into {defender.name}, blood and gore erupting! {damage} damage!"
                elif critical == "failure":
                    response = f"{attacker.name} completely botches the attack, stumbling backward!"
                elif hit:
                    response = f"{attacker.name} strikes {defender.name} with {attacker.weapon}! {damage} damage dealt!"
                else:
                    response = f"{attacker.name}'s attack sails harmlessly past {defender.name}!"
            
            return response[:200]
        except:
            if hit:
                return f"{attacker.name}'s {attacker.weapon} strikes {defender.name}! {damage} damage!"
            else:
                return f"{attacker.name} misses {defender.name}!"

    def narrate_death(self, defeated: CombatCharacter, killer: CombatCharacter) -> str:
        """Generate death narration"""
        prompt = (
            f"Describe the GRITTY, GRAPHIC death of {defeated.name}, defeated by {killer.name}. "
            f"Post-apocalyptic wasteland. Write 2 sentences with gore and finality. Make it brutal."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=60, do_sample=True, temperature=0.85)
            response = output[0]['generated_text'].strip()
            return response[:200] if len(response) > 10 else f"{defeated.name} falls, lifeless."
        except:
            return f"{defeated.name} collapses into the dust, their final breath stolen by the wasteland."

    def narrate_defend(self, character: CombatCharacter) -> str:
        """Generate defend action narration"""
        prompt = (
            f"Describe {character.name} raising their defenses and bracing for impact. "
            f"Write 1-2 sentences. Post-apocalyptic tone."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=40, do_sample=True, temperature=0.7)
            response = output[0]['generated_text'].strip()
            return response if len(response) > 5 else f"{character.name} braces for incoming attacks!"
        except:
            return f"{character.name} raises their defenses!"

    def narrate_inventory_use(self, character: CombatCharacter, item: Item) -> str:
        """Generate narration for using an item"""
        prompt = (
            f"{character.name} uses {item.name} ({item.item_type}). Effect: {item.description}. "
            f"Write 1-2 dramatic sentences. Post-apocalyptic tone."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=40, do_sample=True, temperature=0.75)
            response = output[0]['generated_text'].strip()
            return response if len(response) > 5 else f"{character.name} uses {item.name}!"
        except:
            return f"{character.name} uses {item.name}! {item.description}"

    def narrate_victory(self, player: CombatCharacter) -> str:
        """Generate victory narration"""
        prompt = (
            f"{player.name} has defeated all enemies and stands victorious in the wasteland. "
            f"Write 2-3 dramatic sentences about their triumph. Gritty tone. They're bloodied but alive."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=60, do_sample=True, temperature=0.8)
            response = output[0]['generated_text'].strip()
            return response if len(response) > 10 else "You stand victorious amidst the carnage."
        except:
            return "You stand victorious, the wasteland now yours to command."

    def narrate_defeat(self) -> str:
        """Generate defeat narration"""
        prompt = (
            f"The player has been defeated in combat in the post-apocalyptic wasteland. "
            f"Write 2-3 dramatic, gritty sentences about their death and the darkness claiming them."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=60, do_sample=True, temperature=0.8)
            response = output[0]['generated_text'].strip()
            return response if len(response) > 10 else "Darkness claims you as the wasteland takes another life."
        except:
            return "Your vision fades... the wasteland claims another victim."

# --- PART 2: Enhanced Combat Game with Full Systems ---
class AdvancedCombatGame:
    def __init__(self):
        self.ai = GameAI()
        
        # Initialize player with inventory
        self.player = CombatCharacter(
            name="You",
            hp=100,
            max_hp=100,
            atk=8,
            defense=2,
            weapon="Plasma Rifle",
            weapon_bonus=3,
            is_player=True,
            inventory=[
                Item("Health Pack", "consumable", description="Restores 25 HP"),
                Item("Combat Armor", "armor", armor_bonus=2, description="Kevlar reinforced"),
                Item("Energy Cell", "consumable", description="Refills energy weapon")
            ]
        )
        
        # Initialize enemies with inventory
        self.enemies: List[CombatCharacter] = [
            CombatCharacter(
                name="Super Mutant",
                hp=60,
                max_hp=60,
                atk=7,
                defense=1,
                weapon="Super Sledge",
                weapon_bonus=3,
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
                inventory=[Item("Spoiled Meat", "junk", description="Smells of death")]
            )
        ]
        
        self.all_combatants: List[CombatCharacter] = [self.player] + self.enemies
        self.initiative_order: List[CombatCharacter] = []
        self.round_count = 0
        self.in_combat = True
        
    def roll_initiative(self):
        """Roll and narrate initiative for all combatants"""
        print(f"\n{CLR_YELLOW}{'='*90}")
        print(f"INITIATIVE PHASE - ROLLING FOR COMBAT ORDER")
        print(f"{'='*90}{CLR_RESET}\n")
        
        for combatant in self.all_combatants:
            d20_roll = random.randint(1, 20)
            modifier = combatant.atk // 2
            combatant.initiative_roll = d20_roll + modifier
            
            # AI narration
            narration = self.ai.narrate_initiative_roll(combatant, combatant.initiative_roll)
            
            role = f"{CLR_GREEN}[PLAYER]{CLR_RESET}" if combatant.is_player else f"{CLR_RED}[ENEMY]{CLR_RESET}"
            print(f"{role} {combatant.name:20} | d20: {d20_roll:2} + {modifier} = {combatant.initiative_roll:3}")
            print(f"{CLR_MAGENTA}  └─ {narration}{CLR_RESET}\n")
        
        # Sort by initiative
        self.initiative_order = sorted(self.all_combatants, key=lambda x: x.initiative_roll, reverse=True)
        
        print(f"{CLR_CYAN}TURN ORDER:{CLR_RESET}")
        for i, c in enumerate(self.initiative_order, 1):
            role = "PLAYER" if c.is_player else "ENEMY"
            print(f"  {i}. {CLR_WHITE}{c.name:20}{CLR_RESET} (Initiative: {c.initiative_roll}) [{role}]")
        print()

    def display_combat_status(self):
        """Display all combatants' status"""
        print(f"\n{CLR_BLUE}{'='*90}")
        print(f"ROUND {self.round_count} - BATTLEFIELD STATUS")
        print(f"{'='*90}{CLR_RESET}\n")
        
        for c in self.initiative_order:
            print(c.display_status())
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
        elif action in ["inventory", "inv", "i"]:
            return "inventory", None
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
        """Execute attack roll"""
        d20_roll = random.randint(1, 20)
        total_roll = d20_roll + attacker.atk
        defense_ac = 12 + (defender.defense if defender.active_armor else 0)
        
        if d20_roll == 1:
            return {"hit": False, "d20": d20_roll, "total": total_roll, "damage": 0, "critical": "failure"}
        elif d20_roll == 20:
            base_damage = random.randint(attacker.atk + 5, attacker.atk + 10)
            damage = (base_damage + attacker.weapon_bonus) * 2
            return {"hit": True, "d20": d20_roll, "total": total_roll, "damage": damage, "critical": "success"}
        elif total_roll >= defense_ac:
            base_damage = random.randint(attacker.atk, attacker.atk + 6)
            damage = base_damage + attacker.weapon_bonus
            return {"hit": True, "d20": d20_roll, "total": total_roll, "damage": damage, "critical": None}
        else:
            return {"hit": False, "d20": d20_roll, "total": total_roll, "damage": 0, "critical": None}

    def execute_player_turn(self):
        """Execute player's turn"""
        self.display_combat_status()
        
        while True:
            print(f"{CLR_GREEN}[YOUR TURN]{CLR_RESET}")
            print(f"{CLR_YELLOW}Commands: attack <enemy>, defend, use <item>, inventory, flee, help{CLR_RESET}")
            command = input(f"{CLR_GREEN}> {CLR_RESET}").strip()
            
            action, target = self.parse_player_action(command)
            
            if action == "help":
                print(f"{CLR_CYAN}\n[COMMAND HELP]")
                print(f"  attack <enemy>  - Attack by name (Super Mutant, Feral Ghoul)")
                print(f"  defend          - Raise defenses, reduce damage")
                print(f"  use <item>      - Use consumable from inventory")
                print(f"  inventory       - View your items")
                print(f"  flee            - Attempt to escape (DC 12)")
                print(f"  help            - Show this help{CLR_RESET}\n")
                continue
            
            if action == "inventory":
                print(f"\n{self.player.display_inventory()}\n")
                continue
            
            if action is None:
                print(f"{CLR_RED}Invalid command. Type 'help' for options.{CLR_RESET}\n")
                continue
            
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
                print(f"{CLR_CYAN}d20: {result['d20']} + {self.player.atk} = {result['total']}{CLR_RESET}")
                
                if result["hit"]:
                    enemy.hp = max(enemy.hp - result["damage"], 0)
                    status = "CRITICAL!" if result["critical"] == "success" else "HIT!"
                    print(f"{CLR_GREEN}{status} {result['damage']} damage!{CLR_RESET}\n")
                else:
                    status = "CRITICAL FAILURE!" if result["critical"] == "failure" else "MISS!"
                    print(f"{CLR_RED}{status}{CLR_RESET}\n")
                
                narrative = self.ai.narrate_attack(self.player, enemy, result['d20'], result["hit"], result["damage"], result["critical"])
                print(f"{CLR_MAGENTA}[NARRATION] {narrative}{CLR_RESET}\n")
                
                if enemy.hp <= 0:
                    death_narration = self.ai.narrate_death(enemy, self.player)
                    print(f"{CLR_RED}[DEATH] {death_narration}{CLR_RESET}\n")
                    self.enemies.remove(enemy)
                    if enemy in self.initiative_order:
                        self.initiative_order.remove(enemy)
                
                return "attack"
            
            elif action == "defend":
                print(f"\n{CLR_BLUE}[DEFENSIVE STANCE]{CLR_RESET}\n")
                narration = self.ai.narrate_defend(self.player)
                print(f"{CLR_MAGENTA}[NARRATION] {narration}{CLR_RESET}\n")
                return "defend"
            
            elif action == "use":
                if not target:
                    print(f"{CLR_RED}Use what? Specify item name.{CLR_RESET}\n")
                    continue
                
                item = self.find_item(self.player, target)
                if not item:
                    print(f"{CLR_RED}Item not found in inventory.{CLR_RESET}\n")
                    continue
                
                narration = self.ai.narrate_inventory_use(self.player, item)
                print(f"{CLR_MAGENTA}[NARRATION] {narration}{CLR_RESET}\n")
                
                if item.item_type == "consumable":
                    if "health" in item.name.lower():
                        heal_amount = 25
                        self.player.hp = min(self.player.hp + heal_amount, self.player.max_hp)
                        print(f"{CLR_GREEN}Restored {heal_amount} HP! Current: {self.player.hp}/{self.player.max_hp}{CLR_RESET}\n")
                    self.player.inventory.remove(item)
                
                return "use"
            
            elif action == "flee":
                flee_roll = random.randint(1, 20)
                print(f"\n{CLR_BLUE}[FLEE ATTEMPT]{CLR_RESET}")
                print(f"{CLR_CYAN}Flee Roll: {flee_roll}{CLR_RESET}\n")
                
                if flee_roll > 12:
                    print(f"{CLR_GREEN}You escape the combat!{CLR_RESET}\n")
                    self.in_combat = False
                    return "flee"
                else:
                    print(f"{CLR_RED}Failed! Enemies pursue!{CLR_RESET}\n")
                    continue

    def execute_enemy_turn(self, enemy: CombatCharacter):
        """Execute enemy's turn with full narration"""
        if enemy.hp <= 0 or not self.enemies:
            return
        
        target = self.player
        result = self.roll_attack(enemy, target)
        
        print(f"{CLR_RED}[{enemy.name.upper()} ATTACKS]{CLR_RESET}")
        print(f"{CLR_CYAN}d20: {result['d20']} + {enemy.atk} = {result['total']}{CLR_RESET}")
        
        if result["hit"]:
            target.hp = max(target.hp - result["damage"], 0)
            status = "CRITICAL!" if result["critical"] == "success" else "HIT!"
            print(f"{CLR_RED}{status} You take {result['damage']} damage!{CLR_RESET}\n")
        else:
            status = "CRITICAL FAILURE!" if result["critical"] == "failure" else "MISS!"
            print(f"{CLR_GREEN}{status}{CLR_RESET}\n")
        
        narrative = self.ai.narrate_attack(enemy, target, result['d20'], result["hit"], result["damage"], result["critical"])
        print(f"{CLR_MAGENTA}[NARRATION] {narrative}{CLR_RESET}\n")
        
        if target.hp <= 0:
            death_narration = self.ai.narrate_death(target, enemy)
            print(f"{CLR_RED}[YOUR DEATH] {death_narration}{CLR_RESET}\n")
            self.in_combat = False

    def run(self):
        """Main game loop"""
        print(f"\n{CLR_MAGENTA}{'='*90}")
        print(f"WASTELAND COMBAT SYSTEM - INITIALIZED")
        print(f"{'='*90}{CLR_RESET}\n")
        
        opening = self.ai.narrate_combat_start(self.player, self.enemies)
        print(f"{CLR_MAGENTA}[SCENE] {opening}{CLR_RESET}\n")
        
        self.roll_initiative()
        input(f"{CLR_YELLOW}Press Enter to begin...{CLR_RESET}\n")
        
        while self.in_combat and self.player.hp > 0 and len(self.enemies) > 0:
            self.round_count += 1
            
            round_narration = self.ai.narrate_round_start(self.round_count)
            print(f"{CLR_YELLOW}[ROUND {self.round_count}]{CLR_RESET}")
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
                defeat_text = self.ai.narrate_defeat()
                print(f"{CLR_RED}{defeat_text}{CLR_RESET}\n")
                self.in_combat = False
            else:
                input(f"{CLR_YELLOW}Press Enter for next round...{CLR_RESET}\n")

if __name__ == "__main__":
    try:
        game = AdvancedCombatGame()
        game.run()
    except KeyboardInterrupt:
        print(f"\n{CLR_YELLOW}Game exited.{CLR_RESET}")
    except Exception as e:
        print(f"{CLR_RED}Error occurred:{CLR_RESET}")
        traceback.print_exc()
