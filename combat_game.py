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
        """Generate visual HP bar with color"""
        filled = max(0, self.hp * 10 // self.max_hp)
        bar = "■" * filled + "□" * (10 - filled)
        
        # Color based on HP percentage
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
        armor_str = f" | {CLR_CYAN}Armor: {self.active_armor.name}{CLR_RESET}" if self.active_armor else ""
        
        hp_color = CLR_GREEN if self.hp > self.max_hp * 0.5 else (CLR_YELLOW if self.hp > self.max_hp * 0.25 else CLR_RED)
        
        return f"{name_color}{self.name:20}{CLR_RESET} [{hp_bar}] {hp_color}{self.hp:3}/{self.max_hp:3}{CLR_RESET} HP | {self.weapon}{armor_str}"

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

# --- PART 1: Advanced DM AI with Verbose Combat Narration ---
class GameAI:
    def __init__(self):
        print(f"{CLR_YELLOW}[INIT] Loading AI DM model...{CLR_RESET}", flush=True)
        sys.stdout.flush()
        self.pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
        print(f"{CLR_GREEN}[INIT] AI DM loaded and ready!{CLR_RESET}\n", flush=True)

    def _extract_narrative(self, text: str) -> str:
        """Extract clean narrative from AI response"""
        # Remove prompt remnants
        lines = text.split('\n')
        narrative = ' '.join([line.strip() for line in lines if line.strip()])
        
        # Remove common prompt artifacts at start
        artifacts_start = [
            "You are a gritty", "Write a", "Describe", "Generate", "Create",
            "In this scenario", "The player", "A gritty", "Post-apocalyptic",
            "This is a", "Here is"
        ]
        
        for artifact in artifacts_start:
            if narrative.lower().startswith(artifact.lower()):
                narrative = narrative[len(artifact):].strip()
                break
        
        # Ensure it ends with a complete sentence
        while narrative and narrative[-1] not in '.!?':
            # Find last complete sentence
            last_period = narrative.rfind('.')
            last_exclaim = narrative.rfind('!')
            last_question = narrative.rfind('?')
            
            last_punct = max(last_period, last_exclaim, last_question)
            
            if last_punct > 0:
                narrative = narrative[:last_punct + 1]
            else:
                # No punctuation found, add one at end
                if len(narrative) > 3:
                    narrative = narrative.rstrip() + '.'
                break
        
        return narrative.strip()

    def narrate_combat_start(self, player: 'CombatCharacter', enemies: List['CombatCharacter']) -> str:
        """Generate opening narration for combat - verbose and descriptive"""
        enemy_names = ", ".join([e.name for e in enemies])
        prompt = (
            f"You are a gritty post-apocalyptic DM. Write 3-4 detailed, vivid sentences describing "
            f"the start of an intense combat encounter. The player faces {enemy_names} in the wasteland. "
            f"Use descriptive language about the environment, atmosphere, and the menacing presence of enemies. "
            f"Make it dramatic and immersive. End with a complete sentence."
        )
        
        try:
            output = self.pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.8, top_p=0.9)
            response = output[0]['generated_text']
            narrative = self._extract_narrative(response)
            return narrative if len(narrative) > 50 else prompt.split("describes ")[1][:150]
        except:
            return f"The wasteland trembles as {enemy_names} emerge from the shadows, their weapons gleaming with menace. The air grows thick with tension as you lock eyes with your enemies. This battle will determine who walks away alive—and who becomes another skeleton in the sand."

    def narrate_round_start(self, round_num: int) -> str:
        """Generate verbose narration for round start"""
        prompt = (
            f"You are a gritty post-apocalyptic DM. Write 2-3 detailed, vivid sentences describing "
            f"the start of combat round {round_num}. Use sensory details about the combat intensity, "
            f"movement, tension, and atmosphere. Make it dramatic and immersive. End with a complete sentence."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.8, top_p=0.9)
            response = output[0]['generated_text']
            narrative = self._extract_narrative(response)
            return narrative if len(narrative) > 30 else f"Round {round_num} erupts with renewed fury as combatants clash once more!"
        except:
            return f"Round {round_num} rages on! The wasteland shakes with the violence of combat!"

    def narrate_attack(self, attacker: 'CombatCharacter', defender: 'CombatCharacter', 
                      attack_roll: int, total_roll: int, hit: bool, damage: int = 0, 
                      critical: Optional[str] = None) -> str:
        """Generate verbose attack narration"""
        
        if critical == "failure":
            prompt = (
                f"You are a gritty DM. Write 2-3 detailed, dramatic sentences describing {attacker.name} "
                f"CRITICALLY FAILING their attack against {defender.name}. Make it embarrassing and vivid. "
                f"Describe the botched movement, the weapon swing that goes completely wrong. "
                f"d20 roll was {attack_roll}. End with a complete sentence."
            )
        elif critical == "success":
            prompt = (
                f"You are a gritty DM. Write 3-4 detailed, brutal, graphic sentences describing {attacker.name} "
                f"landing a DEVASTATING CRITICAL HIT on {defender.name} with {attacker.weapon}. "
                f"Deal {damage} damage! Be extremely violent and vivid. Describe blood, impact, and destruction. "
                f"d20 roll was {attack_roll}. End with a complete sentence."
            )
        elif hit:
            prompt = (
                f"You are a gritty DM. Write 2-3 detailed, vivid sentences describing {attacker.name} "
                f"successfully hitting {defender.name} with {attacker.weapon}, dealing {damage} damage. "
                f"Include sensory details about the strike, the impact, and the effect on {defender.name}. "
                f"d20 roll was {attack_roll}. End with a complete sentence."
            )
        else:
            prompt = (
                f"You are a gritty DM. Write 2-3 dramatic sentences describing {attacker.name} "
                f"MISSING their attack against {defender.name}. Make it tense and describe how close "
                f"the weapon comes or how the attack glances off harmlessly. d20 roll was {attack_roll}. "
                f"End with a complete sentence."
            )
        
        try:
            output = self.pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.85, top_p=0.9)
            response = output[0]['generated_text']
            narrative = self._extract_narrative(response)
            return narrative if len(narrative) > 30 else prompt.split("describing ")[1][:150]
        except:
            if hit and damage > 0:
                return f"{attacker.name} unleashes a vicious strike with {attacker.weapon}! {defender.name} recoils as blood sprays across the wasteland. The impact sends shockwaves through the air!"
            elif critical == "failure":
                return f"{attacker.name} stumbles catastrophically, their {attacker.weapon} swinging wildly through empty air. Complete loss of balance!"
            else:
                return f"{attacker.name} swings at {defender.name} but the attack goes completely wide, missing entirely."

    def narrate_death(self, defeated: 'CombatCharacter', killer: 'CombatCharacter') -> str:
        """Generate verbose death narration"""
        prompt = (
            f"You are a gritty post-apocalyptic DM. Write 2-3 detailed, dramatic sentences describing "
            f"the DEATH of {defeated.name}, defeated by {killer.name}. Be graphic and vivid about the finality. "
            f"Describe the wasteland claiming another life, the collapse, the silence. End with a complete sentence."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=120, do_sample=True, temperature=0.8, top_p=0.9)
            response = output[0]['generated_text']
            narrative = self._extract_narrative(response)
            return narrative if len(narrative) > 30 else f"{defeated.name} falls, their body going limp as the wasteland claims them. Death has come."
        except:
            return f"{defeated.name} collapses into the dust, their final breath stolen by the wasteland. They lie motionless as the sand begins to cover their remains."

    def narrate_defend(self, character: 'CombatCharacter') -> str:
        """Generate verbose defend narration"""
        prompt = (
            f"You are a gritty DM. Write 2-3 vivid sentences describing {character.name} "
            f"raising their defenses and bracing for impact. Describe the tactical positioning, "
            f"the readiness, the tensing of muscles. End with a complete sentence."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.8, top_p=0.9)
            response = output[0]['generated_text']
            narrative = self._extract_narrative(response)
            return narrative if len(narrative) > 30 else f"{character.name} raises their defenses and braces for incoming attacks!"
        except:
            return f"{character.name} assumes a defensive stance, muscles tense, ready to weather the coming storm."

    def narrate_inventory_use(self, character: 'CombatCharacter', item: Item) -> str:
        """Generate narration for using an item"""
        prompt = (
            f"You are a gritty DM. Write 2 vivid sentences describing {character.name} "
            f"using {item.name} ({item.item_type}) in combat. Effect: {item.description}. "
            f"Make it dramatic and show the impact of using this item. End with a complete sentence."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.9)
            response = output[0]['generated_text']
            narrative = self._extract_narrative(response)
            return narrative if len(narrative) > 30 else f"{character.name} uses {item.name}! {item.description}"
        except:
            return f"{character.name} deploys {item.name}! {item.description}"

    def narrate_victory(self, player: 'CombatCharacter') -> str:
        """Generate verbose victory narration"""
        prompt = (
            f"You are a gritty post-apocalyptic DM. Write 3-4 detailed, dramatic sentences "
            f"describing {player.name}'s VICTORY. They stand bloodied but triumphant over their defeated enemies. "
            f"Describe the wasteland, their survival, their strength. Make it epic and complete. "
            f"End with a complete sentence."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.8, top_p=0.9)
            response = output[0]['generated_text']
            narrative = self._extract_narrative(response)
            return narrative if len(narrative) > 50 else f"{player.name} stands victorious amidst the ruins of their enemies! The wasteland falls silent. They have survived."
        except:
            return f"{player.name} stands victorious, breathing heavily amid the corpses of fallen enemies. The wasteland trembles beneath their feet—a survivor's domain."

    def narrate_defeat(self) -> str:
        """Generate verbose defeat narration"""
        prompt = (
            f"You are a gritty post-apocalyptic DM. Write 3-4 detailed, dramatic sentences "
            f"describing a warrior's DEFEAT in the wasteland. Their final moments, darkness closing in, "
            f"the wasteland claiming yet another life. Make it poignant and complete. End with a complete sentence."
        )
        try:
            output = self.pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.8, top_p=0.9)
            response = output[0]['generated_text']
            narrative = self._extract_narrative(response)
            return narrative if len(narrative) > 50 else f"Darkness falls as your vision fades. The wasteland has claimed you, adding your bones to its endless graveyard."
        except:
            return f"Your vision blurs as consciousness slips away. The wasteland swallows you whole, and darkness becomes eternal. Another soul lost to the desert."

# --- PART 2: Enhanced Combat Game ---
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
        """Roll and display initiative for all combatants"""
        print(f"\n{CLR_YELLOW}{'='*90}")
        print(f"INITIATIVE PHASE - ROLLING FOR COMBAT ORDER")
        print(f"{'='*90}{CLR_RESET}\n")
        
        for combatant in self.all_combatants:
            d20_roll = random.randint(1, 20)
            modifier = combatant.atk // 2
            combatant.initiative_roll = d20_roll + modifier
            
            role = f"{CLR_GREEN}[PLAYER]{CLR_RESET}" if combatant.is_player else f"{CLR_RED}[ENEMY]{CLR_RESET}"
            print(f"{role} {combatant.name:20} rolled d20: {d20_roll:2} + {modifier} = {combatant.initiative_roll:3}")
        
        # Sort by initiative
        self.initiative_order = sorted(self.all_combatants, key=lambda x: x.initiative_roll, reverse=True)
        
        print(f"\n{CLR_CYAN}{'='*90}")
        print(f"COMBAT ORDER:")
        print(f"{'='*90}{CLR_RESET}")
        for i, c in enumerate(self.initiative_order, 1):
            role = "PLAYER" if c.is_player else "ENEMY"
            print(f"  {i}. {CLR_WHITE}{c.name:20}{CLR_RESET} (Init: {c.initiative_roll}) - {role}")
        print()

    def display_combat_status(self):
        """Display all combatants' status with colors"""
        print(f"\n{CLR_BLUE}{'='*90}")
        print(f"ROUND {self.round_count} - BATTLEFIELD STATUS")
        print(f"{'='*90}{CLR_RESET}\n")
        
        # Player
        print(f"{CLR_GREEN}[PLAYER]{CLR_RESET}")
        print(self.player.display_status())
        
        # Enemies
        print(f"\n{CLR_RED}[ENEMIES]{CLR_RESET}")
        for enemy in self.enemies:
            if enemy.hp > 0:
                print(enemy.display_status())
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
                print(f"  defend          - Raise defenses, reduce damage this round")
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
                    status = f"{CLR_GREEN}CRITICAL HIT!{CLR_RESET}" if result["critical"] == "success" else f"{CLR_GREEN}HIT!{CLR_RESET}"
                    print(f"{status} {result['damage']} damage!\n")
                else:
                    status = f"{CLR_RED}CRITICAL FAILURE!{CLR_RESET}" if result["critical"] == "failure" else f"{CLR_RED}MISS!{CLR_RESET}"
                    print(f"{status}\n")
                
                narrative = self.ai.narrate_attack(self.player, enemy, result['d20'], result['total'], 
                                                  result["hit"], result["damage"], result["critical"])
                print(f"{CLR_MAGENTA}{narrative}{CLR_RESET}\n")
                
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
                print(f"{CLR_MAGENTA}{narration}{CLR_RESET}\n")
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
                print(f"{CLR_MAGENTA}{narration}{CLR_RESET}\n")
                
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
        """Execute enemy's turn with narration"""
        if enemy.hp <= 0 or not self.enemies:
            return
        
        target = self.player
        result = self.roll_attack(enemy, target)
        
        print(f"{CLR_RED}[{enemy.name.upper()} ATTACKS]{CLR_RESET}")
        print(f"{CLR_CYAN}d20: {result['d20']} + {enemy.atk} = {result['total']}{CLR_RESET}")
        
        if result["hit"]:
            target.hp = max(target.hp - result["damage"], 0)
            status = f"{CLR_RED}CRITICAL HIT!{CLR_RESET}" if result["critical"] == "success" else f"{CLR_RED}HIT!{CLR_RESET}"
            print(f"{status} You take {result['damage']} damage!\n")
        else:
            status = f"{CLR_GREEN}CRITICAL FAILURE!{CLR_RESET}" if result["critical"] == "failure" else f"{CLR_GREEN}MISS!{CLR_RESET}"
            print(f"{status}\n")
        
        narrative = self.ai.narrate_attack(enemy, target, result['d20'], result['total'], 
                                          result["hit"], result["damage"], result["critical"])
        print(f"{CLR_MAGENTA}{narrative}{CLR_RESET}\n")
        
        if target.hp <= 0:
            death_narration = self.ai.narrate_defeat()
            print(f"{CLR_RED}[YOUR DEATH] {death_narration}{CLR_RESET}\n")
            self.in_combat = False

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
            
            round_narration = self.ai.narrate_round_start(self.round_count)
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
            # NO MORE "PRESS ENTER" - Combat flows continuously

if __name__ == "__main__":
    try:
        game = AdvancedCombatGame()
        game.run()
    except KeyboardInterrupt:
        print(f"\n{CLR_YELLOW}Game exited.{CLR_RESET}")
    except Exception as e:
        print(f"{CLR_RED}Error occurred:{CLR_RESET}")
        traceback.print_exc()
