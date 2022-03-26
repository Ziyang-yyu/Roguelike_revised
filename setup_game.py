"""Handle the loading and initialization of game sessions."""
from __future__ import annotations

from typing import Optional
import copy
import lzma
import pickle
import traceback

from PIL import Image  # type: ignore
import tcod

from engine import Engine
from game_map import GameWorld
import color
import entity_factories
import input_handlers
import components.ai as ai
import components.config as cfg


# Load the background image.  Pillow returns an object convertable into a NumPy array.
background_image = Image.open("data/menu_background.png")


def new_game() -> Engine:

    """Return a brand new game session as an Engine instance."""
    map_width = cfg.w #80
    map_height = cfg.h #43

    room_max_size = 18 #10
    room_min_size = 18 #6
    max_rooms = 1 # 30


    if entity_factories.player.ai.is_dqn:
        player = entity_factories.player # for dqn agent
    else:
        player = copy.deepcopy(entity_factories.player)



    engine = Engine(player=player)

    engine.game_world = GameWorld(
        engine=engine,
        max_rooms=max_rooms,
        room_min_size=room_min_size,
        room_max_size=room_max_size,
        map_width=map_width,
        map_height=map_height,
    )

    engine.game_world.generate_floor()
    engine.update_fov()

    engine.message_log.add_message("Hello and welcome, adventurer, to yet another dungeon!", color.welcome_text)

    dagger = copy.deepcopy(entity_factories.dagger)
    leather_armor = copy.deepcopy(entity_factories.leather_armor)

    dagger.parent = player.inventory
    leather_armor.parent = player.inventory

    player.inventory.items.append(dagger)
    player.equipment.toggle_equip(dagger, add_message=False)

    player.inventory.items.append(leather_armor)
    player.equipment.toggle_equip(leather_armor, add_message=False)


    # auto movements
    #n = 0
    prev_score = 0

    game = True
    while game:

        if engine.player.is_alive:

            engine.handle_player_turns()

            engine.handle_enemy_turns()

            engine.update_fov()
            engine.game_map.explored |= engine.game_map.visible


            for i in engine.game_map.entities:
                # run 100 rounds on each map
                if i.name == cfg.prey and i.ai.is_new_round() and i.ai.get_rounds()%3==0:

                    print('==================',i.ai.get_rounds(),'======================')
                    # win rate every 100 rounds
                    print('win rate', (i.ai.get_score()-prev_score)/3)
                    prev_score = i.ai.get_score()
                    #print('thief score:',i.ai.get_score())

                    engine.player.ai.save_agent()
                    if i.ai.get_rounds()%3==0:
                        print('------------------------------END--------------------------')
                        game = False
                        break


            if engine.player.level.requires_level_up:
                level_up = input_handlers.LevelUpEventHandler(engine)
        else:
            engine.update_fov()

            break

    ### aie
    return engine


def load_game(filename: str) -> Engine:
    """Load an Engine instance from a file."""

    with open(filename, "rb") as f:
        engine = pickle.loads(lzma.decompress(f.read()))
    assert isinstance(engine, Engine)
    return engine


class MainMenu(input_handlers.BaseEventHandler):
    """Handle the main menu rendering and input."""

    def on_render(self, console: tcod.Console) -> None:
        """Render the main menu on a background image."""
        console.draw_semigraphics(background_image, 0, 0)

        console.print(
            console.width // 2,
            console.height // 2 - 4,
            "TOMBS OF THE ANCIENT KINGS",
            fg=color.menu_title,
            alignment=tcod.CENTER,
        )
        console.print(
            console.width // 2,
            console.height - 2,
            "By Ziyang Yu",
            fg=color.menu_title,
            alignment=tcod.CENTER,
        )

        menu_width = 24
        for i, text in enumerate(["[N] Play a new game", "[C] Continue last game", "[Q] Quit"]):
            console.print(
                console.width // 2,
                console.height // 2 - 2 + i,
                text.ljust(menu_width),
                fg=color.menu_text,
                bg=color.black,
                alignment=tcod.CENTER,
                bg_blend=tcod.BKGND_ALPHA(64),
            )

    def ev_keydown(self, event: tcod.event.KeyDown) -> Optional[input_handlers.BaseEventHandler]:
        if event.sym in (tcod.event.K_q, tcod.event.K_ESCAPE):
            raise SystemExit()
        elif event.sym == tcod.event.K_c:
            try:
                return input_handlers.MainGameEventHandler(load_game("savegame.sav"))
            except FileNotFoundError:
                return input_handlers.PopupMessage(self, "No saved game to load.")
            except Exception as exc:
                traceback.print_exc()  # Print to stderr.
                return input_handlers.PopupMessage(self, f"Failed to load save:\n{exc}")
        elif event.sym == tcod.event.K_n:
            return input_handlers.MainGameEventHandler(new_game())

        return None
