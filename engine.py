from __future__ import annotations

from typing import TYPE_CHECKING
import lzma
import pickle

from tcod.console import Console
from tcod.map import compute_fov

from message_log import MessageLog
import exceptions
import render_functions
import matplotlib.pyplot as plt # for ai
import components.config as cfg
import components.ai as ai




if TYPE_CHECKING:
    from entity import Actor
    from game_map import GameMap, GameWorld


class Engine:
    game_map: GameMap
    game_world: GameWorld

    def __init__(self, player: Actor):
        self.message_log = MessageLog()
        self.mouse_location = (0, 0)
        self.player = player

    def handle_player_turns(self) -> None:
        try:
            self.player.ai.perform()
        except exceptions.Impossible:
            pass   # Ignore impossible action exceptions from AI.

    def handle_enemy_turns(self) -> None:
        for entity in set(self.game_map.actors) - {self.player}:
            if entity.ai:
                try:
                    entity.ai.perform()
                except exceptions.Impossible:
                    pass  # Ignore impossible action exceptions from AI.

    def update_fov(self) -> None:
        """Recompute the visible area based on the players point of view."""
        self.game_map.visible[:] = compute_fov(
            self.game_map.tiles["transparent"],
            (self.player.x, self.player.y),
            radius=cfg.radius*2,
        )

        #self.plot_max_fov() # aie
        # If a tile is "visible" it should be added to "explored".
        self.game_map.explored |= self.game_map.visible

    def render(self, console: Console) -> None:
        self.game_map.render(console)

        self.message_log.render(console=console, x=21, y=45, width=40, height=5)

        render_functions.render_bar(
            console=console,
            current_value=self.player.fighter.hp,
            maximum_value=self.player.fighter.max_hp,
            total_width=20,
        )

        render_functions.render_dungeon_level(
            console=console,
            dungeon_level=self.game_world.current_floor,
            location=(0, 47),
        )

        render_functions.render_names_at_mouse_location(console=console, x=21, y=44, engine=self)


    # TODO: change function for dqn, currently doesnt work for dqn
    def save_as(self, filename: str) -> None:
        """Save this Engine instance as a compressed file."""

        if not self.player.ai.is_dqn:
            save_data = lzma.compress(pickle.dumps(self))
            with open(filename, "wb") as f:
                f.write(save_data)
        else:
            save_data = lzma.compress(pickle.dumps(self))
            with open(filename, "wb") as f:
                f.write(save_data)


    def show_env_state(self,a):
        # aie
        img = plt.imshow(a, interpolation='none', cmap='gray')
        # plt.imshow(a)
        plt.show()

    def plot_max_fov(self):
        # plot the player's focal view
        x1 = self.player.x - 10
        x2 = self.player.x + 10
        y1 = self.player.y - 10
        y2 = self.player.y + 10
        fov = self.game_map.visible
        a = fov[x1 - 1:x2, y1 - 1:y2]
        print("fov:", a)
        img = plt.imshow(a, interpolation='none', cmap='gray')
        plt.show()

    def plot_map_explored(self):
        # aie
        # 画出拓展的地图
        img = plt.imshow(self.game_map.explored, interpolation='none', cmap='gray')
        plt.show()
