from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple
import random


import numpy as np
import tcod

from actions import Action, BumpAction, MeleeAction, MovementAction, WaitAction
# for bfs agent
from queue import Queue
from actions import PickupAction
from entity import Item
import components.qlearn as qlearn
import components.config as cfg

if TYPE_CHECKING:
    from entity import Actor


class BaseAI(Action):
    def perform(self) -> None:
        raise NotImplementedError()

    def get_path_to(self, dest_x: int, dest_y: int) -> List[Tuple[int, int]]:
        """Compute and return a path to the target position.

        If there is no valid path then returns an empty list.
        """
        # Copy the walkable array.
        cost = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)

        for entity in self.entity.gamemap.entities:
            # Check that an enitiy blocks movement and the cost isn't zero (blocking.)
            if entity.blocks_movement and cost[entity.x, entity.y]:
                # Add to the cost of a blocked position.
                # A lower number means more enemies will crowd behind each other in
                # hallways.  A higher number means enemies will take longer paths in
                # order to surround the player.
                cost[entity.x, entity.y] += 10

        # Create a graph from the cost array and pass that graph to a new pathfinder.
        graph = tcod.path.SimpleGraph(cost=cost, cardinal=2, diagonal=3)
        pathfinder = tcod.path.Pathfinder(graph)

        pathfinder.add_root((self.entity.x, self.entity.y))  # Start position.

        # Compute the path to the destination and remove the starting point.
        path: List[List[int]] = pathfinder.path_to((dest_x, dest_y))[1:].tolist()

        # Convert from List[List[int]] to List[Tuple[int, int]].
        return [(index[0], index[1]) for index in path]

    def get_moves(self):
        return [
            (-1, 0),  # West
            (-1, 1),  # Southwest
            (0, 1),  # South
            (1, 1),  # Southeast
            (1, 0),  # East
            (1, -1),  # Northeast
            (0, -1),  # North
            (-1, -1),  # Northwest
        ]

    def get_value(self, mdict, key):
        try:
            return mdict[key]
        except KeyError:
            return 0

    def get_path_bfs(self, dest_x: int, dest_y: int) -> List[Tuple[int, int]]:
        grid_list = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)
        #print(grid_list)

        fh = len(grid_list)
        fw = max([len(x) for x in grid_list])

        start = (self.entity.x, self.entity.y)
        end = (dest_x,dest_y)

        moves = self.get_moves()
        for n in moves:
            if (start[0]+n[0],start[1]+n[1]) == end:
               # if next move can go towards target
                return [n]

        best_action = None
        q = Queue()

        q.put(start)
        step = 1
        V = {}
        preV = {}

        V[start] = 0

        while not q.empty():

            grid = q.get()

            for i in range(len(moves)):

                ny, nx = grid[0] + moves[i][0], grid[1] + moves[i][1]

                if nx < 0 or ny < 0 or nx > (fw-1) or ny > (fh-1):
                    continue

                if self.get_value(V, (ny, nx)) or grid_list[ny][nx] == 0:  # has visit or is wall.

                    continue

                preV[(ny, nx)] = self.get_value(V, (grid[0], grid[1]))

                if ny == end[0] and nx == end[1]:
                    V[(ny, nx)] = step + 1
                    seq = []
                    last = V[(ny, nx)]
                    while last > 1:
                        k = [key for key in V if V[key] == last]
                        seq.append(k[0])
                        assert len(k) == 1
                        last = preV[(k[0][0], k[0][1])]
                    seq.reverse()
                    print("seq: ",seq)

                    best_action = (seq[0][0]-start[0],seq[0][1]-start[1])

                q.put((ny, nx))
                step += 1
                V[(ny, nx)] = step

        if best_action is not None:
            print('the best move:', best_action)

            return [best_action]
            #return seq

        else:
            # no best action found, return a random action
            print("random!!!!!!!!!!!!!!!!!!")
            return [random.choice(self.get_moves())]

# ===================================================================================================
class HostileEnemy(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        self.path: List[Tuple[int, int]] = []

    def perform(self) -> None:
        target = self.engine.player
        dx = target.x - self.entity.x
        dy = target.y - self.entity.y
        distance = max(abs(dx), abs(dy))  # Chebyshev distance.

        if self.engine.game_map.visible[self.entity.x, self.entity.y]:
            if distance <= 1:
                return MeleeAction(self.entity, dx, dy).perform()

            self.path = self.get_path_to(target.x, target.y)

        if self.path:
            dest_x, dest_y = self.path.pop(0)
            return MovementAction(
                self.entity,
                dest_x - self.entity.x,
                dest_y - self.entity.y,
            ).perform()

        return WaitAction(self.entity).perform()


class ConfusedEnemy(BaseAI):
    """
    A confused enemy will stumble around aimlessly for a given number of turns, then revert back to its previous AI.
    If an actor occupies a tile it is randomly moving into, it will attack.
    """

    def __init__(self, entity: Actor, previous_ai: Optional[BaseAI], turns_remaining: int):
        super().__init__(entity)

        self.previous_ai = previous_ai
        self.turns_remaining = turns_remaining

    def perform(self) -> None:
        # Revert the AI back to the original state if the effect has run its course.
        if self.turns_remaining <= 0:
            self.engine.message_log.add_message(f"The {self.entity.name} is no longer confused.")
            self.entity.ai = self.previous_ai
        else:
            # Pick a random direction
            direction_x, direction_y = random.choice(self.get_moves())

            self.turns_remaining -= 1

            # The actor will either try to move or attack in the chosen random direction.
            # Its possible the actor will just bump into the wall, wasting a turn.
            return BumpAction(
                self.entity,
                direction_x,
                direction_y,
            ).perform()


# Added agent: chases thief2 agent
class CompetitiveEnemy(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        self.path: List[Tuple[int, int]] = []

    def perform(self) -> None:

        for ent in self.entity.gamemap.entities:

            if ent.name == "Player":
                dest_x, dest_y = ent.x, ent.y

                dx = dest_x - self.entity.x
                dy = dest_y - self.entity.y
                distance = max(abs(dx), abs(dy))  # Chebyshev distance.

                #if self.engine.game_map.visible[self.entity.x, self.entity.y]:
                if distance <= 1:
                    return MeleeAction(self.entity, dx, dy).perform()

                self.path = self.get_path_to(dest_x, dest_y)

                if self.path:
                    x, y = self.path.pop(0)

                    return MovementAction(
                        self.entity,
                        x - self.entity.x,
                        y - self.entity.y,
                    ).perform()

        return WaitAction(self.entity).perform()

# Added agent:
class ThiefEnemy2(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        self.path: List[Tuple[int, int]] = []
        self.ai = qlearn.QLearn(actions=range(cfg.directions), alpha=0.1, gamma=0.9, epsilon=0.1)
        self.playerScore = 0
        self.thiefScore = 0
        self.lastState = None
        self.lastAction = None

    def get_states_fov(self):
        states = []
        grid_list = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)
        # loop through the entire grid
        for i in range(len(grid_list)):
            sub_states = []
            for j in range(len(grid_list[0])):

                if self.engine.game_map.visible[i, j]:
                    for ent in self.entity.gamemap.entities:
                        if ent.x == i and ent.y == j:
                            if ent.name == "Miner":
                                sub_states.append(3)
                            if ent.name == "Health Potion":
                                sub_states.append(2)
                    if not self.engine.game_map.in_bounds(i, j):

                        # Destination is out of bounds.
                        sub_states.append(1)
                    if not self.engine.game_map.tiles["walkable"][i, j]:
                        # Destination is blocked by a tile.
                        sub_states.append(1)
                    else:
                        # walkable and visible
                        sub_states.append(0)

                else:
                    # not visible
                    sub_states.append(1)
            states.append(sub_states)
        return states


    def calculate_state(self):
        def cell_value(x, y):
            for ent in self.entity.gamemap.entities:
                # TODO: change miner to player later
                if ent.name == "Miner" and (x == ent.x and y == ent.y):
                    return 3
                if ent.name == "Health Potion" and (x == ent.x and y == ent.y):
                    return 2
            if not self.engine.game_map.in_bounds(x, y):
                # Destination is out of bounds.
                return 1
            if not self.engine.game_map.tiles["walkable"][x, y]:
                # Destination is blocked by a tile.
                return 1
            if self.engine.game_map.get_blocking_entity_at_location(x, y):
                # Destination is blocked by an entity.
                return 1
            return 0
            '''
            if cat.cell is not None and (cell.x == cat.cell.x and cell.y == cat.cell.y):
                return 3
            elif cheese.cell is not None and (cell.x == cheese.cell.x and cell.y == cheese.cell.y):
                return 2
            else:
                return 1 if cell.wall else 0
            '''

        #dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        #print(world.get_relative_cell(self.cell.x + dir[0], self.cell.y + dir[1]))
        return tuple([cell_value(self.entity.x+dir[0], self.entity.y+dir[1]) for dir in self.get_moves()])

    def perform(self) -> None:

        #print('mouse update begin...')
        state = self.calculate_state()
        # TODO: all the visible cells
        print(self.get_states_fov())
        reward = cfg.MOVE_REWARD

        for ent in self.entity.gamemap.entities:
            # TODO: change miner to player later
            if ent.name == "Player" and (abs(self.entity.x-ent.x)<=1 and abs(self.entity.y-ent.y)<=1):
                #print(self.entity.name, ' is beaten by', ent.name)
                self.playerScore += 1
                reward = cfg.EATEN_BY_CAT
                if self.lastState is not None:
                    #print("mouse last state:",self.lastState)
                    self.ai.learn(self.lastState, self.lastAction, state, reward)
                    print('mouse learn...')
                self.lastState = None
                # replace the thief

            if ent.name == "Health Potion" and (self.entity.x == ent.x and self.entity.y == ent.y):
                print(self.entity.name, ' picked up a', ent.name)
                self.thiefScore += 1
                reward = 50
                return PickupAction(self.entity).perform()

        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, state, reward)

            # choose a new action and execute it
        action = self.ai.choose_action(state)
        self.lastState = state
        self.lastAction = action # a number
            # 0: left, 1: left down, 2: down, 3: right down, 4: right, 5: right up, 6: up, 7: left up
        dir_x, dir_y = self.get_moves()[action]
        #print("thief move", dir_x, dir_y)
        return MovementAction(
                self.entity,
                dir_x,
                dir_y,
            ).perform()

# stealing the items in the dungeon, but does not attack the player
class ThiefEnemy(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        self.path: List[Tuple[int, int]] = []

    def perform(self) -> None:
        # add fov

        for target in self.entity.gamemap.entities:
            # if there is an item, find the path to the item
            if type(target) == Item:
                # check if the current position is an item
                if (self.entity.x, self.entity.y) == (target.x, target.y):
                    #print("Stealer picked up", target.name)
                    return PickupAction(self.entity).perform()
                #print("item x y:", target.x, target.y)
                #print("stealer x y:", self.entity.x, self.entity.y)
                self.path = self.get_path_to(target.x, target.y)

        if self.path:
            dest_x, dest_y = self.path.pop(0)
            direction_x, direction_y = dest_x - self.entity.x, dest_y - self.entity.y

        #move to the target or random moves if there is no target
        else:
            direction_x, direction_y = random.choice(self.get_moves())

        return MovementAction(
            self.entity,
            direction_x, direction_y,
        ).perform()

        '''
        if random.randint(1,100) >= 60:
            print("moving")
            direction_x, direction_y = random.choice(self.get_moves())
            return MovementAction(
                self.entity,
                direction_x, direction_y,
            ).perform()
        else:
            print("trying to pick up something")
            return PickupAction(self.entity).perform()
        '''
