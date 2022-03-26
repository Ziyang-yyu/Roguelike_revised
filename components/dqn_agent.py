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
from tcod.map import compute_fov
import components.fighter as fighter

# For dqn agent
import time
import sys

# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf
import components.dqn as dqn

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
            #(-1, -1),  # Northwest
            (-1, 0),  # West
            #(-1, 1),  # Southwest

            (0, -1),  # North
            (0, 1),  # South

            #(1, -1),  # Northeast
            (1, 0),  # East
            #(1, 1),  # Southeast
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


                    best_action = (seq[0][0]-start[0],seq[0][1]-start[1])

                q.put((ny, nx))
                step += 1
                V[(ny, nx)] = step

        if best_action is not None:

            return [best_action]

        else:
            # no best action found, return a random action

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
        # attack when 1 or 0
        distance = max(abs(dx), abs(dy))  # Chebyshev distance.

        if self.engine.game_map.visible[self.entity.x, self.entity.y]:
            if distance <= 1:
                return MeleeAction(self.entity, dx, dy).perform()

            self.path = self.get_path_to(target.x, target.y)

        if self.path:
            dest_x, dest_y = self.path.pop(0)
            # 4 actions
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


# Added agent for training: chases thief2 agent
class CompetitiveEnemy(BaseAI):
        def __init__(self, entity: Actor):
            super().__init__(entity)
            self.path: List[Tuple[int, int]] = []

        def perform(self) -> None:
            target = self.engine.player

            if self.engine.game_map.visible[self.entity.x, self.entity.y]:

                self.path = self.get_path_bfs(target.x,target.y)

            if self.path:
                x,y =self.path[0][0],self.path[0][1]

                return MovementAction(
                    self.entity,
                    x,y,
                ).perform()

            direction_x, direction_y = random.choice(self.get_moves())

            return MovementAction(
                self.entity,
                direction_x, direction_y,
            ).perform()

# Random agent with targets
# stealing the items in the dungeon, but does not attack the player
class ThiefEnemy(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        self.path: List[Tuple[int, int]] = []
        self.thiefScore = 0
        self.playerScore = 0
        self.reward = 0
        self.new_round = False


    def is_new_round(self) -> None:
        return self.new_round

    def get_score(self) -> None:
        return self.thiefScore
    def get_rounds(self) -> None:
        return self.thiefScore+self.playerScore

    def perform(self) -> None:
        pickup = False
        beaten = False
        self.new_round = False
        self.reward += cfg.MOVE_REWARD


        for target in self.entity.gamemap.entities:
            if target.name == cfg.predator:
                if (abs(self.entity.x-target.x)==1 and abs(self.entity.y-target.y)==0) or (abs(self.entity.x-target.x)==0 and abs(self.entity.y-target.y)==1):

                    self.playerScore += 1
                    self.reward += cfg.CAUGHT_BY_PLAYER
                    beaten = True
                    break

            if self.engine.game_map.visible[target.x, target.y]:
            # if there is an item, find the path to the item
                if target.name == cfg.target:
                    # check if the current position is an item
                    if (self.entity.x, self.entity.y) == (target.x, target.y):
                        #print("Stealer picked up", target.name)
                        self.thiefScore += 1
                        pickup = True
                        self.reward += cfg.STEAL_POTION

                        break

                    self.path = self.get_path_to(target.x, target.y)

        if pickup or beaten:
            self.new_round = True
            f1 = open('astar_reward.txt', 'a')
            f1.write(str(self.reward)+'\n')

            f1.close()
            self.reward = 0

            if pickup:

                return PickupAction(self.entity).perform()


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

# Added agent:
class ThiefEnemy2(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        #self.path: List[Tuple[int, int]] = []
        self.ai = qlearn.QLearn(actions=range(cfg.directions), alpha=0.1, gamma=0.9, epsilon=0.1)
        self.lastState = None
        self.lastAction = None
        self.new_round = False


    def reset_pos(self):
        # reset to a safe place

        predator_coords = []
        for ent in self.entity.gamemap.entities:
            if ent.name == cfg.predator:
                # get all predator coords
                predator_coords.append([ent.x,ent.y])

        grid = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)

        while True:
            reset_x, reset_y = random.choice(np.argwhere(np.array(grid)==1))
            # ensure it is not 0,0
            if not(reset_x == 0 and reset_y == 0):
                for pos in predator_coords:
                    # ensure new pos is safe
                    if abs(pos[0]-reset_x) <=1 and abs(pos[1]-reset_y)<=1:

                        break

                break

        self.entity.x = reset_x
        self.entity.y = reset_y


    def calculate_state(self):
        def cell_value(x, y):

            for ent in self.entity.gamemap.entities:
                # TODO: change miner to player later
                if ent.name == cfg.predator and (x == ent.x and y == ent.y):
                    return 3
                if ent.name == cfg.target and (x == ent.x and y == ent.y):
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


        # for x,y in all visible states:
        # get player's position
        for ent in self.entity.gamemap.entities:
            if ent.name == cfg.prey:
                center_x, center_y = ent.x, ent.y
        #center_x, center_y = self.engine.player.x, self.engine.player.y
        n = 2*cfg.radius+1
        grid = np.ones((n,n,2))
        # range from center_x-r to center_x+r
        i = 0

        for x in range(center_x-cfg.radius, center_x+cfg.radius+1):
            j = 0
            for y in range(center_y-cfg.radius, center_y+cfg.radius+1):
                grid[i][j] = (x,y)
                j+=1
            i+=1

        # remove center cell
        grid = np.delete(grid.reshape([n*n,2]), n*n//2, axis=0)

        #return tuple([cell_value(self.engine.player.x + dir[0], self.engine.player.y + dir[1]) for dir in self.get_moves()])
        return tuple([cell_value(int(dir[0]), int(dir[1])) for dir in grid])

    def save_agent(self) -> None:
        self.ai.save_qtable()

    # for training

    def get_score(self) -> None:
        return self.thiefScore

    def get_rounds(self) -> None:
        return self.thiefScore+self.playerScore

    def is_new_round(self) -> None:
        return self.new_round

    def perform(self) -> None:

        #print('mouse update begin...')
        state = self.calculate_state()
        pickup = False
        beaten = False
        self.new_round = False

        reward = cfg.MOVE_REWARD

        for ent in self.entity.gamemap.entities:
            if ent.name == cfg.predator:
                if (abs(self.entity.x-ent.x)==0 and abs(self.entity.y-ent.y)==1) or (abs(self.entity.x-ent.x)==1 and abs(self.entity.y-ent.y)==0):
                    reward = cfg.CAUGHT_BY_PLAYER
                    if self.lastState is not None:
                        self.ai.learn(self.lastState, self.lastAction, state, reward)
                    self.lastState = None
                    beaten = True
                    break
            if ent.name == cfg.target and (self.entity.x == ent.x and self.entity.y == ent.y):

                reward = cfg.STEAL_POTION
                pickup = True
                break

        if not beaten:
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, state, reward)
# one round ended

        if pickup or beaten:
            # round ended
            self.new_round = True
            f = open('agent_reward_4_dir_7.txt', 'a')
            f.write(str(reward)+'\n')
            f.close()
            reward = 0
            if pickup:
                return PickupAction(self.entity).perform()
            else: # caught
                self.reset_pos()
            # choose a new action and execute it
        action = self.ai.choose_action(state) # move to main
        self.lastState = state
        self.lastAction = action # a number
            # 0: left, 1: left down, 2: down, 3: right down, 4: right, 5: right up, 6: up, 7: left up
        dir_x, dir_y = self.get_moves()[action]

        return MovementAction(
                self.entity,
                dir_x,
                dir_y,
            ).perform()

# dqn agent
class ThiefEnemy3(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        print("Initialise DQN Agent")
        self.params = {
            # Model backups
            'load_file': None,
            'save_file': None,
            'save_interval' : 10000,

            # Training parameters
            'train_start': 5000,    # Episodes before training starts
            'batch_size': 32,       # Replay memory batch size
            'mem_size': 100000,     # Replay memory size

            'discount': 0.95,       # Discount rate (gamma value)
            'lr': .0002,            # Learning reate
            # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
            # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

            # Epsilon value (epsilon-greedy)
            'eps': 1.0,             # Epsilon start value
            'eps_final': 0.1,       # Epsilon end value
            'eps_step': 34000       # Epsilon steps between start and end (linear)
        }
        # width and height of map
        self.params['width'] = 2*cfg.radius+1
        self.params['height'] = 2*cfg.radius+1
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
        self.qnet = dqn.DQN(self.params)

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = []
        self.cost_disp = 0

        # Stats

        self.cnt = self.qnet.sess.run(self.qnet.global_step)

        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

        self.reward = 0
        self.rounds = 0
        self.score = 0

    def get_rounds(self) -> None:
        return self.rounds

    def is_new_round(self) -> None:
        return self.new_round

    def get_score(self) -> None:
        return self.score

    def getMove(self):
        # Exploit / Explore
        if np.random.rand() > self.params['eps']:
            # Exploit action
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(self.current_state,
                                                     (1, self.params['width'], self.params['height'], 6)),
                             self.qnet.q_t: np.zeros(1),
                             self.qnet.actions: np.zeros((1, 4)),
                             self.qnet.terminals: np.zeros(1),
                             self.qnet.rewards: np.zeros(1)})[0]

            self.Q_global.append(max(self.Q_pred))
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_moves()[a_winner[np.random.randint(0, len(a_winner))][0]]
            else:
                move = self.get_moves()[a_winner[0][0]]
        else:
            # Random:
            #move = self.get_direction(np.random.randint(0, 4))
            move = random.choice(self.get_moves())

        # Save last_action: a value-index of the direction
        self.last_action = self.get_moves().index(move)

        return (move[0],move[1])

    def observation_step(self):
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices()

            # Process current experience reward
            self.current_score = self.score
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 0:
                self.last_reward = 10.    # Eat food    (Yum!)
            elif reward < -10:
                self.last_reward = -500.  # Get eaten   (Ouch!) -500
                self.won = False
            elif reward < 0:
                self.last_reward = -1.    # Punish time (Pff..)

            if(self.terminal and self.won):
                self.last_reward = 100.
            self.ep_rew += self.last_reward

            # Store last experience into memory
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if(params['save_file']):
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    self.qnet.save_ckpt('saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt)/ float(self.params['eps_step']))

    def observationFunction(self):
        # Do observation
        self.terminal = False
        self.observation_step()

        #return state

    # call after a round ends
    def final(self):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step()

        # Print stats
        log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(self.params['height'])+'-x-'+str(self.params['num_training'])+'.log','a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()

    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = [] # States (s)
            batch_r = [] # Rewards (r)
            batch_a = [] # Actions (a)
            batch_n = [] # Next states (s')
            batch_t = [] # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)


    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot


    def getStateMatrices(self):
        def getMatrix():
            # get prey's center
            for ent in self.entity.gamemap.entities:
                if ent.name == cfg.prey:
                    center_x, center_y = ent.x, ent.y

            grid = np.ones((params['width'],params['width'],2))
            # range from center_x-r to center_x+r
            i = 0
            for x in range(center_x-cfg.radius, center_x+cfg.radius+1):
                j = 0
                for y in range(center_y-cfg.radius, center_y+cfg.radius+1):
                    grid[i][j] = (x,y)
                    j+=1
                i+=1
            return grid # a grid of coords
        """ Return wall, ghosts, food, capsules matrices """
        def getWallMatrix():
            """ Return matrix with wall coordinates set to 1 """
            matrix = np.zeros((self.params['width'],self.params['height']), dtype=np.int8)
            full_grid = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)
            fov_grid_coords = self.getMatrix()
            for coord_row in fov_grid_coords:
                for coord in coord_row:
                    # Put cell vertically reversed in matrix
                    cell = 0 if full_grid[coord[0]][coord[1]] else 1
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix():
            """ Return matrix with pacman coordinates set to 1 """
            # find prey coords and replace fov coords == prey coords = 1
            # should be the center of fov
            matrix = np.zeros((self.params['width'],self.params['height']), dtype=np.int8)
            matrix[params['height']//2][params['width']//2] = 1
            return matrix

        def getGhostMatrix():
            """ Return matrix with ghost coordinates set to 1 """
            predator_coords = []
            for ent in self.entity.gamemap.entities:
                if ent.name == cfg.predator:
                    predator_coords.append(ent.x,ent.y)
            matrix = np.zeros((self.params['width'],self.params['height']), dtype=np.int8)
            #full_grid = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)
            fov_grid_coords = self.getMatrix()
            # find conjunction of fov_grid_coords and predator_coords
            for coord_row in fov_grid_coords:
                for coord in coord_row:
                    # if predator coord in fov coord then set to 1
                    if coord in predator_coords:
                        cell = 1
                        matrix[-1-i][j] = cell
            return matrix


        def getFoodMatrix():

            """ Return matrix with food coordinates set to 1 """
            target_coords = []
            for ent in self.entity.gamemap.entities:
                if ent.name == cfg.target:
                    target_coords.append(ent.x,ent.y)
            matrix = np.zeros((self.params['width'],self.params['height']), dtype=np.int8)
            #full_grid = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)
            fov_grid_coords = self.getMatrix()
            # find conjunction of fov_grid_coords and predator_coords
            for coord_row in fov_grid_coords:
                for coord in coord_row:
                    # if predator coord in fov coord then set to 1
                    if coord in target_coords:
                        cell = 1
                        matrix[-1-i][j] = cell
            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((4, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getFoodMatrix(state)
        observation = np.swapaxes(observation, 0, 2)

        return observation

    def registerInitialState(self): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices()

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self):
        move = self.getMove()
        '''
        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP
        '''
        return move

    def perform(self) -> None:

        pickup = False
        beaten = False
        self.new_round = False

        self.reward += cfg.MOVE_REWARD

        for ent in self.entity.gamemap.entities:
            if ent.name == cfg.predator:
                if (abs(self.entity.x-ent.x)==0 and abs(self.entity.y-ent.y)==1) or (abs(self.entity.x-ent.x)==1 and abs(self.entity.y-ent.y)==0):
                    self.reward += cfg.CAUGHT_BY_PLAYER
                    beaten = True
                    break
            if ent.name == cfg.target and (self.entity.x == ent.x and self.entity.y == ent.y):

                self.reward += cfg.STEAL_POTION
                pickup = True
                self.score+=1
                break

        if pickup or beaten:
            # round ended
            self.new_round = True
            self.rounds += 1
            self.final()
            f = open('agent_reward_dqn.txt', 'a')
            f.write(str(reward)+'\n')
            f.close()
            self.reward = 0
            if pickup:
                return PickupAction(self.entity).perform()
            else: # caught
                self.reset_pos()

            # choose a new action and execute it

        # perform action: initialise -> observe -> solicit action -> execute
        self.registerInitialState()
        self.observationFunction()
        dir_x,dir_y = getAction()

        return MovementAction(
                self.entity,
                dir_x,dir_y,
            ).perform()
