from __future__ import annotations

import random
import components.config as cfg
import pickle

class QLearn:
    """
    Q-learning:
        Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s', a') - Q(s,a))

        * alpha is the learning rate.
        * gamma is the value of the future reward.
    It use the best next choice of utility in later state to update the former state.
    """
    def __init__(self, actions, alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon, epsilon_min=cfg.epsilon_min, decay=cfg.epsilon_decay):
        try:
            with open('saved_qtable_4_dir_7.pkl', 'rb') as f:
                print('loaded')
                self.q = pickle.load(f)
                #print('q table:',self.q)
        except FileNotFoundError:
            self.q = {}  # save this
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions  # collection of choices
        self.epsilon = epsilon  # exploration constant
        self.epsilon_min = epsilon_min
        self.decay = decay
    # Get the utility of an action in certain state, default is 0.0.
    def get_utility(self, state, action):
        return self.q.get((state, action), 0.0)  # change to dqn

    # When in certain state, find the best action while explore new grid by chance.
    def choose_action(self, state):
        if random.uniform(0,1) < self.epsilon:
            #print('random action')
            action = random.choice(self.actions)
        else:
            q = [self.get_utility(state, act) for act in self.actions]
            
            max_utility = max(q)

            # In case there're several state-action max values
            # we select a random one among them
            if q.count(max_utility) > 1:
                best_actions = [self.actions[i] for i in range(len(self.actions)) if q[i] == max_utility]
                action = random.choice(best_actions)
            else:
                action = self.actions[q.index(max_utility)]
        #print(action,'action')
        # epsilon decay
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.decay
        
        return action

    # learn
    def learn(self, state1, action, state2, reward):
        old_utility = self.q.get((state1, action), None)
        if old_utility == None:
            self.q[(state1, action)] = reward

        # update utility
        else:
            next_max_utility = max([self.get_utility(state2, a) for a in self.actions])
            self.q[(state1, action)] = old_utility + self.alpha * (reward + self.gamma * next_max_utility - old_utility)

    # save model
    def save_qtable(self):
        #print("q: ",self.q)
        print('Model Saved')
        with open('saved_qtable_4_dir_7.pkl', 'wb') as f:
            pickle.dump(self.q, f)
