import matplotlib.pyplot as plt
import numpy as np
ep_reward = np.loadtxt('agent_reward_4_dir_4.txt')
ave_episode = 2000
n = len(ep_reward) // ave_episode
avg_reward = np.mean(np.reshape(ep_reward[: ave_episode * n], [n, ave_episode]), 1)
plt.title("Agent Reward")

plt.plot(avg_reward)
  
plt.show()