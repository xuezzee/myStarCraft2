import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from utils.worker import RolloutWorker
from utils.replay_buffer import ReplayBuffer
from agent.QMIX_agent import Agents
from utils.logger import Logger

logger = Logger('../logst')
class Runner:
    def __init__(self, env, args):
        self.env = env
        self.buffer = ReplayBuffer(args)
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []


    def run(self, num):
        train_steps = 0
        for epoch in range(self.args.n_epoch):
            print('Run {}, train epoch {}'.format(num, epoch))
            if epoch % self.args.evaluate_cycle == 0:
                win_rate, episode_reward = self.evaluate()
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)

            episodes = []

            for episode_idx in range(self.args.n_episodes):
                episode, episode_reward, _ = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
            logger.scalar_summary('ep_reward', episode_reward, epoch)
                # print(_)

            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(20):
            _, episode_reward, win_tag = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / 20, episode_rewards / 20