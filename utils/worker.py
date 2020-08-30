import torch
import torch.multiprocessing as mp
import numpy as np
import itertools
from utils.update_u import push_and_pull, push_and_pull_Q
from agent.A3C_agent import A3C_agent
# from agent.QMIX_agent import Agents
# from utils.replay_buffer import myReplayBuffer
from torch.distributions import one_hot_categorical


class worker(mp.Process):
    def __init__(self, env, s_dim, a_dim, o_dim, num_agents, gnet, opt, lr_s, sender, device='cpu',name=None):
        super(worker, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.o_dim = o_dim
        self.num_agents = num_agents
        self.gnet = gnet
        self.opt = opt
        self.lr_s = lr_s
        self.device = device
        self.env = env
        self.gamma = 0.9
        self.sender = sender[0]
        self.name = name
        self.lnet = [A3C_agent(
            s_dim=s_dim, o_dim=o_dim, a_dim=a_dim,
            num_agents=num_agents, device=device
        ) for i in range(num_agents)]

    def run(self):
        episode = 1000
        env_info = self.env.get_env_info()

        n_actions = env_info["n_actions"]
        n_agents = env_info["n_agents"]
        state_dim = env_info['state_shape']
        obs_dim = env_info['obs_shape']

        for ep in range(episode):
            terminate = False
            self.env.reset()
            ep_rew = 0
            step = 1
            batch_s, batch_a, batch_r = [], [], []
            obs = self.env.get_obs()
            state = self.env.get_state()

            while not terminate:
                a = []
                for i in range(self.num_agents):
                    avail_actions = self.env.get_avail_agent_actions(i)
                    mask = avail_actions
                    a.append(self.lnet[i].choose_action(obs[i], mask))

                # reward, terminate, _ = self.env.step(a)
                reward, terminate, _ = self.env.step(a)
                # print('step reward:', reward)
                batch_s.append(obs)
                batch_a.append(a)
                batch_r.append([reward for i in range(self.num_agents)])
                ep_rew += reward

                obs = self.env.get_obs()
                state = self.env.get_state()

                if step % 5 == 0:
                    for i in range(self.num_agents):
                        push_and_pull(self.lnet[i], self.gnet[i], self.opt[i], self.lr_s[i], False, obs[i],
                                      batch_s, batch_a, batch_r, self.gamma, i, device=self.device)

                    batch_s, batch_a, batch_r = [], [], []

                step += 1

            print('ep_reward:', ep_rew)
            if self.name == str(0):
                self.sender.send(['ep_r', ep_rew, ep])


import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon)

                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]

        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )

        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        print('episode_reward:',episode_reward)
        return episode, episode_reward, win_tag

