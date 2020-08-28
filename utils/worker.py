import torch
import torch.multiprocessing as mp
import numpy as np
import itertools
from utils.update_u import push_and_pull, push_and_pull_Q
from agent.A3C_agent import A3C_agent
from agent.QMIX_agent import Q_net, QMIX


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


class worker_QMIX(mp.Process):
    def __init__(self, env, s_dim, a_dim, o_dim, num_agents, gnet, opt, lr_s, sender, device='cpu',name=None):
        super(worker_QMIX, self).__init__()
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
        self.lnet = [Q_net(
            s_dim=s_dim, o_dim=o_dim, a_dim=a_dim,
            num_agents=num_agents, device=device
        ) for i in range(num_agents)] + \
        [QMIX(
            s_dim=s_dim, o_dim=o_dim, a_dim=a_dim,
            num_agents=num_agents, device=device
        )]
        # temp = [{'params':self.lnet[i].parameter} for i in range(len(gnet) - 1)]
        # temp = temp + self.lnet[-1].parameter
        # params = itertools.chain(temp)
        # self.lnet[-1].get_all_params(params)

    def run(self):
        episode = 1000
        env_info = self.env.get_env_info()
        epsilon = 0.5
        step = 1

        n_actions = env_info["n_actions"]
        n_agents = env_info["n_agents"]
        state_dim = env_info['state_shape']
        obs_dim = env_info['obs_shape']

        for ep in range(episode):
            terminate = False
            self.env.reset()
            ep_rew = 0
            batch_s, batch_s_, batch_a, batch_r, batch_q, batch_m, batch_m_ = [], [], [], [], [], [], []
            batch_o, batch_o_ = [], []
            obs = self.env.get_obs()
            state = self.env.get_state()

            while not terminate:
                a = []
                for i in range(self.num_agents):
                    avail_actions = self.env.get_avail_agent_actions(i)
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                    mask = avail_actions
                    a.append(self.lnet[i].choose_action(obs[i], mask, avail_actions_ind, epsilon=epsilon, step=step))

                if step % 1000 == 0 and epsilon<=0.9:
                    epsilon = epsilon * 1.05

                q = [a[i][1] for i in range(n_agents)]
                a = [a[i][0] for i in range(n_agents)]

                reward, terminate, _ = self.env.step(a)
                # print(step, 'step reward:', reward, 'epsilon', epsilon)
                batch_m.append(mask)
                batch_q.append(q)
                batch_s.append(state)
                batch_a.append(a)
                batch_o.append(obs)
                batch_r.append([reward for i in range(self.num_agents)])
                ep_rew += reward

                obs = self.env.get_obs()
                state = self.env.get_state()
                avail_actions = self.env.get_avail_agent_actions(i)
                mask = avail_actions

                batch_m_.append(mask)
                batch_o_.append(obs)
                batch_s_.append(state)

                push_and_pull_Q(self.lnet, self.gnet, self.opt, self.lr_s, False, batch_s, batch_s_, batch_o, batch_o_, batch_a,
                                  batch_r, batch_q, batch_m, batch_m_, self.gamma, i, device=self.device)

                batch_s, batch_s_, batch_a, batch_r, batch_q, batch_m, batch_m_ = [], [], [], [], [], [], []

                step += 1

            print(ep, '--ep_reward:', ep_rew, '--step:',step)
            if self.name == str(0):
                self.sender.send(['ep_r', ep_rew, ep])
