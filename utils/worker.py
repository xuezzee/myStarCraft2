import torch
import torch.multiprocessing as mp
import numpy as np
from utils.update_u import push_and_pull
from agent.A3C_agent import A3C_agent


class worker(mp.Process):
    def __init__(self, env, s_dim, a_dim, o_dim, num_agents, gnet, opt, lr_s, sender, device='cpu',name=None):
        super(worker, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.o_dim = o_dim
        self.num_agents = num_agents
        self.gent = gnet
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
                        push_and_pull(self.lnet[i], self.gent[i], self.opt[i], self.lr_s[i], False, obs[i],
                                      batch_s, batch_a, batch_r, self.gamma, i, device=self.device)

                    batch_s, batch_a, batch_r = [], [], []

                step += 1

            print('ep_reward:', ep_rew)
            if self.name == str(0):
                self.sender.send(['ep_r', ep_rew, ep])




