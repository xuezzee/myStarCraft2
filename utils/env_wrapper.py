import torch
import numpy as np
import gym
from gym.spaces import Discrete, Box

class env_wrapper():
    def __init__(self,env,flatten=True):
        self.env = env
        self.flatten = flatten

    def step(self,actions,need_argmax=True):
        def action_convert(action,need_argmax):
            # action = list(action.values())
            act = {}
            for i in range(len(action)):
                if need_argmax:
                    act["agent-%d"%i] = np.argmax(action[i],0)
                else:
                    act["agent-%d"%i] = action[i]
            return act
        n_state_, n_reward, done, info = self.env.step(action_convert(actions,need_argmax))
        if self.flatten:
            n_state_ = np.array([state.reshape(-1) for state in n_state_.values()])
        else:
            n_state_ = np.array([state.reshape((-1,self.channel,self.width,self.height)) for state in n_state_.values()])
        n_reward = np.array([reward for reward in n_reward.values()])
        done = np.array([d for d in done.values()])
        return n_state_/255., n_reward, done, info

    def reset(self):
        n_state = self.env.reset()
        if self.flatten:
            return np.array([state.reshape(-1) for state in n_state.values()])/255.
        else:
            return np.array([state[np.newaxis,:,:,:].transpose(0,3,1,2) for state in n_state.values()])/255.
            # return np.array([state.reshape((-1,self.channel,self.width,self.height)) for state in n_state.values()])/255.

    def seed(self,seed):
        self.env.seed(seed)

    def render(self, filePath = None):
        self.env.render(filePath)

    @property
    def observation_space(self):
        if self.flatten:
            return Box(0., 1., shape=(675,), dtype=np.float32)
        else:
            return Box(0., 1., shape=(15,15,3), dtype=np.float32)

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def num_agents(self):
        return self.env.num_agents

    @property
    def width(self):
        if not self.flatten:
            return self.observation_space.shape[0]
        else: return None

    @property
    def height(self):
        if not self.flatten:
            return self.observation_space.shape[1]
        else: return None

    @property
    def channel(self):
        if not  self.flatten:
            return self.observation_space.shape[2]
        else: return None