import numpy as np
import torch
import smac
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class A3C_agent(nn.Module):
    def __init__(self, s_dim, o_dim, a_dim, num_agents, device='cpu'):
        super(A3C_agent, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.o_dim = o_dim
        self.num_agents = num_agents
        self.device = device
        self._build()

    def _build(self):
        self.policy = nn.Sequential(
            nn.Linear(self.o_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.a_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(self.o_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s):
        s = self.transform_to_Tensor(s)
        pi = self.policy(s)
        v = self.value(s)
        return pi, v

    def choose_action(self, s, mask):
        s = self.transform_to_Tensor(s)
        mask = self.transform_to_Tensor(mask)

        pi, _ = self.forward(s)
        pi = F.softmax(pi)
        pi = torch.mul(pi, mask)
        m = Categorical(pi)
        a = m.sample()
        return int(a.data.cpu().numpy())

    def loss_func(self, s, a, r, v_):
        s = self.transform_to_Tensor(s)
        a = self.transform_to_Tensor(a)

        pi, v = self.forward(s)
        td_err = v_.unsqueeze(-1) - v
        loss_C = td_err.pow(2)
        prob = F.softmax(pi)
        m = Categorical(prob)
        temp = m.log_prob(a).unsqueeze(-1)
        exp_v = -temp * (td_err.detach() + 0.001 * self.entropy(prob))
        loss_A = exp_v

        loss = (loss_C + loss_A).mean()
        return loss

    def entropy(self, distribution):
        entropy = []
        for distribution_per_step in distribution:
            sum = 0
            for prob in distribution_per_step:
                sum -= prob * torch.log(prob)
            entropy.append(sum.unsqueeze(0))
        return torch.cat(entropy).unsqueeze(0).to(self.device)

    def transform_to_Tensor(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.autograd.Variable(torch.Tensor(x), requires_grad=False)
        if x.device != self.device:
            x = x.to(self.device)
        return x




