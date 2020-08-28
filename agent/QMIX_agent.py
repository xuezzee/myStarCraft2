import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import copy
import random

class Q_net(nn.Module):
    def __init__(self, s_dim, a_dim, o_dim, num_agents, device='cpu'):
        super(Q_net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.o_dim = o_dim
        self.num_agents = num_agents
        self.device = device
        self.Q = self._build()
        self.Q_target = self._build()
        self.Q_target.load_state_dict(self.Q.state_dict())

    def _build(self):
        Q = nn.Sequential(
            nn.Linear(self.o_dim + self.a_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        return Q

    @property
    def parameter(self):
        return self.Q.parameters()

    def choose_action(self, s, mask, ava_a, epsilon, step):
        def concat(s, a):
            z = torch.zeros((self.a_dim))
            z[a] = 1
            a = z
            s_a = torch.cat([s,a])
            return s_a
        s = self.convert_to_tensor(s)
        mask = self.convert_to_tensor(mask)
        # mask = torch.cat(mask, dim=-1)
        s_a = [concat(s, a).unsqueeze(0) for a in range(self.a_dim)]
        s_a = torch.cat(s_a, dim=0)
        q = self.Q(s_a)
        q = q.view(-1, self.a_dim)
        bias = torch.ones_like(q) * 50000
        # if step % 100 == 0:
            # print(q)
        q = q + bias

        q = torch.mul(q, mask)
        rand = random.random()
        if rand > epsilon:
            a = np.random.choice(ava_a)
            # print('rand', end='')
        else:
            a = int(torch.argmax(q).cpu().detach().numpy())
            # print('policy', end='')
        q = q - bias
        q = q[0][a]

        return a, q

    def next_Q(self, o_, mask_):
        def concat(o, a):
            z = torch.zeros((self.a_dim))
            z[a] = 1
            a = z
            o_a = torch.cat([o,a])
            return o_a

        o_ = self.convert_to_tensor(o_)
        mask_ = self.convert_to_tensor(mask_)

        o_a = [concat(o_, a).unsqueeze(0) for a in range(self.a_dim)]
        o_a = torch.cat(o_a, dim=0)
        q = self.Q_target(o_a)
        q = q.view(-1, self.a_dim)
        bias = torch.ones_like(q) * 5
        q = q + bias
        q = torch.mul(q, mask_)
        a = int(torch.argmax(q).cpu().detach().numpy())
        q = q - bias
        q = q[0][a]

        return q.detach()

    def convert_to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.autograd.Variable(torch.Tensor(x), requires_grad=False)
        if x.device != self.device:
            x = x.to(self.device)
        return x

    def update_target_Q(self):
        self.Q_target.load_state_dict(self.Q.state_dict())


class QMIX(nn.Module):
    def __init__(self, s_dim, a_dim, o_dim, num_agents, device='cpu'):
        super(QMIX, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.o_dim = o_dim
        self.num_agents = num_agents
        self.device = device
        self.hidden_unit = [32, 1]
        self._build()
        self.params = None

    def _build(self):
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.s_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_agents * self.hidden_unit[0])
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.s_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.hidden_unit[0] * self.hidden_unit[1])
        )

        self.hyper_b1 = nn.Linear(self.s_dim, self.hidden_unit[0])
        self.hyper_b2 = nn.Linear(self.s_dim, self.hidden_unit[1])

        self.target_net = {
            'w1':copy.deepcopy(self.hyper_w1),
            'w2':copy.deepcopy(self.hyper_w2),
            'b1':copy.deepcopy(self.hyper_b1),
            'b2':copy.deepcopy(self.hyper_b2),
        }

    @property
    def parameter(self):
        return [
            {'params': self.hyper_w1.parameters()},
            {'params': self.hyper_w2.parameters()},
            {'params': self.hyper_b1.parameters()},
            {'params': self.hyper_b2.parameters()},
        ]

    def get_all_params(self, params):
        self.params = params


    def forward(self, s):
        s = self.convert_to_tensor(s)
        h_w1 = self.hyper_w1(s)
        h_w2 = self.hyper_w2(s)
        h_b1 = self.hyper_b1(s)
        h_b2 = self.hyper_b2(s)
        return h_w1, h_w2, h_b1, h_b2

    def forward_target(self, s):
        s = self.convert_to_tensor(s)
        h_w1 = self.target_net['w1'](s)
        h_w2 = self.target_net['w2'](s)
        h_b1 = self.target_net['b1'](s)
        h_b2 = self.target_net['b2'](s)
        return h_w1, h_w2, h_b1, h_b2

    def cal_Q_tot(self, s, q, target=False):
        s = self.convert_to_tensor(s)
        q = self.convert_to_tensor(q)
        if target:
            h_w1, h_w2, h_b1, h_b2 = self.forward_target(s)
        else:
            h_w1, h_w2, h_b1, h_b2 = self.forward(s)
        h_w1 = h_w1.view(-1, self.hidden_unit[0])
        h_w2 = h_w2.view(-1, self.hidden_unit[1])
        h_b1 = h_b1.view(-1, self.hidden_unit[0])
        h_b2 = h_b2.view(-1, self.hidden_unit[1])
        h1_out = F.elu(torch.matmul(q, h_w1) + h_b1)
        h2_out = F.elu(torch.matmul(h1_out, h_w2) + h_b2)
        return h2_out

    def loss_func(self, s, q, r, s_, q_):
        s = self.convert_to_tensor(s)
        q = torch.cat([x.unsqueeze(0) for x in q]).unsqueeze(0)
        r = self.convert_to_tensor(np.array(r))
        s_ = self.convert_to_tensor(s_)
        q_ = self.convert_to_tensor(q_).unsqueeze(0)
        q_tot_pred = self.cal_Q_tot(s=s, q=q, target=False)
        q_tot_target = self.cal_Q_tot(s=s_,q=q_, target=True).detach()
        loss = (0.9 * q_tot_target + r - q_tot_pred).pow(2).mean()
        # print(loss)

        return loss


    def convert_to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.autograd.Variable(torch.Tensor(x), requires_grad=False)
        if x.device != self.device:
            x = x.to(self.device)
        return x

    def update_target_Q(self):
        self.target_net = {
            'w1':copy.deepcopy(self.hyper_w1),
            'w2':copy.deepcopy(self.hyper_w2),
            'b1':copy.deepcopy(self.hyper_b1),
            'b2':copy.deepcopy(self.hyper_b2),
        }


