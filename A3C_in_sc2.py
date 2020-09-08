import smac
import torch
import numpy as np
import itertools
from smac.env import StarCraft2Env
from agent.A3C_agent import A3C_agent
from shared_adam import SharedAdam
from utils.worker import worker
import torch.multiprocessing as mp
from utils.logger import Logger, plot
# from agent.QMIX_agent import Q_net, QMIX

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
logger = Logger('./logsA3C3')
def main():
    map_name = ''
    workers = 1
    env = StarCraft2Env(map_name=map_name, difficulty='1')
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    state_dim = env_info['state_shape']
    obs_dim = env_info['obs_shape']

    messager = [mp.Pipe() for i in range(1)]
    sender = [m[0] for m in messager]
    receiver = [m[1] for m in messager]

    opt, lr_s= [], []
    gnet = [A3C_agent(state_dim, obs_dim, n_actions, n_agents, device) for i in range(n_agents)]

    for i in range(n_agents):
        optim = SharedAdam(gnet[i].parameters(), lr=0.001)
        opt.append(optim)
        lr_s.append(
            torch.optim.lr_scheduler.StepLR(optim, step_size=2000, gamma=0.99, last_epoch=-1)
        )

    env_batch = [env] + [StarCraft2Env(map_name=map_name) for i in range(1, workers)]

    worker_batch = [worker(env=env_batch[i],
                           s_dim=state_dim,
                           a_dim=n_actions,
                           o_dim=obs_dim,
                           num_agents=n_agents,
                           gnet=gnet,
                           opt=opt,
                           lr_s=lr_s,
                           sender=sender,
                           device=device,
                           name=str(i)) for i in range(workers)]
    for i in range(workers):
        worker_batch[i].start()

    while True:
        msg = [rec.recv() for rec in receiver]
        logger.scalar_summary(msg[0][0], msg[0][1], msg[0][2])

    [worker.join() for worker in worker_batch]


# def main_QMIX():
#     map_name = '8m'
#     workers = 1
#     env = StarCraft2Env(map_name=map_name, difficulty='1')
#     env_info = env.get_env_info()
#
#     n_actions = env_info["n_actions"]
#     n_agents = env_info["n_agents"]
#     state_dim = env_info['state_shape']
#     obs_dim = env_info['obs_shape']
#
#     messager = [mp.Pipe() for i in range(1)]
#     sender = [m[0] for m in messager]
#     receiver = [m[1] for m in messager]
#
#     gnet = [Q_net(state_dim, obs_dim, n_actions, n_agents, device) for i in range(n_agents)] + \
#            [QMIX(state_dim, obs_dim, n_actions, n_agents, device)]
#
#     temp = [{'params':gnet[i].parameter} for i in range(len(gnet) - 1)]
#     temp = temp + gnet[-1].parameter
#     parameters = itertools.chain(temp)
#     opt = SharedAdam(parameters, lr=0.01)
#     gnet[-1].get_all_params(parameters)
#     lr_s = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.95, last_epoch=-1)
#
#     env_batch = [env] + [StarCraft2Env(map_name=map_name) for i in range(1, workers)]
#
#     worker_batch = [worker_QMIX(env=env_batch[i],
#                            s_dim=state_dim,
#                            a_dim=n_actions,
#                            o_dim=obs_dim,
#                            num_agents=n_agents,
#                            gnet=gnet,
#                            opt=opt,
#                            lr_s=lr_s,
#                            sender=sender,
#                            device=device,
#                            name=str(i)) for i in range(workers)]
#
#     for i in range(workers):
#         worker_batch[i].start()
#
#     while True:
#         msg = [rec.recv() for rec in receiver]
#         logger.scalar_summary(msg[0][0], msg[0][1], msg[0][2])
#
#     [worker.join() for worker in worker_batch]

if __name__ == '__main__':
    mp.set_start_method('spawn', True)
    main()