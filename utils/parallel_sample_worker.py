import torch.multiprocessing as mp
import torch
import numpy as np

# class parallel_sample_worker(mp.Process):
#     def __init__(self, args, agent, env):
#         super(parallel_sample_worker, self).__init__()
#         self.env = env
#         self.agent = agent
#         self.






def init_workers(env_maker, n_workers, agents, args):
    worker = parallel_sample_worker
    envs = [env_maker() for i in range(n_workers)]
    workers = [worker(args, agents[i], envs[i]) for i in range(n_workers)]



