import torch
import numpy as np
import itertools

def v_wrap(np_array, dtype=np.float32, device='cpu'):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array).to(device)

def push_and_pull(lnet, gnet, opt, lr_s, done, s_, batch_s, batch_a, batch_r, gamma, i, device='cpu'):
    bs = [s[i] for s in batch_s]
    ba = [a[i] for a in batch_a]
    br = [r[i] for r in batch_r]
    # bd = [d[i] for d in batch_d]

    if done:
        v_ = 0
    else:
        _, v_ = lnet.forward(s_)
        v_ = v_.detach()

    buffer_v_target = []
    for r in br[::-1]:
        v_ = r + gamma * v_
        buffer_v_target.append(v_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.stack(np.array(bs), axis=0)),
        v_wrap(np.stack(np.array(ba), axis=0)),
        v_wrap(np.stack(np.array(br), axis=0)),
        v_wrap(np.stack(np.array(buffer_v_target), axis=0))
    )

    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp._grad
    opt.step()

    if not isinstance(lr_s, type(None)):
        lr_s.step()

    lnet.load_state_dict(gnet.state_dict())

def push_and_pull_Q(lnet, gnet, opt, lr_s, done, s_, s, batch_o, batch_o_, batch_a, batch_r,
                    batch_q, batch_m, batch_m_, i, num_agents, device='cpu'):
    q_ = []
    for i in range(len(lnet) - 1):
        bo = batch_o[0][i]
        ba = batch_a[0][i]
        br = batch_r[0][i]
        bq = batch_q[0][i]
        bo_ = batch_o_[0][i]
        bm = batch_m[0]
        bm_ = batch_m_[0]
        q_.append(lnet[i].next_Q(bo_, bm_))

    loss = lnet[-1].loss_func(s, batch_q[0], br, s_, q_)

    opt.zero_grad()
    loss.backward()

    for i in range(num_agents + 1):
        for lp, gp in zip(lnet[i].parameters(), gnet[i].parameters()):
            gp._grad = lp._grad

    opt.step()

    if lr_s != None:
        lr_s.step()
    for i in range(num_agents + 1):
        lnet[i].load_state_dict(gnet[i].state_dict())

    for i in range(num_agents + 1):
        lnet[i].update_target_Q()
        gnet[i].update_target_Q()

