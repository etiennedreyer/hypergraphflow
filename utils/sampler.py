import torch

@torch.inference_mode()
def euler_sampler(model, x_0, condition, steps=50, save_seq=False):

    dt = 1 / steps
    times = torch.arange(0, 1, dt, device=x_0.device)

    x_t = x_0
    seq = [x_t]

    for t in times:
        dxdt = model(x_t, t.unsqueeze(0), condition)
        x_t = x_t + dxdt * dt
        seq.append(x_t)

    if save_seq:
        return x_t, torch.stack(seq)
    else:
        return x_t