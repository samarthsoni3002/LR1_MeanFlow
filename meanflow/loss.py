import torch


def sample_r_t(batch_size, device, same_time_prob=0.4):
    a = torch.rand(batch_size, device=device)
    b = torch.rand(batch_size, device=device)

    t = torch.maximum(a, b)
    r = torch.minimum(a, b)
    
    same_mask = torch.rand(batch_size, device=device) < same_time_prob
    r = torch.where(same_mask, t, r)

    return r, t


def meanflow_loss(
    model,
    x,
    y=None,
    same_time_prob=0.4,
):
    B = x.shape[0]
    device = x.device

    eps = torch.randn_like(x)

    r, t = sample_r_t(
        B,
        device=device,
        same_time_prob=same_time_prob,
    )


    t_img = t[:, None, None, None]
    z_t = (1.0 - t_img) * x + t_img * eps

    v = eps - x

    def fn(z_in, r_in, t_in):
        return model(z_in, r_in, t_in, y)

    u, dudt = torch.func.jvp(
        fn,
        (z_t, r, t),
        (v, torch.zeros_like(r), torch.ones_like(t)),
    )

    dt = (t - r)[:, None, None, None]
    u_tgt = v - dt * dudt

    loss = ((u - u_tgt.detach()) ** 2).mean()

    return loss, loss.detach()