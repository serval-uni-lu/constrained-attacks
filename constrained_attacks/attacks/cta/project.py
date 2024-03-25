import torch


# This projection was created for debugging purposes. It is not used in the attacks (yet).
def l2(x, x_adv, mask, eps, device, add_eps):

    perturbation = x_adv - x
    perturbation = perturbation * mask.float()

    f_common = (
        ((perturbation) ** 2)
        .sum(dim=list(range(1, len(x.shape))), keepdim=True)
        .sqrt()
    )

    f1 = f_common + 1e-12

    f2a = eps * torch.ones(x.shape).to(device).detach()

    f2b = f_common
    if add_eps:
        f2b = f2b + 1e-12

    factor = f1 * torch.min(
        f2a,
        f2b,
    )

    perturbation = perturbation / factor

    return torch.clamp(x + perturbation, 0.0, 1.0)
