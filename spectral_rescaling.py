import torch
import torch.nn.functional as F


def compute_spectral_rescaling_dense(M, n_iter=1):
    n_iter = max(n_iter, 1)
    log_curr_norm = 0
    inverse_power = 2**(-n_iter)
    for _ in range(n_iter):
        M_norm = M.norm().detach()
        M = M / M_norm
        M = M.mm(M.T)
        log_curr_norm = 2 * (log_curr_norm + M_norm.log())
    t = M.abs().sum(1)
    norm = ((log_curr_norm * inverse_power).exp())
    t = t.pow(inverse_power)
    t = t * norm
    return t


def compute_spectral_rescaling_residual_dense_with_q(M, q, q_inv, n_iter=1):
    n_iter = max(n_iter, 1)
    log_curr_norm = 0
    inverse_power = 2**(-(n_iter-1))
    for _ in range(n_iter):
        M_norm = M.norm().detach()
        M = M / M_norm
        M = M.mm(M.T)
        log_curr_norm = 2 * (log_curr_norm + M_norm.log())
    t = (q_inv.reshape(-1, 1) * M.abs() * q.reshape(1, -1)).sum(1)
    norm = ((log_curr_norm * 2**(-(n_iter-1))).exp())
    t = t.pow(inverse_power)
    t = t * norm
    return t


def compute_spectral_rescaling_residual_conv_with_q(kernel, q, q_inv, n_iter=1):
    n_iter = max(n_iter, 1)
    effective_iter = 0
    kkt = kernel
    log_curr_norm = 0
    for _ in range(n_iter):
        padding = kkt.shape[-1] - 1
        kkt_norm = kkt.norm().detach()
        kkt = kkt / kkt_norm
        log_curr_norm = 2 * (log_curr_norm + kkt_norm.log())
        kkt = F.conv2d(kkt, kkt, padding=padding)
        effective_iter += 1
    inverse_power = 2**(-(effective_iter-1))
    t = torch.abs(kkt)
    t = (q_inv * t * q).sum(dim=(1, 2, 3)).pow(inverse_power)
    norm = torch.exp(log_curr_norm * inverse_power)
    t = t*norm
    return t


def compute_spectral_rescaling_conv(kernel, n_iter=1):
    n_iter = max(n_iter, 1)
    effective_iter = 0
    kkt = kernel
    log_curr_norm = 0
    for _ in range(n_iter):
        padding = kkt.shape[-1] - 1
        kkt_norm = kkt.norm().detach()
        kkt = kkt / kkt_norm
        log_curr_norm = 2 * (log_curr_norm + kkt_norm.log())
        kkt = F.conv2d(kkt, kkt, padding=padding)
        effective_iter += 1
    inverse_power = 2**(-effective_iter)
    t = torch.abs(kkt)
    t = t.sum(dim=(1, 2, 3)).pow(inverse_power)
    norm = torch.exp(log_curr_norm * inverse_power)
    t = t*norm
    return t
