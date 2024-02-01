from itertools import product
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from spectral_rescaling import compute_spectral_rescaling_conv


###############################################################################


def estimate(X, n=32, n_iter=5, name_func="delattre2023", return_time=False):
    """Estimate spectral norm of convolutional layer with a specific method.

    From a convolutional filter, this function estimates the spectral norm, ie
    the largest singular value, of the convolutional layer.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, h, w)
        Convolutional filter.
    n_iter : int, default=50
        Number of iterations.
    name_func : string, default="ours"
        Name of the method.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.
    """

    if name_func == "delattre2024":
        sigma = compute_delattre2024(X, n_iter=n_iter, return_time=return_time)
    elif name_func == "delattre2023":
        sigma = compute_delattre2023(X, n=n, n_iter=n_iter, return_time=return_time)
    elif name_func == "delattre2023_backward":
        sigma = compute_delattre2023_backward(
            X, n=n, n_iter=n_iter, return_time=return_time
        )
    elif name_func == "araujo2021":
        sigma = compute_araujo2021(X, n_iter=n_iter, return_time=return_time)
    elif name_func == "singla2021":
        sigma = compute_singla2021(X, n_iter=n_iter, return_time=return_time)
    elif name_func == "sedghi2019":
        sigma = compute_sedghi_2019(X, n=n, return_time=return_time)
    elif name_func == "ryu2019":
        sigma = compute_ryu_2019(X, n=n, n_iter=n_iter, return_time=return_time)
    else:
        raise ValueError(f"{name_func} method not implemented")
    return sigma


def estimate_dense(X, n_iter=5, name_func="gi", return_time=False):
    """Estimate spectral norm of dense layer with a specific method.

    From a matrix, this function estimates the spectral norm, ie the largest
    singular value, of the dense layer.

    Parameters
    ----------
    X : ndarray, shape (dim0, dim1)
        Matrix.
    n_iter : int, default=5
        Number of iterations.
    name_func : string, default="ours"
        Name of the method.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.
    """

    if name_func == "gi":
        sigma = gram_iteration_on_matrix(X, n_iter=n_iter, return_time=return_time)
    elif name_func == "gi_backward":
        sigma = gram_iteration_on_matrix_explicit_backward(
            X, n_iter=n_iter, return_time=return_time
        )
    elif name_func == "pi":
        sigma = compute_pm_dense(X, n_iter=n_iter, return_time=return_time)
    else:
        raise ValueError(f"{name_func} method not implemented")
    return sigma


###############################################################################
# Delattre2024
###############################################################################


def compute_delattre2024(X, n_iter=4, return_time=True):
    """Estimate spectral norm of convolutional layer with Delattre2024.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer with zero padding using Delattre2024.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, k, k)
        Convolutional filter.
    n_iter : int, default=4
        Number of iterations.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.

    References
    ----------
    .. [1] `Spectral Norm of Convolutional Layers with Circular and Zero Paddings
        <TODO>`_
        TODO, arXiv, 2024
    """
    cout, cin, _, _ = X.shape
    if cin > cout:
        X = X.transpose(0, 1)
    start_time = time.time()
    rescale_weights = compute_spectral_rescaling_conv(X, n_iter)
    sigma = rescale_weights.max()
    total_time = time.time() - start_time

    if return_time:
        return sigma, total_time
    else:
        return sigma



###############################################################################
# Delattre2023
###############################################################################


def compute_delattre2023(X, n=None, n_iter=4, return_time=True):
    """Estimate spectral norm of convolutional layer with Delattre2023.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer for circular padding using [Section.3, Algo. 3] Delattre2023.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, k, k)
        Convolutional filter.
    n : None | int, default=None
        Size of input image. If None, n is set equal to k.
    n_iter : int, default=4
        Number of iterations.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.
    """
    cout, cin, k, _ = X.shape
    if n is None:
        n = k
    if cin > cout:
        X = X.transpose(0, 1)
        cin, cout = cout, cin
    start_time = time.time()

    crossed_term = (
        torch.fft.rfft2(X, s=(n, n)).reshape(cout, cin, -1).permute(2, 0, 1)
    )
    inverse_power = 1
    log_curr_norm = torch.zeros(crossed_term.shape[0]).cuda()
    for _ in range(n_iter):
        norm_crossed_term = crossed_term.norm(dim=(1, 2))
        crossed_term /= norm_crossed_term.reshape(-1, 1, 1)
        log_curr_norm = 2 * log_curr_norm + norm_crossed_term.log()
        crossed_term = torch.bmm(crossed_term.conj().transpose(1, 2), crossed_term)
        inverse_power /= 2
    sigma = (
        crossed_term.norm(dim=(1, 2)).pow(inverse_power)
        * ((2 * inverse_power * log_curr_norm).exp())
    ).max()
    total_time = time.time() - start_time

    if return_time:
        return sigma, total_time
    else:
        return sigma


class GramIterationConvBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rfft_kernel, n_iter):
        with torch.no_grad():
            inverse_power = 1
            Gt = rfft_kernel
            log_curr_norm = 0.0
            for _ in range(n_iter):
                norm_Gt = Gt.norm(dim=(1, 2))
                Gt = Gt / norm_Gt.reshape(-1, 1, 1)
                log_curr_norm = 2 * (log_curr_norm + norm_Gt.log())
                Gt = torch.bmm(Gt.conj().transpose(1, 2), Gt)
                inverse_power /= 2

            sqrt_Gt_norm = Gt.norm(dim=(1, 2)).pow(inverse_power) * (
                (inverse_power * log_curr_norm).exp()
            )
            max_sqrt_Gt_norm, idx = sqrt_Gt_norm.max(dim=0)

            ctx.save_for_backward(
                rfft_kernel[idx, :, :].detach(), Gt[idx, :, :].detach()
            )
            ctx.idx = idx.item()
            ctx.norm_Gt = max_sqrt_Gt_norm.detach()
            ctx.n_iter = n_iter
            ctx.dtype = rfft_kernel.dtype
            ctx.device = rfft_kernel.device
            ctx.shape_input = rfft_kernel.shape

            return max_sqrt_Gt_norm

    @staticmethod
    def backward(ctx, grad_output):
        G, Gt = ctx.saved_tensors
        idx = ctx.idx
        n_iter = ctx.n_iter
        grad_input = torch.zeros(ctx.shape_input, dtype=ctx.dtype, device=ctx.device)
        if n_iter == 0:
            jac = G / ctx.norm_Gt
        else:
            norm_G_sq = ctx.norm_Gt**2
            Gdag_G = (G.conj().t() @ G) / norm_G_sq
            num = G @ Gdag_G.matrix_power(2**n_iter - 1)
            denom = Gdag_G.matrix_power(2 ** (n_iter - 1)).norm().pow(2 - 0.5**n_iter)
            jac = (num / denom) * norm_G_sq ** (-0.5)
        grad_input[idx, :, :] = jac
        return grad_output * grad_input, None


gram_iteration_conv_backward = GramIterationConvBackward.apply


def compute_delattre2023_backward(kernel, n, n_iter=4, return_time=False):
    """Estimate spectral norm of convolutional layer with Delattre2023.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer for circular padding using 
    [Section.3, Algo. 3] Delattre2023, with explicit backward implementation.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, k, k)
        Convolutional filter.
    n_iter : int, default=4
        Number of iterations.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.
    """
    cout, cin, _, _ = kernel.shape
    start_time = time.time()
    if cin > cout:
        kernel = kernel.transpose(0, 1)
        cin, cout = cout, cin
    crossed_term = torch.fft.rfft2(kernel, s=(n, n))
    crossed_term = crossed_term.reshape(cout, cin, -1).permute(2, 0, 1)
    sigma = gram_iteration_conv_backward(crossed_term, n_iter)
    total_time = time.time() - start_time

    if return_time:
        return sigma, total_time
    else:
        return sigma


###############################################################################
# Gram iteration
###############################################################################


def gram_iteration_on_matrix(M, n_iter=100, return_time=True):
    """Gram iteration on matrix, ie dense layer."""
    n, m = M.shape
    if m > n:
        M = M.T
    inverse_power = 1
    log_curr_norm = 0
    start = time.time()
    for _ in range(n_iter):
        M_norm = M.norm()
        M = M / M_norm
        M = M.T.mm(M)
        log_curr_norm = 2 * (log_curr_norm + M_norm.log())
        inverse_power /= 2
    res = M.norm().pow(inverse_power) * ((log_curr_norm * inverse_power).exp())
    total_time = time.time() - start
    if return_time:
        return res, total_time
    else:
        return res


class GramIterationDenseBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, G, n_iter):
        with torch.no_grad():
            inverse_power = 1
            Gt = G
            log_curr_norm = 0.0
            for _ in range(n_iter):
                Gt_norm = Gt.norm()
                Gt = Gt / Gt_norm
                Gt = Gt.T.mm(Gt)
                log_curr_norm = 2 * (log_curr_norm + Gt_norm.log())
                inverse_power /= 2
                sqrt_Gt_norm = Gt.norm().pow(inverse_power) * (
                    (inverse_power * log_curr_norm).exp()
                )
                ctx.save_for_backward(G, Gt)
                ctx.n_iter = n_iter
                ctx.norm_Gt = sqrt_Gt_norm
                return sqrt_Gt_norm

    @staticmethod
    def backward(ctx, grad_output):
        G, Gt = ctx.saved_tensors
        n_iter = ctx.n_iter
        if n_iter == 0:
            jac = G / ctx.norm_Gt
        elif n_iter == 1:
            jac = G @ Gt * ctx.norm_Gt.pow(-1.5)
        elif n_iter == 2:
            jac = G @ Gt @ G.T @ G * ctx.norm_Gt.pow(-1.75)
        elif n_iter == 3:
            Gdag_G = G.T @ G
            jac = G @ Gt @ Gdag_G.matrix_power(3) * ctx.norm_Gt.pow(-1.875)
        else:
            norm_G_sq = ctx.norm_Gt**2
            Gdag_G = (G.T @ G) / norm_G_sq
            num = G @ Gdag_G.matrix_power(2**n_iter - 1)
            denom = Gdag_G.matrix_power(2 ** (n_iter - 1)).norm().pow(2 - 0.5**n_iter)
            jac = (num / denom) * norm_G_sq ** (-0.5)
        return grad_output * jac, None


gram_iteration_dense_backward = GramIterationDenseBackward.apply


def gram_iteration_on_matrix_explicit_backward(M, n_iter=100, return_time=True):
    """Gram iteration on matrix, ie dense layer,  with explicit backward
    implementation."""
    start = time.time()
    n, m = M.shape
    if m > n:
        M = M.T
    res = gram_iteration_dense_backward(M, n_iter)
    total_time = time.time() - start
    if return_time:
        return res, total_time
    else:
        return res


###############################################################################
# Singla2021
###############################################################################


def compute_singla2021(X, n_iter=50, return_time=True, device="cuda"):
    """Estimate spectral norm of convolutional layer with circular padding 
    with Singla2021.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer using Singla2021 [1]_.

    Code adapted from [2]_.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, h, w)
        Convolutional filter.
    n_iter : int, default=50
        Number of iterations.
    return_time : bool, default True
        Return computational time.
    device : {"cuda", "cpu"}, default "cuda"
        Device.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.

    References
    ----------
    .. [1] `Fantastic Four Differentiable Bounds on Singular Values of
        Convolution Layers
        <https://par.nsf.gov/servlets/purl/10315555>`_
        S Singla & S Feizi, ICLR, 2021
    .. [2] https://github.com/singlasahil14/fantastic-four
    """
    start_time = time.time()
    cout, cint, h, w = X.shape

    permute1 = torch.transpose(X, 1, 2)
    matrix1 = permute1.reshape(cout * h, cint * w)
    u1 = torch.randn(matrix1.shape[1], device=device, requires_grad=False)
    v1 = torch.randn(matrix1.shape[0], device=device, requires_grad=False)

    permute2 = torch.transpose(X, 1, 3)
    matrix2 = permute2.reshape(cout * w, cint * h)
    u2 = torch.randn(matrix2.shape[1], device=device, requires_grad=False)
    v2 = torch.randn(matrix2.shape[0], device=device, requires_grad=False)

    permute3 = X
    matrix3 = permute3.reshape(cout, cint * h * w)
    u3 = torch.randn(matrix3.shape[1], device=device, requires_grad=False)
    v3 = torch.randn(matrix3.shape[0], device=device, requires_grad=False)

    permute4 = torch.transpose(X, 0, 1)
    matrix4 = permute4.reshape(cint, cout * h * w)
    u4 = torch.randn(matrix4.shape[1], device=device, requires_grad=False)
    v4 = torch.randn(matrix4.shape[0], device=device, requires_grad=False)

    for _ in range(n_iter):
        v1.data = F.normalize(torch.mv(matrix1.data, u1.data), dim=0)
        u1.data = F.normalize(torch.mv(torch.t(matrix1.data), v1.data), dim=0)

        v2.data = F.normalize(torch.mv(matrix2.data, u2.data), dim=0)
        u2.data = F.normalize(torch.mv(torch.t(matrix2.data), v2.data), dim=0)

        v3.data = F.normalize(torch.mv(matrix3.data, u3.data), dim=0)
        u3.data = F.normalize(torch.mv(torch.t(matrix3.data), v3.data), dim=0)

        v4.data = F.normalize(torch.mv(matrix4.data, u4.data), dim=0)
        u4.data = F.normalize(torch.mv(torch.t(matrix4.data), v4.data), dim=0)

    sigma1 = torch.mv(v1.unsqueeze(0), torch.mv(matrix1, u1))
    sigma2 = torch.mv(v2.unsqueeze(0), torch.mv(matrix2, u2))
    sigma3 = torch.mv(v3.unsqueeze(0), torch.mv(matrix3, u3))
    sigma4 = torch.mv(v4.unsqueeze(0), torch.mv(matrix4, u4))

    sigma = math.sqrt(h * w) * (
        torch.min(torch.stack([sigma1, sigma2, sigma3, sigma4]))
    )
    total_time = time.time() - start_time

    if return_time:
        return sigma, total_time
    else:
        return sigma


###############################################################################
# Araujo2021
###############################################################################


def compute_araujo2021(X, n_iter=50, *, padding=0, cuda=True, return_time=True):
    """Estimate spectral norm of convolutional layer for circular and zero padding 
    with Araujo2021.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer using Araujo2021 [1]_.

    Code taken from [2]_, algo LipGrid with v2 implementation.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, k, k)
        Convolutional filter.
    n_iter : int, default=50
        Number of samples.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.

    References
    ----------
    .. [1] `On Lipschitz Regularization of Convolutional Layers using Toeplitz
        Matrix Theory
        <https://arxiv.org/abs/2006.08391>`_
        A Araujo, B Negrevergne, Y Chevaleyre & Jamal Atif, AAAI, 2021
    .. [2] https://github.com/MILES-PSL/Upper-Bound-Lipschitz-Convolutional-Layers/blob/master/lipschitz_bound/lipschitz_bound.py
    """
    cout, cin, k, k2 = X.shape
    if k != k2:  # verify if kernel is square
        raise ValueError("The last 2 dim of the kernel must be equal.")
    if not k % 2 == 1:  # verify if kernel have odd shape
        raise ValueError("The dimension of the kernel must be odd.")
    device = X.device
    n_sample = n_iter
    start_time = time.time()
    # special case kernel 1x1
    if k == 1:
        ker = X.reshape(-1)
        res = torch.sqrt(torch.einsum("i,i->", ker, ker))
        res = res
        total_time = time.time() - start_time
        if return_time:
            return res, total_time
        else:
            return res
    # define search space
    x = np.linspace(0, 2 * np.pi, num=n_sample)
    w = np.array(list(product(x, x)))
    w0 = w[:, 0].reshape(-1, 1)
    w1 = w[:, 1].reshape(-1, 1)
    w0 = torch.FloatTensor(np.float32(w0))
    w1 = torch.FloatTensor(np.float32(w1))
    if cuda:
        w0 = w0.cuda()
        w1 = w1.cuda()
    p_index = torch.arange(-k + 1.0, 1.0) + padding
    H0 = p_index.repeat(k).reshape(k, k).T.reshape(-1)
    H1 = p_index.repeat(k)
    if cuda:
        H0 = H0.cuda()
        H1 = H1.cuda()
    real = torch.cos(w0 * H0 + w1 * H1).T
    imag = torch.sin(w0 * H0 + w1 * H1).T
    samples = (real, imag)
    real, imag = samples
    real = real.to(device)
    imag = imag.to(device)
    ker = X.reshape(cout * cin, -1)
    poly_real = torch.matmul(ker, real).view(cout, cin, -1)
    poly_imag = torch.matmul(ker, imag).view(cout, cin, -1)
    poly1 = torch.einsum("ijk,ijk->k", poly_real, poly_real)
    poly2 = torch.einsum("ijk,ijk->k", poly_imag, poly_imag)
    poly = poly1 + poly2
    sv_max = torch.sqrt(poly.max())
    d = (k - 1) / 2
    denom = 1 - (2 * d) / n_sample
    if denom:
        alpha = 1 / denom
    else:
        alpha = 1
    res = alpha * sv_max
    total_time = time.time() - start_time

    if return_time:
        return res, total_time
    else:
        return res


###############################################################################
# Sedghi2019
###############################################################################


def compute_sedghi_2019(X, n=None, n_iter=None, return_time=True):
    """Estimate spectral norm of convolutional layer with Sedghi2019 for circular padding.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer using Sedghi2019 [1]_.

    Code adapted from [2]_.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, h, w)
        Convolutional filter.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.

    References
    ----------
    .. [1] `The Singular Values of Convolutional Layers
        <https://arxiv.org/abs/1805.10408>`_
        H Sedghi, V Gupta & P M Long, ICLR, 2019
    .. [2] https://github.com/brain-research/conv-sv/blob/master/conv2d_singular_values.py
    """
    cout, cin, _, _ = X.shape
    start_time = time.time()
    X = torch.permute(X, (2, 3, 0, 1))
    fft_X = torch.fft.fft2(X, s=(n, n), dim=(0, 1))
    sigma = torch.linalg.matrix_norm(fft_X, ord=2, dim=(2, 3)).max()
    total_time = time.time() - start_time

    if return_time:
        return sigma, total_time
    else:
        return sigma


###############################################################################
#  Ryu2019
###############################################################################


def normalize(arr):
    norm = torch.sqrt((arr**2).sum())
    return arr / (norm + 1e-12)


def compute_ryu_2019(X, n, n_iter=100, eps=1e-8, return_time=True):
    """Estimate spectral norm of convolutional layer with zero padding Ryu2019.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer using Ryu2019 [1]_ or Farnia2019.

    Code adapted from [2]_.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, h, w)
        Convolutional filter.
    n_iter : int, default=100
        Number of iterations.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.

    References
    ----------
    .. [1] `Plug-and-Play Methods Provably Converge with Properly Trained
        Denoisers
        <https://par.nsf.gov/servlets/purl/10315555>`_
        E K Ryu, J Liu, S Wang, X Chen, Z Wang & W Yin, ICML, 2019
    .. [2] https://github.com/uclaopt/Provable_Plug_and_Play/blob/master/model/Spectral_Normalize.py

    """
    start_time = time.time()
    cout, cin, _, _ = X.shape
    input_size = (1, cin, n, n)
    u = torch.randn(input_size, dtype=X.dtype, device=X.device)
    u = u / u.norm(p=2)
    pad = (1, 1, 1, 1)
    pad_ = (-1, -1, -1, -1)
    for _ in range(n_iter):
        v = normalize(F.conv2d(F.pad(u, pad), X))
        u.data = normalize(F.pad(F.conv_transpose2d(v, X), pad_))
    u_hat, v_hat = u, v
    z = F.conv2d(F.pad(u_hat, pad), X)
    sigma = torch.mul(z, v_hat).sum()
    total_time = time.time() - start_time

    if return_time:
        return sigma, total_time
    else:
        return sigma


###############################################################################
# Power method
###############################################################################


def compute_pm_dense(M, n_iter=50, return_time=True):
    """Power iteration for matrix, ie dense layer."""
    u = torch.rand(M.shape[1], dtype=M.dtype, device=M.device)
    start = time.time()
    for idx in range(n_iter):
        v = M @ u
        v = v / torch.norm(v, p=2)
        u = M.T @ v
        u = u / torch.norm(u, p=2)
    z = M @ u
    res = torch.inner(z, v)
    total_time = time.time() - start

    if return_time:
        return res, total_time
    else:
        return res
