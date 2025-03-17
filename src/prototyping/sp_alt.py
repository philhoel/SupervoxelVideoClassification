import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from typing import Sequence, Union, Optional, Tuple
from scipy.sparse import coo_matrix as cpu_coo_matrix
from scipy.sparse.csgraph import connected_components as cpu_concom
from matplotlib.animation import FuncAnimation
from matplotlib import colors

from time import perf_counter


class InterpolationExtractor(nn.Module):

    '''Extract interpolated features from volume given bounding boxes and segmentation.
    Attributes
    ----------
    patch_size : Tensor
        Desired size of the features we want to extract (cube with patch_size**3).
    channels : int
        Num. channels
    return_masks : bool
        Whether to return the masks for the segmentation within each bounding box
    _vi : Tensor
        Shape of trilinear output, precomputed
    _vm : Tensor
        Shape of trilinear mask, precomputed
    dims : Tensor (buffer)
        An arange for num channels
    ygrid : Tensor (buffer)
        Self explanatory, shape [1, P^3, 1]
    xgrid : Tensor (buffer)
        Self explanatory, shape [1, P^3, 1]
    zgrid : Tensor (buffer)
        Self explanatory, shape [1, P^3, 1]
    '''

    def __init__(self, space_patch, time_patch, channels, return_masks=True):
        super().__init__()
        # self.patch_size = patch_size
        self.space_patch = space_patch
        self.time_patch = time_patch
        self.channels = channels
        self.return_masks = return_masks
        grid_base = torch.linspace(0, 1, space_patch)
        grid_base_z = torch.linspace(0, 1, time_patch)
        self._vi = (-1, self.time_patch, self.space_patch,
                    self.space_patch, channels)
        self._vm = (-1, self.time_patch, self.space_patch, self.space_patch, 1)
        ygrid, xgrid, zgrid = torch.meshgrid(
            grid_base_z, grid_base, grid_base, indexing='ij')
        self.register_buffer('dims', torch.arange(channels), persistent=False)
        self.register_buffer(
            'ygrid', ygrid.reshape(-1, self.space_patch**2 * self.time_patch, 1), persistent=False)
        self.register_buffer(
            'xgrid', xgrid.reshape(-1, self.space_patch**2 * self.time_patch, 1), persistent=False)
        self.register_buffer(
            'zgrid', zgrid.reshape(-1, self.space_patch**2 * self.time_patch, 1), persistent=False)

    def forward(
        self, flatvid: torch.Tensor, seg: torch.Tensor, coord: torch.Tensor, bbox: torch.Tensor
    ):
        '''Forward function.
        Parameters
        ----------
        flatvid : Tensor
            The flattened video tensor, of shape [BTHW, C]
        seg : Tensor
            Supervoxel segmentation [B,T,H,W]
        coord : Tensor
            Voxel indices, shape [4, BTHW]
        bbox : Tensor
            Bounding boxes, of shape [6, N] where N is num_regions.
        '''
        B, T, H, W = seg.shape
        C = flatvid.shape[-1]
        device = flatvid.device

        # Setup
        b = coord[0]
        b_idx = seg.view(-1).mul(B).add(b).unique() % B  # Shape [N]
        # Shape [N, P^3, 1]
        b_idx = b_idx.view(-1, 1, 1).expand(-1,
                                            self.space_patch**2 * self.time_patch, -1)
        c_idx = self.dims.view(
            1, 1, -1).expand(*b_idx.shape[:2], -1)  # type: ignore
        # Shape [N, P^3, C]
        img = flatvid.view(B, T, H, W, -1)
        ymin, xmin, zmin, ymax, xmax, zmax = bbox.view(6, -1, 1, 1)
        t_pos = self.ygrid * (ymax - ymin) + ymin
        h_pos = self.xgrid * (xmax - xmin) + xmin
        w_pos = self.zgrid * (zmax - zmin) + zmin

        # Construct lower and upper bounds
        t_floor = t_pos.floor().long().clamp(0, T-1)
        h_floor = h_pos.floor().long().clamp(0, H-1)
        w_floor = w_pos.floor().long().clamp(0, W-1)
        t_ceil = (t_floor + 1).clamp(0, T-1)
        h_ceil = (h_floor + 1).clamp(0, H-1)
        w_ceil = (w_floor + 1).clamp(0, W-1)

        # Construct fractional parts of bilinear coordinates
        Ut, Uh, Uw = t_pos - t_floor, h_pos - h_floor, w_pos - w_floor
        Lt, Lh, Lw = 1 - Ut, 1 - Uh, 1 - Uw
        tfhfwf, tfhfwc, tfhcwf, tfhcwc = Lt*Lh*Lw, Lt*Lh*Uw, Lt*Uh*Lw, Lt*Uh*Uw
        tchfwf, tchfwc, tchcwf, tchcwc = Ut*Lh*Lw, Ut*Lh*Uw, Ut*Uh*Lw, Ut*Uh*Uw

        # Get interpolated features
        trilinear = (
            img[b_idx, t_floor, h_floor, w_floor, c_idx] * tfhfwf +
            img[b_idx, t_floor, h_floor, w_ceil, c_idx] * tfhfwc +
            img[b_idx, t_floor, h_ceil,  w_floor, c_idx] * tfhcwf +
            img[b_idx, t_floor, h_ceil,  w_ceil, c_idx] * tfhcwc +
            img[b_idx, t_ceil, h_floor, w_floor, c_idx] * tchfwf +
            img[b_idx, t_ceil, h_floor, w_ceil, c_idx] * tchfwc +
            img[b_idx, t_ceil, h_ceil,  w_floor, c_idx] * tchcwf +
            img[b_idx, t_ceil, h_ceil,  w_ceil, c_idx] * tchcwc
        ).view(*self._vi).permute(0, 4, 1, 2, 3)

        if not self.return_masks:
            return trilinear

        # Get masks
        srange = torch.arange(b_idx.shape[0], device=device).view(-1, 1)
        masks = (
            (seg[b_idx[..., 0], t_floor[..., 0], h_floor[..., 0], w_floor[..., 0]] == srange).unsqueeze(-1) * tfhfwf +
            (seg[b_idx[..., 0], t_floor[..., 0], h_floor[..., 0], w_ceil[..., 0]] == srange).unsqueeze(-1) * tfhfwc +
            (seg[b_idx[..., 0], t_floor[..., 0], h_ceil[..., 0], w_floor[..., 0]] == srange).unsqueeze(-1) * tfhcwf +
            (seg[b_idx[..., 0], t_floor[..., 0], h_ceil[..., 0], w_ceil[..., 0]] == srange).unsqueeze(-1) * tfhcwc +
            (seg[b_idx[..., 0], t_ceil[..., 0], h_floor[..., 0], w_floor[..., 0]] == srange).unsqueeze(-1) * tchfwf +
            (seg[b_idx[..., 0], t_ceil[..., 0], h_floor[..., 0], w_ceil[..., 0]] == srange).unsqueeze(-1) * tchfwc +
            (seg[b_idx[..., 0], t_ceil[..., 0], h_ceil[..., 0], w_floor[..., 0]] == srange).unsqueeze(-1) * tchcwf +
            (seg[b_idx[..., 0], t_ceil[..., 0], h_ceil[..., 0],
             w_ceil[..., 0]] == srange).unsqueeze(-1) * tchcwc
        ).view(*self._vm).permute(0, 4, 1, 2, 3)

        return trilinear, masks


def adjust_saturation(rgb: torch.Tensor, mul: float):

    weights = rgb.new_tensor([0.299, 0.587, 0.114])
    grayscale = (
        torch.matmul(rgb, weights)
        .unsqueeze(dim=-1)
        .expand_as(rgb)
        .to(dtype=rgb.dtype)
    )

    return torch.lerp(grayscale, rgb, mul).clip(0, 1)


def pernomalik1(img, niter=5, kappa=0.0275, gamma=0.275):

    deltaS, deltaE = img.new_zeros(2, *img.shape)

    for _ in range(niter):
        deltaS[..., :-1, :] = torch.diff(img, dim=-2)
        deltaE[..., :, :-1] = torch.diff(img, dim=-1)

        gS = torch.exp(-(deltaS/kappa)**2.)
        gE = torch.exp(-(deltaE/kappa)**2.)

        S, E = gS*deltaS, gE*deltaE

        S[..., 1:, :] = S.diff(dim=-2)
        E[..., :, 1:] = E.diff(dim=-1)
        img = img + gamma*(S+E)

    return img


def pernomalik3d(img, niter=5, kappa=0.00275, gamma=0.275):

    # print(img.shape)

    deltaZ, deltaY, deltaX = img.new_zeros(3, *img.shape)

    for _ in range(niter):

        deltaZ[..., :-1, :, :] = torch.diff(img, dim=-3)
        deltaY[..., :, :-1, :] = torch.diff(img, dim=-2)
        deltaX[..., :, :, :-1] = torch.diff(img, dim=-1)

        gZ = torch.exp(-(deltaZ/kappa)**2.)
        gY = torch.exp(-(deltaY/kappa)**2.)
        gX = torch.exp(-(deltaX/kappa)**2.)

        Z, Y, X = gZ * deltaZ, gY * deltaY, gX * deltaX

        Z[..., 1:, :, :] = Z.diff(dim=-3)
        Y[..., :, 1:, :] = Y.diff(dim=-2)
        X[..., :, :, 1:] = X.diff(dim=-1)

        img = img + gamma*(Z+Y+X)

    return img


def rgb_to_ycbcr(feat: torch.Tensor, dim=-1) -> torch.Tensor:

    r, g, b = feat.unbind(dim)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    delta = 0.5
    cb = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta

    return torch.stack([y, cb, cr], dim)


def fast_uidx_1d(ar: torch.Tensor) -> torch.Tensor:

    assert ar.ndim == 1, f'Need dim of 1, got: {ar.ndim}!'
    perm = ar.argsort()
    aux = ar[perm]
    mask = ar.new_zeros(aux.shape[0], dtype=torch.bool)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    return perm[mask]


def fast_uidx_long2d(ar: torch.Tensor) -> torch.Tensor:

    assert ar.ndim == 2, "Wrong dim"
    m = ar.max() + 1
    r, c = ar
    cons = r*m + c
    return fast_uidx_1d(cons)


def scatter_add_1d(src: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:

    assert src.ndim == 1
    assert len(src) == len(idx)

    out = src.new_zeros(n)
    return out.scatter_add_(0, idx, src)


def scatter_mean_2d(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:

    assert src.ndim == 2
    assert len(src) == len(idx)

    if idx.ndim == 1:
        idx = idx.unsqueeze(1).expand(*src.shape)

    out = src.new_zeros(idx.max()+1, src.shape[1])
    return out.scatter_reduce_(0, idx, src, 'mean', include_self=False)


def cosine_similarity_argmax(
        vfeat: torch.Tensor, edges: torch.Tensor, sizes: torch.Tensor,
        nnz: int, lvl: Optional[int] = None, valbit: int = 16
) -> torch.Tensor:

    idxbit = 63 - valbit
    u, v = edges
    sfl = sizes.contiguous().to(dtype=vfeat.dtype)

    if lvl is None:
        mu = sfl.mean()
    else:
        mu = sfl.new_tensor(4**(lvl-1))

    std = sfl.std().clip(min=1e-6)

    u_v_bool = (u == v)
    stdwt = ((sfl - mu) / std).clamp(-.75, .75)[u]

    cosine_sim = torch.cosine_similarity(vfeat[u], vfeat[v], -1, 1e-4)
    sim = torch.where(u_v_bool, stdwt, cosine_sim)
    del cosine_sim
    del stdwt
    sim.clamp(-1, 1)

    shorts = (((sim + 1.0) / 2) * (2**valbit - 1)).long()
    packed_u = (shorts << idxbit) | u
    packed_v = (shorts << idxbit) | v
    del shorts
    packed_values = torch.zeros(nnz, dtype=torch.long, device=v.device)
    packed_values.scatter_reduce_(
        0, v, packed_u, 'amax', include_self=False
    )
    packed_values.scatter_reduce_(
        0, u, packed_v, 'amax', include_self=True
    )

    out = packed_values & (2**(idxbit) - 1)

    assert (out.max().item() < nnz)
    assert (out.min().item() >= 0)
    return out


# 3D format
def shape_proc(x): return x.permute(
    1, 0, 2, 3, 4).reshape(x.shape[1], -1).unbind(0)


def shape_proc_2d(x): return x.permute(
    1, 0, 2, 3).reshape(x.shape[1], -1).unbind(0)


def center_(x): return x.mul_(2).sub_(1)


def constrast1_(x, mu, lambda_):
    x.clip_(0, 1)
    m, a = x.new_tensor(mu).clip_(0, 1), x.new_tensor(lambda_).clip_(0)
    b = - (x.new_tensor(2)).log() / (1 - m**a).log()
    return x.pow_(a).mul_(-1).add_(1).pow_(b).mul_(-1).add_(1)


def contrast2_(x, lambda_):

    if lambda_ == 0:
        return x

    tmul = x.new_tensor(lambda_)
    m, d = tmul, torch.arcsinh(tmul)

    if lambda_ > 0:
        return x.mul_(m).arcsinh_().div(d)

    return x.mul_(d).sinh_().div_(m)


def dgrad(img, lambda_):

    img = img.mean(1, keepdim=True)
    print(img.shape)
    kernel = img.new_tensor([[[[-3., -10, -3.], [0., 0., 0.,], [3., 10, 3.]]]])
    kernel = torch.cat([kernel, kernel.mT], dim=0)
    out = F.conv2d(
        F.pad(img, 4*[1], mode='replicate'),
        kernel,
        stride=1,
    ).div_(16)
    return contrast2_(out, lambda_)


def dgrad3d(img, lambda_):
    img = img.mean(1, keepdim=True)

    scharr_x = torch.tensor([[[[-3., -10, -3.], [0., 0., 0.], [3., 10, 3.]],
                              [[-3., -10, -3.], [0., 0., 0.], [3., 10, 3.]],
                              [[-3., -10, -3.], [0., 0., 0.], [3., 10, 3.]]]])

    scharr_y = torch.tensor([[[[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]],
                              [[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]],
                              [[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]]])

    scharr_z = torch.tensor([[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                              [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                              [[-1., -1., -1.], [-1., -1., -1.], [-1., -1., -1.]]]])

    kernel = torch.stack(
        [scharr_x, scharr_y, scharr_z], dim=0
    ).to(img.device)

    img_padded = F.pad(img, (1, 1, 1, 1, 1, 1), mode='replicate')

    out = F.conv3d(img_padded, kernel, stride=1).div_(16)

    return contrast2_(out, lambda_)


def col_transform(colfeat, shape, lambda_col):

    # 3D format

    device = colfeat.device
    b, _, d, h, w = shape
    c = colfeat.shape[-1]
    f = adjust_saturation(colfeat.add(1).div_(2), 2.718)
    f = rgb_to_ycbcr(f, -1).mul_(2).sub_(1)
    contrast2_(f, lambda_col)
    return pernomalik3d(
        f.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).cpu(),
        4,
        0.1,
        0.5
    ).permute(0, 2, 3, 4, 1).view(-1, c).clip_(-1, 1).to(device)


def col_transform_2d(colfeat, shape, lambda_col):

    device = colfeat.device
    b, _, h, w = shape
    c = colfeat.shape[-1]
    f = adjust_saturation(colfeat.add(1).div_(2), 2.718)
    f = rgb_to_ycbcr(f, -1).mul_(2).sub_(1)
    contrast2_(f, lambda_col)
    return pernomalik1(
        f.view(b, h, w, c).permute(0, 3, 1, 2).cpu(),
        4,
        0.1,
        0.5
    ).permute(0, 2, 3, 1).view(-1, c).clip_(-1, 1).to(device)


def spstep(
        lab: torch.Tensor, edges: torch.Tensor, vfeat: torch.Tensor,
        sizes: torch.Tensor, nnz: int, lvl: int
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    int, torch.Tensor
]:
    sim = cosine_similarity_argmax(vfeat, edges, sizes, nnz, lvl=lvl)
    device = edges.device

    # CPU connected components
    ones = torch.ones_like(lab, device='cpu').numpy()
    adj = (lab.cpu().numpy(), sim.cpu().numpy())
    csr = cpu_coo_matrix((ones, adj), shape=(nnz, nnz)).tocsr()
    cc = lab.new_tensor(cpu_concom(csr)[1]).to(device)

    # print(cc)

    # print(cc.shape)

    vfeat_new = scatter_mean_2d(vfeat, cc)
    edges_new = cc[edges].contiguous()
    edges_new = edges_new[:, fast_uidx_long2d(edges_new)]
    lab_new = cc.unique()
    nnz_new = len(lab_new)
    sizes_new = scatter_add_1d(sizes, cc, nnz_new)
    return lab_new, edges_new, vfeat_new, sizes_new, nnz_new, cc


def get_supervoxel_segmentation(img: torch.Tensor, maxlvl=4) -> torch.Tensor:

    # 3D format

    lvl = 0
    lambda_grad = 27.8
    lambda_col = 10.0
    batch_size, _, depth, height, width = img.shape
    nnz = batch_size * height * width * depth

    labels = torch.arange(nnz, device='cpu')
    lr = labels.view(batch_size, depth, height,
                     width).unfold(-1, 2, 1).reshape(-1, 2).mT
    ud = labels.view(batch_size, depth, height,
                     width).unfold(-2, 2, 1).reshape(-1, 2).mT
    bf = labels.view(batch_size, depth, height,
                     width).unfold(-3, 2, 1).reshape(-1, 2).mT

    edges = torch.cat([lr, ud, bf], -1)
    sizes = torch.ones_like(labels)
    hierograph = [labels]

    r, g, b = shape_proc(img.clone())
    center_(constrast1_(r, .485, .539))
    center_(constrast1_(g, .456, .507))
    center_(constrast1_(b, .406, .404))
    gx, gy, gz = shape_proc(dgrad3d(img, lambda_grad))
    features = torch.stack([r, g, b, gx, gy, gz], -1)

    maxgrad = contrast2_(
        img.new_tensor(13/16), lambda_grad).mul_(2**.5)
    features = torch.cat([
        col_transform(features[:, :3], img.shape, lambda_col),
        center_(features[:, -3:].norm(2, dim=1, keepdim=True).div_(maxgrad)),
    ], -1).float()

    while lvl < maxlvl:
        lvl += 1
        labels, edges, features, sizes, nnz, cc = spstep(
            labels, edges, features, sizes, nnz, lvl
        )

        hierograph.append(cc)

    segmentation = hierograph[0]
    for i in range(1, lvl + 1):
        segmentation = hierograph[i][segmentation]

    return segmentation.view(batch_size, depth, height, width), edges


def altered_sv_alg(img: torch.Tensor, maxlvl_space=4, maxlvl_time=2):

    lvl = 0
    lambda_grad = 27.8
    lambda_col = 10.0

    batch_s, _, depth, height, width = img.shape

    new_img = img.reshape(batch_s*depth, _, height, width)
    batch_size = batch_s*depth
    nnz = batch_size * height * width

    labels = torch.arange(nnz, device='cpu')
    lr = labels.view(batch_size, height, width).unfold(-1,
                                                       2, 1).reshape(-1, 2).mT
    ud = labels.view(batch_size, height, width).unfold(-2,
                                                       2, 1).reshape(-1, 2).mT

    # print(f"lr: \n\n")
    # print(lr)
    # print("\n\n")
    # print(f"lr shape: {lr.shape}")

    # print(f"ud: \n\n")
    # print(ud)
    # print("\n\n")
    # print(f"ud shape: {ud.shape}")

    edges = torch.cat([lr, ud], dim=-1)
    sizes = torch.ones_like(labels)
    hierograph = [labels]

    r, g, b = shape_proc(img.clone())
    center_(constrast1_(r, .485, .539))
    center_(constrast1_(g, .456, .507))
    center_(constrast1_(b, .406, .404))
    gx, gy, gz = shape_proc(dgrad3d(img, lambda_grad))
    features = torch.stack([r, g, b, gx, gy, gz], -1)

    maxgrad = contrast2_(img.new_tensor(13/16), lambda_grad).mul_(2**.5)
    features = torch.cat([
        col_transform(features[:, :3], img.shape, lambda_col),
        center_(features[:, -3:].norm(2, dim=1, keepdim=True).div_(maxgrad)),
    ], -1).float()

    # print(f"features shape before: {features.shape}")

    while lvl < maxlvl_space:
        lvl += 1
        labels, edges, features, sizes, nnz, cc = spstep(
            labels, edges, features, sizes, nnz, lvl
        )

        hierograph.append(cc)

    # print(f"features shape after: {features.shape}")

    segmentation = hierograph[0]
    for i in range(1, lvl + 1):
        segmentation = hierograph[i][segmentation]

    segmentation = segmentation.view(batch_s, depth, height, width)

    # print(f"segmentation shape: {segmentation.shape}")

    edges_t = torch.stack([segmentation[:, :-1].flatten(),
                           segmentation[:, 1:].flatten()], dim=0)
    # print(f"edges_t shape: {edges_t.shape}")
    edges_t = edges_t[:, fast_uidx_long2d(edges_t)]

    # print(f"edges_t unique shape: {edges_t.shape}")
    edges = torch.cat([edges, edges_t], dim=-1)
    # edges = edges_t

    # print(f"edges shape: {edges.shape}")

    lvl = 0

    while lvl < maxlvl_time:

        lvl += 1

        labels, edges, features, sizes, nnz, cc = spstep(
            labels, edges, features, sizes, nnz, lvl
        )

        hierograph.append(cc)

    segmentation = hierograph[0]
    for i in range(1, maxlvl_space + maxlvl_time + 1):
        # print(i)
        # print(hierograph[i].shape[0])
        # print(segmentation.max())
        segmentation = hierograph[i][segmentation]

    return segmentation.view(batch_s, depth, height, width), edges


# def altered_sv_alg2(img: torch.Tensor, maxlvl_space=4, maxlvl_time=2):

    lvl = 0
    lambda_grad = 27.8
    lambda_col = 10.0

    batch_s, _, depth, height, width = img.shape

    new_img = img.reshape(batch_s*depth, _, height, width)
    batch_size = batch_s*depth
    nnz = batch_size * height * width

    labels = torch.arange(nnz, device='cpu')
    # bf = labels.view(batch_s, depth).unfold(-1, 2, 1).reshape(-1, 2).mT
    # lr = labels.view(batch_size, height, width).unfold(-1,
    #                                                    2, 1).reshape(-1, 2).mT
    # ud = labels.view(batch_size, height, width).unfold(-2,
    #                                                    2, 1).reshape(-1, 2).mT

    # edges = torch.cat([lr, ud], dim=-1)
    edges = labels.view(batch_s, depth).unfold(-3, 2, 1).reshape(-1, 2).mT
    sizes = torch.ones_like(labels)
    hierograph = [labels]

    r, g, b = shape_proc_2d(new_img.clone())
    center_(constrast1_(r, .485, .539))
    center_(constrast1_(g, .456, .507))
    center_(constrast1_(b, .406, .404))
    gx, gy = shape_proc_2d(dgrad(new_img, lambda_grad))
    features = torch.stack([r, g, b, gx, gy], -1)

    maxgrad = contrast2_(new_img.new_tensor(13/16), lambda_grad).mul_(2**.5)
    features = torch.cat([
        col_transform_2d(features[:, :3], new_img.shape, lambda_col),
        center_(features[:, -2:].norm(2, dim=1, keepdim=True).div_(maxgrad)),
    ], -1).float()

    print(f"features shape before: {features.shape}")

    while lvl < maxlvl_time:
        lvl += 1
        labels, edges, features, sizes, nnz, cc = spstep(
            labels, edges, features, sizes, nnz, lvl
        )

        hierograph.append(cc)

    print(f"features shape after: {features.shape}")

    segmentation = hierograph[0]
    for i in range(1, lvl + 1):
        segmentation = hierograph[i][segmentation]

    # segmentation = segmentation.view(batch_size, height, width)

    # print(f"segmentation shape: {segmentation.shape}")

    # edges_t = torch.stack([segmentation[:, :-1].flatten(),
    #                        segmentation[:, 1:].flatten()], dim=0)
    # print(f"edges_t shape: {edges_t.shape}")
    # edges_t = edges_t.unique(dim=-1)

    # print(f"edges_t unique shape: {edges_t.shape}")
    # # edges = torch.cat([edges, edges_t], dim=-1)
    # edges = edges_t

    # print(f"edges shape: {edges.shape}")

    # lvl = 0

    # while lvl < maxlvl_time:

    #     lvl += 1

    #     labels, edges, features, sizes, nnz, cc = spstep(
    #         labels, edges, features, sizes, nnz, lvl
    #     )

    #     hierograph.append(cc)

    # segmentation = hierograph[0]
    # for i in range(1, maxlvl_space + maxlvl_time + 1):
    #     print(i)
    #     print(hierograph[i].shape[0])
    #     print(segmentation.max())
    #     segmentation = hierograph[i][segmentation]

    return segmentation.view(batch_s, depth, height, width), edges


def unravel_index(idx: torch.Tensor, shape: Union[Sequence[int], torch.Tensor]) -> torch.Tensor:

    try:
        shape = idx.new_tensor(torch.Size(shape))[:, None]
    except Exception:
        pass

    shape = F.pad(shape, (0, 0, 0, 1), value=1)
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()
    return torch.div(idx[None], coefs, rounding_mode='trunc') % shape[:-1]


def init_graph_coords(vid: torch.Tensor) -> torch.Tensor:

    B, C, T, H, W = vid.shape
    nnz = B*T*H*W
    lab = torch.arange(nnz, device=vid.device)
    coords = unravel_index(lab, (B, T, H, W))
    return coords


def bbox_coords(seg: torch.Tensor, coord: torch.Tensor):
    nb, t, h, w = seg.shape
    _, x, y, z = coord
    bbox = seg.new_zeros(6, int(seg.max() + 1))
    bbox[0].scatter_reduce_(0, seg.view(-1), x, 'amin', include_self=False)
    bbox[1].scatter_reduce_(0, seg.view(-1), y, 'amin', include_self=False)
    bbox[2].scatter_reduce_(0, seg.view(-1), z, 'amin', include_self=False)
    bbox[3].scatter_reduce_(0, seg.view(-1), x, 'amax', include_self=False)
    bbox[4].scatter_reduce_(0, seg.view(-1), y, 'amax', include_self=False)
    bbox[5].scatter_reduce_(0, seg.view(-1), z, 'amax', include_self=False)

    return bbox


def process(vid: torch.Tensor, maxlvl: int = 8, interp: bool = True):

    output = []

    segs, edges = get_supervoxel_segmentation(vid, maxlvl)

    output.append(segs)
    output.append(edges)

    flatvid = vid.permute(0, 2, 3, 4, 1).flatten(start_dim=0, end_dim=-2)
    coords = init_graph_coords(vid)
    bbox = bbox_coords(segs, coords)

    output.append(bbox)

    IE = InterpolationExtractor(10, 5, 3)

    trilinear, mask = IE.forward(flatvid, segs, coords, bbox)

    output.append(trilinear @ mask)
    output.append(
        segs.view(-1).mul(segs.shape[0]
                          ).add(coords[0]).unique() % segs.shape[0]
    )

    return output


def print_results(info: list):

    print()
    print("--------------")
    print("Results")
    print("--------------")
    print()
    print(f"initial video shape: {info[0]}")
    print()
    print(f"Torch shape: {info[1]}")
    print()
    print(f"segs shape: {info[2]}")
    print()
    print(f"#sp: {info[3]}")
    print()
    print(f"edges shape: {info[4]}")
    print()
    print("Edges:")
    print()
    print(info[5])


def plot_image(video, segs, idx):
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    color = np.random.rand(256, 3)
    cmap = colors.ListedColormap(color)

    fvid_3d = video.permute(0, 2, 3, 4, 1).reshape(-1, 3)
    means_3d = scatter_mean_2d(fvid_3d, segs.view(-1))
    sv_vid_3d = means_3d[segs]

    video_3d = video.permute(0, 2, 3, 4, 1)

    axes[0].imshow(video_3d[0, idx, :, :, :])
    axes[1].imshow(sv_vid_3d[0, idx, :, :, :])
    axes[2].imshow(segs[0, idx, :, :], cmap=cmap)

    plt.show()


def plot_animation(video, segs):

    fig, ax = plt.subplots()
    ax.axis('off')

    fvid = video.permute(0, 2, 3, 4, 1).reshape(-1, 3)
    means = scatter_mean_2d(fvid, segs.view(-1))
    sv_vid = means[segs]
    img = ax.imshow(sv_vid[0, 0, :, :, :])

    def animate(i):
        img.set_data(sv_vid[0, i, :, :, :])
        return img,

    ani = FuncAnimation(fig, animate, frames=142, interval=100, blit=True)

    plt.show()


def image_loss(video: torch.Tensor, segs: torch.Tensor):

    fvid = t.permute(0, 2, 3, 4, 1).reshape(-1, 3)
    means = scatter_mean_2d(fvid, segs.view(-1))
    sv_vid = means[segs]

    sv_vid = sv_vid.permute(0, 4, 1, 2, 3)

    print(video.shape)

    print(sv_vid.shape)

    l = F.mse_loss(sv_vid, video)

    return l


def statistics_of_segs(video: torch.Tensor, segs: torch.Tensor):
    # What type of statistics?
    # Which dataset? Which config of the dataset (resolution, frames etc)?
    # Define length of supervoxel (not very well shaped)
    # Find length of supervoxels in specific direction
    # Per class or total or both
    # Average length of supervoxels or average length of highest and lowest
    #

    pass


if __name__ == "__main__":

    pass
    # from moviepy.editor import VideoFileClip

    # info_list = []

    # clip = VideoFileClip("v_Archery_g01_c02.avi")
    # clip = clip.resize(height=240, width=320)
    # vid = np.array([frame for frame in clip.iter_frames()])

    # vid = vid / 255

    # info_list.append(vid.shape)

    # t = torch.tensor(vid, dtype=torch.float32)
    # t = t.unsqueeze(0)
    # t = t.permute(0, -1, 1, 2, 3)

    # segs1, edges1 = get_supervoxel_segmentation(t, 10)
    # segs2, edges3 = get_supervoxel_segmentation(t, 11)
    # segs3, edges3 = get_supervoxel_segmentation(t, 12)

    # print(segs)

    # print(segs.shape)

    # print(segs.view(-1))
    # print(segs.view(-1).shape)

    # print(f"num sv: {segs.max() + 1}")

    # temp = segs.view(-1)
    # print()

    # print(f"segs.view: {temp}")

    # print()

    # print(f"segs.view shape: {temp.shape}")

    # print(f"num of unique: {temp.unique()}")

    # new_arr = torch.zeros(temp.unique().shape)

    # print(f"segs: {segs}")

    # print(f"maxlvl 10: {image_loss(t, segs1)}")
    # print(f"maxlvl 11: {image_loss(t, segs2)}")
    # print(f"maxlvl 12: {image_loss(t, segs3)}")

    # for num in temp:
    #    new_arr[num] += 1

    # plt.bar(range(0, 119), new_arr)
    # plt.show()

    # fvid = t.permute(0, 2, 3, 4, 1).reshape(-1, 3)
    # means = scatter_mean_2d(fvid, segs.view(-1))

    # print(f"means: {means}")

    # print()

    # print(f"shape means: {means.shape}")

    # print()

    # print(f"means[segs]: {means[segs]}")

    # print()

    # print(f"means[segs] shape: {means[segs].shape}")

    # print(coords.shape)

    # print(coords)

    # print(output[-1])

    # segs, edges = altered_sv_alg(t, 4, 4)

    # plot image

    # plot_image(t, segs, 0)

    # plot_image(t, segs, 0)

    # Standard deviation og mean på lengde av supervoxler
