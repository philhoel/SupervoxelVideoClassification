import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional


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
        self, flatvid: Tensor, seg: Tensor, coord: Tensor, bbox: Tensor
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


class PositionalHistogramExtractor(nn.Module):

    '''Extracts positional histograms for supervoxels
    Attributes
    ----------
    patch_size : Tensor
        Desired size of the features we want to extract (cube with patch_size**3).    
    '''

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(
        self, flatvid: Tensor, seg: Tensor, coord: Tensor,
        bbox: Tensor, num_regions: int, sizes: Optional[Tensor] = None
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
        num_regions : int
            The number of supervoxels.
        sizes : Tensor
            Number of voxels in each supervoxel.
        '''
        B, T, H, W = seg.shape
        if sizes is None:
            sizes = seg.view(-1).bincount()
        t_pos, h_pos, w_pos = self.patch_size * \
            coord[1:] / coord.new_tensor([[T], [H], [W]])
        grid = flatvid.new_zeros(num_regions * self.patch_size**3)
        t_pos = t_pos.floor().long()
        h_pos = h_pos.floor().long()
        w_pos = w_pos.floor().long()
        pos = (
            seg.view(-1) * self.patch_size**3 +
            t_pos * self.patch_size**2 +
            h_pos * self.patch_size +
            w_pos
        )
        grid.scatter_add_(-1, pos, flatvid.new_ones(len(pos)))
        # There are other ways of doing this...
        den = sizes.mul((self.patch_size/32)**2)
        return (
            grid.view(
                num_regions, 1, self.patch_size, self.patch_size, self.patch_size
            ) / den.view(-1, 1, 1, 1, 1)
        )


class GradientHistogramExtractor(nn.Module):

    '''Extracts positional histograms for supervoxels
    NOTE: This expects the gradients to be (to some degree) normalized between -1,1.
          Ideally, find the norm of the gradient operator used to compute the gradients
          and normalize with this before passing to the extractor.

    Attributes
    ----------
    patch_size : Tensor
        Desired size of the features we want to extract (cube with patch_size**3). 
    eps : float
        Threshold for edge values...
    '''

    def __init__(self, patch_size, eps=1e-7):
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps

    def forward(
        self, grad: Tensor, seg: Tensor, coord: Tensor,
        bbox: Tensor, num_regions: int, sizes: Optional[Tensor] = None
    ):
        '''Forward function.
        Parameters
        ----------
        grad : Tensor
            The gradient video tensor, of shape [B,3,T,H,W] (3 dims)
        seg : Tensor
            Supervoxel segmentation [B,T,H,W]
        coord : Tensor
            Voxel indices, shape [4, BTHW]
        bbox : Tensor
            Bounding boxes, of shape [6, N] where N is num_regions.
        num_regions : int
            The number of supervoxels.
        sizes : Tensor
            Number of voxels in each supervoxel.
        '''
        if sizes is None:
            sizes = seg.view(-1).bincount()
        grad_y, grad_x, grad_z = (
            self.patch_size *
            grad.clip(self.eps-1, 1-self.eps)
            .permute(1, 0, 2, 3, 4)
            .reshape(3, -1)
            .add(1)
            .div(2)  # Vals in [0,1], shape [3, BTHW]
        )
        grad_out = grad.new_zeros(num_regions * self.patch_size**3)
        grad_y = grad_y.floor().long()
        grad_x = grad_x.floor().long()
        grad_z = grad_z.floor().long()
        pos = (
            seg.view(-1) * self.patch_size**3 +
            grad_y*self.patch_size**2 +
            grad_x*self.patch_size +
            grad_z
        )
        grad_out.scatter_add_(-1, pos, grad.new_ones(len(pos)))
        # This could be done in a better way?
        den = sizes.mul((self.patch_size/32)**2)
        return (
            grad_out.view(
                num_regions, 1, self.patch_size, self.patch_size, self.patch_size
            ) / den.view(-1, 1, 1, 1, 1)
        )
