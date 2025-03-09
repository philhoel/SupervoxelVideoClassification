from time import perf_counter
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from typing import Sequence, Union, Optional, Tuple
from scipy.sparse import coo_matrix as cpu_coo_matrix
from scipy.sparse.csgraph import connected_components as cpu_concom
from .interpolation import InterpolationExtractor, PositionalHistogramExtractor, GradientHistogramExtractor
from utils import format_time
from .concom import connected_components
from .cossimargmax import cosine_similarity_argmax


class Supervoxel:

    def __init__(self,
                 device='cpu',
                 lambda_grad=27.8,
                 lambda_col=10.,
                 kappa=0.0275,
                 gamma=0.0275,
                 time_patch=5,
                 space_patch=5,
                 channels=3):

        self.device = device
        self.lambda_grad = lambda_grad
        self.lambda_col = lambda_col
        self.kappa = kappa  # 0.1
        self.gamma = gamma  # 0.5
        # self.patch_size = patch_size
        self.time_patch = time_patch
        self.space_patch = space_patch
        self.channels = channels
        self.IE = InterpolationExtractor(
            self.space_patch, self.time_patch, self.channels)
        self.IE = self.IE.to(torch.device(device))

    def adjust_saturation(self, rgb: torch.Tensor, mul: float):
        '''Adjusts saturation via interpolation / extrapolation.

        Args:
            rgb (torch.Tensor): An input tensor of shape (..., 3) representing the RGB values of an image.
            mul (float): Saturation adjustment factor. A value of 1.0 will keep the saturation unchanged.

        Returns:
            torch.Tensor: A tensor of the same shape as the input, with adjusted saturation.
        '''
        weights = rgb.new_tensor([0.299, 0.587, 0.114])
        grayscale = (
            torch.matmul(rgb, weights)
            .unsqueeze(dim=-1)
            .expand_as(rgb)
            .to(dtype=rgb.dtype)
        )
        return torch.lerp(grayscale, rgb, mul).clip(0, 1)

    def peronamalik1(self, img, niter=5, kappa=0.0275, gamma=0.275):
        '''Anisotropic diffusion.

        Perona-Malik anisotropic diffusion type 1, which favours high contrast
        edges over low contrast ones.

        `kappa` controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.

        Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.

        Args:
            img (torch.Tensor): input image
            niter (int): number of iterations
            kappa (float): conduction coefficient.
            gamma (float): controls speed of diffusion (generally max 0.25)

        Returns:
        Diffused image.
        '''

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

    def rgb_to_ycbcr(self, feat: torch.Tensor, dim=-1) -> torch.Tensor:
        '''Convert RGB features to YCbCr.

        Args:
            feat (torch.Tensor): Pixels to be converted YCbCr.

        Returns:
            torch.Tensor: YCbCr converted features.
        '''
        r, g, b = feat.unbind(dim)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        delta = 0.5
        cb = (b - y) * 0.564 + delta
        cr = (r - y) * 0.713 + delta
        return torch.stack([y, cb, cr], dim)

    def fast_uidx_1d(self, ar: torch.Tensor) -> torch.Tensor:
        '''Pretty fast unique index calculation for 1d tensors.

        Args:
            ar (torch.Tensor): Tensor to compute unique indices for.

        Returns:
            torch.Tensor: Tensor (long) of indices.
        '''
        assert ar.ndim == 1, f'Need dim of 1, got: {ar.ndim}!'
        perm = ar.argsort()
        aux = ar[perm]
        mask = ar.new_zeros(aux.shape[0], dtype=torch.bool)
        mask[:1] = True
        mask[1:] = aux[1:] != aux[:-1]
        return perm[mask]

    def fast_uidx_long2d(self, ar: torch.Tensor) -> torch.Tensor:
        '''Pretty fast unique index calculation for 2d long tensors (row wise).

        Args:
            ar (torch.Tensor): Tensor to compute unique indices for.

        Returns:
            torch.Tensor: Tensor (long) of indices.
        '''
        assert ar.ndim == 2, f'Need dim of 2, got: {ar.ndim}!'
        m = ar.max() + 1
        r, c = ar
        cons = r*m + c
        return self.fast_uidx_1d(cons)

    def scatter_add_1d(self, src: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
        '''Computes scatter add with 1d source and 1d index.

        Args:
            src (torch.Tensor): Source tensor.
            idx (torch.Tensor): Index tensor.
            n (int): No. outputs.

        Returns:
            torch.Tensor: Output tensor.
        '''
        assert src.ndim == 1
        assert len(src) == len(idx)
        out = src.new_zeros(n)
        return out.scatter_add_(0, idx, src)

    def scatter_mean_2d(self, src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        '''Computes scatter mean with 2d source and 1d index over first dimension.

        Args:
            src (torch.Tensor): Source tensor.
            idx (torch.Tensor): Index tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        assert src.ndim == 2
        assert len(src) == len(idx)
        if idx.ndim == 1:
            idx = idx.unsqueeze(1).expand(*src.shape)
        out = src.new_zeros(idx.max()+1, src.shape[1])  # type: ignore
        return out.scatter_reduce_(0, idx, src, 'mean', include_self=False)

    def cosine_similarity_argmax(
        self, vfeat: torch.Tensor, edges: torch.Tensor, sizes: torch.Tensor,
        nnz: int, lvl: Optional[int] = None, valbit: int = 16
    ) -> torch.Tensor:
        '''Compute the cosine similarity between edge-connected vertex features

        Uses Bit-Packing to perform argmax. We pack the first `valbit` bits in a
        signed `int64` (torch.long) tensor, and the indices are packed in the
        remaining bits (defaults to 47). Note that this implementation is device
        agnostic, but could result in some computational overhead from storing all
        similarities.

        NOTE: Original code from Superpixel Transformers by Aasan et al. 2023.

        Args:
            vfeat (torch.Tensor): Vertex features of shape (N, D), where N is the
                number of vertices and D is the dimension of the feature vector.
            edges (torch.Tensor): Edge indices as a tuple of tensors (u, v), where
                u and v are 1-D tensors containing source and target vertex
                indices, respectively.
            sizes (torch.Tensor): Sizes tensor used to normalize similarity measures.
            nnz (int): Number of non-zero values.
            lvl (Optional[int]): Level parameter to adjust the mean normalization,
                defaults to None, in which case the mean is computed from sizes.
            valbit (int): Number of bits used to represent the packed similarity
                value, defaults to 16.

        Returns:
            torch.Tensor: Packed similarity values as a tensor of long integers.
                The packed format includes both similarity values and vertex
                indices, and is suitable for further processing or storage.

        Note:
            The function includes assertions to ensure that the resulting packed
            values are within valid bounds. The compressed format allows for efficient
            storage and manipulation of large graph structures with associated
            similarity measures.

        Examples:
            >>> vfeat = torch.rand((100, 50))
            >>> edges = (torch.randint(100, (200,)), torch.randint(100, (200,)))
            >>> sizes = torch.rand(100)
            >>> nnz = 200
            >>> packed_sim = cosine_similarity_argmax(vfeat, edges, sizes, nnz)
        '''
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

        # sim = torch.where(
        #    u == v,
        #    stdwt.clip(-.75, .75)[u],
        #    torch.cosine_similarity(vfeat[u], vfeat[v], -1, 1e-4)
        # ).clip(-1, 1)

        shorts = (((sim + 1.0) / 2) * (2**valbit - 1)).long()
        packed_u = (shorts << idxbit) | u
        packed_v = (shorts << idxbit) | v
        del shorts
        packed_values = torch.zeros(nnz, dtype=torch.long, device=v.device)
        packed_values.scatter_reduce_(
            0, v, packed_u, 'amax', include_self=False)
        packed_values.scatter_reduce_(
            0, u, packed_v, 'amax', include_self=True)
        out = packed_values & (2**(idxbit)-1)

        assert (out.max().item() < nnz)
        assert (out.min().item() >= 0)
        return out

    def shape_proc(self, x): return x.permute(
        1, 0, 2, 3, 4).reshape(x.shape[1], -1).unbind(0)

    def center_(self, x): return x.mul_(2).sub_(1)

    def contrast1_(self, x, mu, lambda_):
        '''Kuwaraswamy contrast.
        '''
        x.clip_(0, 1)
        m, a = x.new_tensor(mu).clip_(0, 1), x.new_tensor(lambda_).clip_(0)
        b = -(x.new_tensor(2)).log_() / (1-m**a).log_()
        return x.pow_(a).mul_(-1).add_(1).pow_(b).mul_(-1).add_(1)

    def contrast2_(self, x, lambda_):
        '''Arcsinh contrast.
        '''
        if lambda_ == 0:
            return x
        tmul = x.new_tensor(lambda_)
        m, d = tmul, torch.arcsinh(tmul)
        if lambda_ > 0:
            return x.mul_(m).arcsinh_().div_(d)
        return x.mul_(d).sinh_().div_(m)

    def dgrad3d(self, img, lambda_):
        '''Discrete gradients with 3D Scharr Kernel.'''

        # Convert image to grayscale if it's a multi-channel image
        # Assuming input has shape [batch, channels, depth, height, width]
        img = img.mean(1, keepdim=True)

        # Scharr kernels for 3D (x, y, z gradients)
        scharr_x = torch.tensor([[[[-3., -10, -3.], [0., 0., 0.], [3., 10, 3.]],
                                  [[-3., -10, -3.], [0., 0., 0.], [3., 10, 3.]],
                                  [[-3., -10, -3.], [0., 0., 0.], [3., 10, 3.]]]])

        scharr_y = torch.tensor([[[[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]],
                                  [[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]],
                                  [[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]]])

        scharr_z = torch.tensor([[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                                  [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                  [[-1., -1., -1.], [-1., -1., -1.], [-1., -1., -1.]]]])

        # print(scharr_x.dtype)
        # print(img.dtype)

        # Stack kernels along the output dimension
        kernel = torch.stack(
            [scharr_x, scharr_y, scharr_z], dim=0).to(img.device)

        # Pad the image to maintain the original size after convolution
        img_padded = F.pad(img, (1, 1, 1, 1, 1, 1), mode='replicate')

        # Apply 3D convolution
        out = F.conv3d(img_padded, kernel, stride=1).div_(16)

        # Process the output with contrast function
        return self.contrast2_(out, lambda_)

    def col_transform(self, colfeat, shape, lambda_col):
        '''Color normalization.
        '''
        device = colfeat.device
        b, _, d, h, w = shape
        c = colfeat.shape[-1]
        f = self.adjust_saturation(colfeat.add(1).div_(2), 2.718)
        f = self.rgb_to_ycbcr(f, -1).mul_(2).sub_(1)
        self.contrast2_(f, lambda_col)
        return self.peronamalik1(
            f.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).cpu(),
            4,
            self.kappa,
            self.gamma
        ).permute(0, 2, 3, 4, 1).view(-1, c).clip_(-1, 1).to(device)

    def spstep(
        self, lab: torch.Tensor, edges: torch.Tensor, vfeat: torch.Tensor,
        sizes: torch.Tensor, nnz: int, lvl: int,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        int, torch.Tensor
    ]:
        '''Superpixel edge contraction step.
        '''

        # print("Hello")
        # Compute argmax over cosine similarities
        sim_start_time = perf_counter()
        sim = cosine_similarity_argmax(vfeat, edges, sizes, nnz, lvl=lvl)
        # print(f"Similarity time: {format_time(perf_counter() - sim_start_time)}")

        # Connected Components (performed on CPU)
        connected_start_time = perf_counter()
        # ones = torch.ones_like(lab, device='cpu').numpy()
        # adj = (lab.cpu().numpy(), sim.cpu().numpy())
        # csr = cpu_coo_matrix((ones, adj), shape=(nnz, nnz)).tocsr()
        # cc = lab.new_tensor(cpu_concom(csr)[1]).to(self.device)

        cc = connected_components(lab, sim, lab.shape[0])

        # print(f"Connected components time: {format_time(perf_counter() - connected_start_time)}")

        # Update Parameters
        vfeat_new = self.scatter_mean_2d(vfeat, cc)
        edges_new = cc[edges].contiguous()
        edges_new = edges_new[:, self.fast_uidx_long2d(edges_new)]
        lab_new = cc.unique()
        nnz_new = len(lab_new)
        sizes_new = self.scatter_add_1d(sizes, cc, nnz_new)
        return lab_new, edges_new, vfeat_new, sizes_new, nnz_new, cc

    def get_supervoxel_segmentation(self, img: torch.Tensor, maxlvl=4) -> torch.Tensor:
        '''Custom graph based superpixel segmentation.

        Hierarchically builds a superpixel segmentation in 5 levels. In this
        example, only the top level is extracted.

        There are lots of optimized code packed into one big function here, for
        more details, email `mariuaas(at)ifi.uio.no.

        NOTE: Original code from Superpixel Transformers by Aasan et al. 2023.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Superpixel segmentation.
        '''

        # Intialize Parameters
        lvl = 0
        lambda_grad = self.lambda_grad
        lambda_col = self.lambda_col
        batch_size, _, depth, height, width = img.shape
        nnz = batch_size * height * width * depth

        # print(f"img: {img.ndim}")

        # Initialize Segmentation
        labels = torch.arange(nnz, device=self.device)
        lr = labels.view(batch_size, depth, height,
                         width).unfold(-1, 2, 1).reshape(-1, 2).mT
        ud = labels.view(batch_size, depth, height,
                         width).unfold(-2, 2, 1).reshape(-1, 2).mT
        bf = labels.view(batch_size, depth, height,
                         width).unfold(-3, 2, 1).reshape(-1, 2).mT
        edges = torch.cat([lr, ud, bf], -1)
        sizes = torch.ones_like(labels)
        hierograph = [labels]

        # print(f"edges: {edges.ndim}")
        # print(f"edges: {edges.shape}")

        # Preprocess Features
        den = max(height, width)
        r, g, b = self.shape_proc(img.clone())
        self.center_(self.contrast1_(r, .485, .539))
        self.center_(self.contrast1_(g, .456, .507))
        self.center_(self.contrast1_(b, .406, .404))
        gx, gy, gz = self.shape_proc(self.dgrad3d(img, lambda_grad))
        features = torch.stack([r, g, b, gx, gy, gz], -1)
        # print(f"features: {features.ndim}")
        # print(f"features: {features.shape}")

        maxgrad = self.contrast2_(
            img.new_tensor(13/16), lambda_grad).mul_(2**.5)
        features = torch.cat([
            self.col_transform(features[:, :3], img.shape, lambda_col),
            self.center_(features[:, -3:].norm(2, dim=1,
                                               keepdim=True).div_(maxgrad)),
        ], -1).float()

        # Construct superpixel hierarchy
        while lvl < maxlvl:
            lvl += 1
            labels, edges, features, sizes, nnz, cc = self.spstep(
                labels, edges, features, sizes, nnz, lvl
            )
            hierograph.append(cc)

        # Collapse hierarchy to top level
        segmentation = hierograph[0]
        for i in range(1, lvl + 1):
            segmentation = hierograph[i][segmentation]

        return segmentation.view(batch_size, depth, height, width), edges

    def unravel_index(
        self, idx: torch.Tensor, shape: Union[Sequence[int], torch.Tensor]
    ) -> torch.Tensor:
        '''Converts a tensor of flat indices into a tensor of coordinate vectors.

        Parameters
        ----------
        idx : Tensor 
            Indices to unravel.
        shape : tuple[int] 
            Shape of tensor.

        Returns
        -------
        Tensor
            Tensor (long) of unraveled indices.
        '''
        try:
            shape = idx.new_tensor(torch.Size(shape))[:, None]  # type: ignore
        except Exception:
            pass
        # type: ignore
        shape = F.pad(shape, (0, 0, 0, 1), value=1)
        coefs = shape[1:].flipud().cumprod(dim=0).flipud()
        return torch.div(idx[None], coefs, rounding_mode='trunc') % shape[:-1]

    def init_graph_coords(self, vid: torch.Tensor) -> torch.Tensor:
        '''Initializes video coordinates.

        Parameters
        ----------
        vid (Tensor):
            Input video.

        Returns
        -------
        Tensor: Set of coordinates.
        '''
        B, C, T, H, W = vid.shape
        nnz = B*T*H*W
        lab = torch.arange(nnz, device=vid.device)
        coords = self.unravel_index(lab, (B, T, H, W))
        return coords

    def bbox_coords(self, seg: torch.Tensor, coord: torch.Tensor):

        nb, t, h, w = seg.shape
        _, x, y, z = coord
        bbox = seg.new_zeros(6, int(seg.max().item() + 1))
        bbox[0].scatter_reduce_(0, seg.view(-1), x, 'amin', include_self=False)
        bbox[1].scatter_reduce_(0, seg.view(-1), y, 'amin', include_self=False)
        bbox[2].scatter_reduce_(0, seg.view(-1), z, 'amin', include_self=False)
        bbox[3].scatter_reduce_(0, seg.view(-1), x, 'amax', include_self=False)
        bbox[4].scatter_reduce_(0, seg.view(-1), y, 'amax', include_self=False)
        bbox[5].scatter_reduce_(0, seg.view(-1), z, 'amax', include_self=False)
        return bbox

    def process(self, vid: torch.Tensor, maxlvl: int = 8, interp: bool = True, histo: bool = False, gradient: bool = False):

        output = []

        segs, edges = self.get_supervoxel_segmentation(vid, maxlvl)

        output.append(segs)
        output.append(edges)

        flatvid = vid.permute(0, 2, 3, 4, 1).flatten(
            start_dim=0, end_dim=-2)
        coords = self.init_graph_coords(vid)
        bbox = self.bbox_coords(segs, coords)

        if interp:

            trilinear, masks = self.IE.forward(flatvid, segs, coords, bbox)
            # print(f"trilinear shape: {trilinear.shape}")
            # print(f"masks shape: {masks.shape}")
            output.append(trilinear @ masks)
            output.append(
                segs.view(-1).mul(segs.shape[0]).add(coords[0]).unique() % segs.shape[0])

        # if histo:

            # PHE = PositionalHistogramExtractor(self.patch_size)
            # grid = PHE.forward(flatvid, segs, coords, bbox, (segs.max() + 1))
            # output.append(grid)

        return output


if __name__ == "__main__":

    from moviepy.editor import VideoFileClip

    clip = VideoFileClip("v_ApplyEyeMakeup_g01_c01.avi")
    clip = clip.resize(height=120, width=160)
    vid = np.array([frame for frame in clip.iter_frames()])

    clip2 = VideoFileClip("v_Archery_g01_c02.avi")
    clip2 = clip2.resize(height=120, width=160)
    vid2 = np.array([frame for frame in clip2.iter_frames()])

    vid = vid / 255
    vid2 = vid2 / 255

    t = torch.tensor(vid, dtype=torch.float32)
    t = t.unsqueeze(0)
    t = t.permute(0, -1, 1, 2, 3)

    t = t[:, :, :142, :, :]

    t2 = torch.tensor(vid2, dtype=torch.float32)
    t2 = t2.unsqueeze(0)
    t2 = t2.permute(0, -1, 1, 2, 3)

    tn = torch.cat([t, t2], 0)

    # lmbd_grad_values = [10, 20, 30, 40, 50]
    # lmbd_col_values = [5., 9., 10., 12., 15.]

    # print(tn.shape)

    # for lg in lmbd_grad_values:
    #     for lc in lmbd_col_values:
    #         myclass = Supervoxel(
    #             lambda_grad=lg, lambda_col=lc, kappa=0.1, gamma=0.5)
    #         segs = myclass.get_superpixel_segmentation(t, maxlvl=9)
    #         plot_multiple_frames(segs, lg, lc)

    sv = Supervoxel(kappa=0.1, gamma=0.5)
    # segs = supervoxel.get_superpixel_segmentation(t, maxlvl=9)

    # print(f"first video shape super: {segs.shape} || {segs.max() + 1}")

    output = sv.process(tn, maxlvl=9, interp=True)

    segs, edges, features, sv_indices = output[0], output[1], output[2], output[3], output[4]

    # print(f"Edges shape: {edges.shape}")
    # print(f"Edges: {edges}")
    # print(f"Num of SV: {segs.max() + 1}")

    print(features.flatten(start_dim=1, end_dim=-1).shape)

    # print(masks)

    # print(trilinear)

    # print(features.shape)

    # print(out)

    # print(sv_indices)

    # print(features[sv_indices == 0, :, :, :, :].shape)

    # for i in range(edges.shape[1]):
    #    print(f"vi: {edges[0,i]}, vj: {edges[1,i]} || i = {i}")
