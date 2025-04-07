from torch import Tensor

def scatter_reduce_2d(src:Tensor, idx:Tensor, red:str, nnz:int|None=None) -> Tensor:
    '''Scatter reduction over dim 0 with 1/2d source and 1d index.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    red : str
        Reduction method.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    add_dim = False
    if src.ndim == 1:
        add_dim = True
        src = src.unsqueeze(-1)
    if idx.ndim == 1:
        idx = idx.unsqueeze(1).expand(*src.shape)
    if nnz is None:
        nnz = int(idx.max().item()) + 1
    out = src.new_empty(nnz, src.shape[1])
    out.scatter_reduce_(0, idx, src, red, include_self=False)
    if not add_dim: return out
    return out.squeeze(-1)

def scatter_reduce_func(src: Tensor, idx: Tensor, nnz: int | None = None) -> Tensor:
    B, E = idx.shape
    D = None
    if src.ndim == 3:
        D = src.size(-1)

    if nnz is None:
        nnz = int(idx.max().item()) + 1
    if D is None:
        out = torch.zeros(B, nnz, device=src.device, dtype=src.dtype)
        count = torch.zeros_like(out)
    else:
        out = torch.zeros(B, nnz, D, device=src.device, dtype=src.dtype)
        count = torch.zeros(B, nnz, 1, device=src.device, dtype=src.dtype)

    for i in range(E):
        ix = idx[:,i]
        if D is None:
            out.scatter_add_(1, ix.unsqueeze(1), src[:,1].unsqueeze(1))
            count.scatter_add_(1, ix.unsqueeze(1), torch.ones_like(src[:,i].unsqueeze(1)))
        else:
            out.scatter_add_(1, ix.unsqueeze(1).expand(-1,D).unsqueeze(1), src[:,1].unsqueeze(1))
            count.scatter_add_(1, ix.unsqueeze(1).unsqueeze(-1), torch.ones_like(B, 1, 1, device=src.device))
    out.fill_(-float('inf'))
    for i in range(E):
        ix = idx[:, i]
        if D is None:
            out.scatter_(1, ix.unsqueeze(1), torch.max(out.gather(1, ix.unsqueeze(1)), src[:, i].unsqueeze(1)))
        else:
            out.scatter_(1, ix.unsqueeze(1).expand(-1, D).unsqueeze(1),
                         torch.max(out.gather(1, ix.unsqueeze(1).expand(-1, D).unsqueeze(1)), src[:, i].unsqueeze(1)))
    
    return out



def scatter_sum_2d(src:Tensor, idx:Tensor, nnz:int|None=None):
    '''Scatter sum reduction. Convenience function.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    return scatter_reduce_2d(src, idx, 'sum', nnz=nnz)

#def scatter_sum_func(src: Tensor, idx: Tensor, 

def scatter_softmax_2d(src:Tensor, idx:Tensor, nnz:int|None=None):
    '''Scatter sum reduction. Convenience function.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    mx = scatter_reduce_2d(src, idx, 'amax', nnz=nnz)
    src = (src - mx[idx]).exp()
    expsum = scatter_sum_2d(src, idx, nnz)
    return src / expsum[idx]