import torch
import numpy as np


def get_tensor_elem_idx(*dims):
    """Returns a tuple of index arrays for every element in a tensor with the input dimensions"""
    if not isinstance(dims, np.ndarray):
        dims = np.array(dims)
    return tuple(
        torch.arange(dims[i])
            .repeat(dims[:i].prod())
            .repeat_interleave(dims[i+1:].prod())
        for i in range(len(dims))
    )

def sample_elem_idx(shape, sample_size):
    """Samples elements from the input tensor"""
    numel = np.array(shape).prod()
    idxs = get_tensor_elem_idx(*shape)
    idx_tensor = torch.stack(idxs)  # [ n dims X n elements ]
    sample_elems = torch.randperm(numel)[:sample_size]
    return tuple(idx_tensor[:,sample_elems])

def sample(tensor, n_samples, dim=-1):
    """Samples from the given tensor of normalized likelihoods along the given dimension"""
    upper_bound = tensor.cumsum(dim=dim)
    lower_bound = torch.cat(
        [
            torch.zeros(
                tuple(d if dim % len(tensor.shape) != i else 1 for i, d in enumerate(tensor.shape)),
                dtype=upper_bound.dtype,
                device=upper_bound.device,
            ),
            upper_bound[
                tuple(slice(None) if dim % len(tensor.shape) != i else slice(-1) for i in range(len(tensor.shape)))
            ],
        ],
        dim=dim,
    )
    n_broadcast = np.array([d for i, d in enumerate(tensor.shape) if dim % len(tensor.shape) != i]).prod()
    
    # expand bound tensors to add sample dimension
    upper_bound = upper_bound.unsqueeze(-1)
    lower_bound = lower_bound.unsqueeze(-1)
    
    # sample
    samples = np.random.rand(
        *(d if dim % len(tensor.shape) != i else 1 for i, d in enumerate(tensor.shape)), n_samples
    )
    samples = torch.Tensor(samples).to(upper_bound.device)
    
    return np.where(
        torch.bitwise_and(samples >= lower_bound, samples < upper_bound).cpu()  # [ ... X sample dim X ... X n samples ]
    )[:-1]

def test_sample(tensor, dim=-1, n_samples=10000):
    totals = torch.zeros_like(tensor, device=tensor.device)
    for i in range(n_samples):
        totals[sample(tensor, 1, dim=dim)] += 1
    return (totals / n_samples - tensor).abs()

def topk(tensor, k, dim=-1):
    """Returns topk values of input tensor along specified dimension, returning tuple of index arrays"""
    idx_tensor = tensor.topk(k, dim=dim).indices
    *indices, _ = get_tensor_elem_idx(*idx_tensor.shape)
    return *indices, idx_tensor.flatten()

def create_tril_mask(d, device: str = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    mask = torch.ones((d, d), device=device).type(torch.bool)
    return torch.tril(mask, diagonal=0)
