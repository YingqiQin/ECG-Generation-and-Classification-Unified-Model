import torch
import torch.distributed as dist

def ddp_gather_to_rank0(x: torch.Tensor, dst: int = 0) -> torch.Tensor | None:
    """
    Gather a tensor with variable first-dimension length from all ranks to rank0.

    Args:
      x: local tensor [N, ...] on GPU (or CPU if backend supports, but GPU is safest).
      dst: destination rank (usually 0)

    Returns:
      On rank dst: concatenated tensor [sum_i N_i, ...]
      On other ranks: None
    """
    assert dist.is_initialized(), "torch.distributed not initialized"
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = x.device

    # 1) gather lengths
    n_local = torch.tensor([x.shape[0]], device=device, dtype=torch.long)
    n_list = [torch.zeros_like(n_local) for _ in range(world_size)]
    dist.all_gather(n_list, n_local)
    sizes = [int(t.item()) for t in n_list]
    max_n = max(sizes)

    # 2) pad to max length
    if x.shape[0] < max_n:
        pad_shape = (max_n - x.shape[0],) + x.shape[1:]
        pad = torch.zeros(pad_shape, device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    # 3) gather padded tensors to rank0
    if rank == dst:
        gather_list = [torch.empty_like(x_pad) for _ in range(world_size)]
        dist.gather(x_pad, gather_list=gather_list, dst=dst)
        # 4) unpad and concat
        out = torch.cat([t[:sz] for t, sz in zip(gather_list, sizes)], dim=0)
        return out
    else:
        dist.gather(x_pad, gather_list=None, dst=dst)
        return None
