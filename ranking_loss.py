import torch
import torch.nn.functional as F


def pairwise_ranking_loss(
    pred: torch.Tensor,         # [B] or [B,1]
    y: torch.Tensor,            # [B] (float)
    *,
    min_delta: float = 5.0,     # only use pairs with |yi - yj| >= min_delta
    margin: float = 0.0,        # hinge/logistic margin (same unit as pred)
    mode: str = "logistic",     # "hinge" or "logistic"
    max_pairs: int = 4096,      # subsample eligible pairs to control O(B^2)
    weight_by_delta: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Enforces order consistency:
      if y_j > y_i then pred_j should be > pred_i (by margin optionally).
    Constant predictions -> non-zero loss (prevents collapse).

    Returns scalar loss.
    """
    pred = pred.squeeze(-1)  # [B]
    y = y.squeeze(-1)

    B = pred.shape[0]
    if B < 2:
        return pred.new_tensor(0.0)

    # pairwise label difference
    dy = y.unsqueeze(1) - y.unsqueeze(0)  # [B,B]
    # eligible pairs: strong label separation + exclude diagonal
    mask = dy.abs() >= min_delta
    mask.fill_diagonal_(False)

    # only keep i<j to avoid double counting
    iu = torch.triu(torch.ones((B, B), dtype=torch.bool, device=pred.device), diagonal=1)
    mask = mask & iu

    idx = mask.nonzero(as_tuple=False)  # [N,2] with i<j
    if idx.numel() == 0:
        return pred.new_tensor(0.0)

    # subsample pairs if too many
    if idx.shape[0] > max_pairs:
        perm = torch.randperm(idx.shape[0], device=pred.device)[:max_pairs]
        idx = idx[perm]

    i = idx[:, 0]
    j = idx[:, 1]

    # sign: +1 if y_j > y_i else -1
    s = torch.sign(y[j] - y[i])  # [N]
    # pred diff: pred_j - pred_i
    dp = pred[j] - pred[i]       # [N]

    # loss on each pair
    # want s*dp >= margin
    if mode == "hinge":
        # relu(margin - s*dp)
        per = F.relu(margin - s * dp)
    elif mode == "logistic":
        # softplus(margin - s*dp) = log(1+exp(margin - s*dp))
        per = F.softplus(margin - s * dp)
    else:
        raise ValueError("mode must be 'hinge' or 'logistic'")

    if weight_by_delta:
        # stronger separated labels get higher weight (cap to avoid huge weights)
        w = (y[j] - y[i]).abs()
        # normalize weights to mean~1
        w = w / (w.mean() + 1e-12)
        per = per * w

    if reduction == "mean":
        return per.mean()
    elif reduction == "sum":
        return per.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")



def huber_plus_ranking_loss(
    pred: torch.Tensor,      # [B] or [B,1]
    y: torch.Tensor,         # [B] float
    *,
    huber_beta: float = 5.0, # SmoothL1 beta in same unit as y (e.g., 5 LVEF points)
    lambda_rank: float = 0.2,
    min_delta: float = 8.0,
    rank_margin: float = 0.0,
    rank_mode: str = "logistic",
    max_pairs: int = 4096,
) -> torch.Tensor:
    pred = pred.squeeze(-1)
    y = y.squeeze(-1)

    # Huber / SmoothL1
    huber = F.smooth_l1_loss(pred, y, beta=huber_beta, reduction="mean")

    # Ranking
    rank = pairwise_ranking_loss(
        pred, y,
        min_delta=min_delta,
        margin=rank_margin,
        mode=rank_mode,
        max_pairs=max_pairs,
        weight_by_delta=True,
        reduction="mean",
    )

    return huber + lambda_rank * rank


#####################################################################################

# x: [B,C,L] (或 MIL 时 [B,K,C,L] 先经 pooling 得到 pred)
# y: [B] float LVEF

pred = model(x).squeeze(-1)  # [B]
loss = huber_plus_ranking_loss(
    pred, y.float(),
    huber_beta=5.0,
    lambda_rank=0.2,
    min_delta=8.0,
    rank_margin=0.0,
    rank_mode="logistic",
)

optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()

