import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Window extraction (5000 -> K x 1024)
# ---------------------------
def make_window_starts(L: int, win: int, stride: int):
    starts = list(range(0, L - win + 1, stride))
    last = L - win
    if starts[-1] != last:
        starts.append(last)
    return starts

def extract_windows(x: torch.Tensor, win: int = 1024, stride: int = 256):
    """
    x: (B,1,L)
    return windows: (B,K,1,win), starts(list)
    """
    B, C, L = x.shape
    starts = make_window_starts(L, win, stride)
    ws = [x[:, :, s:s+win] for s in starts]  # list of (B,1,win)
    windows = torch.stack(ws, dim=1)         # (B,K,1,win)
    return windows, starts

def sample_k_windows(windows: torch.Tensor, k_sample: int):
    """
    windows: (B,K,1,win) -> (B,ks,1,win)
    """
    B, K, _, _ = windows.shape
    if k_sample is None or k_sample >= K:
        return windows
    idx = torch.randperm(K, device=windows.device)[:k_sample]
    return windows[:, idx, :, :]

def single_to_fake12(x_single_win: torch.Tensor, lead_idx: int = 0) -> torch.Tensor:
    """
    x_single_win: (N,1,win) -> (N,12,win), only lead_idx filled
    """
    assert x_single_win.dim() == 3 and x_single_win.size(1) == 1
    N, _, W = x_single_win.shape
    x12 = x_single_win.new_zeros((N, 12, W))
    x12[:, lead_idx, :] = x_single_win[:, 0, :]
    return x12


# ---------------------------
# MCMA encoder wrapper: output tokens (N,T,D)
# ---------------------------
class MCMAEncWrapper(nn.Module):
    def __init__(self, mcma_encoder: nn.Module, d_model: int = 512):
        super().__init__()
        self.enc = mcma_encoder
        self.d_model = d_model
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x12: torch.Tensor) -> torch.Tensor:
        """
        x12: (N,12,win)
        returns tokens: (N,T,D)
        """
        t = self.enc(x12)

        if t.dim() == 3 and t.size(-1) == self.d_model:
            tokens = t                       # (N,T,D)
        elif t.dim() == 3 and t.size(1) == self.d_model:
            tokens = t.transpose(1, 2).contiguous()  # (N,D,T)->(N,T,D)
        elif t.dim() == 2 and t.size(1) == self.d_model:
            tokens = t.unsqueeze(1)          # (N,D)->(N,1,D)
        else:
            raise ValueError(f"Unsupported MCMA output shape: {tuple(t.shape)}")

        return self.ln(tokens)


# ---------------------------
# ASL loss (multi-label long-tail friendly)
# ---------------------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gn = gamma_neg
        self.gp = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: (B,C), targets: (B,C) float {0,1}
        """
        p = torch.sigmoid(logits)
        if self.clip and self.clip > 0:
            p = torch.clamp(p, self.clip, 1.0 - self.clip)

        pos = targets * torch.log(p + self.eps)
        neg = (1.0 - targets) * torch.log(1.0 - p + self.eps)

        w_pos = torch.pow(1.0 - p, self.gp)
        w_neg = torch.pow(p, self.gn)

        loss = -(w_pos * pos + w_neg * neg)
        return loss.mean()


def kd_bce_distill_weighted(student_logits, teacher_logits, class_w, T=2.0):
    """
    Multi-label KD with soft targets + class weights.
    student_logits, teacher_logits: (B,C)
    class_w: (C,)
    """
    with torch.no_grad():
        t_prob = torch.sigmoid(teacher_logits / T)

    s_logit = student_logits / T
    bce = F.binary_cross_entropy_with_logits(s_logit, t_prob, reduction="none")  # (B,C)
    loss = (bce * class_w.view(1, -1)).mean() * (T * T)
    return loss


# ---------------------------
# Window-conditioned attention over window vectors
# ---------------------------
class WindowCondAttnPool(nn.Module):
    """
    Query: student_emb (B,D)
    Keys/Values: window_vecs (B,K,D)
    Output: gen_emb (B,D)
    """
    def __init__(self, d_model=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)

    def forward(self, q_vec: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q_vec: (B,D)
        kv: (B,K,D)
        """
        q = self.ln_q(q_vec).unsqueeze(1)        # (B,1,D)
        kv = self.ln_kv(kv)                      # (B,K,D)
        out, _ = self.mha(q, kv, kv, need_weights=False)  # (B,1,D)
        return out.squeeze(1)                    # (B,D)


# ---------------------------
# Unified model: student + MCMA windows -> attention pooled gen -> proj+concat -> MLP -> logits
# ---------------------------
class SplusG_AttnMLP(nn.Module):
    def __init__(
        self,
        student_encoder: nn.Module,       # expects (B,1,5000) -> (B,512)
        mcma_encoder: nn.Module,          # expects (N,12,1024) -> tokens
        num_classes: int = 9,
        d_model: int = 512,
        d_proj: int = 256,
        win: int = 1024,
        stride: int = 256,
        k_train: int = 8,                 # train-time sampled windows; set None to use all
        lead_idx: int = 0,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.student = student_encoder
        self.mcma = MCMAEncWrapper(mcma_encoder, d_model=d_model)

        self.win = win
        self.stride = stride
        self.k_train = k_train
        self.lead_idx = lead_idx

        self.win_attn = WindowCondAttnPool(d_model=d_model, num_heads=num_heads, dropout=0.1)

        # project to remove "dimension trick"
        self.proj_s = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_proj))
        self.proj_g = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_proj))

        self.mlp = nn.Sequential(
            nn.LayerNorm(2 * d_proj),
            nn.Linear(2 * d_proj, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_single_5000: torch.Tensor):
        """
        x_single_5000: (B,1,5000)
        returns logits: (B,9)
        """
        # student emb
        s = self.student(x_single_5000)              # (B,512)
        # keep it stable; you can remove normalize if you want, but LN is already used
        # s = F.normalize(s, p=2, dim=1)

        # windows
        windows, _ = extract_windows(x_single_5000, win=self.win, stride=self.stride)  # (B,K,1,win)
        if self.training and self.k_train is not None:
            windows = sample_k_windows(windows, self.k_train)                          # (B,ks,1,win)
        B, K, _, W = windows.shape

        xw = windows.reshape(B * K, 1, W)                                               # (B*K,1,win)
        x12 = single_to_fake12(xw, lead_idx=self.lead_idx)                              # (B*K,12,win)

        # MCMA tokens -> token mean => per-window vectors
        tokens = self.mcma(x12)                                                         # (B*K,T,512)
        v = tokens.mean(dim=1).reshape(B, K, -1)                                        # (B,K,512)

        # window-conditioned attention pooling
        g = self.win_attn(s, v)                                                         # (B,512)

        # proj + concat + MLP
        zs = self.proj_s(s)                                                             # (B,256)
        zg = self.proj_g(g)                                                             # (B,256)
        z = torch.cat([zs, zg], dim=1)                                                  # (B,512)
        logits = self.mlp(z)                                                            # (B,9)
        return logits
