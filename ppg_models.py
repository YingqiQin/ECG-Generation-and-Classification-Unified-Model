import torch
class BPRegressorHeteroMIL(nn.Module):
    """
    输入：feats [B,K,D]（D=256）
    输出：事件级预测 mu [B,2] 以及事件级方差 v [B,2]（可选用于推理置信度/加权）
    """
    def __init__(self, d_model=256, attn_hidden=128, dropout=0.1, head_hidden=128):
        super().__init__()
        self.pool = SegmentAttentionPooling(d_model, attn_hidden, dropout)

        # 片段级均值 head: mu_i
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 2),
        )
        # 片段级 log-variance head: logvar_i
        self.logvar_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, feats, mask=None):
        """
        feats: [B,K,D]
        mask: [B,K] bool or None
        """
        _, w = self.pool(feats, mask=mask)          # w: [B,K]
        w2 = w.unsqueeze(-1)                        # [B,K,1]

        mu_i = self.mu_head(feats)                  # [B,K,2]
        logvar_i = self.logvar_head(feats)          # [B,K,2]

        # 数值稳定：限制 logvar 范围
        logvar_i = torch.clamp(logvar_i, min=-6.0, max=4.0)
        var_i = torch.exp(logvar_i) + 1e-6          # [B,K,2]

        # 事件级均值
        mu = (w2 * mu_i).sum(dim=1)                 # [B,2]

        # 事件级方差：全概率方差 = E[var] + Var[E]
        v_intra = (w2 * var_i).sum(dim=1)           # [B,2]
        v_inter = (w2 * (mu_i - mu.unsqueeze(1))**2).sum(dim=1)  # [B,2]
        v = (v_intra + v_inter).clamp_min(1e-6)     # [B,2]

        return mu, v, w

def hetero_gaussian_nll(mu, v, y):
    # mu,v,y: [B,2]
    return 0.5 * (torch.log(v) + (y - mu)**2 / v).mean()

