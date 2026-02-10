import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

@torch.no_grad()
def collect_sample_stats(model, loader, device="cuda"):
    model.eval()
    rows = []
    for batch in tqdm(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        sample_id = batch["sample_id"]          # list[str] or list[int]
        patient_id = batch.get("patient_id", None)

        logits = model(x)                       # [B, 4]
        prob = F.softmax(logits, dim=-1)        # [B, 4]

        # per-sample CE loss
        loss = F.cross_entropy(logits, y, reduction="none")  # [B]

        p_true = prob.gather(1, y.view(-1,1)).squeeze(1)     # [B]
        p_max, pred = prob.max(dim=1)                        # [B]
        p_sorted, _ = prob.sort(dim=1, descending=True)
        margin = (p_sorted[:,0] - p_sorted[:,1])             # [B]
        entropy = -(prob * (prob.clamp_min(1e-12)).log()).sum(dim=1)

        for i in range(len(sample_id)):
            rows.append({
                "sample_id": sample_id[i],
                "patient_id": patient_id[i] if patient_id is not None else None,
                "y": int(y[i].item()),
                "pred": int(pred[i].item()),
                "loss": float(loss[i].item()),
                "p_true": float(p_true[i].item()),
                "p_max": float(p_max[i].item()),
                "margin": float(margin[i].item()),
                "entropy": float(entropy[i].item()),
                "prob0": float(prob[i,0].item()),
                "prob1": float(prob[i,1].item()),
                "prob2": float(prob[i,2].item()),
                "prob3": float(prob[i,3].item()),
            })

    df = pd.DataFrame(rows)
    return df

def main():
    # model, loader = ...
    df = collect_sample_stats(model, train_eval_loader, device="cuda")
    df.to_csv("train_oof_stats.csv", index=False)


import pandas as pd

def flag_suspects(df: pd.DataFrame):
    df = df.copy()

    # Rule A1: high-confidence disagreement
    df["flag_A1"] = (df["pred"] != df["y"]) & (df["p_max"] >= 0.90) & (df["margin"] >= 0.30)

    # Rule B1: per-class top-loss q%
    df["flag_B1"] = False
    q_default = 0.02
    q_by_class = {0: 0.02, 1: 0.04, 2: 0.02, 3: 0.02}  # class1更激进一点
    for c in sorted(df["y"].unique()):
        sub = df[df["y"] == c]
        q = q_by_class.get(int(c), q_default)
        thr = sub["loss"].quantile(1 - q)
        df.loc[sub.index, "flag_B1"] = sub["loss"] >= thr

    # Rule C1: very low p_true (optional)
    df["flag_C1"] = df["p_true"] <= 0.10

    # Focus on 0<->1 conflicts (optional tag)
    df["flag_01_conflict"] = (
        ((df["y"] == 0) & (df["pred"] == 1)) |
        ((df["y"] == 1) & (df["pred"] == 0))
    )

    # A practical "drop" suggestion
    df["suggest_drop"] = df["flag_A1"] | df["flag_B1"]

    return df

def main():
    df = pd.read_csv("train_oof_stats.csv")
    out = flag_suspects(df)

    out.to_csv("train_oof_stats_flagged.csv", index=False)

    drop_df = out[out["suggest_drop"]].copy()
    drop_df = drop_df.sort_values(["flag_A1","flag_B1","loss"], ascending=[False, False, False])

    # 输出要剔除的 sample_id 列表
    drop_df[["sample_id","patient_id","y","pred","loss","p_true","p_max","margin","flag_A1","flag_B1","flag_01_conflict"]] \
        .to_csv("drop_list.csv", index=False)

    # 也可以只导出最关键的 0<->1 高置信冲突给医生复核
    review_df = out[out["flag_A1"] & out["flag_01_conflict"]].sort_values("p_max", ascending=False).head(300)
    review_df.to_csv("review_01_highconf.csv", index=False)

if __name__ == "__main__":
    main()


