import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

def compute_class_weights_from_loader(train_loader):
    """
    Computes class weights from segment labels in the train loader.
    For extreme imbalance, patient-level weights are better, but this is a decent start.
    """
    n0 = 0
    n1 = 0
    for batch in train_loader:
        # batch could be (x,y,subject_id) or (x,y)
        y = batch[1]
        y = y.detach().cpu().numpy().astype(int)
        n1 += int((y == 1).sum())
        n0 += int((y == 0).sum())
    # weights inversely proportional to class frequency
    # w0 for class 0, w1 for class 1
    w0 = (n0 + n1) / max(n0, 1)
    w1 = (n0 + n1) / max(n1, 1)
    return float(w0), float(w1), n0, n1


def weighted_bce_with_logits(logits, y, w0: float, w1: float):
    """
    For your case where minority might be label 0 (depending on how you define positive),
    this is safer than pos_weight because it supports weighting BOTH classes.
    logits: [B]
    y: [B] float 0/1
    """
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    w = torch.where(y > 0.5, torch.tensor(w1, device=y.device), torch.tensor(w0, device=y.device))
    return (loss * w).mean()


@torch.no_grad()
def evaluate(model, loader, device, patient_agg="mean_logit"):
    """
    Returns:
      - segment-level metrics
      - patient-level metrics (by aggregating segment logits per subject)
    """
    model.eval()
    all_logits = []
    all_y = []
    all_sid = []

    for batch in loader:
        if len(batch) == 3:
            x, y, sid = batch
        else:
            x, y = batch
            sid = ["unknown"] * len(y)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        logits = model(x).squeeze(-1)  # [B]
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
        all_sid.extend(list(sid))

    logits = torch.cat(all_logits).numpy()
    y = torch.cat(all_y).numpy().astype(int)
    prob = 1 / (1 + np.exp(-logits))

    # ----- segment-level -----
    seg = compute_binary_metrics(y, prob, threshold=0.5)

    # ----- patient-level aggregation -----
    pat_prob, pat_y = aggregate_by_patient(all_sid, logits, y, mode=patient_agg)
    pat = compute_binary_metrics(pat_y, pat_prob, threshold=0.5)

    return seg, pat


def aggregate_by_patient(subject_ids, segment_logits, segment_y, mode="mean_logit"):
    """
    Aggregate segment predictions into patient-level prediction.
    Best default: mean of logits, then sigmoid.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    y_buckets = defaultdict(list)

    for sid, logit, yy in zip(subject_ids, segment_logits, segment_y):
        buckets[sid].append(float(logit))
        y_buckets[sid].append(int(yy))

    pat_logits = []
    pat_y = []
    for sid in buckets.keys():
        ls = np.array(buckets[sid], dtype=np.float32)
        ys = np.array(y_buckets[sid], dtype=np.int64)
        # patient label should be consistent; use majority vote just in case
        y_pat = int(np.round(ys.mean()))

        if mode == "mean_logit":
            logit_pat = float(ls.mean())
        elif mode == "median_logit":
            logit_pat = float(np.median(ls))
        elif mode == "topk_mean_logit":
            k = max(1, int(0.2 * len(ls)))  # top 20%
            logit_pat = float(np.sort(ls)[-k:].mean())
        elif mode == "logsumexp":
            # smooth max; more sensitive to “salient” segments
            logit_pat = float(np.log(np.mean(np.exp(ls))))
        else:
            raise ValueError(f"Unknown patient agg mode: {mode}")

        pat_logits.append(logit_pat)
        pat_y.append(y_pat)

    pat_logits = np.array(pat_logits, dtype=np.float32)
    pat_prob = 1 / (1 + np.exp(-pat_logits))
    return pat_prob, np.array(pat_y, dtype=np.int64)


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    # Guard for rare-class folds: AUROC/AUPRC undefined if only one class present
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "confusion_matrix": cm,
        "n": int(len(y_true))
    }


def train_one_epoch(model, loader, optimizer, device, scaler, w0, w1, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(scaler is not None)):
            logits = model(x).squeeze(-1)  # [B]
            loss = weighted_bce_with_logits(logits, y, w0=w0, w1=w1)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(n, 1)


def fit(model, train_loader, val_loader, device, epochs=30, lr=1e-3, patient_agg="mean_logit"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler() if device.type == "cuda" else None

    # class weights from TRAIN only
    w0, w1, n0, n1 = compute_class_weights_from_loader(train_loader)
    print(f"[train class counts] n0={n0}, n1={n1}  => weights w0={w0:.3f}, w1={w1:.3f}")

    best_val = -np.inf
    best_state = None

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, w0=w0, w1=w1)

        seg_val, pat_val = evaluate(model, val_loader, device, patient_agg=patient_agg)
        # choose selection metric — for your extreme imbalance, patient-level AUPRC is often most meaningful
        sel = pat_val["auprc"] if not np.isnan(pat_val["auprc"]) else pat_val["f1"]

        print(
            f"Epoch {ep:03d} | loss {tr_loss:.4f} | "
            f"VAL(seg) AUROC {seg_val['auroc']:.3f} AUPRC {seg_val['auprc']:.3f} F1 {seg_val['f1']:.3f} | "
            f"VAL(pat) AUROC {pat_val['auroc']:.3f} AUPRC {pat_val['auprc']:.3f} F1 {pat_val['f1']:.3f}"
        )

        if sel > best_val:
            best_val = sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
