import numpy as np
import matplotlib.pyplot as plt

def plot_segments_with_attention(
    segments, w, fs=500,
    y_true=None, y_pred=None, prob=None,
    class_names=None, topk=4, savepath=None
):
    """
    segments: [K, 1, T] or [K, T] (torch/numpy)
    w: [K]
    """
    if hasattr(segments, "detach"):
        segments = segments.detach().cpu().numpy()
    if hasattr(w, "detach"):
        w = w.detach().cpu().numpy()

    segments = np.asarray(segments)
    if segments.ndim == 3:  # [K,1,T]
        segments = segments[:, 0, :]
    K, T = segments.shape
    t = np.arange(T) / fs

    # sort by weight
    order = np.argsort(-w)
    top_idx = order[:topk]

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.bar(np.arange(K), w)
    ax1.set_title("Attention weights over K segments")
    ax1.set_xlabel("Segment index")
    ax1.set_ylabel("w")

    txt = []
    if y_true is not None:
        txt.append(f"GT={class_names[y_true] if class_names else y_true}")
    if y_pred is not None:
        txt.append(f"Pred={class_names[y_pred] if class_names else y_pred}")
    if prob is not None and y_pred is not None:
        txt.append(f"p={float(prob[y_pred]):.3f}")
    if txt:
        ax1.text(0.01, 0.95, " | ".join(txt), transform=ax1.transAxes, va="top")

    # plot top-k segments stacked
    ax2 = fig.add_subplot(2, 1, 2)
    for rank, idx in enumerate(top_idx, start=1):
        ax2.plot(t, segments[idx] + rank * 1.2 * np.std(segments[idx]), linewidth=1,
                 label=f"seg {idx} (w={w[idx]:.3f})")
    ax2.set_title(f"Top-{topk} segments waveforms (offset for display)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("ECG (a.u.)")
    ax2.legend(loc="upper right", ncol=2, fontsize=9)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig