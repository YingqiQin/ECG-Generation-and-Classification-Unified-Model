import numpy as np
import math


def correct(result, uc, is_bce=True):
    """
    result: 1D 序列 (T,)
    uc: 1D 序列 (T,) —— 不确定性
    """
    if is_bce:
        pos = np.argwhere(result >= .5).flatten()
    else:
        pos = np.argwhere(result >= 0).flatten()
    diff_pos = np.diff(pos)
    diff_uc = np.diff(uc)

    rpos = []
    pre = 0
    last = len(pos) - 1
    cand_poses = np.where(diff_pos >= 2)[0]
    if last > 1:
        for index, cand_pos in enumerate(cand_poses):
            if cand_pos - pre >= 2:
                r_cand = round((pos[pre] + pos[cand_pos]) / 2)
                rpos.append(r_cand)
            pre = cand_pos + 1

        rpos.append(round((pos[pre] + pos[last]) / 2))
        qrs = np.array(rpos).astype(int)
    else:
        qrs = np.array([])

    # 合并过近的 R 位置
    check = True
    while check:
        qrs_diff = np.diff(qrs)
        if len(qrs_diff[qrs_diff <= 4]) == 0:
            check = False
            break
        r = 0
        while r < len(qrs_diff):
            if qrs_diff[r] <= 4:
                prev = qrs[r]
                nxt = qrs[r + 1]
                qrs = np.delete(qrs, r)
                qrs = np.delete(qrs, r)
                qrs = np.insert(qrs, r, (prev + nxt) / 2)
                qrs_diff = np.diff(qrs)
            r += 1
        check = False

    # # 噪声段检测
    # if np.mean(uc) >= 10:
    #     noise_starts = np.array([0])
    #     noise_ends = np.array([len(uc) - 1])
    # else:
    #     noise_starts = np.argwhere(diff_uc >= 5).flatten() + 1
    #     noise_ends = np.argwhere(diff_uc <= -5).flatten() + 1

    #     if uc[0] >= 5:
    #         noise_starts = np.insert(noise_starts, 0, 0)
    #     if uc[-1] >= 5:
    #         noise_ends = np.insert(noise_ends, len(noise_ends), len(uc) - 1)

    # noise_starts = np.expand_dims(noise_starts, -1)
    # noise_ends = np.expand_dims(noise_ends, -1)
    # noise_idx = np.concatenate((noise_starts, noise_ends), axis=-1)

    return qrs


def entropy(p):
    """
    p: [B, C, T] 或 [C, T]，均在“通道为第 2 维”的约定下。
    这里按 channel 维（axis=1）做信息熵。
    """
    logp = np.log(p)
    plogp = p * logp
    # 通道在前 -> 对 axis=1 求和
    ent = -np.sum(plogp, axis=1)
    return ent


def en_est(logits):
    """
    logits: [B, C, T]，C=3（0: o, 1: p, 2: n）
    返回: [B, T] —— 每个 time step 的熵
    """
    # 取出 p / n 通道，形状 [C, T]
    logits_p = logits[1, :]   # P 通道
    logits_n = logits[2, :]   # N 通道

    # 扩展维度到 [T, 1]，方便后面拼接
    logits_p = np.expand_dims(logits_p, -1)
    logits_n = np.expand_dims(logits_n, -1)

    logits_p_s = logits_p / (logits_p - logits_n + 1.)
    logits_n_s = (1. - logits_n) / (logits_p - logits_n + 1.)

    # p: [T, 2]
    p = np.concatenate((logits_p_s, logits_n_s), axis=-1)

    logp = np.log(p + 1e-5)
    plogp = p * logp
    entropy_t = -np.sum(plogp, axis=-1)  # [T]

    return entropy_t


def mi_est(logits):
    """
    logits: [B, C, T]，C=3
    返回: [B] —— 每条序列的 MI-based 不确定性
    """
    logits_o = logits[0, :]   # [T]
    logits_p = logits[1, :]
    logits_n = logits[2, :]

    eu_p = logits_p * np.log(logits_p / (logits_o + 1e-5) + 1e-5)
    eu_n = (1. - logits_n) * np.log(
        (1. - logits_n) / (1. - logits_o + 1e-5) + 1e-5
    )

    eu = eu_p + eu_n           # [T]
    # eu = np.mean(eu, axis=-1)  # [B]
    
    return eu


def uncertain_est(logits, thr=0.12):
    """
    logits: [B, C, T]，C=3
    返回: 不确定性序列 [B, T]
    """
    au = en_est(logits)     # [T]
    eu = mi_est(logits)     # [T]

    # 噪声标记：超过阈值的整条序列标成高不确定
    eu[eu > thr] = 10

    # # broadcast 到时间维
    # eu = np.expand_dims(eu, 1)                # [B, 1]
    # eu = np.repeat(eu, logits.shape[-1], 1)   # [B, T]

    uncertain = eu + au
    
    return uncertain


def mc_dropout_uncertain_est(logits):
    """
    logits: [B, C, T] 概率（经过 softmax/sigmoid）
    """
    uc = entropy(logits + 1e-6)  # [B, T]
    return uc


def qrs_uncertain_score(logits, uncertain_seq):
    """
    logits: [B, C, T]，这里用第 0 个 batch、0 通道作为 QRS 概率序列
    uncertain_seq: [T] 或 [T, ...]，这里按一维使用
    """
    logits_o = logits[0, :]   # [T]
    qrs_o, _ = correct(logits_o, uncertain_seq)

    def ngrams(arr, qrs_loc, length, step):
        grams = []
        for i in range(0, length - 2, step):
            start = qrs_loc[i]
            end = qrs_loc[i + 2]
            grams.append(arr[start: end])
        return grams

    loc_screen = np.insert(qrs_o, len(qrs_o), len(logits_o))
    loc_screen = np.insert(loc_screen, 0, 0)
    total_beats_num = len(loc_screen)

    screen_periods = ngrams(uncertain_seq, loc_screen, total_beats_num, 1)
    uncertain_r = []
    for screen_period in screen_periods:
        m = len(screen_period)
        n = np.sum(screen_period)
        r = n / math.log(m + 1e-6)
        uncertain_r.append(r)

    return uncertain_r, qrs_o


def multi_lead_select(logits_o_leads, uncertain_leads):
    """
    logits_o_leads: [n_leads, T] —— 每个导联的 QRS logits 序列（单通道）
    uncertain_leads: [n_leads, T] —— 每个导联的 uncertainty 序列
    现在按“导联在前 (channel-first)” 假设。
    返回:
        logits_o: [T]   —— 融合后的 QRS logits 序列
        uc:      [T]   —— 对应的不确定性值
    """
    # 对每个时间点，在各导联上找到不确定性最小的导联
    lead_select = np.argmin(uncertain_leads, axis=1)  # [T]

    T = logits_o_leads.shape[0]
    time_idx = np.arange(T)

    uc = uncertain_leads[time_idx, lead_select]       # [T]
    logits_o = logits_o_leads[time_idx, lead_select]  # [T]

    return logits_o, uc