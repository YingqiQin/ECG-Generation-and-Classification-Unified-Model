import torch
import torch.nn as nn

class BPModelWithSleepAffine(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, sleep, a_day, b_day, a_night, b_night):
        """
        x:        [B,K,1,L]
        sleep:    [B,1] or [B]   0=day, 1=night
        a_day:    [B,2] or [1,2]
        b_day:    [B,2] or [1,2]
        a_night:  [B,2] or [1,2]
        b_night:  [B,2] or [1,2]
        """
        raw = self.base_model(x)  # [B,2]

        if sleep.dim() == 1:
            sleep = sleep.unsqueeze(1)
        sleep = sleep.float()     # [B,1]

        a = a_day * (1.0 - sleep) + a_night * sleep
        b = b_day * (1.0 - sleep) + b_night * sleep

        y = raw * a + b
        return y, raw

def load_clean_state_dict(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    clean = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean[k] = v
    return clean