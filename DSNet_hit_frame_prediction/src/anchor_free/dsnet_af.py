from torch import nn
import torch
import math
from sklearn.metrics import roc_auc_score

from modules.models import build_base_model



class DSNetAF(nn.Module):
    def __init__(self, base_model, num_feature, num_hidden, num_head):
        super().__init__()
        self.base_model = build_base_model(base_model, num_feature, num_head)
        self.layer_norm = nn.LayerNorm(num_feature)

        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden)
        )
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_dr = nn.Linear(num_hidden, 3)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)
        self.soft_max = nn.Softmax(dim=2)

    def forward(self, x):
        _, seq_len, _ = x.shape
        
        # x = self.layer_norm(x)
        if torch.isnan(x).any():
            print(x.shape)
        out = self.base_model(x)
        if torch.isnan(out).any():
            print(out.shape)
        out = out + x
        out = self.layer_norm(out)

        out = self.fc1(out)

        pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        pred_dr = self.fc_dr(out)
        pred_dr = self.soft_max(pred_dr).view(seq_len, 3)
        # pred_dr = self.soft_max(pred_dr).view(seq_len, 3)
        # pred_loc = self.fc_loc(out).exp().view(seq_len, 2)

        # pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)
        

        return pred_cls, pred_dr

    def predict(self, seq):
        pred_cls, pred_dr = self(seq)

        # pred_cls *= pred_ctr
        # pred_cls /= pred_cls.max() + 1e-8

        pred_cls = pred_cls.cpu().numpy()
        pred_dr = pred_dr.cpu().numpy()

        return pred_cls, pred_dr
