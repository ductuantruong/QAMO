import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class QAMO(nn.Module):
    def __init__(self, feat_dim, r_real=0.9, r_fake=0.2, alpha=20.0, scale=32.0, margin=0.4):
        super(QAMO, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(2, self.feat_dim))
        self.center_loss_w = 2.
        self.softplus = nn.Softplus()
        self.scale = scale
        self.margin = margin

    def quality_clst_forward(self, input, label):
        cosine = F.linear(input, F.normalize(self.center))
        phi = cosine - self.margin
        # ---------------- convert label to one-hot ---------------
        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output
    
    def forward(self, x, labels=None, q_score=None):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        maxscores, _ = torch.max(scores, dim=1, keepdim=True)
        maxscores = maxscores.squeeze(1)
        output_scores = maxscores.clone()
        if labels is None:
            return None, scores
        else:
            if q_score is None:
                maxscores[labels == 1] = self.r_real - maxscores[labels == 1]
                maxscores[labels == 0] = maxscores[labels == 0] - self.r_fake
                loss = self.softplus(self.alpha * maxscores).mean()
            else:
                maxscores[labels == 1] = self.r_real - scores[labels == 1, q_score[labels == 1]]
                maxscores[labels == 0] = maxscores[labels == 0] - self.r_fake
                loss_cls = self.softplus(self.alpha * maxscores).mean()
                quality_cls_output = self.quality_clst_forward(x[labels == 1], q_score[labels == 1])
                loss = [loss_cls, quality_cls_output]
            return loss, output_scores