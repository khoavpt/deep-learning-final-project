import torch
import numpy as np
from scipy.stats import spearmanr
from torch import nn

class ManhattanSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, s1, s2):
        """
        s1: (batch_size, tx, embedding_dim)
        s2: (batch_size, tx, embedding_dim)
        """
        sim = torch.linalg.norm(s1 - s2, ord=1, dim=(1,))
        sim = torch.exp(-sim)
        return sim

class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s1, s2):
        """
        s1: (batch_size, embedding_dim)
        s2: (batch_size, embedding_dim)
        """
        s1_norm = torch.nn.functional.normalize(s1, p=2, dim=1)
        s2_norm = torch.nn.functional.normalize(s2, p=2, dim=1)
        sim = torch.sum(s1_norm * s2_norm, dim=1)
        return sim
    

def calculate_correlations(y_pred, score):
    y_pred_np = y_pred.detach().cpu().numpy()
    score_np = score.detach().cpu().numpy()

    # Calculate correlations
    pearson_corr = np.corrcoef(y_pred_np, score_np)[0, 1]
    spearman_corr, _ = spearmanr(y_pred_np, score_np)

    return pearson_corr * 100, spearman_corr * 100