import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import GPT2Model
from ...utils.utils import calculate_correlations, ManhattanSimilarity, CosineSimilarity

class GptEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained('gpt2', return_dict=True)
        self.gpt.resize_token_embeddings(50258)
        
    def forward(self, input_data):
        output = self.gpt(**input_data).last_hidden_state
        return output[:, -1, :] # (batch_size, hidden_size)

class Gpt(pl.LightningModule):
    def __init__(self, lr=0.003, optimizer_name='adam'):
        super().__init__()
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.encoder = GptEncoder()
        self.scorer = ManhattanSimilarity()
        
    def forward(self, s1, s2):
        """
        s1: (batch_size, tx)
        s2: (batch_size, tx)
        """
        s1 = self.encoder(s1)
        s2 = self.encoder(s2)
        sim = self.scorer(s1, s2)
        return sim

    def training_step(self, batch, batch_idx):
        s1, s2, score = batch
        y_pred = self(s1, s2)

        loss = nn.MSELoss()(y_pred.float(), score.float())
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        s1, s2, score = batch
        y_pred = self(s1, s2)

        loss = nn.MSELoss()(y_pred, score)  # Ensure loss computation is Float
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        pearson_corr, spearman_corr = calculate_correlations(y_pred, score)  # Ensure correlations are computed with Float tensors
        self.log('pearson_corr', pearson_corr, on_epoch=True, prog_bar=True)
        self.log('spearman_corr', spearman_corr, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        s1, s2, score = batch
        y_pred = self(s1, s2)

        loss = nn.MSELoss()(y_pred, score)  # Ensure loss computation is Float
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)

        pearson_corr, spearman_corr = calculate_correlations(y_pred, score)  # Ensure correlations are computed with Float tensors
        self.log('test_pearson_corr', pearson_corr, on_epoch=True, prog_bar=True)
        self.log('test_spearman_corr', spearman_corr, on_epoch=True, prog_bar=True)
        return loss
    

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        return optimizer