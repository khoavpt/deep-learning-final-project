import torch
import torch.nn as nn
import pytorch_lightning as pl
from ...utils.utils import calculate_correlations, CosineSimilarity, ManhattanSimilarity

class LSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, embed_model, lr=0.003, optimizer_name='adam'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.optimizer_name = optimizer_name

        self.embedding_layer = nn.Embedding(embedding_dim=embed_model.embedding_dim, num_embeddings=embed_model.vocab_size)
        self.embedding_layer.weight = nn.Parameter(embed_model.get_embedding_matrix())  # all vectors

        self.embedding_layer.weight.requires_grad = False  # Freeze the embedding layer

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.scorer = CosineSimilarity()

    def forward(self, s1, s2):
        """
        s1: (batch_size, tx)
        s2: (batch_size, tx)
        """
        batch_size = s1.shape[0]
        s1 = self.embedding_layer(s1) # (batch_size, tx, embedding_dim)
        s2 = self.embedding_layer(s2) # (batch_size, tx, embedding_dim)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=s1.device) # (num_layers, batch_size, hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=s1.device) # (num_layers, batch_size, hidden_size)

        _, (s1, _)= self.lstm(s1, (h0, c0)) # (num_layers, batch_size, hidden_size)
        _, (s2, _)= self.lstm(s2, (h0, c0)) # (num_layers, batch_size, hidden_size)

        s1 = s1.squeeze() # (batch_size, hidden_size)
        s2 = s2.squeeze() # (batch_size, hidden_size)

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
    