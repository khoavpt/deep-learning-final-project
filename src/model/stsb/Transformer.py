import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ...utils.utils import calculate_correlations, CosineSimilarity, ManhattanSimilarity

class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=5, n_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)
       
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self,key,query,value,mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)
        
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
       
        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
       
        k_adjusted = k.transpose(-1,-2)
        product = torch.matmul(q, k_adjusted)
      
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / math.sqrt(self.single_head_dim)
        scores = F.softmax(product, dim=-1)
        scores = torch.matmul(scores, v)
        
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)
        
        output = self.out(concat)
       
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,key,query,value):
        attention_out = self.attention(key,query,value)
        attention_residual_out = attention_out + value
        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out

class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, embedding_dim, num_layers=2, expansion_factor=4, n_heads=5):
        super(TransformerEncoder, self).__init__()        
        self.positional_encoder = PositionalEmbedding(seq_len, embedding_dim)

        self.layers = nn.ModuleList([TransformerBlock(embedding_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
    def forward(self, x):
        out = self.positional_encoder(x)
        for layer in self.layers:
            out = layer(out,out,out)

        return out

class Transformer(pl.LightningModule):
    def __init__(self, seq_len, embed_model, num_layers=2, expansion_factor=4, n_heads=5, lr=0.003, optimizer_name='adam'):
        super().__init__()
        self.num_layers = num_layers
        self.lr = lr
        self.optimizer_name = optimizer_name

        self.embedding_layer = nn.Embedding(embedding_dim=embed_model.embedding_dim, num_embeddings=embed_model.vocab_size)
        self.embedding_layer.weight = nn.Parameter(embed_model.get_embedding_matrix())  # all vectors

        self.embedding_layer.weight.requires_grad = False  # Freeze the embedding layer

        self.transformer_encoder = TransformerEncoder(
            seq_len=seq_len, 
            embedding_dim=embed_model.embedding_dim, 
            num_layers=num_layers, 
            expansion_factor=expansion_factor, 
            n_heads=n_heads
        )
        
    def forward(self, s1, s2):
        """
        s1: (batch_size, tx)
        s2: (batch_size, tx)
        """
        batch_size = s1.shape[0]
        s1 = self.embedding_layer(s1) # (batch_size, tx, embedding_dim)
        s2 = self.embedding_layer(s2) # (batch_size, tx, embedding_dim)

        s1 = self.transformer_encoder(s1) # (batch_size, tx, embedding_dim)
        s2 = self.transformer_encoder(s2) # (batch_size, tx, embedding_dim)

        # Mean pooling
        s1 = s1.mean(dim=1)  # (batch_size, embedding_dim)
        s2 = s2.mean(dim=1)  # (batch_size, embedding_dim)
        
        sim = torch.linalg.norm(s1 - s2, ord=1, dim=(1,))
        sim = 5 * torch.exp(-sim) # (batch_size, )
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

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        return optimizer