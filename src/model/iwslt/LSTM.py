import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from nltk.translate.bleu_score import corpus_bleu

class LstmEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, embed_model):        
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
   
        self.embedding_layer = nn.Embedding(embedding_dim=embed_model.embedding_dim, num_embeddings=embed_model.vocab_size)
        self.embedding_layer.weight = nn.Parameter(embed_model.get_embedding_matrix())  # all vectors
        self.embedding_layer.weight.requires_grad = False  # Freeze the embedding layer
       
        self.lstm = nn.LSTM(embed_model.embedding_dim, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        """
        x: (batch_size, tx)
        """
        batch_size = x.shape[0]
        x = self.embedding_layer(x)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device) # (num_layers, batch_size, hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device) # (num_layers, batch_size, hidden_size)
        
        _, out = self.lstm(x, (h0, c0))

        return out # tuple (h,c) each of shape (num_layers, batch_size, hidden_size)

class LstmDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, embed_model):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = embed_model.vocab_size
        self.index_to_key = embed_model.model.index_to_key
        self.sos_token_id = embed_model.model.key_to_index['<UNK>']
        
        self.embedding_layer = nn.Embedding(embedding_dim=embed_model.embedding_dim, num_embeddings=embed_model.vocab_size)
        self.embedding_layer.weight = nn.Parameter(embed_model.get_embedding_matrix())
        self.embedding_layer.weight.requires_grad = False

        self.lstm = nn.LSTM(embed_model.embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, hidden, max_length):
        """
        tuple (h,c) each of shape (num_layers, batch_size, hidden_size)
        """
        batch_size = hidden[0].size(1)
        inputs = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=hidden[0].device) # (batch_size, 1)

        outputs = []
        for _ in range(max_length):
            embedded = self.embedding_layer(inputs) # (batch_size, 1, embedding_dim)
            output, hidden = self.lstm(embedded, hidden) # output: (batch_size, 1, hiddenn_size); hidden: tuple of (h, c) shape (num_layers, batch_size, hidden_size)
            output = self.fc(output.squeeze(1)) # (batch_size, output_size)
            _, topi = output.topk(1) # (batch_size, 1)
            inputs = topi.detach()  # detach from history as input
            outputs.append(output.unsqueeze(1)) 

        outputs = torch.cat(outputs, dim=1) # (batch_size, max_length, output_size)
        return outputs

class Seq2Seq(pl.LightningModule):
    def __init__(self, encoder, decoder, lr=0.0003, optimizer_name='adam'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=231486)  # 231486 is the padding index

    def forward(self, src, trg, max_length):
        hidden = self.encoder(src)
        outputs = self.decoder(hidden, max_length)
        return outputs

    def training_step(self, batch, batch_idx):
        src, trg = batch
        max_length = trg.size(1)
        y_pred = self(src, trg, max_length)
        # Exclude the first token of trg for loss calculation
        loss = self.loss_fn(y_pred[:, :-1].reshape(-1, y_pred.size(-1)), trg[:, 1:].reshape(-1))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        max_length = trg.size(1)
        y_pred = self(src, trg, max_length)
        # Exclude the first token of trg for loss calculation
        loss = self.loss_fn(y_pred[:, :-1].reshape(-1, y_pred.size(-1)), trg[:, 1:].reshape(-1))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        src, trg = batch
        max_length = trg.size(1)
        y_pred = self(src, trg, max_length)
        loss = self.loss_fn(y_pred[:, :-1].reshape(-1, y_pred.size(-1)), trg[:, 1:].reshape(-1))
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)

        # Convert predictions to tokens
        _, top_indices = y_pred.topk(1, dim=2)
        predictions = top_indices.squeeze(-1)

        # Calculate BLEU score
        predictions_words = [self.decoder.index_to_key[pred.item()] for pred in predictions.cpu().numpy().flatten()]
        actuals_words = [[self.decoder.index_to_key[act.item()]] for act in trg.cpu().numpy().flatten()]
        bleu_score = corpus_bleu(actuals_words, predictions_words)
        self.log('test_bleu', bleu_score, on_epoch=True, prog_bar=True)

        return {'test_loss': loss, 'test_bleu': bleu_score}

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        return optimizer