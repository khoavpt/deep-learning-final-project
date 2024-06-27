import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from underthesea import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import re

def build_vi_vocab(data_path):
    def vi_tokenizer(sentence):
        tokens = word_tokenize(sentence)
        return tokens
    
    tokenizer = get_tokenizer(vi_tokenizer)
    def yield_tokens(train_iter):
        for index, row in train_iter:
            tokens = tokenizer(row['vi'])
            yield tokens
            
    df = pd.read_csv('/kaggle/input/translation-data/train.csv').dropna()
    train_iter = df.iterrows()
    special_symbols = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
    vocab = build_vocab_from_iterator(yield_tokens(train_iter),
                                      min_freq=1,
                                      specials=special_symbols,
                                      special_first=True)
    return vocab

def custom_vi_preprocess_pipeline_for_transformer(sentence, vocab, tokenizer, tx): 
    vocab_stoi = vocab.get_stoi()
#     sentence = '<SOS> ' + sentence + ' <EOS>'    
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = [vocab_stoi[word] if word in vocab_stoi else vocab_stoi['<UNK>'] for word in tokenizer(sentence)]
    sentence.insert(0, vocab_stoi['<SOS>'])
    sentence.append(vocab_stoi['<EOS>'])

    if len(sentence) > tx:
        sentence = sentence[:tx]
    else:
        sentence += [vocab_stoi['<PAD>']] * (tx - len(sentence))
        
    return torch.tensor(sentence, dtype=torch.long)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device="cuda" if torch.cuda.is_available() else "cpu")) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(en, vi):
    en_seq_len = en.shape[1]
    vi_seq_len = vi.shape[1]
    vi_mask = generate_square_subsequent_mask(vi_seq_len)
    en_mask = torch.zeros((en_seq_len, en_seq_len), device="cuda" if torch.cuda.is_available() else "cpu").type(torch.bool)

    en_padding_mask = (en == 0)
    vi_padding_mask = (vi == 0)
    return en_mask, vi_mask, en_padding_mask, vi_padding_mask

class PositionalEmbedding(nn.Module):
    def __init__(self,tx,embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(tx,self.embed_dim)
        for pos in range(tx):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x # (batch_size, tx, embedding_dim)
    
class Transformer(pl.LightningModule):
    def __init__(self, en_embed_model, vocab, tx, num_encoder_layers, num_decoder_layers, nhead, dim_feedforward, dropout, lr=0.0003, optimizer_name='adam'):
        super().__init__()
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 0 is the padding index
        self.vocab = vocab
        
        d_model = en_embed_model.embedding_dim
        vi_vocab_size = len(vocab.get_stoi())
        
        self.positional_encoding = PositionalEmbedding(tx, d_model)
        
        self.en_embedding_layer = nn.Embedding(embedding_dim=d_model, num_embeddings=en_embed_model.vocab_size)
        self.en_embedding_layer.weight = nn.Parameter(en_embed_model.get_embedding_matrix())
        self.en_embedding_layer.weight.requires_grad = False
        
        self.vi_embedding_layer = nn.Embedding(embedding_dim=d_model, num_embeddings=vi_vocab_size)
        self.transformer = nn.Transformer(d_model=d_model,
                                         nhead=nhead,
                                         num_encoder_layers=num_decoder_layers,
                                         num_decoder_layers=num_decoder_layers,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout,
                                         batch_first=True)
        self.fc = nn.Linear(d_model, vi_vocab_size)
    
    def forward(self, en, vi, en_mask, vi_mask, en_padding_mask, vi_padding_mask, memory_key_padding_mask):
        """
        en: (batch_size, seq_len)
        vi: (batch_size, seq_len)
        """
        en = self.positional_encoding(self.en_embedding_layer(en))
        vi = self.positional_encoding(self.vi_embedding_layer(vi))
        
        out = self.transformer(en, vi, en_mask, vi_mask, None,
                                en_padding_mask, vi_padding_mask, memory_key_padding_mask)
        out = self.fc(out)
        return out         
    
    def training_step(self, batch, batch_idx):
        en, vi = batch
        en_mask, vi_mask, en_padding_mask, vi_padding_mask = create_mask(en, vi)   
        y_pred = self(en, vi, en_mask, vi_mask, en_padding_mask, vi_padding_mask, en_padding_mask)

        loss = self.loss_fn(y_pred[:, :-1].reshape(-1, y_pred.size(-1)), vi[:, 1:].reshape(-1))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        en, vi = batch
        en_mask, vi_mask, en_padding_mask, vi_padding_mask = create_mask(en, vi)   
        y_pred = self(en, vi, en_mask, vi_mask, en_padding_mask, vi_padding_mask, en_padding_mask)

        loss = self.loss_fn(y_pred[:, :-1].reshape(-1, y_pred.size(-1)), vi[:, 1:].reshape(-1))
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        en, vi = batch
        en_mask, vi_mask, en_padding_mask, vi_padding_mask = create_mask(en, vi)
        y_pred = self(en, vi, en_mask, vi_mask, en_padding_mask, vi_padding_mask, en_padding_mask)
        
        loss = self.loss_fn(y_pred[:, :-1].reshape(-1, y_pred.size(-1)), vi[:, 1:].reshape(-1))
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        
        # Convert predictions to tokens
        _, top_indices = y_pred.topk(1, dim=2)
        predictions = top_indices.squeeze(-1)
        
        predictions_words = [[self.vocab.get_itos()[idx] for idx in pred] for pred in predictions]
        actuals_words = [[[self.vocab.get_itos()[idx] for idx in act]] for act in vi]
        
        print(f'preds: {predictions_words}')
        print(f'actuals: {actuals_words}')
        
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