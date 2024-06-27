import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import corpus_bleu

class T5TranslationModule(pl.LightningModule):
    def __init__(self, lr=0.0003, optimizer_name='adam'):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], decoder_input_ids=batch['decoder_input_ids'], labels=batch['labels'])
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], decoder_input_ids=batch['decoder_input_ids'], labels=batch['labels'])
        val_loss = outputs.loss
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self.model.generate(input_ids=inputs, attention_mask=attention_mask, max_length=self.tokenizer.model_max_length)
        preds = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
        targets = [self.tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=True) for label in labels]

        # Log BLEU score
        references = [[target.split()] for target in targets]  # list of lists of reference sentences
        candidates = [pred.split() for pred in preds]  # list of candidate sentences

        bleu_score = corpus_bleu(references, candidates)
        self.log('bleu_score', bleu_score, prog_bar=True, logger=True)

        return {'bleu_score': bleu_score}
    
    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        return optimizer