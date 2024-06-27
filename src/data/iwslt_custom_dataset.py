import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
import re

def custom_vi_preprocess_pipeline(sentence, embed_model, tx): 
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence= sentence.lower().split()
    sentence.insert(0, '<SOS>')
    sentence.append('<EOS>')
    
    if len(sentence) > tx:
        sentence = sentence[:tx]
    else:
        sentence += ['<PAD>'] * (tx - len(sentence))
        
    sentence = [embed_model.key_to_index[word] if word in embed_model.key_to_index else embed_model.key_to_index['<UNK>'] for word in sentence]
    return torch.tensor(sentence, dtype=torch.long)

def custom_en_preprocess_pipeline(sentence, embed_model, tx):
    word2index = embed_model.word2index
    
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence= sentence.lower().split()
    sentence.insert(0, '<SOS>')
    sentence.append('<EOS>')
    
    if len(sentence) > tx:
        sentence = sentence[:tx]
    else:
        sentence += ['<PAD>'] * (tx - len(sentence))
        
    sentence = [word2index[word] if word in embed_model.vocab else 0 for word in sentence]

    return torch.tensor(sentence, dtype=torch.long)

class CustomDataset(Dataset):
    def __init__(self, data_path, en_embed_model, vi_embed_model, tx):
        data_frame = pd.read_csv(data_path)
        data_frame = data_frame.dropna()
        
        self.data = []
        for i in range(len(data_frame)):
            row_at_index = data_frame.iloc[i]
            en_sentence = custom_en_preprocess_pipeline(row_at_index['en'], en_embed_model, tx)
            vi_sentence = custom_vi_preprocess_pipeline(row_at_index['vi'], vi_embed_model, tx)
            self.data.append((en_sentence, vi_sentence))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        en_sentence, vi_sentence = self.data[index]
        
        return en_sentence, vi_sentence
    
    def get_data_loader(self, batch_size=32, shuffle=False, num_workers=3):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=3)
    
class CustomDatasetForT5(Dataset):
    def __init__(self, data_path, tx):
        self.data_frame = pd.read_csv(data_path)
        self.data_frame = self.data_frame.dropna()
        self.tx = tx
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
            
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        en_sentence = self.tokenizer("translate English to Vietnamese: " + row['en'], padding='max_length', max_length=self.tx, truncation=True, return_tensors="pt")
        vi_sentence = self.tokenizer(row['vi'], padding='max_length', max_length=self.tx, truncation=True, return_tensors="pt")
        
        # Shift the target_ids to create decoder_input_ids
        target_ids = vi_sentence['input_ids'].squeeze()
        decoder_input_ids = target_ids.clone()
        decoder_input_ids[1:] = target_ids[:-1]
        decoder_input_ids[0] = self.tokenizer.pad_token_id  # Usually, the start token for T5 is the padding token in the label space

        return {
            'input_ids': en_sentence['input_ids'].squeeze(),
            'attention_mask': en_sentence['attention_mask'].squeeze(),
            'decoder_input_ids': decoder_input_ids,
            'labels': target_ids
        }
    
    def get_data_loader(self, batch_size=32, shuffle=False, num_workers=3):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)