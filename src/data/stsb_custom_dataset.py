import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, GPT2Tokenizer

def custom_preprocess_pipeline(sentence, embed_model, tx):
    word2index = embed_model.word2index
    
    sentence= sentence.lower().split()
    if len(sentence) > tx:
        sentence = sentence[:tx]
    else:
        sentence += ['<PAD>'] * (tx - len(sentence))
        
    sentence = [word2index[word] if word in embed_model.vocab else 0 for word in sentence]
    return torch.tensor(sentence, dtype=torch.int32)

class GLUESTSBDataset(Dataset):
    def __init__(self, data_path, embed_model, tx):
        data_frame = pd.read_csv(data_path, sep='\t', on_bad_lines='skip')
        data_frame = data_frame.dropna()
        
        self.data = []
        for i in range(len(data_frame)):
            row_at_index = data_frame.iloc[i]
            sentence1 = custom_preprocess_pipeline(row_at_index['sentence1'], embed_model, 10)
            sentence2 = custom_preprocess_pipeline(row_at_index['sentence2'], embed_model, 10)
            score = row_at_index['score'] / 5
            self.data.append((sentence1, sentence2, score))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence1, sentence2, score = self.data[index]
        return sentence1, sentence2, score

    def get_data_loader(self, batch_size=32, shuffle=False, num_workers=3):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=3)
    

class CustomDatasetForBERT(Dataset):
    def __init__(self, data_path, tx):
        data_frame = pd.read_csv(data_path, sep='\t', on_bad_lines='skip')
        data_frame = data_frame.dropna()
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)
        self.tx = tx
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data_frame = data_frame

    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        sentence1, sentence2, score = row['sentence1'], row['sentence2'], row['score']
        
        sentence1 = self.tokenizer(sentence1, padding='max_length', max_length=self.tx, truncation=True, return_tensors="pt")
        sentence2 = self.tokenizer(sentence2, padding='max_length', max_length=self.tx, truncation=True, return_tensors="pt")
        
        sentence1 = {key: val.squeeze() for key, val in sentence1.items()}
        sentence2 = {key: val.squeeze() for key, val in sentence2.items()}
        
        return sentence1, sentence2, score / 5.0

    def get_data_loader(self, batch_size=32, shuffle=False, num_workers=3):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=3)
    
class CustomDatasetForGPT(Dataset):
    def __init__(self, data_path, tx):
        data_frame = pd.read_csv(data_path, sep='\t', on_bad_lines='skip')
        data_frame = data_frame.dropna()
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)
        self.tx = tx
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<pad>')
        
        self.data_frame = data_frame
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        sentence1, sentence2, score = row['sentence1'], row['sentence2'], row['score']
        
        sentence1 = self.tokenizer(sentence1, padding='max_length', max_length=self.tx, truncation=True, return_tensors="pt")
        sentence2 = self.tokenizer(sentence2, padding='max_length', max_length=self.tx, truncation=True, return_tensors="pt")
        
        sentence1 = {key: val.squeeze() for key, val in sentence1.items()}
        sentence2 = {key: val.squeeze() for key, val in sentence2.items()}
        score = score / 5
        
        return sentence1, sentence2, score

    def get_data_loader(self, batch_size=32, shuffle=False, num_workers=3):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=3)