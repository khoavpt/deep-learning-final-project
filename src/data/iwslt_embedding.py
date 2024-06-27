import torch
import numpy as np
from gensim.models import KeyedVectors

class W2VModel:
    def __init__(self, w2v_path):
        self.model = KeyedVectors.load_word2vec_format( w2v_path, binary=True)
        self.embedding_dim = self.model.vectors.shape[1]
        unknown_vec = np.zeros(shape=(1, self.embedding_dim))
        padding_vec = np.zeros(shape=(1, self.embedding_dim))
        sos_vec = np.random.rand(1, self.embedding_dim)
        eos_vec = np.random.rand(1, self.embedding_dim)
        
        # Add '<PAD>' token
        self.model.add_vectors(['<PAD>'], padding_vec)
        # Add '<UNK>' token
        self.model.add_vectors(['<UNK>'], unknown_vec)
        # Add '<SOS>' token
        self.model.add_vectors(['<SOS>'], sos_vec)
        # Add '<EOS>' token
        self.model.add_vectors(['<EOS>'], eos_vec)
        
        self.vocab_size = len(self.model.key_to_index)
    
    def get_embedding_matrix(self):
        return torch.from_numpy(self.model.vectors)
    
class GloveEmbeddings():
    def __init__(self, path):
        self.vocab = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']
        self.embed_dict = {}
        with open(path) as f:
            for line in f:
                line = line.split()
                self.vocab.append(line[0])
                self.embed_dict[line[0]] = torch.Tensor([float(i) for i in line[1:]])        
        self.embedding_dim = len(self.embed_dict[self.vocab[-1]])
        self.vocab_size = len(self.vocab)
        self.word2index = {word: index for index, word in enumerate(self.vocab)}
        
        self.embed_dict['<PAD>'] = torch.zeros(size=(self.embedding_dim, ))
        self.embed_dict['<UNK>'] = torch.randn(size=(self.embedding_dim, ))
        self.embed_dict['<SOS>'] = torch.randn(size=(self.embedding_dim, ))
        self.embed_dict['<EOS>'] = torch.randn(size=(self.embedding_dim, ))
        
        
        self.embedding_matrix = None
    
    def get_embedding_matrix(self):
        if self.embedding_matrix == None:
            self.embedding_matrix = torch.zeros(self.vocab_size, self.embedding_dim)
    
            for idx, word in enumerate(self.vocab):
                self.embedding_matrix[idx] = self.embed_dict[word]
        return self.embedding_matrix
        