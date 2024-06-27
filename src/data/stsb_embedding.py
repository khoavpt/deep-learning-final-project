import torch

class GloveEmbeddings():
    def __init__(self, path):
        self.vocab = ['<UNK>', '<PAD>']
        self.embed_dict = {}
        with open(path) as f:
            for line in f:
                line = line.split()
                self.vocab.append(line[0])
                self.embed_dict[line[0]] = torch.Tensor([float(i) for i in line[1:]])        
        self.embedding_dim = len(self.embed_dict[self.vocab[-1]])
        self.vocab_size = len(self.vocab)
        self.word2index = {word: index for index, word in enumerate(self.vocab)}

        self.embed_dict['<UNK>'] = torch.zeros(size=(self.embedding_dim, ))
        self.embed_dict['<PAD>'] = torch.zeros(size=(self.embedding_dim, ))
        
        self.embedding_matrix = None
    
    def get_embedding_matrix(self):
        if self.embedding_matrix == None:
            self.embedding_matrix = torch.zeros(self.vocab_size, self.embedding_dim)
    
            for idx, word in enumerate(self.vocab):
                self.embedding_matrix[idx] = self.embed_dict[word]
        return self.embedding_matrix
        