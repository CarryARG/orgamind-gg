import torch
import torch.nn as nn

class ModeloConversacional(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super(ModeloConversacional, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden
