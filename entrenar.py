import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from modelo import ModeloConversacional
from utils import tokenizar, construir_vocabulario

class ConversacionDataset(Dataset):
    def __init__(self, textos, word2idx, seq_len=5):
        self.secuencias = []
        for linea in textos:
            tokens = tokenizar(linea)
            ids = [word2idx[t] for t in tokens if t in word2idx]
            for i in range(len(ids) - seq_len):
                self.secuencias.append((ids[i:i+seq_len], ids[i+seq_len]))

    def __len__(self):
        return len(self.secuencias)

    def __getitem__(self, idx):
        entrada, objetivo = self.secuencias[idx]
        return torch.tensor(entrada), torch.tensor(objetivo)

# Cargar datos
with open("dataset.txt", "r", encoding="utf-8") as f:
    textos = [line.strip() for line in f.readlines() if line.strip()]

word2idx, idx2word = construir_vocabulario(textos)
dataset = ConversacionDataset(textos, word2idx)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

modelo = ModeloConversacional(len(word2idx))
criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

# Entrenamiento simple
for epoch in range(20):
    total_loss = 0
    for x, y in loader:
        salida, _ = modelo(x)
        loss = criterio(salida[:, -1, :], y)
        optimizador.zero_grad()
        loss.backward()
        optimizador.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(modelo.state_dict(), "modelo_ia.pth")
