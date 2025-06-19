import torch
from modelo import ModeloConversacional
from utils import tokenizar, construir_vocabulario

with open("dataset.txt", "r", encoding="utf-8") as f:
    textos = [line.strip() for line in f.readlines() if line.strip()]

word2idx, idx2word = construir_vocabulario(textos)
modelo = ModeloConversacional(len(word2idx))
modelo.load_state_dict(torch.load("modelo_ia.pth"))
modelo.eval()

def responder(prompt, max_palabras=20):
    tokens = tokenizar(prompt)
    entrada = [word2idx.get(t, 0) for t in tokens][-5:]
    entrada = torch.tensor([entrada])
    hidden = None
    respuesta = tokens.copy()
    for _ in range(max_palabras):
        salida, hidden = modelo(entrada, hidden)
        siguiente_id = torch.argmax(salida[:, -1, :], dim=1).item()
        siguiente_palabra = idx2word.get(siguiente_id, "")
        respuesta.append(siguiente_palabra)
        entrada = torch.tensor([[siguiente_id]])
        if siguiente_palabra in [".", "?"]:
            break
    return " ".join(respuesta)

while True:
    entrada = input("Tú: ")
    print("IA:", responder(entrada))
