import re

def tokenizar(texto):
    texto = texto.lower()
    return re.findall(r"\b\w+\b|[^\w\s]", texto)

def construir_vocabulario(textos):
    palabras = set()
    for linea in textos:
        palabras.update(tokenizar(linea))
    vocab = sorted(list(palabras))
    return {w: i for i, w in enumerate(vocab)}, {i: w for w, i in zip(vocab, range(len(vocab)))}
