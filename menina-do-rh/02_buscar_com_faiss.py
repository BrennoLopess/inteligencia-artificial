# 02_buscar_com_faiss.py

"""
Este script permite realizar buscas semânticas usando FAISS + embeddings.
Ele recebe uma pergunta, gera o embedding da consulta e retorna os trechos mais similares.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔹 Modelo utilizado para os embeddings
modelo_embeddings = "intfloat/multilingual-e5-base"
model = SentenceTransformer(modelo_embeddings)

# 🔹 Carrega o índice FAISS salvo previamente
index = faiss.read_index("faiss_index.idx")

# 🔹 Valida se a dimensão do modelo bate com a do índice
embedding_dim = model.get_sentence_embedding_dimension()
if embedding_dim != index.d:
    raise ValueError(f"🚨 Dimensão incompatível: modelo={embedding_dim}, FAISS={index.d}")

# 🔹 Carrega os textos associados aos vetores do índice
textos_indexados = []
with open("textos_indexados.txt", "r", encoding="utf-8") as f:
    for linha in f:
        try:
            _, texto = linha.strip().split("\t", 1)
            textos_indexados.append(texto)
        except ValueError:
            continue

# 🔍 Função principal para buscar os trechos mais relevantes
def buscar_resposta(pergunta, k=5, threshold=0.85):
    """
    Realiza a busca vetorial no índice FAISS, filtrando por similaridade mínima (threshold).
    """
    entrada_modelo = f"query: {pergunta}"
    embedding_pergunta = model.encode(
        entrada_modelo,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).reshape(1, -1)

    distancias, indices = index.search(embedding_pergunta, k)

    resultados = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(textos_indexados) and distancias[0][i] >= threshold:
            resultados.append((textos_indexados[idx], distancias[0][i]))

    resultados = sorted(resultados, key=lambda x: x[1], reverse=True)

    # Exibir os resultados
    print("\n🔍 Resultados da busca:\n")
    if not resultados:
        print("❌ Nenhum trecho relevante encontrado.")
    else:
        for i, (trecho, score) in enumerate(resultados):
            print(f"[{i+1}] 📌 Trecho:\n{trecho}")
            print(f"🔹 Similaridade: {score:.4f}\n")

# ✅ Teste manual
if __name__ == "__main__":
    pergunta_teste = "Quais são as regras para solicitar o vale alimentação?"
    buscar_resposta(pergunta_teste)
