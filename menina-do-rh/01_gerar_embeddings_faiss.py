# 01_gerar_embeddings_faiss.py

"""
Gera embeddings vetoriais a partir de textos segmentados e os indexa usando FAISS.
Modelo usado: intfloat/multilingual-e5-base (Sentence Transformers).
Esta versão é genérica e não utiliza documentos reais.
"""

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔹 Modelo de embeddings utilizado
modelo_embeddings = "intfloat/multilingual-e5-base"
model = SentenceTransformer(modelo_embeddings)

# 🔹 Verificar a dimensão dos embeddings
embedding_dim = model.get_sentence_embedding_dimension()
print(f"✅ Dimensão dos embeddings: {embedding_dim}")

# 🔹 Pasta contendo os arquivos de chunks (simulados)
pasta_chunks = "exemplo_chunks"

# 🔹 Armazenar textos e embeddings gerados
textos = []
embeddings = []

# 🔹 Iterar sobre cada arquivo da pasta e gerar embeddings
for nome_arquivo in os.listdir(pasta_chunks):
    caminho_arquivo = os.path.join(pasta_chunks, nome_arquivo)

    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        chunks = f.read().strip().split("\n\n")

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        textos.append(chunk)

        entrada_modelo = f"passage: {chunk}"
        vetor = model.encode(entrada_modelo, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(vetor)

# 🔹 Converter lista de embeddings para array NumPy
embeddings_np = np.array(embeddings, dtype=np.float32)

# 🔹 Criar índice FAISS (Inner Product para embeddings normalizados)
index = faiss.IndexFlatIP(embedding_dim)
index.add(embeddings_np)

# 🔹 Salvar índice FAISS no disco
faiss.write_index(index, "faiss_index.idx")

# 🔹 Salvar os textos associados aos embeddings
with open("textos_indexados.txt", "w", encoding="utf-8") as f:
    for i, texto in enumerate(textos):
        f.write(f"{i}\t{texto.replace('\n', ' ')}\n")

print("✅ Embeddings gerados e indexados com sucesso!")
