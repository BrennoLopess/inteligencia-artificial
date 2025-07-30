# 03_gerar_resposta_claude.py

"""
Este script realiza busca semântica (RAG) e utiliza a IA Claude 3 Haiku via OpenRouter para gerar respostas baseadas em documentos.
"""

import os
import json
import faiss
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔹 Modelo de embeddings
modelo_embeddings = "intfloat/multilingual-e5-base"
model = SentenceTransformer(modelo_embeddings)

# 🔹 Carregar índice FAISS e verificar dimensões
index = faiss.read_index("faiss_index.idx")
embedding_dim = model.get_sentence_embedding_dimension()
if embedding_dim != index.d:
    raise ValueError(f"🚨 Dimensão incompatível: embeddings={embedding_dim}, FAISS={index.d}")

# 🔹 Carregar os textos indexados
textos = []
with open("textos_indexados.txt", "r", encoding="utf-8") as f:
    for linha in f:
        partes = linha.strip().split("\t", 1)
        if len(partes) > 1:
            textos.append(partes[1])

# 🔍 Buscar trechos relevantes com base na pergunta
def buscar_no_faiss(pergunta, k=5):
    entrada = f"query: {pergunta}"
    vetor = model.encode(entrada, convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(np.array([vetor], dtype=np.float32), k)
    return [textos[i] for i in I[0] if i < len(textos) and len(textos[i]) > 100]

# 🤖 Gerar resposta com Claude 3 Haiku (OpenRouter)
def gerar_resposta(pergunta):
    trechos = buscar_no_faiss(pergunta, k=5)

    if not trechos:
        return "❌ Nenhum trecho relevante encontrado. Tente reformular a pergunta."

    contexto = "\n\n".join(trechos)[:4000]  # Limite de segurança

    prompt = (
        "Você é um assistente virtual treinado para responder dúvidas com base em documentos internos. "
        "Use apenas os trechos abaixo como base de conhecimento e responda de forma objetiva, simpática e 100% em português.\n\n"
        f"### DOCUMENTAÇÃO:\n{contexto}\n\n"
        f"### PERGUNTA:\n{pergunta}\n\n"
        "### RESPOSTA:"
    )

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "anthropic/claude-3-haiku",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
        "temperature": 0.1
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"❌ Erro {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"❌ Erro de conexão: {str(e)}"

# 🧪 Teste manual (remova ao integrar em produção)
if __name__ == "__main__":
    perguntas_teste = [
        "Quais são as regras para solicitar o vale alimentação?",
        "Como funciona o plano odontológico da empresa?",
        "Existe auxílio creche? Como solicito?",
        "Faltas justificadas precisam de atestado?"
    ]

    for pergunta in perguntas_teste:
        resposta = gerar_resposta(pergunta)
        print(f"\n🔍 Pergunta: {pergunta}\n📝 Resposta: {resposta}\n" + "="*80)
