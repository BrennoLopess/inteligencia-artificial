# 00_segmentar_txt_em_chunks.py

"""
Este script segmenta arquivos .txt em pequenos blocos (chunks) com base na estrutura numérica dos títulos.
Ideal para transformar documentos institucionais em dados utilizáveis por IA (como RAG e busca vetorial).
"""

import os
import re

# 🔹 Função que segmenta com base em seções numeradas (1., 2., 3.1 etc.)
def segmentar_por_numeros(texto, nome_documento):
    padrao_secao = re.compile(r'^(\d+)\.\s(.+)$', re.MULTILINE)
    padrao_subsecao = re.compile(r'^(\d+\.\d+)\s')

    chunks = []
    chunk_atual = []
    titulo_atual = None

    linhas = texto.split("\n")

    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue

        match_secao = padrao_secao.match(linha)
        if match_secao:
            if chunk_atual and len("\n".join(chunk_atual)) > 100:
                chunks.append((titulo_atual, "\n".join(chunk_atual)))

            titulo_atual = f"{match_secao.group(1)} - {match_secao.group(2)}"
            chunk_atual = [linha]
            continue

        if padrao_subsecao.match(linha) or (chunk_atual and len(chunk_atual[-1]) < 200):
            chunk_atual.append(linha)
        else:
            if len("\n".join(chunk_atual)) > 100:
                chunks.append((titulo_atual, "\n".join(chunk_atual)))
                chunk_atual = [linha]

    if chunk_atual and len("\n".join(chunk_atual)) > 100:
        chunks.append((titulo_atual, "\n".join(chunk_atual)))

    # Remover chunks muito pequenos
    return [(titulo, conteudo) for titulo, conteudo in chunks if titulo and len(conteudo) > 100]

# 🔹 Lista de arquivos de exemplo para segmentar
arquivos_txt = [
    "documento_exemplo_1.txt",
    "documento_exemplo_2.txt"
]

# 🔹 Pasta com os arquivos .txt de entrada
pasta_entrada = "exemplo_txts"

# 🔹 Pasta de saída com os arquivos segmentados (chunks)
pasta_saida = "exemplo_chunks"
os.makedirs(pasta_saida, exist_ok=True)

# 🔹 Processar cada documento da lista
for arquivo in arquivos_txt:
    caminho_entrada = os.path.join(pasta_entrada, arquivo)
    nome_documento = arquivo.replace(".txt", "").replace("_", " ")

    if os.path.exists(caminho_entrada):
        with open(caminho_entrada, "r", encoding="utf-8") as f:
            texto = f.read()

        chunks = segmentar_por_numeros(texto, nome_documento)

        nome_base = os.path.splitext(arquivo)[0]
        caminho_saida = os.path.join(pasta_saida, f"{nome_base}_chunks.txt")

        with open(caminho_saida, "w", encoding="utf-8") as f:
            for i, (titulo, chunk) in enumerate(chunks):
                f.write(f"[CHUNK {i}] {titulo}\n\n")
                f.write(f"{chunk}\n\n")

        print(f"✅ Segmentado: {caminho_saida}")
    else:
        print(f"❌ Arquivo não encontrado: {caminho_entrada}")
