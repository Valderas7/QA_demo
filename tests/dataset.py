# Librerías
import requests
from pathlib import Path

# Lista de nombres de PDFs y sus enlaces
pdfs = [
    ("attention.pdf", "https://arxiv.org/pdf/1706.03762.pdf"),      # Attention is All You Need
    ("bert.pdf",      "https://arxiv.org/pdf/1810.04805.pdf"),      # BERT
    ("rag_original.pdf", "https://arxiv.org/pdf/2005.11401.pdf"),   # Retrieval-Augmented Generation (original)
    ("ragprobe.pdf",  "https://arxiv.org/pdf/2409.19019.pdf"),      # Evaluación de RAG (2024)
    ("llama.pdf",     "https://arxiv.org/pdf/2302.13971.pdf"),      # Llama 2 paper
    ("eu_ai_act.pdf", "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689"),  # EU AI Act (oficial)
]

# Se crea el directorio de 'data/pdfs'
Path("data/pdfs").mkdir(parents=True, exist_ok=True)

# Para cada PDF, se hace una petición 'GET' para descargarlos y escribirlos en 'data/pdfs'
for name, url in pdfs:
    print(f"Descargando {name}...")
    r = requests.get(url)
    (Path("data/pdfs") / name).write_bytes(r.content)