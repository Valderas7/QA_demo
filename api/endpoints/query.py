# Librerías
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.ingestion.ingest import IngestService
from services.ingestion.chunking import ChunkService
from services.ingestion.embedding import EmbeddingService
from services.ingestion.vector_store import VectorStoreService

# Se obtiene el logger para este módulo
logger = logging.getLogger(__name__)

# Se crea un enrutador
router = APIRouter()

# @router.post("/query", tags="Consulta")
# async def query_rag(payload: ):

#     try:
#         question = payload.question

#         # 1. Retrieval (top-k)
#         docs = vectorstore.search(
#             query=question,
#             k=payload.top_k
#         )

#         if not docs:
#             return {
#                 "answer": "No se encontraron documentos relevantes.",
#                 "sources": []
#             }

#         # 2. Reranking (mejora precisión)
#         reranked_docs = reranker.rerank(
#             query=question,
#             docs=docs,
#             top_k=3
#         )

#         # 3. Construcción de contexto con citas
#         context = "\n\n".join([
#             f"[Fuente: {d.metadata.get('source')} | pág {d.metadata.get('page')}]\n{d.page_content}"
#             for d in reranked_docs
#         ])

#         # 4. Prompt al LLM
#         prompt = f"""
# Responde usando SOLO el contexto.

# Contexto:
# {context}

# Pregunta:
# {question}

# Reglas:
# - Cita fuente y página
# - Si no está en el contexto, di "no encontrado"
# """

#         # 5. Generación LLM
#         answer = llm.invoke(prompt)

#         # 6. Formato de salida estructurado
#         return {
#             "question": question,
#             "answer": answer,
#             "sources": [
#                 {
#                     "source": d.metadata.get("source"),
#                     "page": d.metadata.get("page")
#                 }
#                 for d in reranked_docs
#             ]
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error en query: {str(e)}"
#         )