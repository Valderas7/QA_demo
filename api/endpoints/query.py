# Librerías
from core.dependencies import vectorstore
from fastapi import APIRouter, HTTPException, Query
from services.ingestion.vector_store import VectorStoreService
from services.ingestion.embedding import EmbeddingService
from services.query.rag_retriever import Retriever
from services.query.reranker import Reranker
from services.query.llm_service import LLMService
import logging

# Se obtiene el logger para este módulo
logger = logging.getLogger(__name__)

# Se crea un enrutador
router = APIRouter()

# Inicializamos servicios
embedder = EmbeddingService()
retriever = Retriever(embedder, vectorstore)
reranker = Reranker()
llm_service = LLMService()


# Endpoint para hacer consulta
@router.get("/query", tags=["Consulta"])
async def query(
    q: str = Query(..., description="Consulta en lenguaje natural"),
    top_k: int = Query(5, description="Número de documentos a recuperar antes del reranking"),
    rerank_top_k: int = Query(3, description="Número de documentos finales tras reranking")
):
    """
    Endpoint para realizar consultas sobre los PDFs ingeridos.
    """
    # Se intenta...
    try:

        # Recupera chunks relevantes
        docs = retriever.retrieve(q, top_k=top_k)

        # Re-ranking con esos documentos de Langchain
        top_docs = reranker.rerank(q, docs, top_k=rerank_top_k)

        # Generar la respuesta
        response = llm_service.generate(q, top_docs)

        # Preparar salida (ya usa correctamente Document)
        doc_output = [
            {
                "text": d.page_content,
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "chunk_id": d.metadata.get("chunk_id")
            }
            for d in top_docs
        ]

        # Devuelve el diccionario de respuesta con la consulta, la respuesta y
        # los documentos de referencia con sus metadatos
        return {
            "query": q,
            "response": response,
            "documents": doc_output
        }

    # Excepción
    except Exception as e:
        logger.error(f"Error procesando consulta: {e}.")
        raise HTTPException(
            status_code=500,
            detail="Error procesando la consulta"
        )