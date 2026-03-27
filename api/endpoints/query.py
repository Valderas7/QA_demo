# Librerías
import logging
from core.dependencies import get_services, Services
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

# Se obtiene el logger para este módulo
logger = logging.getLogger(__name__)

# Se crea un enrutador
router = APIRouter()


# Endpoint para hacer consulta
@router.get("/query", tags=["Consulta"])
async def query(
    query: str,
    services: Services = Depends(get_services)
):
    """
    Endpoint para hacer una consulta. El endpoint recibe una consulta de
    texto, realiza una búsqueda semántica en la base de datos vectorial
    para recuperar los chunks más relevantes, los reordena utilizando un
    modelo de reranking para priorizar los más relevantes, y luego genera
    una respuesta utilizando un modelo de lenguaje que sintetiza la
    información de los chunks reordenados para responder a la consulta.
    """
    # Se intenta...
    try:

        # Se realiza una búsqueda semántica en la base de datos vectorial
        # obteniendo los 10 chunks más relevantes para la consulta
        docs = services.vectorstore.search(query, k=10)

        # Se reordenan los chunks obtenidos utilizando el servicio de
        # reranking, priorizando los más relevantes para la consulta y
        # quedándose con los 3 mejores para generar la respuesta
        reranked = services.reranker.rerank(query, docs, top_k=3)

        # Se genera una respuesta utilizando el servicio de LLM, que sintetiza
        # la información de los chunks reordenados para responder a la
        # consulta
        answer = services.llm.generate(query, reranked)

        # Se devuelve la respuesta generada junto con las fuentes de los chunks
        # reordenados, incluyendo la página y la fuente de cada chunk para que el
        # usuario pueda verificar la información si lo desea
        return {
            "answer": answer,
            "sources": [
                {
                    "page": chunk.metadata["page"],
                    "source": chunk.metadata["source"]
                }
                for chunk in reranked
            ]
        }

    # Excepción
    except Exception as e:
        logger.exception(f"Error procesando consulta: {e}.")
        raise HTTPException(
            status_code=500,
            detail="Error procesando la consulta"
        )