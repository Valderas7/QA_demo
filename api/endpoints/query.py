# Librerías
import asyncio
import logging
from core.dependencies import get_services, Services
from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, Depends, HTTPException

# Se obtiene el logger para este módulo
logger = logging.getLogger(__name__)

# Se crea un enrutador
router = APIRouter()


@router.post(
    "/query",
    tags=["Consulta"],
    responses={
        200: {"description": "Respuesta generada correctamente."},
        400: {"description": "Consulta inválida."},
        500: {"description": "Error procesando la consulta."},
        504: {"description": "Tiempo de espera agotado para generar la respuesta."}
    }
)
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

        # Si no se recibe una consulta o la consulta es demasiado larga, se
        # lanza una excepción
        if not query or len(query) > 1000:
            raise HTTPException(400, "Consulta inválida")

        # Se realiza una búsqueda semántica en la base de datos vectorial
        # obteniendo los 10 chunks más relevantes para la consulta
        docs = await run_in_threadpool(
            services.vectorstore.search,
            query,
            10
        )

        # Se reordenan los chunks obtenidos utilizando el servicio de
        # reranking, priorizando los más relevantes para la consulta y
        # quedándose con los 3 mejores para generar la respuesta
        reranked = await run_in_threadpool(
            services.reranker.rerank,
            query,
            docs,
            3
        )

        # Se genera una respuesta utilizando el servicio de LLM, que sintetiza
        # la información de los chunks reordenados para responder a la
        # consulta, esperando un máximo de 5 minutos para evitar que el proceso
        # se quede colgado indefinidamente
        answer = await asyncio.wait_for(
            run_in_threadpool(
                services.llm.generate,
                query,
                reranked
            ),
            timeout=600
        )

        # Se devuelve la respuesta generada junto con las fuentes de los chunks
        # reordenados, incluyendo la página y la fuente de cada chunk para que
        # el usuario pueda verificar la información si lo desea
        return {
            "answer": answer,
            "sources": [
                {
                    "text": chunk.page_content,
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata["page"]
                }
                for chunk in reranked
            ]
        }
    
    # Si el tiempo de espera para generar la respuesta se agota, se lanza una
    # excepción de timeout
    except asyncio.TimeoutError:
        logger.error("Tiempo de espera agotado para generar la respuesta.")
        raise HTTPException(
            status_code=504,
            detail="Tiempo de espera agotado para generar la respuesta"
        )

    # Excepción general para capturar cualquier error que ocurra durante
    # el procesamiento de la consulta
    except Exception as e:
        logger.exception(f"Error procesando consulta: {e}.")
        raise HTTPException(
            status_code=500,
            detail="Error procesando la consulta"
        )