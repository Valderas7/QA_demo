# Librerías
import logging
from core.constants import Constants
from core.dependencies import get_services, Services
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from typing import Annotated

# Se obtiene el logger para este módulo
logger = logging.getLogger(__name__)

# Se crea un enrutador
router = APIRouter()


@router.post(
    "/ingest",
    tags=["Ingesta"],
    responses={
        200: {"description": "PDF ingestado e indexado correctamente."},
        400: {"description": "Archivo inválido o demasiado grande."},
        413: {"description": "Archivo demasiado grande, excede el límite."},
        415: {"description": "Archivo no soportado, se requiere PDF."},
        500: {"description": "Error procesando el PDF."}
    }
)
async def ingest_pdf(
    file: Annotated[UploadFile, File(...)],
    services: Annotated[Services, Depends(get_services)]
):
    """
    Endpoint para ingestar un PDF y crear un índice vectorial con los chunks
    del PDF. El endpoint recibe un archivo PDF, lo procesa para extraer su
    contenido, lo divide en chunks y luego guarda esos chunks en una base de
    datos vectorial para su posterior consulta.
    """
    # Se intenta...
    try:

        # Se lee el contenido del archivo PDF
        content = await file.read()

        # Se valida que el archivo sea un PDF
        if (
            file.content_type != Constants.PDF_CONTENT_TYPE
            or not content.startswith(b"%PDF")
        ):
            logger.error("El archivo subido no es un PDF: %s", file.filename)
            raise HTTPException(
                status_code=415,
                detail="Solo se permiten archivos PDF"
            )

        # Si el archivo es demasiado grande, se lanza una excepción para evitar
        # problemas de rendimiento o seguridad
        if len(content) > Constants.MAX_FILE_SIZE:
            logger.error(
                "PDF demasiado grande: %s (%d bytes)",
                file.filename,
                len(content)
            )
            raise HTTPException(
                status_code=413,
                detail="Archivo demasiado grande"
            )
        
        # Se almacena el nombre del documento de la petición
        source = file.filename

        # Comprobamos en ambos servicios
        already_in_semantic = await run_in_threadpool(
            services.semantic_search.has_source,
            source
        )
        already_in_lexical = await run_in_threadpool(
            services.lexical_search.has_source,
            source
        )

        # Si el PDF ya existe completamente indexado en ambos servicios...
        if already_in_semantic and already_in_lexical:

            # Se omite la ingesta
            logger.info(f"El PDF '{source}' ya estaba indexado. Se omite.")
            return {
                "status": "Ya indexado",
                "message": f"El documento '{source}' ya existe en el índice.",
                "pages": 0,
                "chunks": 0
            }

        # Mensaje de información
        logger.info(f"Indexando nuevo PDF: '{source}'.")

        # Se llama al método para ingestar PDFs, obteniendo una lista de
        # páginas con su texto limpio y metadatos de página y fuente
        pages = await run_in_threadpool(
            services.ingestion.load_pdf,
            content,
            source
        )

        # Se fragmenta el texto del PDF en chunks utilizando el servicio de
        # chunking, obteniendo una lista de Documentos de Langchain
        chunks = await run_in_threadpool(
            services.chunking.chunk_text,
            pages
        )

        # Se guarda en la base de datos vectorial los embeddings a partir de
        # los chunks obtenidos utilizando el servicio de vector store
        await run_in_threadpool(
            services.semantic_search.add_chunks,
            chunks
        )

        # Se guarda en el índice BM25 los chunks obtenidos utilizando el
        # servicio de búsqueda léxica
        await run_in_threadpool(
            services.lexical_search.add_chunks,
            chunks
        )

        # Devuelve un mensaje de éxito con el número de páginas procesadas, el
        # número de chunks generados y el estado de la ingesta
        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "status": "Indexado"
        }

    # Si se lanza una excepción HTTP, se vuelve a lanzar para que FastAPI
    # devuelva la respuesta adecuada al cliente
    except HTTPException:
        raise

    # Excepción genérica
    except Exception:
        logger.exception("Error ingestando el documento.")
        raise HTTPException(
            status_code=500,
            detail="Error procesando el PDF"
        )



