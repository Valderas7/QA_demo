# Librerías
import logging
from core.constants import Constants
from core.dependencies import get_services, Services
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

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
    file: UploadFile = File(...),
    services: Services = Depends(get_services)
):
    """
    Endpoint para ingestar un PDF y crear un índice vectorial con los chunks
    del PDF. El endpoint recibe un archivo PDF, lo procesa para extraer su
    contenido, lo divide en chunks y luego guarda esos chunks en una base de
    datos vectorial para su posterior consulta.
    """
    # Se intenta...
    try:

        # Se valida que el archivo sea un PDF
        if not file.filename.endswith(".pdf"):
            logger.error("El archivo subido no es un PDF: %s", file.filename)
            raise HTTPException(
                status_code=415,
                detail="Solo se permiten archivos PDF"
            )
        
        # Se lee el contenido del archivo PDF
        content = await file.read()

        # Si el archivo es demasiado grande, se lanza una excepción para evitar
        # problemas de rendimiento o seguridad
        if len(content) > Constants.MAX_FILE_SIZE:
            logger.error("El PDF es demasiado grande.")
            raise HTTPException(413, "Archivo demasiado grande")

        # Se llama al método para ingestar PDFs, obteniendo una lista de
        # páginas con su texto limpio y metadatos de página y fuente
        pages = await run_in_threadpool(
            services.ingest.load_pdf,
            content,
            file.filename
        )

        # Se fragmenta el texto del PDF en chunks utilizando el servicio de
        # chunking, obteniendo una lista de Documentos de Langchain
        chunks = services.chunk.chunk_text(pages)

        # Se guarda en la base de datos vectorial los embeddings a partir de
        # los chunks obtenidos utilizando el servicio de vector store
        services.vectorstore.add(chunks)

        # Devuelve un mensaje de éxito con el número de páginas procesadas, el
        # número de chunks generados y el estado de la ingesta
        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "status": "indexado correctamente"
        }

    # Excepción
    except Exception as e:
        logger.exception("Error ingestando el documento.")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando PDF: {str(e)}"
        )



